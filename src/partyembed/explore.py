from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec
from sklearn.decomposition import PCA

from partyembed.utils.guided import custom_projection_2D, custom_projection_1D
from partyembed.utils.interpret import Interpret
from partyembed.utils.issues import issue_ownership
from partyembed.utils.polarization import polarization_metric

logger = logging.getLogger(__name__)

_DEFAULT_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def _default_group_fn(tag: str) -> str:
    """Extract group from a tag by taking everything before the last separator."""
    return tag.rsplit("_", 1)[0] if "_" in tag else tag


def _default_time_fn(tag: str) -> str:
    """Extract time period from a tag by taking everything after the last separator."""
    return tag.rsplit("_", 1)[1] if "_" in tag else ""


class Explore:
    """Visualise and analyse ideological placement from a fitted Doc2Vec model.

    Parameters
    ----------
    model:
        A trained :class:`gensim.models.doc2vec.Doc2Vec` instance, or a path
        string to load one with ``Doc2Vec.load``.
    tags:
        Document tags to include in the analysis.  If ``None``, every tag in
        the model's document-vector table is used.
    group_fn:
        Maps a tag to a *group key* (e.g. ``'D_113'`` -> ``'D'``).
        Defaults to everything before the last ``'_'``.
    time_fn:
        Maps a tag to a *time label* (e.g. ``'D_113'`` -> ``'113'``).
        Defaults to everything after the last ``'_'``.
    group_labels:
        Human-readable label for each group key
        (e.g. ``{'D': 'Democrats', 'R': 'Republicans'}``).
    group_colors:
        Matplotlib colour for each group key
        (e.g. ``{'D': 'blue', 'R': 'red'}``).
    method:
        Dimensionality-reduction method: ``'pca'`` or ``'guided'``.
    dimensions:
        Number of dimensions for PCA (ignored for guided projection, which
        always returns 2).
    orientation:
        Dict specifying which tag should be *left / right / bottom / top* for
        axis orientation, e.g.::

            {'dim1': ('D_2015', 'R_2015'),   # (left_tag, right_tag)
             'dim2': ('R_2015', 'D_2015')}   # (bottom_tag, top_tag)

    custom_lexicon:
        Lexicon used with ``method='guided'``.  Must be a list of 4 word lists
        for 2-D projection, or 2 word lists for 1-D projection.

    Examples
    --------
    >>> from partyembed import Explore
    >>> m = Explore('models/mymodel', group_labels={'D': 'Democrats', 'R': 'Republicans'})
    >>> m.plot()
    >>> m.polarization(group1_prefix='D_', group2_prefix='R_')
    """

    def __init__(
        self,
        model,
        tags: Optional[List[str]] = None,
        group_fn: Callable[[str], str] = _default_group_fn,
        time_fn: Callable[[str], str] = _default_time_fn,
        group_labels: Optional[Dict[str, str]] = None,
        group_colors: Optional[Dict[str, str]] = None,
        method: str = "pca",
        dimensions: int = 2,
        orientation: Optional[Dict[str, Tuple[str, str]]] = None,
        custom_lexicon=None,
    ):
        if isinstance(model, str):
            self.model = Doc2Vec.load(model)
        elif isinstance(model, Doc2Vec):
            self.model = model
        else:
            raise TypeError("model must be a Doc2Vec instance or a path string.")

        self.M = self.model.vector_size
        self.method = method
        self.components = dimensions
        self.custom_lexicon = custom_lexicon
        self.group_fn = group_fn
        self.time_fn = time_fn
        self._orientation = orientation or {}

        # Resolve tags
        all_model_tags = list(self.model.dv.index_to_key)
        self.tags: List[str] = tags if tags is not None else all_model_tags
        # Keep only tags that actually exist in the model
        missing = [t for t in self.tags if t not in self.model.dv]
        if missing:
            logger.warning("Ignoring %d tags not found in model: %s", len(missing), missing[:5])
        self.tags = [t for t in self.tags if t in self.model.dv]

        self.P = len(self.tags)
        if self.P == 0:
            raise ValueError("No valid tags found in the model.")

        # Group/time metadata
        self.groups = [group_fn(t) for t in self.tags]
        self.times  = [time_fn(t)  for t in self.tags]

        unique_groups = list(dict.fromkeys(self.groups))  # stable order
        self.group_labels = group_labels or {g: g for g in unique_groups}
        # Build display labels: "Group Time" (e.g. "Democrats 2015")
        self.display_labels = [
            f"{self.group_labels.get(g, g)} {t}"
            for g, t in zip(self.groups, self.times)
        ]

        # Assign colors
        if group_colors:
            self.group_colors = group_colors
        else:
            self.group_colors = {
                g: _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)]
                for i, g in enumerate(unique_groups)
            }
        self.colors = [self.group_colors.get(g, "#333333") for g in self.groups]

        # Run dimensionality reduction
        self._dr: Optional[PCA] = None
        self._flip_dim1 = False
        self._flip_dim2 = False
        self.placement = self._reduce()

    # ------------------------------------------------------------------
    # Dimensionality reduction
    # ------------------------------------------------------------------

    def _reduce(self) -> pd.DataFrame:
        Z_raw = np.stack([self.model.dv[t] for t in self.tags])

        if self.method == "pca":
            self._dr = PCA(n_components=self.components)
            Z = self._dr.fit_transform(Z_raw)
        elif self.method == "guided":
            Z = custom_projection_2D(Z_raw, self.model, self.custom_lexicon)
        else:
            raise ValueError("method must be 'pca' or 'guided'.")

        df = pd.DataFrame(Z[:, :2], columns=["dim1", "dim2"])
        df["tag"] = self.tags
        df["group"] = self.groups
        df["time"] = self.times
        df["label"] = self.display_labels

        # Axis orientation
        tag_index = {t: i for i, t in enumerate(self.tags)}
        for dim, (left_tag, right_tag) in self._orientation.items():
            if left_tag in tag_index and right_tag in tag_index:
                li, ri = tag_index[left_tag], tag_index[right_tag]
                if dim == "dim1" and df.dim1.iloc[li] > df.dim1.iloc[ri]:
                    df["dim1"] *= -1
                    self._flip_dim1 = True
                if dim == "dim2" and df.dim2.iloc[li] > df.dim2.iloc[ri]:
                    df["dim2"] *= -1
                    self._flip_dim2 = True

        return df

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plot(
        self,
        axis_labels: Optional[Tuple[str, str]] = None,
        save_path: Optional[str] = None,
        xlim: Optional[Tuple[float, float]] = None,
        figsize: Tuple[int, int] = (22, 15),
    ) -> None:
        """Scatter-plot of party/group positions in embedding space."""
        import matplotlib as mpl
        import matplotlib.pyplot as plt

        mpl.rcParams.update({"axes.titlesize": 20, "axes.labelsize": 20, "font.size": 14})
        plt.figure(figsize=figsize)
        plt.scatter(self.placement.dim1, self.placement.dim2, color=self.colors)
        for label, x, y, c in zip(
            self.display_labels, self.placement.dim1, self.placement.dim2, self.colors
        ):
            plt.annotate(
                label,
                xy=(x, y),
                xytext=(-20, 20),
                textcoords="offset points",
                ha="right",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.5", fc=c, alpha=0.3),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
            )
        if xlim:
            plt.xlim(xlim)
        xlabel, ylabel = axis_labels if axis_labels else (
            ("Economic Left-Right", "Social Left-Right")
            if self.method == "guided"
            else ("Component 1", "Component 2")
        )
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if save_path:
            plt.savefig(save_path, dpi=600, bbox_inches="tight")
        plt.show()

    def plot_timeseries(
        self,
        dimension: int = 1,
        axis_labels: Optional[Tuple[str, str]] = None,
        save_path: Optional[str] = None,
        legend: str = "upper left",
        figsize: Tuple[int, int] = (22, 15),
    ) -> None:
        """Line plot of ideological scores over time, one line per group."""
        import matplotlib as mpl
        import matplotlib.pyplot as plt

        mpl.rcParams.update({"axes.titlesize": 20, "axes.labelsize": 20, "font.size": 14})
        dim_col = "dim1" if dimension == 1 else "dim2"
        df = self.placement.copy()
        df["color"] = self.colors

        try:
            df["_time_num"] = pd.to_numeric(df["time"])
        except (ValueError, TypeError):
            df["_time_num"] = df["time"]

        fig, ax = plt.subplots(figsize=figsize)
        for grp_key, grp_df in df.groupby("group"):
            grp_df = grp_df.sort_values("_time_num")
            color = self.group_colors.get(grp_key, "#333333")
            label = self.group_labels.get(grp_key, grp_key)
            grp_df.plot(ax=ax, kind="line", x="time", y=dim_col, linewidth=5, c=color, label=label)

        plt.legend(loc=legend)
        if axis_labels:
            plt.xlabel(axis_labels[0])
            plt.ylabel(axis_labels[1])
        else:
            plt.xlabel("Time period")
            plt.ylabel(
                "Ideological Placement (First Component)"
                if dimension == 1
                else "Second Component"
            )
        if save_path:
            plt.savefig(save_path, dpi=600, bbox_inches="tight")
        plt.show()

    def interpret(
        self,
        top_words: int = 20,
        min_count: int = 100,
        max_count: int = 1_000_000,
        max_features: int = 1_000_000,
    ) -> None:
        """Print words most associated with each dimension."""
        if self._dr is None:
            raise RuntimeError("interpret() requires method='pca'.")
        Interpret(
            self.model,
            self.tags,
            self._dr,
            self.placement,
            self.display_labels,
            rev1=self._flip_dim1,
            rev2=self._flip_dim2,
            min_count=min_count,
            max_count=max_count,
            max_features=max_features,
        ).top_words_list(top_words)

    def polarization(
        self,
        group1_prefix: str = "",
        group2_prefix: str = "",
    ) -> pd.DataFrame:
        """Return pairwise Euclidean distance between two groups over time.

        Parameters
        ----------
        group1_prefix / group2_prefix:
            Prefixes to filter tags belonging to each of the two groups
            (e.g. ``'D_'`` and ``'R_'``).  When both are empty, all tags
            within each unique group are compared pairwise.
        """
        return polarization_metric(
            self.model,
            self.tags,
            group1_prefix=group1_prefix,
            group2_prefix=group2_prefix,
            time_fn=self.time_fn,
        )

    def issue(
        self,
        topic_word: str,
        n: int = 20,
        bootstrap: bool = True,
        n_boot: int = 1000,
        smooth: int = 0,
    ) -> pd.DataFrame:
        """Measure how closely each group aligns with a topic over time.

        Parameters
        ----------
        topic_word:
            Seed word defining the topic (must be in the model vocabulary).
        n:
            Number of words used to build the topic vector.
        bootstrap:
            Compute 95 % confidence intervals via bootstrapping.
        n_boot:
            Number of bootstrap resamples.
        smooth:
            Rolling-window size over time periods (0 = no smoothing).

        Returns
        -------
        DataFrame with columns ``tag``, ``group``, ``time``, ``similarity``
        and (if ``bootstrap=True``) ``lb``, ``ub``.

        Examples
        --------
        >>> df = m.issue('healthcare', n=30)
        >>> df.pivot(index='time', columns='group', values='similarity')
        """
        return issue_ownership(
            self.model,
            self.tags,
            topic_word=topic_word,
            group_fn=self.group_fn,
            time_fn=self.time_fn,
            n=n,
            bootstrap=bootstrap,
            n_boot=n_boot,
            smooth=smooth,
        )

    def get_placement(self) -> pd.DataFrame:
        """Return a DataFrame with tag, group, time, dim1, dim2 columns."""
        return self.placement.copy()
