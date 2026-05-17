from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def _topic_vector(model, topic_word: str, n: int) -> np.ndarray:
    """Centroid of topic_word and its n-1 nearest neighbours."""
    M = model.vector_size
    vocab = model.wv.key_to_index
    if topic_word not in vocab:
        raise ValueError(f"'{topic_word}' not in model vocabulary.")
    neighbours = [w for w, _ in model.wv.most_similar(topic_word, topn=n - 1)]
    words = [topic_word] + neighbours
    mat = np.stack([model.wv[w] for w in words])
    return mat.mean(axis=0)


def _bootstrap_topic_vector(
    model, topic_word: str, n: int, n_boot: int
) -> np.ndarray:
    """Bootstrap distribution of topic centroids (shape: n_boot × M)."""
    M = model.vector_size
    if topic_word not in model.wv.key_to_index:
        raise ValueError(f"'{topic_word}' not in model vocabulary.")
    neighbours = [w for w, _ in model.wv.most_similar(topic_word, topn=n - 1)]
    words = [topic_word] + neighbours
    results = np.zeros((n_boot, M))
    for s in range(n_boot):
        sample = np.random.choice(words, size=n, replace=True)
        results[s] = np.stack([model.wv[w] for w in sample]).mean(axis=0)
    return results


def issue_ownership(
    model,
    tags: List[str],
    topic_word: str,
    group_fn: Callable[[str], str],
    time_fn: Callable[[str], str],
    n: int = 20,
    bootstrap: bool = True,
    n_boot: int = 1000,
    smooth: int = 0,
) -> pd.DataFrame:
    """Measure how closely each group's embedding aligns with a topic over time.

    For each document tag the function computes the cosine similarity between
    that group-period vector and a *topic vector* built from the ``n`` words
    most similar to ``topic_word`` in the embedding space.

    Parameters
    ----------
    model:
        Fitted Doc2Vec model.
    tags:
        Document tags to analyse (typically ``Explore.tags``).
    topic_word:
        Seed word defining the topic (must be in the model vocabulary).
    group_fn / time_fn:
        Same callables used in :class:`~partyembed.explore.Explore` to parse
        group and time period from each tag.
    n:
        Number of words used to build the topic vector (seed + n-1 neighbours).
    bootstrap:
        If ``True``, compute 95 % confidence intervals via bootstrapping.
    n_boot:
        Number of bootstrap resamples (ignored when ``bootstrap=False``).
    smooth:
        Rolling-window size for smoothing similarity scores over time.
        ``0`` means no smoothing.

    Returns
    -------
    DataFrame with columns ``tag``, ``group``, ``time``, ``similarity``,
    and (when ``bootstrap=True``) ``lb`` and ``ub`` for the 95 % CI.

    Examples
    --------
    >>> df = m.issue('sanita', n=30)
    >>> df.pivot(index='time', columns='group', values='similarity')
    """
    M = model.vector_size

    if bootstrap:
        topic_mat = _bootstrap_topic_vector(model, topic_word, n=n, n_boot=n_boot)
    else:
        topic_mat = _topic_vector(model, topic_word, n=n).reshape(1, -1)

    rows = []
    for tag in tags:
        vec = model.dv[tag].reshape(1, -1)
        sims = cosine_similarity(vec, topic_mat)[0]  # shape: (n_boot,) or (1,)
        row = {
            "tag": tag,
            "group": group_fn(tag),
            "time": time_fn(tag),
            "similarity": float(sims.mean()),
        }
        if bootstrap:
            row["lb"] = float(np.percentile(sims, 2.5))
            row["ub"] = float(np.percentile(sims, 97.5))
        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by group then time (numeric if possible)
    try:
        df["_t"] = pd.to_numeric(df["time"])
    except (ValueError, TypeError):
        df["_t"] = df["time"]
    df = df.sort_values(["group", "_t"]).drop(columns="_t").reset_index(drop=True)

    if smooth > 1:
        sim_cols = ["similarity"] + (["lb", "ub"] if bootstrap else [])
        df[sim_cols] = (
            df.groupby("group")[sim_cols]
            .transform(lambda s: s.rolling(smooth, min_periods=1, center=False).mean())
        )

    return df
