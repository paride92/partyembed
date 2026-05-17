from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances


def polarization_metric(
    model,
    tags: List[str],
    group1_prefix: str = "",
    group2_prefix: str = "",
    time_fn: Callable[[str], str] = lambda t: t.rsplit("_", 1)[-1],
) -> pd.DataFrame:
    """Compute Euclidean distance between two groups over time.

    Parameters
    ----------
    model:
        Fitted Doc2Vec model.
    tags:
        All document tags present in the model to consider.
    group1_prefix / group2_prefix:
        Filter tags by prefix to select each group.  When both are empty,
        the first two unique groups (derived by ``time_fn``) are compared.
    time_fn:
        Maps a tag to its time-period label.

    Returns
    -------
    DataFrame with columns ``time``, ``tag1``, ``tag2``, ``euclidean_distance``.
    """
    M = model.vector_size

    if group1_prefix or group2_prefix:
        g1_tags = sorted([t for t in tags if t.startswith(group1_prefix)])
        g2_tags = sorted([t for t in tags if t.startswith(group2_prefix)])
    else:
        # Auto-detect: pair tags that share the same time period
        from collections import defaultdict
        by_time: dict = defaultdict(list)
        for t in tags:
            by_time[time_fn(t)].append(t)
        # Use the first two groups found in the first time period
        first_period_tags = next(iter(by_time.values()))
        if len(first_period_tags) < 2:
            raise ValueError(
                "Cannot auto-detect two groups. Provide group1_prefix/group2_prefix."
            )
        prefix1 = first_period_tags[0].rsplit("_", 1)[0] + "_"
        prefix2 = first_period_tags[1].rsplit("_", 1)[0] + "_"
        g1_tags = sorted([t for t in tags if t.startswith(prefix1)])
        g2_tags = sorted([t for t in tags if t.startswith(prefix2)])

    # Match by time period
    time_to_g2 = {time_fn(t): t for t in g2_tags}
    rows = []
    for t1 in g1_tags:
        period = time_fn(t1)
        t2 = time_to_g2.get(period)
        if t2 is None:
            continue
        v1 = model.dv[t1].reshape(1, -1)
        v2 = model.dv[t2].reshape(1, -1)
        dist = float(euclidean_distances(v1, v2)[0, 0])
        rows.append({"time": period, "tag1": t1, "tag2": t2, "euclidean_distance": dist})

    return pd.DataFrame(rows)
