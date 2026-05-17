from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances


class Interpret:
    """Find words most associated with each PCA dimension."""

    def __init__(
        self,
        model,
        tags: List[str],
        dr: PCA,
        placement: pd.DataFrame,
        labels: List[str],
        rev1: bool = False,
        rev2: bool = False,
        min_count: int = 100,
        max_count: int = 1_000_000,
        max_features: int = 10_000,
    ):
        self.model = model
        self.pca = dr
        self.rev1 = rev1
        self.rev2 = rev2
        self.max_dim1 = float(placement["dim1"].max())
        self.min_dim1 = float(placement["dim1"].min())
        self.max_dim2 = float(placement["dim2"].max())
        self.min_dim2 = float(placement["dim2"].min())
        self.vocab = self._sorted_vocab(min_count, max_count, max_features)
        self.sims = self._compute_sims()

    def _sorted_vocab(self, min_count: int, max_count: int, max_features: int) -> List[str]:
        entries = []
        for word, idx in self.model.wv.key_to_index.items():
            count = self.model.wv.get_vecattr(word, "count")
            entries.append((word, count))
        entries.sort(key=lambda x: x[1], reverse=True)
        return [
            w for w, c in entries
            if min_count < c < max_count and w.count("_") < 3
        ][:max_features]

    def _compute_sims(self) -> pd.DataFrame:
        V = len(self.vocab)
        Z = np.zeros((V, 2))
        for i, w in enumerate(self.vocab):
            Z[i] = self.pca.transform(self.model.wv[w].reshape(1, -1))[0, :2]
        right_anchor = np.array([[self.max_dim1, 0]])
        left_anchor  = np.array([[self.min_dim1, 0]])
        up_anchor    = np.array([[0, self.max_dim2]])
        down_anchor  = np.array([[0, self.min_dim2]])
        return pd.DataFrame({
            "word":  self.vocab,
            "right": euclidean_distances(Z, right_anchor)[:, 0],
            "left":  euclidean_distances(Z, left_anchor)[:, 0],
            "up":    euclidean_distances(Z, up_anchor)[:, 0],
            "down":  euclidean_distances(Z, down_anchor)[:, 0],
        })

    def top_words_list(self, topn: int = 20) -> None:
        bar = "-" * 80
        d1_pos, d1_neg = ("left", "right") if self.rev1 else ("right", "left")
        d2_pos, d2_neg = ("down", "up")   if self.rev2 else ("up",   "down")
        for direction, label in [
            (d1_pos, "Positive (Right) on Dim 1"),
            (d1_neg, "Negative (Left) on Dim 1"),
            (d2_pos, "Positive (North) on Dim 2"),
            (d2_neg, "Negative (South) on Dim 2"),
        ]:
            words = self.sims.sort_values(direction).word.tolist()[:topn]
            print(bar)
            print(f"Words associated with {label}:")
            print(bar)
            print(", ".join(w.replace("_", " ") for w in words))
        print(bar)
