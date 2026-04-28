"""
DTW 相似度 — 基於 impact window 時序路徑的 Dynamic Time Warping

distance function:
    d = w1*(Δr)² + w2*(Δθ)² + w3*(Δwind)² + w4*(Δpressure)²
"""

import numpy as np
from .base import SimilarityBase, SimilarityResult


def _dtw_distance(
    seq1: np.ndarray, seq2: np.ndarray, weights: np.ndarray | None = None
) -> float:
    """
    計算兩個多維時序的 DTW 距離

    Args:
        seq1: (T1, D) 時序矩陣
        seq2: (T2, D) 時序矩陣
        weights: (D,) 各維度的權重

    Returns:
        DTW 距離
    """
    n, m = len(seq1), len(seq2)
    if n == 0 or m == 0:
        return float("inf")

    if weights is None:
        weights = np.ones(seq1.shape[1])

    # 標準化權重
    weights = weights / np.sum(weights)

    # 計算成本矩陣
    cost = np.full((n + 1, m + 1), float("inf"))
    cost[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diff = seq1[i - 1] - seq2[j - 1]
            d = np.sum(weights * diff**2)
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])

    # 歸一化（除以路徑長度）
    path_len = n + m
    return float(np.sqrt(cost[n, m] / path_len))


class DTWSimilarity(SimilarityBase):
    """
    基於 DTW 的時序路徑相似度

    比較 impact window 內的 [r, theta, wind, pressure] 序列
    """

    def __init__(self, dtw_weights: np.ndarray | None = None):
        """
        Args:
            dtw_weights: DTW 各維度權重 [w_r, w_theta, w_wind, w_pressure]
        """
        if dtw_weights is None:
            self.dtw_weights = np.array([1.0, 1.0, 1.0, 1.0])
        else:
            self.dtw_weights = np.array(dtw_weights)

        self._features_dict = {}
        self._ids: list[str] = []
        self._matrices: dict[str, np.ndarray] = {}
        self._distance_cache: dict[tuple, float] = {}

    def fit(self, feature_dict: dict):
        self._features_dict = feature_dict
        self._ids = list(feature_dict.keys())
        self._distance_cache = {}

        # 預先標準化 impact window 矩陣
        all_data = []
        for tid in self._ids:
            mat = feature_dict[tid].get_impact_window_matrix()
            all_data.append(mat)

        # 計算全域均值和標準差
        concatenated = np.vstack(all_data)
        self._mean = concatenated.mean(axis=0)
        self._std = concatenated.std(axis=0) + 1e-8

        for tid, mat in zip(self._ids, all_data):
            self._matrices[tid] = (mat - self._mean) / self._std

        print(f"✓ DTW 已擬合 {len(self._ids)} 筆颱風（impact window 路徑）")

    def compute_distance(self, id_a: str, id_b: str) -> float:
        key = tuple(sorted([id_a, id_b]))
        if key not in self._distance_cache:
            mat_a = self._matrices[id_a]
            mat_b = self._matrices[id_b]
            dist = _dtw_distance(mat_a, mat_b, self.dtw_weights)
            self._distance_cache[key] = dist
        return self._distance_cache[key]

    def find_similar(
        self, query_id: str, k: int = 5, exclude_self: bool = True
    ) -> SimilarityResult:
        if query_id not in self._matrices:
            raise KeyError(f"找不到颱風：{query_id}")

        distances = {}
        for tid in self._ids:
            if exclude_self and tid == query_id:
                continue
            distances[tid] = self.compute_distance(query_id, tid)

        sorted_items = sorted(distances.items(), key=lambda x: x[1])[:k]

        result_ids = [item[0] for item in sorted_items]
        result_dists = [item[1] for item in sorted_items]
        max_d = max(result_dists) if result_dists else 1.0
        scores = [1.0 - d / (max_d + 1e-8) for d in result_dists]

        return SimilarityResult(
            query_id=query_id,
            similar_ids=result_ids,
            distances=result_dists,
            scores=scores,
        )
