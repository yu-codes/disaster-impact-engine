"""
Combined Similarity — 結合 KNN 特徵距離 + DTW 路徑距離

final_score = alpha * knn_score + (1 - alpha) * dtw_score

這是 strategy.md 第六步的實作。
"""

import numpy as np
from .base import SimilarityBase, SimilarityResult
from .knn import KNNSimilarity
from .dtw import DTWSimilarity


class CombinedSimilarity(SimilarityBase):
    """
    複合相似度 = α * 特徵距離 + (1-α) * DTW 路徑距離
    """

    def __init__(
        self,
        alpha: float = 0.5,
        feature_weights: np.ndarray | None = None,
        dtw_weights: np.ndarray | None = None,
    ):
        """
        Args:
            alpha: 特徵距離的權重 (0~1)，剩餘給 DTW
            feature_weights: KNN 特徵權重
            dtw_weights: DTW 各維度權重
        """
        self.alpha = alpha
        self.knn = KNNSimilarity(feature_weights=feature_weights)
        self.dtw = DTWSimilarity(dtw_weights=dtw_weights)
        self._ids: list[str] = []
        self._features_dict = {}

    def fit(self, feature_dict: dict):
        self._features_dict = feature_dict
        self._ids = list(feature_dict.keys())
        self.knn.fit(feature_dict)
        self.dtw.fit(feature_dict)
        print(f"✓ Combined Similarity 已擬合（α={self.alpha}）")

    def compute_distance(self, id_a: str, id_b: str) -> float:
        knn_d = self.knn.compute_distance(id_a, id_b)
        dtw_d = self.dtw.compute_distance(id_a, id_b)
        return self.alpha * knn_d + (1 - self.alpha) * dtw_d

    def find_similar(
        self, query_id: str, k: int = 5, exclude_self: bool = True
    ) -> SimilarityResult:
        if query_id not in self._features_dict:
            raise KeyError(f"找不到颱風：{query_id}")

        # 取 KNN 和 DTW 的候選（各取 2*k，合併後取 top-k）
        pool_size = min(len(self._ids), k * 3)
        knn_result = self.knn.find_similar(query_id, pool_size, exclude_self)
        dtw_result = self.dtw.find_similar(query_id, pool_size, exclude_self)

        # 合併候選
        candidates = set(knn_result.similar_ids + dtw_result.similar_ids)
        if exclude_self:
            candidates.discard(query_id)

        # 計算組合距離
        # 先歸一化各自的距離
        knn_dists = {
            tid: self.knn.compute_distance(query_id, tid) for tid in candidates
        }
        dtw_dists = {
            tid: self.dtw.compute_distance(query_id, tid) for tid in candidates
        }

        knn_max = max(knn_dists.values()) if knn_dists else 1.0
        dtw_max = max(dtw_dists.values()) if dtw_dists else 1.0

        combined = {}
        for tid in candidates:
            norm_knn = knn_dists[tid] / (knn_max + 1e-8)
            norm_dtw = dtw_dists[tid] / (dtw_max + 1e-8)
            combined[tid] = self.alpha * norm_knn + (1 - self.alpha) * norm_dtw

        sorted_items = sorted(combined.items(), key=lambda x: x[1])[:k]

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
