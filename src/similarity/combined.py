"""
Combined Similarity v2 — 整合 Rule-Based 前置篩選 + KNN + DTW

Pipeline:
1. Rule-based 分類 → 前置篩選（soft filter: 同類加權）
2. DTW 路徑相似度（物理標準化 + 環形距離）
3. KNN 特徵距離（摘要特徵）
4. final_score = 0.6 * DTW + 0.4 * KNN (同類獎勵)

改進：
- Rule-based 作為 soft filter（非 hard filter）
- DTW 權重提高（路徑形狀更重要）
- 同類別的颱風獲得距離折扣
"""

import numpy as np
from .base import SimilarityBase, SimilarityResult
from .knn import KNNSimilarity
from .dtw import DTWSimilarity


class CombinedSimilarity(SimilarityBase):
    """
    複合相似度 v2 = rule_based soft filter + DTW + KNN

    Pipeline:
    1. Rule-based 前置分類（同類加權）
    2. 0.6 * DTW_normalized + 0.4 * KNN_normalized
    3. 同類別折扣 0.85
    """

    def __init__(
        self,
        alpha: float = 0.2,
        feature_weights: np.ndarray | None = None,
        dtw_weights: np.ndarray | None = None,
        category_discount: float = 1.0,
    ):
        """
        Args:
            alpha: KNN 特徵距離的權重 (0~1)，DTW 權重 = 1 - alpha
            feature_weights: KNN 特徵權重
            dtw_weights: DTW 各維度權重
            category_discount: 同分類的距離折扣率
        """
        self.alpha = alpha
        self.category_discount = category_discount
        self.knn = KNNSimilarity(feature_weights=feature_weights)
        self.dtw = DTWSimilarity(dtw_weights=dtw_weights)
        self._ids: list[str] = []
        self._features_dict = {}
        self._categories: dict[str, str] = {}  # rule-based 分類結果

    def fit(self, feature_dict: dict, loader=None):
        self._features_dict = feature_dict
        self._ids = list(feature_dict.keys())
        self.knn.fit(feature_dict)
        self.dtw.fit(feature_dict)

        # 如果有 loader，計算 rule-based 分類
        if loader is not None:
            from .rule_based import classify_typhoon_by_rules

            for rec in loader.records:
                if rec.typhoon_id in feature_dict:
                    result = classify_typhoon_by_rules(rec.track, rec.landfall_location)
                    self._categories[rec.typhoon_id] = result["predicted_category"]

        print(f"✓ Combined v2 已擬合（α={self.alpha}, DTW={1-self.alpha}）")

    def compute_distance(self, id_a: str, id_b: str) -> float:
        knn_d = self.knn.compute_distance(id_a, id_b)
        dtw_d = self.dtw.compute_distance(id_a, id_b)
        combined = self.alpha * knn_d + (1 - self.alpha) * dtw_d

        # 同類折扣
        if (
            self._categories.get(id_a)
            and self._categories.get(id_b)
            and self._categories[id_a] == self._categories[id_b]
        ):
            combined *= self.category_discount

        return combined

    def find_similar(
        self, query_id: str, k: int = 5, exclude_self: bool = True
    ) -> SimilarityResult:
        if query_id not in self._features_dict:
            raise KeyError(f"找不到颱風：{query_id}")

        # Reciprocal Rank Fusion (RRF)
        pool_size = min(len(self._ids) - 1, max(k * 5, 50))
        knn_result = self.knn.find_similar(query_id, pool_size, exclude_self)
        dtw_result = self.dtw.find_similar(query_id, pool_size, exclude_self)

        # Build rank maps (rank 0 = best)
        knn_ranks = {tid: rank for rank, tid in enumerate(knn_result.similar_ids)}
        dtw_ranks = {tid: rank for rank, tid in enumerate(dtw_result.similar_ids)}

        # All candidates
        candidates = set(knn_result.similar_ids + dtw_result.similar_ids)
        if exclude_self:
            candidates.discard(query_id)

        rrf_k = 60  # RRF constant
        query_cat = self._categories.get(query_id)

        combined = {}
        for tid in candidates:
            # RRF score (higher = more similar)
            knn_rank = knn_ranks.get(tid, pool_size)
            dtw_rank = dtw_ranks.get(tid, pool_size)
            rrf_score = self.alpha / (rrf_k + knn_rank) + (1 - self.alpha) / (
                rrf_k + dtw_rank
            )

            # 同類獎勵
            if query_cat and self._categories.get(tid) == query_cat:
                rrf_score /= self.category_discount

            combined[tid] = rrf_score

        # Sort descending (higher RRF = more similar)
        sorted_items = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:k]

        result_ids = [item[0] for item in sorted_items]
        result_scores = [item[1] for item in sorted_items]
        max_s = max(result_scores) if result_scores else 1.0
        scores = [s / (max_s + 1e-8) for s in result_scores]
        # Convert to distances (lower = more similar)
        distances = [1.0 - s for s in scores]

        return SimilarityResult(
            query_id=query_id,
            similar_ids=result_ids,
            distances=distances,
            scores=scores,
        )
