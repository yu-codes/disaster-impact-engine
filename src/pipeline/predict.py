"""
完整預測流程

data → features → similarity → model → evaluation → visualization
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

from ..data.loader import DataLoader
from ..features.typhoon import TyphoonFeatureExtractor, TyphoonFeatures
from ..similarity.base import SimilarityBase, SimilarityResult
from ..similarity.knn import KNNSimilarity
from ..similarity.dtw import DTWSimilarity
from ..similarity.combined import CombinedSimilarity
from ..similarity.baseline import BaselineSimilarity
from ..similarity.rule_based import RuleBasedSimilarity
from ..models.analog import AnalogModel
from ..impact.mapping import ImpactMapper


@dataclass
class PredictionResult:
    """單一颱風的預測結果"""

    typhoon_id: str
    name_zh: str
    name_en: str
    true_category: str | None
    predicted_category: str | None
    confidence: float
    is_correct: bool | None
    similar_typhoons: list[dict]
    category_votes: dict[str, float]


class DisasterImpactPipeline:
    """
    完整的颱風類比預測流程
    """

    def __init__(
        self,
        similarity_method: str = "combined",
        alpha: float = 0.5,
        feature_weights: np.ndarray | None = None,
        dtw_weights: np.ndarray | None = None,
        impact_radius_km: float = 500.0,
    ):
        """
        Args:
            similarity_method: "knn", "dtw", "combined"
            alpha: combined 模式下特徵距離的權重
            feature_weights: KNN 特徵權重
            dtw_weights: DTW 維度權重
            impact_radius_km: impact window 半徑
        """
        self.similarity_method = similarity_method
        self.alpha = alpha
        self.feature_weights = feature_weights
        self.dtw_weights = dtw_weights
        self.impact_radius_km = impact_radius_km

        self.loader: DataLoader | None = None
        self.extractor = TyphoonFeatureExtractor(impact_radius_km=impact_radius_km)
        self.similarity: SimilarityBase | None = None
        self.model: AnalogModel | None = None
        self.features: dict[str, TyphoonFeatures] = {}
        self.label_dict: dict[str, str] = {}

    def initialize(self, processed_dir: str = "data/processed"):
        """載入資料並建立模型"""
        print("=" * 60)
        print("🌀 初始化颱風類比預測系統")
        print("=" * 60)

        # 1. 載入資料
        print("\n📂 載入資料...")
        self.loader = DataLoader(processed_dir)
        self.loader.load()

        # 2. 提取特徵
        print("\n🔧 提取特徵...")
        self.features = self.extractor.extract_all(self.loader)

        # 3. 建立標籤
        self.label_dict = ImpactMapper.build_label_dict(self.loader)

        # 4. 建立相似度模型
        print("\n📐 建立相似度模型...")
        if self.similarity_method == "knn":
            self.similarity = KNNSimilarity(feature_weights=self.feature_weights)
        elif self.similarity_method == "dtw":
            self.similarity = DTWSimilarity(dtw_weights=self.dtw_weights)
        elif self.similarity_method == "combined":
            self.similarity = CombinedSimilarity(
                alpha=self.alpha,
                feature_weights=self.feature_weights,
                dtw_weights=self.dtw_weights,
            )
        elif self.similarity_method == "baseline":
            self.similarity = BaselineSimilarity(seed=42)
        elif self.similarity_method == "rule_based":
            self.similarity = RuleBasedSimilarity()
        else:
            raise ValueError(f"不支援的相似度方法：{self.similarity_method}")

        if self.similarity_method == "rule_based":
            self.similarity.fit(self.features, loader=self.loader)
        else:
            self.similarity.fit(self.features)

        # 5. 建立預測模型
        self.model = AnalogModel(label_dict=self.label_dict)

        print(f"\n✅ 系統初始化完成（方法={self.similarity_method}）")

    def predict(self, query_id: str, k: int = 5) -> PredictionResult:
        """對單一颱風做預測"""
        rec = self.loader.get(query_id)

        # 找相似颱風
        sim_result = self.similarity.find_similar(query_id, k=k)

        # 預測
        pred = self.model.predict(
            query_id=query_id,
            similar_ids=sim_result.similar_ids,
            distances=sim_result.distances,
        )

        # 組裝相似颱風資訊
        similar_info = []
        for analog in pred.get("analogs", []):
            tid = analog["typhoon_id"]
            analog_rec = self.loader.get(tid)
            similar_info.append(
                {
                    "typhoon_id": tid,
                    "name_zh": analog_rec.name_zh,
                    "name_en": analog_rec.name_en,
                    "year": analog_rec.year,
                    "category": analog.get("category"),
                    "distance": analog.get("distance"),
                }
            )

        return PredictionResult(
            typhoon_id=query_id,
            name_zh=rec.name_zh,
            name_en=rec.name_en,
            true_category=rec.taiwan_track_category,
            predicted_category=pred.get("predicted_category"),
            confidence=pred.get("confidence", 0.0),
            is_correct=pred.get("is_correct"),
            similar_typhoons=similar_info,
            category_votes=pred.get("category_votes", {}),
        )

    def evaluate(self, k: int = 5, verbose: bool = True) -> dict[str, Any]:
        """
        Leave-one-out 評估

        Returns:
            {
                "accuracy": float,
                "total": int,
                "correct": int,
                "per_category": {cat: {total, correct, accuracy}},
                "predictions": [PredictionResult],
                "confusion_data": {(true, pred): count},
            }
        """
        all_ids = self.loader.get_all_ids()
        results: list[PredictionResult] = []
        correct = 0
        total = 0

        per_category: dict[str, dict] = {}
        confusion: dict[tuple, int] = {}

        for tid in all_ids:
            result = self.predict(tid, k=k)
            results.append(result)

            if result.true_category and result.predicted_category:
                total += 1
                true_cat = result.true_category
                pred_cat = result.predicted_category

                if result.is_correct:
                    correct += 1

                # 每類統計
                if true_cat not in per_category:
                    per_category[true_cat] = {"total": 0, "correct": 0}
                per_category[true_cat]["total"] += 1
                if result.is_correct:
                    per_category[true_cat]["correct"] += 1

                # 混淆矩陣
                key = (true_cat, pred_cat)
                confusion[key] = confusion.get(key, 0) + 1

        accuracy = correct / total if total > 0 else 0.0

        # 每類正確率
        for cat_info in per_category.values():
            cat_info["accuracy"] = (
                cat_info["correct"] / cat_info["total"]
                if cat_info["total"] > 0
                else 0.0
            )

        if verbose:
            print(f"\n{'='*60}")
            print(f"📊 評估結果（k={k}, method={self.similarity_method}）")
            print(f"{'='*60}")
            print(f"  總準確率：{accuracy:.1%} ({correct}/{total})")
            print(f"\n  各類準確率：")
            for cat in sorted(per_category.keys()):
                info = per_category[cat]
                print(
                    f"    類型 {cat}: {info['accuracy']:.1%} ({info['correct']}/{info['total']})"
                )

        return {
            "accuracy": accuracy,
            "total": total,
            "correct": correct,
            "per_category": per_category,
            "predictions": results,
            "confusion_data": confusion,
        }

    def save_results(self, eval_result: dict, output_dir: str = "outputs/predictions"):
        """儲存評估結果"""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # 預測明細
        details = []
        for r in eval_result["predictions"]:
            details.append(
                {
                    "typhoon_id": r.typhoon_id,
                    "name_zh": r.name_zh,
                    "name_en": r.name_en,
                    "true_category": r.true_category,
                    "predicted_category": r.predicted_category,
                    "confidence": round(r.confidence, 4),
                    "is_correct": r.is_correct,
                    "similar_typhoons": r.similar_typhoons,
                    "category_votes": {
                        k: round(v, 4) for k, v in r.category_votes.items()
                    },
                }
            )

        with open(out / "prediction_details.json", "w", encoding="utf-8") as f:
            json.dump(details, f, ensure_ascii=False, indent=2)

        # 摘要
        summary = {
            "method": self.similarity_method,
            "alpha": self.alpha,
            "accuracy": round(eval_result["accuracy"], 4),
            "total": eval_result["total"],
            "correct": eval_result["correct"],
            "per_category": eval_result["per_category"],
        }
        with open(out / "evaluation_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"✓ 結果已儲存至 {out}/")
