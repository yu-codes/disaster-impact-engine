"""
預測流程模組
職責：串整完整的預測流程
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from ..data.loader import DataLoader
from ..features.typhoon import TyphoonFeatureExtractor, TyphoonFeatures
from ..similarity.base import SimilarityBase
from ..models.base import ModelBase
from ..impact.mapping import ImpactMapper


@dataclass
class PredictionResult:
    """預測結果容器"""
    typhoon_id: str
    query_features: TyphoonFeatures
    similar_typhoon_ids: List[str]
    similar_distances: List[float]
    predictions: Dict[str, Dict[str, Any]]
    timestamp: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """轉換為字典"""
        return {
            'typhoon_id': self.typhoon_id,
            'features': {
                'distance_to_taiwan': self.query_features.distance_to_taiwan,
                'azimuth': self.query_features.azimuth,
                'max_wind': self.query_features.max_wind,
                'speed': self.query_features.speed,
            },
            'similar_typhoons': self.similar_typhoon_ids,
            'similar_distances': self.similar_distances,
            'predictions': self.predictions,
            'timestamp': self.timestamp
        }


class DisasterImpactPipeline:
    """
    完整的災害預測流程
    
    流程：
    颱風數據 → 特徵提取 → 相似度計算 → 類比預測 → 結果
    """
    
    def __init__(self, 
                 similarity_model: SimilarityBase,
                 prediction_model: ModelBase,
                 feature_extractor: Optional[TyphoonFeatureExtractor] = None,
                 impact_mapper: Optional[ImpactMapper] = None):
        """
        初始化預測流程
        
        Args:
            similarity_model: 相似度計算模型（如 KNNSimilarity）
            prediction_model: 預測模型（如 AnalogModel）
            feature_extractor: 特徵提取器
            impact_mapper: 災害標籤映射器
        """
        self.similarity_model = similarity_model
        self.prediction_model = prediction_model
        self.feature_extractor = feature_extractor or TyphoonFeatureExtractor()
        self.impact_mapper = impact_mapper or ImpactMapper()
        
        self.data_loader = None
        self.reference_typhoons = None
        self.reference_features = None
        self.reference_indices = {}
        self.impact_labels = {}
    
    def initialize(self, typhoon_data_path: str, impact_data_path: str):
        """
        初始化流程（加載數據）
        
        Args:
            typhoon_data_path: 颱風軌跡數據路徑
            impact_data_path: 災害影響數據路徑
        """
        self.data_loader = DataLoader()
        
        # 加載數據
        typhoon_df = self.data_loader.load_typhoon_data(typhoon_data_path)
        impact_df = self.data_loader.load_impact_data(impact_data_path)
        
        # 提取參考颱風特徵
        self._build_reference_features(typhoon_df)
        
        # 構建影響標籤
        self._build_impact_labels(impact_df)
        
        print(f"✓ 流程已初始化")
        print(f"  - 參考颱風: {len(self.reference_indices)}")
        print(f"  - 災害類型: {list(self.impact_labels.keys())}")
    
    def _build_reference_features(self, typhoon_df: pd.DataFrame):
        """構建參考颱風特徵"""
        features_dict = self.feature_extractor.extract_batch(typhoon_df)
        
        self.reference_typhoons = list(features_dict.keys())
        self.reference_features = np.array([
            features_dict[tid].to_vector() for tid in self.reference_typhoons
        ])
        
        # 建立索引映射
        for idx, tid in enumerate(self.reference_typhoons):
            self.reference_indices[tid] = idx
        
        # 擬合相似度模型
        self.similarity_model.fit(self.reference_features)
    
    def _build_impact_labels(self, impact_df: pd.DataFrame):
        """構建災害標籤"""
        # 為每個參考颱風創建災害標籤
        n_typhoons = len(self.reference_typhoons)
        
        # 按災害類型分類
        for impact_type in impact_df['impact_type'].unique():
            labels = np.zeros(n_typhoons, dtype=int)
            
            # 為有該類型災害的颱風標記為 1
            for idx, typhoon_id in enumerate(self.reference_typhoons):
                if typhoon_id in impact_df[impact_df['impact_type'] == impact_type]['typhoon_id'].values:
                    labels[idx] = 1
            
            self.impact_labels[impact_type] = labels
    
    def predict(self, query_typhoon_id: str, k: int = 5) -> PredictionResult:
        """
        對單個颱風進行預測
        
        Args:
            query_typhoon_id: 要預測的颱風 ID
            k: 返回的相似颱風數
            
        Returns:
            PredictionResult 物件
        """
        if self.data_loader is None:
            raise ValueError("流程尚未初始化，請先調用 initialize()")
        
        # 取得查詢颱風數據
        query_typhoon_data = self.data_loader.get_typhoon_by_id(query_typhoon_id)
        if query_typhoon_data.empty:
            raise ValueError(f"找不到颱風: {query_typhoon_id}")
        
        # 提取查詢颱風特徵
        query_features = self.feature_extractor.extract(query_typhoon_data)
        query_vector = query_features.to_vector()
        
        # 找相似颱風
        similar_indices, similar_distances = self.similarity_model.find_similar(query_vector, k)
        similar_typhoon_ids = [self.reference_typhoons[idx] for idx in similar_indices]
        
        # 執行預測
        predictions = {}
        for impact_type, impact_labels in self.impact_labels.items():
            result = self.prediction_model.predict(
                similar_indices, 
                similar_distances, 
                impact_labels
            )
            predictions[impact_type] = result
        
        # 構建結果
        result = PredictionResult(
            typhoon_id=query_typhoon_id,
            query_features=query_features,
            similar_typhoon_ids=similar_typhoon_ids,
            similar_distances=similar_distances,
            predictions=predictions
        )
        
        return result
    
    def predict_batch(self, query_typhoon_ids: List[str], k: int = 5) -> List[PredictionResult]:
        """
        批量進行預測
        
        Args:
            query_typhoon_ids: 颱風 ID 列表
            k: 相似颱風數
            
        Returns:
            PredictionResult 列表
        """
        results = []
        
        for typhoon_id in query_typhoon_ids:
            try:
                result = self.predict(typhoon_id, k)
                results.append(result)
            except Exception as e:
                print(f"✗ 預測失敗 {typhoon_id}: {e}")
        
        return results
    
    def get_prediction_summary(self, result: PredictionResult) -> Dict[str, str]:
        """
        生成預測摘要（人類可讀）
        
        Args:
            result: 預測結果
            
        Returns:
            摘要字典
        """
        summary = {
            'typhoon_id': result.typhoon_id,
            'distance_to_taiwan_km': f"{result.query_features.distance_to_taiwan:.1f}",
            'max_wind_kmh': f"{result.query_features.max_wind:.1f}",
            'similar_typhoons': ', '.join(result.similar_typhoon_ids),
        }
        
        # 加入災害預測
        for impact_type, prediction in result.predictions.items():
            pred_value = prediction.get('prediction', 0)
            confidence = prediction.get('confidence', 0)
            summary[f'{impact_type}_risk'] = f"{pred_value:.2f} (信心: {confidence:.2f})"
        
        return summary
