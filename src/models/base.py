"""
預測模型基類
職責：定義 model 介面（可替換）
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any
import numpy as np


class ModelBase(ABC):
    """
    預測模型的抽象基類
    
    所有預測模型都應繼承此類
    """
    
    @abstractmethod
    def predict(self, similar_indices: List[int], 
                similar_distances: List[float],
                impact_labels: np.ndarray) -> Dict[str, Any]:
        """
        基於相似颱風進行預測
        
        Args:
            similar_indices: 相似颱風的索引列表
            similar_distances: 相應的距離列表
            impact_labels: 所有颱風的災害標籤 (N,)
            
        Returns:
            預測結果字典，包含：
                - 'prediction': 預測結果
                - 'confidence': 信心度
                - 'details': 詳細信息
        """
        pass
    
    def predict_batch(self, similar_indices_list: List[List[int]], 
                     similar_distances_list: List[List[float]],
                     impact_labels: np.ndarray) -> List[Dict[str, Any]]:
        """
        批量進行預測
        
        Args:
            similar_indices_list: 每個查詢的相似颱風索引
            similar_distances_list: 對應的距離列表
            impact_labels: 所有颱風的災害標籤
            
        Returns:
            預測結果列表
        """
        results = []
        for indices, distances in zip(similar_indices_list, similar_distances_list):
            result = self.predict(indices, distances, impact_labels)
            results.append(result)
        return results
