"""
類比模型實現
職責：基於相似颱風的災害記錄進行預測
"""

import numpy as np
from typing import Dict, List, Any
from .base import ModelBase


class AnalogModel(ModelBase):
    """
    類比預測模型
    
    原理：
    - 找相似歷史颱風
    - 基於這些颱風的災害記錄進行加權預測
    - 支援多種聚合方式（平均、加權平均、投票等）
    """
    
    def __init__(self, aggregation_method: str = 'weighted_mean'):
        """
        初始化類比模型
        
        Args:
            aggregation_method: 聚合方式
                - 'mean': 簡單平均
                - 'weighted_mean': 距離加權平均（推薦）
                - 'majority_vote': 多數投票（用於分類）
                - 'max': 取最大值
        """
        valid_methods = ['mean', 'weighted_mean', 'majority_vote', 'max']
        if aggregation_method not in valid_methods:
            raise ValueError(f"不支援的聚合方式: {aggregation_method}。支援: {valid_methods}")
        
        self.aggregation_method = aggregation_method
    
    def predict(self, similar_indices: List[int], 
                similar_distances: List[float],
                impact_labels: np.ndarray) -> Dict[str, Any]:
        """
        基於相似颱風進行類比預測
        
        Args:
            similar_indices: 相似颱風的索引列表
            similar_distances: 相應的距離列表（小=相似）
            impact_labels: 所有颱風的災害標籤或數值 (N,)
            
        Returns:
            預測結果字典
        """
        if not similar_indices:
            return {
                'prediction': None,
                'confidence': 0.0,
                'error': 'No similar analogs found'
            }
        
        # 提取相似颱風的標籤
        analog_labels = impact_labels[similar_indices]
        
        # 根據聚合方式進行預測
        if self.aggregation_method == 'mean':
            prediction, confidence = self._predict_mean(analog_labels, similar_distances)
        
        elif self.aggregation_method == 'weighted_mean':
            prediction, confidence = self._predict_weighted_mean(analog_labels, similar_distances)
        
        elif self.aggregation_method == 'majority_vote':
            prediction, confidence = self._predict_majority_vote(analog_labels, similar_distances)
        
        elif self.aggregation_method == 'max':
            prediction, confidence = self._predict_max(analog_labels, similar_distances)
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'method': self.aggregation_method,
            'num_analogs': len(similar_indices),
            'analog_indices': similar_indices,
            'analog_distances': similar_distances,
            'analog_values': analog_labels.tolist()
        }
    
    def _predict_mean(self, labels: np.ndarray, distances: List[float]) -> tuple:
        """簡單平均預測"""
        prediction = np.mean(labels)
        confidence = 1.0 - (np.std(labels) / (np.mean(labels) + 1e-6))
        return float(prediction), float(max(0, min(1, confidence)))
    
    def _predict_weighted_mean(self, labels: np.ndarray, distances: List[float]) -> tuple:
        """
        距離加權平均預測
        
        距離越小，權重越大
        """
        distances = np.array(distances)
        
        # 轉換距離為權重（距離倒數）
        # 避免除以 0
        weights = 1.0 / (distances + 1e-6)
        weights = weights / np.sum(weights)
        
        # 加權平均
        prediction = np.sum(labels * weights)
        
        # 信心度基於權重的集中度
        # 權重越集中（一個很大的權重），信心度越高
        confidence = np.max(weights)
        
        return float(prediction), float(confidence)
    
    def _predict_majority_vote(self, labels: np.ndarray, distances: List[float]) -> tuple:
        """
        多數投票預測
        
        用於分類問題（標籤為 0/1）
        """
        # 確定是分類任務
        unique_labels = np.unique(labels)
        if len(unique_labels) > 10:
            # 如果有太多不同的值，使用加權平均
            return self._predict_weighted_mean(labels, distances)
        
        # 多數投票
        prediction = np.argmax(np.bincount(labels.astype(int)))
        
        # 信心度 = 得票比例
        votes = np.bincount(labels.astype(int))
        confidence = votes[prediction] / len(labels)
        
        return float(prediction), float(confidence)
    
    def _predict_max(self, labels: np.ndarray, distances: List[float]) -> tuple:
        """
        取最大值預測
        
        用於極端災害預測（最壞情況）
        """
        prediction = np.max(labels)
        
        # 信心度基於最大值的頻次
        count_max = np.sum(labels == prediction)
        confidence = count_max / len(labels)
        
        return float(prediction), float(confidence)
    
    def predict_by_impact_type(self, similar_indices: List[int],
                               similar_distances: List[float],
                               impact_data_dict: Dict[str, np.ndarray]) -> Dict[str, Dict]:
        """
        按災害類型進行多個預測
        
        Args:
            similar_indices: 相似颱風的索引
            similar_distances: 相應的距離
            impact_data_dict: {impact_type: impact_labels} 的字典
            
        Returns:
            {impact_type: prediction_result} 的字典
        """
        results = {}
        
        for impact_type, impact_labels in impact_data_dict.items():
            result = self.predict(similar_indices, similar_distances, impact_labels)
            results[impact_type] = result
        
        return results
