"""
災害數據處理模組
職責：定義災害標籤、轉換和驗證
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from enum import Enum


class ImpactType(Enum):
    """災害類型定義"""
    FLOODING = "flooding"           # 淹水
    BLACKOUT = "blackout"           # 停電
    DAMAGE = "damage"               # 損害
    LANDSLIDE = "landslide"         # 山崩
    WIND_DAMAGE = "wind_damage"     # 風災
    OTHER = "other"                 # 其他


class SeverityLevel(Enum):
    """嚴重程度定義"""
    NONE = 0
    MINOR = 1
    MODERATE = 2
    SEVERE = 3
    CRITICAL = 4


class ImpactMapper:
    """
    災害數據映射器
    
    責任：
    - 將原始災害數據轉換為可預測的標籤
    - 定義不同災害類型的嚴重程度
    - 支援多種標籤格式（0-1, 0-4, 分類等）
    """
    
    def __init__(self):
        """初始化災害映射器"""
        self.impact_type_map = {}
        self.severity_thresholds = {}
    
    def create_binary_label(self, impact_data: pd.DataFrame, 
                           impact_type: str,
                           threshold: Optional[float] = None) -> np.ndarray:
        """
        創建二元災害標籤 (0 = 無, 1 = 有)
        
        Args:
            impact_data: 災害數據
            impact_type: 災害類型
            threshold: 嚴重程度閾值（若提供則用於判定）
            
        Returns:
            二元標籤陣列
        """
        mask = impact_data['impact_type'] == impact_type
        
        if threshold is not None:
            # 基於嚴重程度的閾值
            labels = ((mask) & (impact_data['severity'] >= threshold)).astype(int).values
        else:
            # 簡單的有無判定
            labels = mask.astype(int).values
        
        return labels
    
    def create_severity_label(self, impact_data: pd.DataFrame,
                             impact_type: str) -> np.ndarray:
        """
        創建嚴重程度標籤 (0-4)
        
        Args:
            impact_data: 災害數據
            impact_type: 災害類型
            
        Returns:
            嚴重程度標籤陣列
        """
        # 初始化為 0（無災害）
        labels = np.zeros(len(impact_data), dtype=int)
        
        # 該類型的記錄
        mask = impact_data['impact_type'] == impact_type
        labels[mask] = impact_data.loc[mask, 'severity'].values
        
        return labels
    
    def create_composite_label(self, impact_data: pd.DataFrame,
                              impact_types: List[str],
                              weights: Optional[List[float]] = None) -> np.ndarray:
        """
        創建複合災害標籤（多個災害類型的加權組合）
        
        Args:
            impact_data: 災害數據
            impact_types: 要組合的災害類型列表
            weights: 各類型的權重（預設平均）
            
        Returns:
            複合標籤陣列 (0-max(weights))
        """
        if weights is None:
            weights = [1.0 / len(impact_types)] * len(impact_types)
        
        weights = np.array(weights)
        composite = np.zeros(len(impact_data), dtype=float)
        
        for impact_type, weight in zip(impact_types, weights):
            severity = self.create_severity_label(impact_data, impact_type)
            composite += severity * weight
        
        return composite
    
    def normalize_label(self, labels: np.ndarray, target_range: Tuple[float, float] = (0, 1)) -> np.ndarray:
        """
        標準化標籤到指定範圍
        
        Args:
            labels: 原始標籤
            target_range: 目標範圍
            
        Returns:
            標準化後的標籤
        """
        if np.max(labels) == np.min(labels):
            return np.ones_like(labels) * (target_range[0] + target_range[1]) / 2
        
        normalized = (labels - np.min(labels)) / (np.max(labels) - np.min(labels))
        normalized = normalized * (target_range[1] - target_range[0]) + target_range[0]
        
        return normalized
    
    def categorize_severity(self, severity_values: np.ndarray, 
                           custom_thresholds: Optional[List[float]] = None) -> np.ndarray:
        """
        將連續的嚴重程度值轉換為分類
        
        Args:
            severity_values: 連續的嚴重程度值
            custom_thresholds: 自定義分類閾值
                              預設: [1, 2, 3, 4]（對應 NONE, MINOR, MODERATE, SEVERE, CRITICAL）
            
        Returns:
            分類標籤
        """
        if custom_thresholds is None:
            thresholds = [1, 2, 3, 4]
        else:
            thresholds = custom_thresholds
        
        categories = np.digitize(severity_values, thresholds)
        return categories
    
    def get_label_description(self, label_value: float, label_type: str = 'severity') -> str:
        """
        根據標籤值返回描述
        
        Args:
            label_value: 標籤數值
            label_type: 標籤類型 ('severity' 或 'binary')
            
        Returns:
            描述字串
        """
        if label_type == 'severity':
            level = int(label_value)
            descriptions = {
                0: '無災害',
                1: '輕微',
                2: '中度',
                3: '嚴重',
                4: '災難性'
            }
            return descriptions.get(level, '未知')
        
        elif label_type == 'binary':
            return '有災害' if label_value > 0.5 else '無災害'
        
        return '未知'
    
    def validate_labels(self, labels: np.ndarray, expected_range: Tuple[float, float] = None) -> bool:
        """
        驗證標籤的有效性
        
        Args:
            labels: 標籤陣列
            expected_range: 期望的值域範圍
            
        Returns:
            是否有效
        """
        if not isinstance(labels, np.ndarray):
            return False
        
        if labels.size == 0:
            return False
        
        if expected_range is not None:
            min_val, max_val = expected_range
            if np.any((labels < min_val) | (labels > max_val)):
                return False
        
        return True
    
    def summary(self, impact_data: pd.DataFrame) -> Dict[str, Dict]:
        """
        生成災害數據摘要
        
        Args:
            impact_data: 災害數據
            
        Returns:
            摘要統計字典
        """
        summary = {}
        
        for impact_type in impact_data['impact_type'].unique():
            subset = impact_data[impact_data['impact_type'] == impact_type]
            summary[impact_type] = {
                'count': len(subset),
                'avg_severity': subset['severity'].mean(),
                'max_severity': subset['severity'].max(),
                'affected_typhoons': subset['typhoon_id'].nunique()
            }
        
        return summary
