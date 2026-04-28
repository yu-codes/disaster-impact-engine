"""
特徵工程模組
職責：把颱風原始數據轉換成可用於預測的特徵
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TyphoonFeatures:
    """颱風特徵容器"""
    typhoon_id: str
    distance_to_taiwan: float          # 與台灣距離
    azimuth: float                     # 方位角 (度)
    max_wind: float                    # 最大風速 (km/h)
    speed: float                       # 移動速度 (km/h)
    central_pressure: float            # 中心氣壓 (hPa)
    time_to_impact: float              # 距離台灣預計時間 (小時)
    wind_change_rate: float            # 風速變化率
    pressure_change_rate: float        # 氣壓變化率
    
    def to_vector(self) -> np.ndarray:
        """轉換為特徵向量"""
        return np.array([
            self.distance_to_taiwan,
            self.azimuth,
            self.max_wind,
            self.speed,
            self.central_pressure,
            self.time_to_impact,
            self.wind_change_rate,
            self.pressure_change_rate
        ], dtype=np.float32)


class TyphoonFeatureExtractor:
    """
    颱風特徵提取器
    
    責任：
    - 計算相對位置 (距離、方位角)
    - 計算動態特徵 (速度、變化率)
    - 標準化數據
    """
    
    # 台灣中心座標 (約)
    TAIWAN_LAT = 23.5
    TAIWAN_LON = 120.5
    
    # 特徵的標準化參數（用於歸一化）
    FEATURE_STATS = {
        'distance_to_taiwan': {'min': 0, 'max': 2000},      # km
        'max_wind': {'min': 20, 'max': 250},                # km/h
        'speed': {'min': 0, 'max': 100},                    # km/h
        'central_pressure': {'min': 800, 'max': 1010},      # hPa
        'time_to_impact': {'min': 0, 'max': 168},           # hours
    }
    
    def __init__(self, use_normalization: bool = True):
        """
        初始化特徵提取器
        
        Args:
            use_normalization: 是否進行特徵標準化
        """
        self.use_normalization = use_normalization
    
    def extract(self, typhoon_trajectory: pd.DataFrame) -> TyphoonFeatures:
        """
        從完整颱風軌跡提取特徵
        
        Args:
            typhoon_trajectory: 包含 lat, lon, max_wind, central_pressure 等列的 DataFrame
            
        Returns:
            TyphoonFeatures 物件
        """
        # 取最新位置（通常是最後一筆記錄）
        latest = typhoon_trajectory.iloc[-1]
        typhoon_id = latest.get('typhoon_id', 'unknown')
        
        # 計算與台灣的距離
        distance = self._calculate_distance(latest['lat'], latest['lon'])
        
        # 計算方位角
        azimuth = self._calculate_azimuth(latest['lat'], latest['lon'])
        
        # 計算移動速度
        speed = self._calculate_speed(typhoon_trajectory)
        
        # 計算風速變化率
        wind_change_rate = self._calculate_wind_change_rate(typhoon_trajectory)
        
        # 計算氣壓變化率
        pressure_change_rate = self._calculate_pressure_change_rate(typhoon_trajectory)
        
        # 估計距離台灣的時間
        time_to_impact = self._estimate_time_to_impact(distance, speed)
        
        features = TyphoonFeatures(
            typhoon_id=typhoon_id,
            distance_to_taiwan=distance,
            azimuth=azimuth,
            max_wind=float(latest['max_wind']),
            speed=speed,
            central_pressure=float(latest.get('central_pressure', 950)),
            time_to_impact=time_to_impact,
            wind_change_rate=wind_change_rate,
            pressure_change_rate=pressure_change_rate
        )
        
        return features
    
    def extract_batch(self, typhoon_data: pd.DataFrame) -> Dict[str, TyphoonFeatures]:
        """
        批量提取多個颱風的特徵
        
        Args:
            typhoon_data: 包含所有颱風軌跡的 DataFrame
            
        Returns:
            {typhoon_id: TyphoonFeatures} 字典
        """
        features_dict = {}
        
        for typhoon_id in typhoon_data['typhoon_id'].unique():
            trajectory = typhoon_data[typhoon_data['typhoon_id'] == typhoon_id]
            features = self.extract(trajectory)
            features_dict[typhoon_id] = features
        
        return features_dict
    
    def _calculate_distance(self, lat: float, lon: float) -> float:
        """
        計算颱風與台灣中心的距離 (Haversine公式)
        
        Args:
            lat: 颱風緯度
            lon: 颱風經度
            
        Returns:
            距離 (km)
        """
        R = 6371  # 地球半徑 (km)
        
        lat1, lon1 = np.radians(lat), np.radians(lon)
        lat2, lon2 = np.radians(self.TAIWAN_LAT), np.radians(self.TAIWAN_LON)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def _calculate_azimuth(self, lat: float, lon: float) -> float:
        """
        計算颱風相對於台灣的方位角 (0-360度)
        
        Args:
            lat: 颱風緯度
            lon: 颱風經度
            
        Returns:
            方位角 (度, 0=北, 90=東, 180=南, 270=西)
        """
        lat1, lon1 = np.radians(lat), np.radians(lon)
        lat2, lon2 = np.radians(self.TAIWAN_LAT), np.radians(self.TAIWAN_LON)
        
        dlon = lon2 - lon1
        y = np.sin(dlon) * np.cos(lat2)
        x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
        
        azimuth = np.degrees(np.arctan2(y, x))
        return (azimuth + 360) % 360
    
    def _calculate_speed(self, trajectory: pd.DataFrame) -> float:
        """
        計算颱風移動速度
        
        Args:
            trajectory: 颱風軌跡 DataFrame
            
        Returns:
            移動速度 (km/h)
        """
        if len(trajectory) < 2:
            return 0.0
        
        # 使用最後兩個記錄計算速度
        latest = trajectory.iloc[-1]
        previous = trajectory.iloc[-2]
        
        distance = self._calculate_distance_between_points(
            latest['lat'], latest['lon'],
            previous['lat'], previous['lon']
        )
        
        # 假設時間間隔為 6 小時
        time_diff = 6  # hours
        speed = distance / time_diff
        
        return max(0, speed)
    
    def _calculate_distance_between_points(self, lat1: float, lon1: float, 
                                          lat2: float, lon2: float) -> float:
        """計算兩個點之間的距離"""
        R = 6371
        
        lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
        lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def _calculate_wind_change_rate(self, trajectory: pd.DataFrame) -> float:
        """
        計算風速的變化率
        
        Args:
            trajectory: 颱風軌跡 DataFrame
            
        Returns:
            風速變化率 (km/h per 6 hours)
        """
        if len(trajectory) < 2:
            return 0.0
        
        latest_wind = trajectory.iloc[-1]['max_wind']
        previous_wind = trajectory.iloc[-2]['max_wind']
        
        return float(latest_wind - previous_wind)
    
    def _calculate_pressure_change_rate(self, trajectory: pd.DataFrame) -> float:
        """
        計算氣壓的變化率
        
        Args:
            trajectory: 颱風軌跡 DataFrame
            
        Returns:
            氣壓變化率 (hPa per 6 hours)
        """
        if len(trajectory) < 2 or 'central_pressure' not in trajectory.columns:
            return 0.0
        
        latest_pressure = trajectory.iloc[-1].get('central_pressure', 950)
        previous_pressure = trajectory.iloc[-2].get('central_pressure', 950)
        
        return float(latest_pressure - previous_pressure)
    
    def _estimate_time_to_impact(self, distance: float, speed: float) -> float:
        """
        估計颱風到達台灣所需時間
        
        Args:
            distance: 距離 (km)
            speed: 移動速度 (km/h)
            
        Returns:
            時間 (小時), 若無法到達則返回很大的值
        """
        if speed < 1:  # 速度過低
            return 999.0
        
        time_hours = distance / speed
        return max(0, time_hours)
    
    def normalize_features(self, features: TyphoonFeatures) -> np.ndarray:
        """
        將特徵向量標準化到 [0, 1]
        
        Args:
            features: 特徵物件
            
        Returns:
            標準化後的特徵向量
        """
        feature_dict = {
            'distance_to_taiwan': features.distance_to_taiwan,
            'max_wind': features.max_wind,
            'speed': features.speed,
            'central_pressure': features.central_pressure,
            'time_to_impact': features.time_to_impact,
        }
        
        normalized = np.zeros(8, dtype=np.float32)
        
        for i, (key, value) in enumerate([
            ('distance_to_taiwan', features.distance_to_taiwan),
            ('azimuth', features.azimuth / 360),
            ('max_wind', features.max_wind),
            ('speed', features.speed),
            ('central_pressure', features.central_pressure),
            ('time_to_impact', features.time_to_impact),
        ]):
            if key in self.FEATURE_STATS:
                stats = self.FEATURE_STATS[key]
                normalized[i] = (value - stats['min']) / (stats['max'] - stats['min'])
            else:
                normalized[i] = value
        
        return normalized
