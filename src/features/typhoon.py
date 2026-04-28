"""
特徵工程模組 — 依據 strategy.md 實作

核心策略：
1. 空間標準化 → 以台灣為中心的相對座標 / 極座標
2. Impact Window → 只取距台灣 < 500 km 的時間段
3. 動態特徵 → min_distance, mean_angle, speed, intensity, rain_proxy
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

# 台灣中心座標
TAIWAN_LAT = 23.7
TAIWAN_LON = 121.0

# 地球半徑 (km)
EARTH_RADIUS_KM = 6371.0

# Impact window 閾值 (km)
IMPACT_WINDOW_RADIUS_KM = 500.0


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine 球面距離 (km)"""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return EARTH_RADIUS_KM * 2 * np.arcsin(np.sqrt(a))


def haversine_vec(
    lats: np.ndarray,
    lons: np.ndarray,
    ref_lat: float = TAIWAN_LAT,
    ref_lon: float = TAIWAN_LON,
) -> np.ndarray:
    """向量化 Haversine"""
    lat1 = np.radians(lats)
    lon1 = np.radians(lons)
    lat2 = np.radians(ref_lat)
    lon2 = np.radians(ref_lon)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return EARTH_RADIUS_KM * 2 * np.arcsin(np.sqrt(a))


@dataclass
class TyphoonFeatures:
    """颱風特徵向量（用於相似度比較）"""

    typhoon_id: str

    # === 靜態 / 摘要特徵 ===
    min_distance_to_taiwan: float  # 最接近台灣的距離 (km)
    mean_angle: float  # 通過台灣時的平均方位角 (deg)
    max_wind_kt: float  # 生命週期最大風速 (kt)
    max_wind_in_window_kt: float  # impact window 內最大風速 (kt)
    approach_speed_kmh: float  # 接近台灣的平均移動速度 (km/h)
    min_pressure_mb: float  # 最低氣壓 (mb)
    intensification_rate: float  # 增強率 (kt / 6h) — impact window 前段
    rain_proxy: float  # 降雨代理指標 wind / distance
    is_landfall: bool  # 是否登陸
    birth_lon: float  # 生成經度
    birth_lat: float  # 生成緯度

    # === Impact Window 路徑（用於 DTW）===
    impact_window_r: np.ndarray = field(repr=False)  # 距台灣距離序列
    impact_window_theta: np.ndarray = field(repr=False)  # 方位角序列
    impact_window_wind: np.ndarray = field(repr=False)  # 風速序列
    impact_window_pressure: np.ndarray = field(repr=False)  # 氣壓序列

    def to_feature_vector(self) -> np.ndarray:
        """轉換為用於 KNN 比較的摘要特徵向量（11 維）"""
        return np.array(
            [
                self.min_distance_to_taiwan,
                self.mean_angle,
                self.max_wind_kt,
                self.max_wind_in_window_kt,
                self.approach_speed_kmh,
                self.min_pressure_mb,
                self.intensification_rate,
                self.rain_proxy,
                float(self.is_landfall),
                self.birth_lon,
                self.birth_lat,
            ],
            dtype=np.float64,
        )

    @staticmethod
    def feature_names() -> list[str]:
        return [
            "min_distance_to_taiwan",
            "mean_angle",
            "max_wind_kt",
            "max_wind_in_window_kt",
            "approach_speed_kmh",
            "min_pressure_mb",
            "intensification_rate",
            "rain_proxy",
            "is_landfall",
            "birth_lon",
            "birth_lat",
        ]

    def get_impact_window_matrix(self) -> np.ndarray:
        """取得 impact window 的多維時序矩陣 (T, 4): [r, theta, wind, pressure]"""
        return np.column_stack(
            [
                self.impact_window_r,
                self.impact_window_theta,
                self.impact_window_wind,
                self.impact_window_pressure,
            ]
        )


class TyphoonFeatureExtractor:
    """
    依據 strategy.md 的特徵提取器

    步驟：
    1. 轉換為相對台灣座標（極座標 r, theta）
    2. 提取 impact window（距台灣 < 500km）
    3. 計算摘要特徵
    """

    def __init__(self, impact_radius_km: float = IMPACT_WINDOW_RADIUS_KM):
        self.impact_radius_km = impact_radius_km

    def extract(
        self,
        typhoon_id: str,
        track: pd.DataFrame,
        birth_lon: float = None,
        birth_lat: float = None,
        landfall_location: str = None,
    ) -> TyphoonFeatures:
        """
        從單一颱風軌跡提取完整特徵

        Args:
            typhoon_id: 颱風編號
            track: DataFrame with columns [latitude, longitude, wind_kt, pressure_mb, timestamp_utc]
            birth_lon, birth_lat: 生成位置
            landfall_location: 登陸地段

        Returns:
            TyphoonFeatures
        """
        track = track.copy()

        # --- Step 1: 空間標準化（相對座標 → 極座標）---
        lats = track["latitude"].values.astype(float)
        lons = track["longitude"].values.astype(float)

        # 相對座標
        dx = lons - TAIWAN_LON
        dy = lats - TAIWAN_LAT

        # 極座標
        r = haversine_vec(lats, lons)
        theta = np.degrees(np.arctan2(dy, dx))  # 方位角

        track["distance_km"] = r
        track["azimuth_deg"] = theta

        # 風速 / 氣壓（處理 None）
        winds = track["wind_kt"].fillna(0).values.astype(float)
        pressures = track["pressure_mb"].fillna(1013).values.astype(float)

        # --- Step 2: 提取 Impact Window ---
        in_window = r < self.impact_radius_km
        if in_window.sum() < 2:
            # 如果路徑沒有進入 500km 範圍，取最近的 5 個點
            nearest_indices = np.argsort(r)[: max(5, len(r) // 5)]
            in_window = np.zeros(len(r), dtype=bool)
            in_window[nearest_indices] = True

        window_r = r[in_window]
        window_theta = theta[in_window]
        window_wind = winds[in_window]
        window_pressure = pressures[in_window]

        # --- Step 3: 摘要特徵 ---

        # 最接近台灣的距離
        min_distance = float(np.min(r))

        # 通過台灣時的平均方位角
        mean_angle = float(np.mean(window_theta))

        # 最大風速（全生命週期 & impact window）
        max_wind = float(np.max(winds)) if len(winds) > 0 else 0.0
        max_wind_window = float(np.max(window_wind)) if len(window_wind) > 0 else 0.0

        # 移動速度（接近台灣階段）
        approach_speed = self._compute_approach_speed(track, in_window)

        # 最低氣壓
        valid_pressures = pressures[pressures < 1013]
        min_pressure = (
            float(np.min(valid_pressures)) if len(valid_pressures) > 0 else 1013.0
        )

        # 增強率（impact window 前段）
        intensification = self._compute_intensification(winds, in_window)

        # Rain proxy: wind / distance（在 impact window 內的平均值）
        safe_r = np.maximum(window_r, 1.0)
        rain_proxy = float(np.mean(window_wind / safe_r))

        # 是否登陸
        is_landfall = landfall_location is not None and str(
            landfall_location
        ).strip() not in ("", "---", "nan", "None")

        # 生成位置
        _birth_lon = birth_lon if birth_lon is not None else float(lons[0])
        _birth_lat = birth_lat if birth_lat is not None else float(lats[0])

        return TyphoonFeatures(
            typhoon_id=typhoon_id,
            min_distance_to_taiwan=min_distance,
            mean_angle=mean_angle,
            max_wind_kt=max_wind,
            max_wind_in_window_kt=max_wind_window,
            approach_speed_kmh=approach_speed,
            min_pressure_mb=min_pressure,
            intensification_rate=intensification,
            rain_proxy=rain_proxy,
            is_landfall=is_landfall,
            birth_lon=_birth_lon,
            birth_lat=_birth_lat,
            impact_window_r=window_r,
            impact_window_theta=window_theta,
            impact_window_wind=window_wind,
            impact_window_pressure=window_pressure,
        )

    def extract_all(self, loader) -> dict[str, TyphoonFeatures]:
        """
        從 DataLoader 批量提取所有颱風的特徵

        Args:
            loader: DataLoader instance (已 load)

        Returns:
            {typhoon_id: TyphoonFeatures}
        """
        features = {}
        for rec in loader.records:
            feat = self.extract(
                typhoon_id=rec.typhoon_id,
                track=rec.track,
                birth_lon=rec.birth_lon,
                birth_lat=rec.birth_lat,
                landfall_location=rec.landfall_location,
            )
            features[rec.typhoon_id] = feat
        print(f"✓ 已提取 {len(features)} 筆颱風特徵")
        return features

    # ---- 內部計算方法 ----

    def _compute_approach_speed(
        self, track: pd.DataFrame, in_window: np.ndarray
    ) -> float:
        """計算在 impact window 內的平均移動速度 (km/h)"""
        if in_window.sum() < 2:
            return 0.0

        window_track = track[in_window].reset_index(drop=True)
        lats = window_track["latitude"].values
        lons = window_track["longitude"].values

        total_dist = 0.0
        for i in range(1, len(lats)):
            total_dist += haversine(lats[i - 1], lons[i - 1], lats[i], lons[i])

        # 假設每步 3 或 6 小時（IBTrACS 通常 3h）
        if "timestamp_utc" in window_track.columns and pd.notna(
            window_track["timestamp_utc"].iloc[0]
        ):
            times = pd.to_datetime(window_track["timestamp_utc"])
            dt_hours = (times.iloc[-1] - times.iloc[0]).total_seconds() / 3600
            if dt_hours > 0:
                return total_dist / dt_hours

        # fallback: 假設 3 小時間隔
        n_steps = len(lats) - 1
        hours = n_steps * 3
        return total_dist / max(hours, 1)

    def _compute_intensification(
        self, winds: np.ndarray, in_window: np.ndarray
    ) -> float:
        """計算 impact window 前半段的風速增強率 (kt / step)"""
        window_indices = np.where(in_window)[0]
        if len(window_indices) < 2:
            return 0.0

        # 取 impact window 前半段
        half = max(2, len(window_indices) // 2)
        first_half = winds[window_indices[:half]]

        if len(first_half) < 2:
            return 0.0

        # 線性回歸的斜率
        x = np.arange(len(first_half))
        slope = np.polyfit(x, first_half, 1)[0]
        return float(slope)
