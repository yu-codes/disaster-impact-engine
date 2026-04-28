"""
規則式路徑分類 — 基於 CWA 官方分類定義

根據颱風路徑相對於台灣的幾何特徵判斷侵臺路徑分類：
  1: 通過台灣北部海面向西或西北西
  2: 通過台灣北部向西或西北（含登陸北部）
  3: 通過台灣中部向西（含登陸中部）
  4: 通過台灣南部向西（含登陸南部）
  5: 通過台灣南部海面向西
  6: 沿台灣東岸或東部海面北上
  7: 通過台灣南部海面向東或東北
  8: 通過台灣南部海面向北或北北西
  9: 對台灣無侵襲但有影響（含特殊路徑）
  特殊: 特殊路徑

核心判斷邏輯:
- 找出颱風軌跡最接近台灣的段落 (impact window)
- 計算通過時的緯度帶（北部/中部/南部/海面）
- 計算移動方向（向西/向東/向北/向南）
- 判斷是否登陸
- 組合上述特徵 → 分類
"""

import numpy as np
import pandas as pd
from ..features.typhoon import haversine, haversine_vec, TAIWAN_LAT, TAIWAN_LON
from .base import SimilarityBase, SimilarityResult


# 台灣地理邊界（概略）
TAIWAN_NORTH_LAT = 25.3  # 台灣北端
TAIWAN_SOUTH_LAT = 21.9  # 台灣南端
TAIWAN_WEST_LON = 120.2  # 台灣西岸
TAIWAN_EAST_LON = 121.8  # 台灣東岸
TAIWAN_CENTER_LAT = 23.5  # 台灣中部

# 緯度帶劃分
NORTH_THRESHOLD = 24.2  # 北部 > 24.2°N
SOUTH_THRESHOLD = 22.8  # 南部 < 22.8°N
# 中部 = [22.8, 24.2]

# 距離閾值
LANDFALL_DISTANCE_KM = 50  # 距離 < 50km 視為登陸
NEAR_COAST_KM = 100  # 距離 < 100km 視為近岸
NO_IMPACT_KM = 300  # 距離 > 300km 視為無直接侵襲


def classify_typhoon_by_rules(
    track: pd.DataFrame, landfall_location: str = None
) -> dict:
    """
    根據軌跡幾何特徵分類颱風

    Args:
        track: DataFrame with latitude, longitude columns
        landfall_location: 登陸地段（輔助判斷）

    Returns:
        {
            "predicted_category": str,
            "confidence": float,
            "reasoning": str,
            "features": dict  # 判斷用的特徵值
        }
    """
    lats = track["latitude"].values.astype(float)
    lons = track["longitude"].values.astype(float)

    # 1. 計算距台灣距離
    distances = haversine_vec(lats, lons)
    min_dist = float(np.min(distances))
    closest_idx = int(np.argmin(distances))

    # 2. 找 impact window（距台灣 < 500km 的段落）
    in_window = distances < 500
    if in_window.sum() < 2:
        # 沒進入 500km → 類型 9 或特殊
        return {
            "predicted_category": "9",
            "confidence": 0.7,
            "reasoning": "軌跡未進入台灣 500km 範圍",
            "features": {"min_distance_km": min_dist},
        }

    window_lats = lats[in_window]
    window_lons = lons[in_window]

    # 3. 最接近台灣時的位置
    closest_lat = float(lats[closest_idx])
    closest_lon = float(lons[closest_idx])

    # 4. 移動方向：取 impact window 的整體移動向量
    if len(window_lats) >= 2:
        # 入窗點 → 出窗點
        entry_lat, entry_lon = window_lats[0], window_lons[0]
        exit_lat, exit_lon = window_lats[-1], window_lons[-1]
        dlat = exit_lat - entry_lat
        dlon = exit_lon - entry_lon
        heading_deg = float(
            np.degrees(np.arctan2(dlat, dlon))
        )  # 0=東, 90=北, -90=南, 180=西
    else:
        heading_deg = 0.0
        dlat, dlon = 0.0, 0.0

    # 5. 判斷通過的緯度帶
    mean_lat_near = float(
        np.mean(window_lats[distances[in_window] < max(200, min_dist + 50)])
    )
    is_north_of_taiwan = mean_lat_near > TAIWAN_NORTH_LAT
    is_south_of_taiwan = mean_lat_near < TAIWAN_SOUTH_LAT
    is_north_part = mean_lat_near >= NORTH_THRESHOLD
    is_south_part = mean_lat_near <= SOUTH_THRESHOLD
    is_central = not is_north_part and not is_south_part

    # 6. 判斷是否登陸（距離判斷 + 經度穿越）
    has_landfall = landfall_location is not None and str(
        landfall_location
    ).strip() not in ("", "---", "nan", "None")
    passes_through = min_dist < LANDFALL_DISTANCE_KM
    near_coast = min_dist < NEAR_COAST_KM

    # 7. 判斷是否在台灣東側/西側通過
    passes_east = closest_lon > TAIWAN_EAST_LON
    passes_west = closest_lon < TAIWAN_WEST_LON

    # 8. 主要移動方向分類
    is_westward = dlon < -0.5 and abs(dlon) > abs(dlat) * 0.3
    is_eastward = dlon > 0.5 and abs(dlon) > abs(dlat) * 0.3
    is_northward = dlat > 0.5 and abs(dlat) > abs(dlon) * 0.3
    is_southward = dlat < -0.5

    # 9. 檢查是否沿東岸北上（類型 6 的特徵）
    # 特點：在台灣東側，整體往北移動
    east_side_points = window_lons > TAIWAN_EAST_LON - 0.5
    is_along_east_coast = (
        east_side_points.sum() > len(window_lons) * 0.5
        and dlat > 1.0
        and closest_lon >= TAIWAN_EAST_LON - 1.0
    )

    # === 分類規則 ===
    features = {
        "min_distance_km": round(min_dist, 1),
        "closest_lat": round(closest_lat, 2),
        "closest_lon": round(closest_lon, 2),
        "mean_lat_near": round(mean_lat_near, 2),
        "heading_deg": round(heading_deg, 1),
        "dlat": round(dlat, 2),
        "dlon": round(dlon, 2),
        "has_landfall": has_landfall or passes_through,
        "passes_east": passes_east,
    }

    # 類型 6：沿東岸或東部海面北上
    if is_along_east_coast:
        return {
            "predicted_category": "6",
            "confidence": 0.8,
            "reasoning": "沿台灣東岸或東部海面北上",
            "features": features,
        }

    # 距離太遠 → 類型 9
    if min_dist > NO_IMPACT_KM:
        return {
            "predicted_category": "9",
            "confidence": 0.7,
            "reasoning": f"最近距離 {min_dist:.0f}km > {NO_IMPACT_KM}km，無直接侵襲",
            "features": features,
        }

    # 通過南部海面的情況（closest_lat < 台灣南端）
    if is_south_of_taiwan or (
        closest_lat < SOUTH_THRESHOLD and not (has_landfall or passes_through)
    ):
        if is_westward:
            return {
                "predicted_category": "5",
                "confidence": 0.75,
                "reasoning": "通過台灣南部海面向西",
                "features": features,
            }
        elif is_eastward:
            return {
                "predicted_category": "7",
                "confidence": 0.75,
                "reasoning": "通過台灣南部海面向東或東北",
                "features": features,
            }
        elif is_northward:
            return {
                "predicted_category": "8",
                "confidence": 0.75,
                "reasoning": "通過台灣南部海面向北或北北西",
                "features": features,
            }
        else:
            # 南部海面但方向不明確
            return {
                "predicted_category": "5",
                "confidence": 0.5,
                "reasoning": f"通過台灣南部海面，方向不明確 (heading={heading_deg:.0f}°)",
                "features": features,
            }

    # 通過北部海面（closest_lat > 台灣北端，未登陸）
    if is_north_of_taiwan and not (has_landfall or passes_through):
        if is_westward:
            return {
                "predicted_category": "1",
                "confidence": 0.75,
                "reasoning": "通過台灣北部海面向西或西北西",
                "features": features,
            }
        else:
            return {
                "predicted_category": "1",
                "confidence": 0.5,
                "reasoning": f"通過台灣北部海面 (heading={heading_deg:.0f}°)",
                "features": features,
            }

    # 登陸或非常接近台灣
    if has_landfall or passes_through or near_coast:
        if is_north_part:
            return {
                "predicted_category": "2",
                "confidence": 0.75,
                "reasoning": "通過台灣北部向西或西北（含登陸北部）",
                "features": features,
            }
        elif is_south_part:
            return {
                "predicted_category": "4",
                "confidence": 0.75,
                "reasoning": "通過台灣南部向西（含登陸南部）",
                "features": features,
            }
        else:
            return {
                "predicted_category": "3",
                "confidence": 0.75,
                "reasoning": "通過台灣中部向西（含登陸中部）",
                "features": features,
            }

    # 介於登陸與海面之間的模糊地帶
    if is_north_part and is_westward:
        return {
            "predicted_category": "1" if not near_coast else "2",
            "confidence": 0.5,
            "reasoning": "北部附近向西",
            "features": features,
        }
    if is_south_part and is_westward:
        return {
            "predicted_category": "5" if not near_coast else "4",
            "confidence": 0.5,
            "reasoning": "南部附近向西",
            "features": features,
        }

    # 預設：根據最接近位置做最後判斷
    if is_central:
        return {
            "predicted_category": "3",
            "confidence": 0.4,
            "reasoning": f"中部附近 (lat={closest_lat:.1f}, heading={heading_deg:.0f}°)",
            "features": features,
        }

    return {
        "predicted_category": "特殊",
        "confidence": 0.3,
        "reasoning": f"無法明確分類 (lat={closest_lat:.1f}, heading={heading_deg:.0f}°, dist={min_dist:.0f}km)",
        "features": features,
    }


class RuleBasedSimilarity(SimilarityBase):
    """
    規則式分類器包裝為 SimilarityBase 介面

    工作方式：
    1. 對查詢颱風用規則判斷分類
    2. 找同分類的歷史颱風作為「相似」颱風
    3. 在同分類中用簡單的距離排序
    """

    def __init__(self):
        self._ids: list[str] = []
        self._features_dict = {}
        self._tracks: dict[str, pd.DataFrame] = {}
        self._categories: dict[str, str] = {}  # rule-predicted category per typhoon
        self._landfall: dict[str, str] = {}

    def fit(self, feature_dict: dict, loader=None):
        self._features_dict = feature_dict
        self._ids = list(feature_dict.keys())

        if loader is not None:
            for rec in loader.records:
                self._tracks[rec.typhoon_id] = rec.track
                self._landfall[rec.typhoon_id] = rec.landfall_location

            # 預計算所有颱風的規則分類
            for tid in self._ids:
                result = classify_typhoon_by_rules(
                    self._tracks[tid], self._landfall.get(tid)
                )
                self._categories[tid] = result["predicted_category"]

        print(f"✓ 規則式分類器已擬合 {len(self._ids)} 筆颱風")

    def find_similar(
        self, query_id: str, k: int = 5, exclude_self: bool = True
    ) -> SimilarityResult:
        # 用規則分類查詢颱風
        query_cat = self._categories.get(query_id, "特殊")

        # 找同分類的颱風
        same_cat = [
            tid
            for tid in self._ids
            if self._categories.get(tid) == query_cat and tid != query_id
        ]

        if not same_cat:
            # 如果沒有同分類的，退回到所有颱風
            same_cat = [tid for tid in self._ids if tid != query_id]

        # 在同分類中用特徵距離排序
        query_vec = self._features_dict[query_id].to_feature_vector()
        dists = []
        for tid in same_cat:
            other_vec = self._features_dict[tid].to_feature_vector()
            d = float(np.linalg.norm(query_vec - other_vec))
            dists.append((tid, d))
        dists.sort(key=lambda x: x[1])

        result_ids = [x[0] for x in dists[:k]]
        result_dists = [x[1] for x in dists[:k]]
        max_d = max(result_dists) if result_dists else 1.0
        scores = [1.0 - d / (max_d + 1e-8) for d in result_dists]

        return SimilarityResult(
            query_id=query_id,
            similar_ids=result_ids,
            distances=result_dists,
            scores=scores,
        )

    def compute_distance(self, id_a: str, id_b: str) -> float:
        vec_a = self._features_dict[id_a].to_feature_vector()
        vec_b = self._features_dict[id_b].to_feature_vector()
        return float(np.linalg.norm(vec_a - vec_b))

    def get_rule_category(self, typhoon_id: str) -> str:
        """取得規則判斷的分類"""
        return self._categories.get(typhoon_id, "特殊")

    def classify_track(
        self, track: pd.DataFrame, landfall_location: str = None
    ) -> dict:
        """對外部軌跡做規則分類"""
        return classify_typhoon_by_rules(track, landfall_location)
