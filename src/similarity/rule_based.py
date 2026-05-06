"""
規則式路徑分類 v2 — 基於 CWA 官方分類定義

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
  特殊: 特殊路徑（迴旋、U-turn、滯留、二次穿越）

v2 改進：
- 雙層 impact window（core=200km, context=400km）
- 向量平均方向取代單一向量
- 角度判斷取代固定閾值
- 修正優先順序：type6 → no-impact → landfall → sea-types
- 使用 closest_lat 取代 mean_lat
- 強化 type6 判斷條件
- 資料驅動的信心分數
- 定義特殊路徑模式
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

# 距離閾值（雙層 window）
CORE_RADIUS_KM = 200  # 核心窗口：方向判斷用
CONTEXT_RADIUS_KM = 400  # 脈絡窗口：路徑趨勢用
LANDFALL_DISTANCE_KM = 50  # 距離 < 50km 視為登陸
NEAR_COAST_KM = 100  # 距離 < 100km 視為近岸
NO_IMPACT_KM = 300  # 距離 > 300km 視為無直接侵襲


def _compute_vector_mean_heading(lats: np.ndarray, lons: np.ndarray) -> float:
    """
    計算路徑段落的向量平均方向（角度）。
    使用相鄰點間的向量取平均，避免曲線路徑被單一向量誤判。

    Returns:
        heading in degrees: 0=東, 90=北, -90=南, ±180=西
    """
    if len(lats) < 2:
        return 0.0

    dlats = np.diff(lats)
    dlons = np.diff(lons)

    # 向量平均
    mean_dlon = np.mean(dlons)
    mean_dlat = np.mean(dlats)

    return float(np.degrees(np.arctan2(mean_dlat, mean_dlon)))


def _detect_special_patterns(
    lats: np.ndarray, lons: np.ndarray, distances: np.ndarray
) -> str | None:
    """
    檢測特殊路徑模式：迴旋、U-turn、滯留、二次穿越

    Returns:
        特殊類型字串，或 None
    """
    if len(lats) < 5:
        return None

    # 計算逐步方向變化
    dlats = np.diff(lats)
    dlons = np.diff(lons)
    headings = np.degrees(np.arctan2(dlats, dlons))

    # 方向變化量
    dheadings = np.diff(headings)
    # 處理環狀（-180 到 180）
    dheadings = (dheadings + 180) % 360 - 180

    # 迴旋（loop）: 累積方向變化 > 360°（在影響區域內才算）
    cumulative_turn = np.abs(np.sum(dheadings))
    if cumulative_turn > 360:
        return "迴旋"

    # U-turn: 前後方向相反（差 > 120°）
    if len(headings) >= 4:
        first_quarter = np.mean(headings[: len(headings) // 4])
        last_quarter = np.mean(headings[-len(headings) // 4 :])
        turn_diff = abs((last_quarter - first_quarter + 180) % 360 - 180)
        if turn_diff > 120:
            return "U型轉向"

    # 滯留: 在 300km 內停留很久但移動距離很小
    near_taiwan = distances < 300
    if near_taiwan.sum() >= 8:  # 至少 24 小時（每 3 小時一個點）
        near_lats = lats[near_taiwan]
        near_lons = lons[near_taiwan]
        lat_range = near_lats.max() - near_lats.min()
        lon_range = near_lons.max() - near_lons.min()
        if lat_range < 1.5 and lon_range < 1.5 and near_taiwan.sum() >= 12:
            return "滯留"

    # 二次穿越: 距離曲線有兩個谷底
    if len(distances) > 10:
        local_mins = []
        for i in range(1, len(distances) - 1):
            if distances[i] < distances[i - 1] and distances[i] < distances[i + 1]:
                if distances[i] < 200:
                    local_mins.append(i)
        if len(local_mins) >= 2:
            for m in range(len(local_mins) - 1):
                between = distances[local_mins[m] : local_mins[m + 1]]
                if between.max() > 200:
                    return "二次穿越"

    return None


def _crosses_taiwan_bbox(lats: np.ndarray, lons: np.ndarray) -> bool:
    """判斷路徑是否穿越台灣 bounding box"""
    return (
        lons.min() < TAIWAN_EAST_LON
        and lons.max() > TAIWAN_WEST_LON
        and lats.max() > TAIWAN_SOUTH_LAT
        and lats.min() < TAIWAN_NORTH_LAT
    )


def _compute_confidence(
    min_dist: float, direction_consistency: float, has_landfall: bool
) -> float:
    """
    資料驅動的信心分數

    Args:
        min_dist: 最近距離 (km)
        direction_consistency: 方向一致性 (0-1)
        has_landfall: 是否有登陸資料
    """
    score = 0.0
    score += (1 - min(min_dist / 500, 1.0)) * 0.3
    score += direction_consistency * 0.4
    score += 0.3 if has_landfall else 0.1
    return min(max(score, 0.2), 0.95)


def _direction_consistency(lats: np.ndarray, lons: np.ndarray) -> float:
    """計算方向一致性（0-1），值越高代表路徑越直"""
    if len(lats) < 3:
        return 0.5

    dlats = np.diff(lats)
    dlons = np.diff(lons)
    headings = np.arctan2(dlats, dlons)

    mean_sin = np.mean(np.sin(headings))
    mean_cos = np.mean(np.cos(headings))
    r = np.sqrt(mean_sin**2 + mean_cos**2)
    return float(r)


def classify_typhoon_by_rules(
    track: pd.DataFrame, landfall_location: str = None
) -> dict:
    """
    根據軌跡幾何特徵分類颱風 (v2)

    Args:
        track: DataFrame with latitude, longitude columns
        landfall_location: 登陸地段（輔助判斷）

    Returns:
        {
            "predicted_category": str,
            "confidence": float,
            "reasoning": str,
            "features": dict
        }
    """
    lats = track["latitude"].values.astype(float)
    lons = track["longitude"].values.astype(float)

    # 1. 計算距台灣距離
    distances = haversine_vec(lats, lons)
    min_dist = float(np.min(distances))
    closest_idx = int(np.argmin(distances))

    # 2. 雙層 impact window
    in_context = distances < CONTEXT_RADIUS_KM
    in_core = distances < CORE_RADIUS_KM

    if in_context.sum() < 2:
        return {
            "predicted_category": "9",
            "confidence": 0.7,
            "reasoning": "軌跡未進入台灣 400km 範圍",
            "features": {"min_distance_km": min_dist},
        }

    context_lats = lats[in_context]
    context_lons = lons[in_context]

    if in_core.sum() >= 2:
        core_lats = lats[in_core]
        core_lons = lons[in_core]
    else:
        context_dists = distances[in_context]
        median_d = np.median(context_dists)
        near_mask = context_dists <= median_d
        core_lats = context_lats[near_mask]
        core_lons = context_lons[near_mask]

    # 3. 最接近台灣時的位置
    closest_lat = float(lats[closest_idx])
    closest_lon = float(lons[closest_idx])

    # 4. 向量平均方向
    heading_deg = _compute_vector_mean_heading(core_lats, core_lons)
    context_heading_deg = _compute_vector_mean_heading(context_lats, context_lons)

    # 5. 方向一致性
    dir_consistency = _direction_consistency(core_lats, core_lons)

    # 6. 登陸判斷（優先：data > bbox > distance）
    has_landfall_data = landfall_location is not None and str(
        landfall_location
    ).strip() not in ("", "---", "nan", "None")

    if has_landfall_data:
        is_landfall = True
    elif (distances < 150).any() and _crosses_taiwan_bbox(
        lats[distances < 150], lons[distances < 150]
    ):
        is_landfall = True
    else:
        is_landfall = min_dist < LANDFALL_DISTANCE_KM

    near_coast = min_dist < NEAR_COAST_KM

    # 7. 角度判斷方向
    is_westward = abs(heading_deg) > 135
    is_eastward = abs(heading_deg) < 45
    is_northward = 45 <= heading_deg <= 135
    is_southward = -135 <= heading_deg <= -45

    # 8. 類型 6 強化判斷
    east_side_points = context_lons > TAIWAN_EAST_LON - 0.5
    is_along_east_coast = (
        east_side_points.sum() > len(context_lons) * 0.5
        and heading_deg > 30
        and np.std(context_lons) < 1.0
        and min_dist < 150
        and closest_lon >= TAIWAN_EAST_LON - 1.0
    )

    # 9. 位置判斷
    is_north_of_taiwan = closest_lat > TAIWAN_NORTH_LAT
    is_south_of_taiwan = closest_lat < TAIWAN_SOUTH_LAT
    is_north_part = closest_lat >= NORTH_THRESHOLD
    is_south_part = closest_lat <= SOUTH_THRESHOLD
    is_central = not is_north_part and not is_south_part

    # 10. 特殊路徑（只檢查 context window 內的路徑）
    special_pattern = _detect_special_patterns(
        context_lats, context_lons, distances[in_context]
    )

    # 信心分數
    confidence = _compute_confidence(min_dist, dir_consistency, has_landfall_data)

    features = {
        "min_distance_km": round(min_dist, 1),
        "closest_lat": round(closest_lat, 2),
        "closest_lon": round(closest_lon, 2),
        "heading_deg": round(heading_deg, 1),
        "context_heading_deg": round(context_heading_deg, 1),
        "direction_consistency": round(dir_consistency, 3),
        "has_landfall": is_landfall,
        "passes_east": closest_lon > TAIWAN_EAST_LON,
        "special_pattern": special_pattern,
    }

    # === 分類規則（修正優先順序）===

    # [1] 類型 6
    if is_along_east_coast:
        return {
            "predicted_category": "6",
            "confidence": confidence,
            "reasoning": "沿台灣東岸或東部海面北上",
            "features": features,
        }

    # [2] 特殊路徑
    if special_pattern is not None and min_dist < NO_IMPACT_KM:
        return {
            "predicted_category": "特殊",
            "confidence": max(confidence * 0.8, 0.4),
            "reasoning": f"特殊路徑模式：{special_pattern}",
            "features": features,
        }

    # [3] 無影響
    if min_dist > NO_IMPACT_KM:
        return {
            "predicted_category": "9",
            "confidence": 0.7,
            "reasoning": f"最近距離 {min_dist:.0f}km > {NO_IMPACT_KM}km",
            "features": features,
        }

    # [4] 登陸 / 穿越（2/3/4）
    if is_landfall or near_coast:
        if is_north_part or closest_lat >= NORTH_THRESHOLD:
            return {
                "predicted_category": "2",
                "confidence": confidence,
                "reasoning": "通過台灣北部（含登陸）",
                "features": features,
            }
        elif is_south_part or closest_lat <= SOUTH_THRESHOLD:
            return {
                "predicted_category": "4",
                "confidence": confidence,
                "reasoning": "通過台灣南部（含登陸）",
                "features": features,
            }
        else:
            return {
                "predicted_category": "3",
                "confidence": confidence,
                "reasoning": "通過台灣中部（含登陸）",
                "features": features,
            }

    # [5] 海面分類（1/5/7/8）
    if is_north_of_taiwan:
        return {
            "predicted_category": "1",
            "confidence": confidence,
            "reasoning": f"通過台灣北部海面 (heading={heading_deg:.0f}°)",
            "features": features,
        }

    if is_south_of_taiwan:
        if is_westward:
            return {
                "predicted_category": "5",
                "confidence": confidence,
                "reasoning": "通過台灣南部海面向西",
                "features": features,
            }
        elif is_eastward:
            return {
                "predicted_category": "7",
                "confidence": confidence,
                "reasoning": "通過台灣南部海面向東或東北",
                "features": features,
            }
        elif is_northward:
            return {
                "predicted_category": "8",
                "confidence": confidence,
                "reasoning": "通過台灣南部海面向北",
                "features": features,
            }
        else:
            return {
                "predicted_category": "5",
                "confidence": confidence * 0.7,
                "reasoning": f"南部海面方向不明確 (heading={heading_deg:.0f}°)",
                "features": features,
            }

    # [6] 模糊地帶
    if is_north_part:
        if is_westward:
            return {
                "predicted_category": "1",
                "confidence": confidence * 0.8,
                "reasoning": "北部附近向西（偏海面）",
                "features": features,
            }
        return {
            "predicted_category": "2",
            "confidence": confidence * 0.7,
            "reasoning": f"北部附近 (heading={heading_deg:.0f}°)",
            "features": features,
        }

    if is_south_part:
        if is_westward:
            return {
                "predicted_category": "5",
                "confidence": confidence * 0.8,
                "reasoning": "南部附近向西（偏海面）",
                "features": features,
            }
        if is_northward:
            return {
                "predicted_category": "8",
                "confidence": confidence * 0.8,
                "reasoning": "南部附近向北",
                "features": features,
            }
        return {
            "predicted_category": "4",
            "confidence": confidence * 0.7,
            "reasoning": f"南部附近 (heading={heading_deg:.0f}°)",
            "features": features,
        }

    if is_central:
        return {
            "predicted_category": "3",
            "confidence": confidence * 0.7,
            "reasoning": f"中部附近 (lat={closest_lat:.1f}, heading={heading_deg:.0f}°)",
            "features": features,
        }

    return {
        "predicted_category": "9",
        "confidence": 0.3,
        "reasoning": f"無法明確分類 (lat={closest_lat:.1f}, heading={heading_deg:.0f}°, dist={min_dist:.0f}km)",
        "features": features,
    }


class RuleBasedSimilarity(SimilarityBase):
    """
    規則式分類器 v2 — 加權相似度排序

    similarity = 0.6 * path_distance + 0.3 * category_match + 0.1 * intensity_diff
    """

    def __init__(self):
        self._ids: list[str] = []
        self._features_dict = {}
        self._tracks: dict[str, pd.DataFrame] = {}
        self._categories: dict[str, str] = {}
        self._landfall: dict[str, str] = {}

    def fit(self, feature_dict: dict, loader=None):
        self._features_dict = feature_dict
        self._ids = list(feature_dict.keys())

        if loader is not None:
            for rec in loader.records:
                self._tracks[rec.typhoon_id] = rec.track
                self._landfall[rec.typhoon_id] = rec.landfall_location

            for tid in self._ids:
                result = classify_typhoon_by_rules(
                    self._tracks[tid], self._landfall.get(tid)
                )
                self._categories[tid] = result["predicted_category"]

        print(f"✓ 規則式分類器 v2 已擬合 {len(self._ids)} 筆颱風")

    def find_similar(
        self, query_id: str, k: int = 5, exclude_self: bool = True
    ) -> SimilarityResult:
        query_cat = self._categories.get(query_id, "特殊")
        query_vec = self._features_dict[query_id].to_feature_vector()

        candidates = (
            [tid for tid in self._ids if tid != query_id] if exclude_self else self._ids
        )

        # First pass: compute raw distances for normalization
        raw_dists = []
        for tid in candidates:
            other_vec = self._features_dict[tid].to_feature_vector()
            raw_dists.append(float(np.linalg.norm(query_vec - other_vec)))
        max_raw = max(raw_dists) if raw_dists else 1.0

        scored = []
        for i, tid in enumerate(candidates):
            other_vec = self._features_dict[tid].to_feature_vector()
            # Normalized path distance (0-1)
            norm_path = raw_dists[i] / (max_raw + 1e-8)
            cat_penalty = 0.0 if self._categories.get(tid) == query_cat else 1.0
            intensity_diff = abs(query_vec[2] - other_vec[2]) / 100.0
            weighted_dist = 0.4 * norm_path + 0.5 * cat_penalty + 0.1 * intensity_diff
            scored.append((tid, weighted_dist))

        scored.sort(key=lambda x: x[1])
        top_k = scored[:k]

        result_ids = [x[0] for x in top_k]
        result_dists = [x[1] for x in top_k]
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
        path_dist = float(np.linalg.norm(vec_a - vec_b))
        cat_penalty = (
            0.0 if self._categories.get(id_a) == self._categories.get(id_b) else 1.0
        )
        intensity_diff = abs(vec_a[2] - vec_b[2]) / 100.0
        return 0.6 * path_dist + 0.3 * cat_penalty + 0.1 * intensity_diff

    def get_rule_category(self, typhoon_id: str) -> str:
        return self._categories.get(typhoon_id, "特殊")

    def classify_track(
        self, track: pd.DataFrame, landfall_location: str = None
    ) -> dict:
        return classify_typhoon_by_rules(track, landfall_location)
