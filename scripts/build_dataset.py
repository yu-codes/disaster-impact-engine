"""
數據集構建腳本
職責：從 typhoon_information_overview.xlsx 篩選有「侵臺路徑分類」的颱風，
      合併 ibtracs 路徑資料，輸出整合後的 JSON 到 data/processed/
"""

import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path
from datetime import datetime


RAW_DIR = Path("data/raw")
IBTRACS_DIR = RAW_DIR / "typhoon_information_ibtracs"
OVERVIEW_FILE = RAW_DIR / "typhoon_information_overview.xlsx"
PROCESSED_DIR = Path("data/processed")


def load_overview() -> pd.DataFrame:
    return pd.read_excel(OVERVIEW_FILE)


def filter_typhoons_with_track_category(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["侵臺路徑分類"].notna() & (df["侵臺路徑分類"] != "---")
    filtered = df[mask].copy()
    print(f"✓ 篩選完成：{len(filtered)} / {len(df)} 筆颱風有侵臺路徑分類")
    return filtered


def load_ibtracs_track(year: int, typhoon_id: str) -> list | None:
    json_path = (
        IBTRACS_DIR / str(year) / str(typhoon_id) / "ibtracs_position_intensity.json"
    )
    if not json_path.exists():
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("position_intensity", None)


def parse_wind_speed(raw) -> float | None:
    if pd.isna(raw):
        return None
    try:
        return float(str(raw).split("(")[0].strip())
    except (ValueError, IndexError):
        return None


def build_typhoon_record(row: pd.Series) -> dict | None:
    typhoon_id = str(row["颱風編號"])
    year = int(row["年份"])

    track = load_ibtracs_track(year, typhoon_id)
    if track is None or len(track) == 0:
        return None

    return {
        "typhoon_id": typhoon_id,
        "year": year,
        "name_zh": row["中文名稱"],
        "name_en": row["英文名稱"],
        "genesis_time": str(row["生成時間"]) if pd.notna(row["生成時間"]) else None,
        "dissipation_time": str(row["消散時間"]) if pd.notna(row["消散時間"]) else None,
        "birth_location": {
            "longitude": float(row["生成經度"]) if pd.notna(row["生成經度"]) else None,
            "latitude": float(row["生成緯度"]) if pd.notna(row["生成緯度"]) else None,
        },
        "max_intensity_value": (
            float(row["最大強度值"]) if pd.notna(row["最大強度值"]) else None
        ),
        "max_intensity_class": row["最大強度"] if pd.notna(row["最大強度"]) else None,
        "max_sustained_wind_ms": parse_wind_speed(row["近中心最大風速"]),
        "min_pressure": float(row["最低氣壓"]) if pd.notna(row["最低氣壓"]) else None,
        "taiwan_track_category": str(row["侵臺路徑分類"]),
        "landfall_location": row["登陸地段"] if pd.notna(row["登陸地段"]) else None,
        "movement_summary": row["動態"] if pd.notna(row["動態"]) else None,
        "disaster_summary": row["災情"] if pd.notna(row["災情"]) else None,
        "warning_report_count": (
            str(row["發布報數"]) if pd.notna(row["發布報數"]) else None
        ),
        "track_point_count": len(track),
        "track": track,
    }


def build_dataset(df: pd.DataFrame) -> list:
    records = []
    skipped = []
    for _, row in df.iterrows():
        record = build_typhoon_record(row)
        if record is not None:
            records.append(record)
        else:
            skipped.append(f"{row['颱風編號']} {row['英文名稱']} ({row['年份']})")

    if skipped:
        print(f"⚠ 跳過 {len(skipped)} 筆（無路徑資料）：")
        for s in skipped:
            print(f"    - {s}")
    print(f"✓ 成功構建 {len(records)} 筆颱風記錄")
    return records


def save_dataset(records: list, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    # 完整資料集
    full_path = output_dir / "typhoons_with_tracks.json"
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at": datetime.now().isoformat(),
                "description": "侵臺颱風完整資料（含 IBTrACS 路徑）",
                "typhoon_count": len(records),
                "track_categories": sorted(
                    set(r["taiwan_track_category"] for r in records)
                ),
                "typhoons": records,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"✓ 完整資料集已儲存：{full_path}")

    # 索引檔
    index_records = [
        {
            "typhoon_id": r["typhoon_id"],
            "year": r["year"],
            "name_zh": r["name_zh"],
            "name_en": r["name_en"],
            "taiwan_track_category": r["taiwan_track_category"],
            "birth_lon": r["birth_location"]["longitude"],
            "birth_lat": r["birth_location"]["latitude"],
            "max_sustained_wind_ms": r["max_sustained_wind_ms"],
            "min_pressure": r["min_pressure"],
            "max_intensity_class": r["max_intensity_class"],
            "landfall_location": r["landfall_location"],
            "track_point_count": r["track_point_count"],
        }
        for r in records
    ]

    index_path = output_dir / "typhoons_index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_records, f, ensure_ascii=False, indent=2)
    print(f"✓ 索引檔已儲存：{index_path}")

    # 統計摘要
    cat_counts = {}
    for r in records:
        cat = r["taiwan_track_category"]
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    summary = {
        "total_typhoons": len(records),
        "category_distribution": dict(sorted(cat_counts.items())),
        "year_range": [
            min(r["year"] for r in records),
            max(r["year"] for r in records),
        ],
    }
    summary_path = output_dir / "dataset_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"✓ 摘要已儲存：{summary_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="構建侵臺颱風資料集")
    parser.add_argument("--output-dir", type=str, default="data/processed")
    args = parser.parse_args()

    print("=" * 60)
    print("🌀 構建侵臺颱風資料集")
    print("=" * 60)

    print("\n📂 載入颱風總覽...")
    overview_df = load_overview()

    print("\n🔍 篩選有侵臺路徑分類的颱風...")
    filtered_df = filter_typhoons_with_track_category(overview_df)

    # 只保留有 IBTrACS 匹配的
    matched_df = filtered_df[filtered_df["IBTrACS是否匹配"] == "是"].copy()
    print(f"✓ 有 IBTrACS 路徑資料的：{len(matched_df)} 筆")

    print("\n🔨 構建資料集...")
    records = build_dataset(matched_df)

    print("\n💾 儲存資料集...")
    summary = save_dataset(records, Path(args.output_dir))

    print(f"\n{'='*60}")
    print(f"📊 資料集摘要")
    print(f"{'='*60}")
    print(f"  總颱風數：{summary['total_typhoons']}")
    print(f"  年份範圍：{summary['year_range'][0]} ~ {summary['year_range'][1]}")
    print(f"  路徑分類分布：")
    for cat, count in sorted(summary["category_distribution"].items()):
        print(f"    類型 {cat}：{count} 筆")
    print("\n✅ 資料集構建完成！")


if __name__ == "__main__":
    main()
