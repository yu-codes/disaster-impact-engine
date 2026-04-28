"""
執行完整預測流程

支援版本控制：每次執行自動建立時間戳子目錄，並記錄執行參數。
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.predict import DisasterImpactPipeline
from src.visualization.plots import TyphoonVisualizer


def get_fixed_example_ids(loader) -> dict[str, str]:
    """
    為每個分類選一個固定的範例颱風 ID。
    規則：每個分類取資料集中第一筆（按 typhoon_id 排序），確保每次相同。
    """
    by_cat: dict[str, list[str]] = {}
    for rec in loader.records:
        cat = rec.taiwan_track_category
        by_cat.setdefault(cat, []).append(rec.typhoon_id)
    # 每類取排序後的第一個
    return {cat: sorted(ids)[0] for cat, ids in by_cat.items()}


def run_single(pipeline, args):
    """單一預測"""
    result = pipeline.predict(args.typhoon_id, k=args.k)
    print(f"\n{'='*60}")
    print(f"颱風：{result.name_zh} {result.name_en} ({result.typhoon_id})")
    print(f"真實分類：{result.true_category}")
    print(f"預測分類：{result.predicted_category} (信心度: {result.confidence:.1%})")
    print(f"預測{'✓ 正確' if result.is_correct else '✗ 錯誤'}")
    print(f"\n相似颱風：")
    for st in result.similar_typhoons:
        print(
            f"  - {st['name_zh']} ({st['year']}) 類型{st['category']} 距離={st['distance']:.2f}"
        )
    return result


def run_evaluation(pipeline, args, output_dir: Path):
    """完整評估 + 版本控制輸出"""
    # 建立時間戳子目錄
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    # 完整評估
    eval_result = pipeline.evaluate(k=args.k, verbose=True)

    # 儲存結果
    pipeline.save_results(eval_result, str(run_dir))

    # 儲存執行元資料
    meta = {
        "timestamp": timestamp,
        "method": args.method,
        "alpha": args.alpha,
        "k": args.k,
        "accuracy": round(eval_result["accuracy"], 4),
        "total": eval_result["total"],
        "correct": eval_result["correct"],
        "per_category": eval_result["per_category"],
    }
    with open(run_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"✓ 執行元資料已儲存至 {run_dir / 'run_meta.json'}")

    # 固定範例 ID
    fixed_ids = get_fixed_example_ids(pipeline.loader)
    with open(run_dir / "fixed_example_ids.json", "w", encoding="utf-8") as f:
        json.dump(fixed_ids, f, ensure_ascii=False, indent=2)

    # 視覺化
    viz = TyphoonVisualizer(str(run_dir))
    viz.generate_all_prediction_plots(
        eval_result, pipeline.loader, fixed_example_ids=fixed_ids
    )

    print(f"\n✅ 完成！結果已儲存至 {run_dir}/")
    return run_dir


def main():
    parser = argparse.ArgumentParser(description="颱風類比預測系統")
    parser.add_argument(
        "--method",
        type=str,
        default="knn",
        choices=["knn", "dtw", "combined", "baseline", "rule_based"],
        help="相似度計算方法",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5, help="combined 模式下特徵權重 (0~1)"
    )
    parser.add_argument("--k", type=int, default=5, help="相似颱風數量")
    parser.add_argument(
        "--typhoon-id",
        type=str,
        default=None,
        help="特定颱風 ID（若指定，只預測該颱風）",
    )
    parser.add_argument("--processed-dir", type=str, default="data/processed")
    parser.add_argument("--output-dir", type=str, default="outputs/predictions")
    args = parser.parse_args()

    print("=" * 60)
    print("🌀 颱風類比災害預測系統")
    print("=" * 60)

    # 初始化 pipeline
    pipeline = DisasterImpactPipeline(
        similarity_method=args.method,
        alpha=args.alpha,
    )
    pipeline.initialize(args.processed_dir)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.typhoon_id:
        run_single(pipeline, args)
    else:
        run_evaluation(pipeline, args, output_dir)


if __name__ == "__main__":
    main()
