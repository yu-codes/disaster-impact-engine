"""
執行完整預測流程
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.predict import DisasterImpactPipeline
from src.visualization.plots import TyphoonVisualizer


def main():
    parser = argparse.ArgumentParser(description="颱風類比預測系統")
    parser.add_argument(
        "--method",
        type=str,
        default="knn",
        choices=["knn", "dtw", "combined"],
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

    if args.typhoon_id:
        # 單一預測
        result = pipeline.predict(args.typhoon_id, k=args.k)
        print(f"\n{'='*60}")
        print(f"颱風：{result.name_zh} {result.name_en} ({result.typhoon_id})")
        print(f"真實分類：{result.true_category}")
        print(
            f"預測分類：{result.predicted_category} (信心度: {result.confidence:.1%})"
        )
        print(f"預測{'✓ 正確' if result.is_correct else '✗ 錯誤'}")
        print(f"\n相似颱風：")
        for st in result.similar_typhoons:
            print(
                f"  - {st['name_zh']} ({st['year']}) 類型{st['category']} 距離={st['distance']:.2f}"
            )

        # 視覺化
        viz = TyphoonVisualizer(args.output_dir)
        query_rec = pipeline.loader.get(args.typhoon_id)
        similar_recs = [
            pipeline.loader.get(st["typhoon_id"]) for st in result.similar_typhoons
        ]
        viz.plot_prediction_example(
            query_rec, similar_recs, result.predicted_category, result.confidence
        )
    else:
        # 完整評估
        eval_result = pipeline.evaluate(k=args.k, verbose=True)

        # 儲存結果
        pipeline.save_results(eval_result, args.output_dir)

        # 視覺化
        viz = TyphoonVisualizer(args.output_dir)
        viz.generate_all_prediction_plots(eval_result, pipeline.loader)

    print(f"\n✅ 完成！結果已儲存至 {args.output_dir}/")


if __name__ == "__main__":
    main()
