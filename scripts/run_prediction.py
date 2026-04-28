"""
主預測運行腳本
職責：端到端的預測流程示範
"""

import sys
import json
import argparse
from pathlib import Path

# 添加 src 到 Python 路徑
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from pipeline.predict import DisasterImpactPipeline
from similarity.knn import KNNSimilarity
from models.analog import AnalogModel
from features.typhoon import TyphoonFeatureExtractor
from impact.mapping import ImpactMapper


def run_single_prediction(pipeline, typhoon_id: str, verbose: bool = True):
    """
    對單個颱風進行預測
    
    Args:
        pipeline: 預測流程
        typhoon_id: 颱風 ID
        verbose: 是否輸出詳細信息
    """
    try:
        result = pipeline.predict(typhoon_id, k=5)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"颱風: {result.typhoon_id}")
            print(f"{'='*60}")
            
            print(f"\n📍 特徵:")
            print(f"  - 距台灣: {result.query_features.distance_to_taiwan:.1f} km")
            print(f"  - 方位角: {result.query_features.azimuth:.1f}°")
            print(f"  - 最大風速: {result.query_features.max_wind:.1f} km/h")
            print(f"  - 移動速度: {result.query_features.speed:.1f} km/h")
            
            print(f"\n🔍 相似颱風:")
            for i, (tid, dist) in enumerate(zip(result.similar_typhoon_ids, result.similar_distances), 1):
                print(f"  {i}. {tid} (距離: {dist:.2f})")
            
            print(f"\n🎯 災害預測:")
            for impact_type, prediction in result.predictions.items():
                pred_val = prediction.get('prediction', 0)
                confidence = prediction.get('confidence', 0)
                print(f"  - {impact_type}: {pred_val:.2f} (信心: {confidence:.2%})")
        
        return result
    
    except Exception as e:
        print(f"✗ 預測失敗 {typhoon_id}: {e}")
        return None


def run_batch_prediction(pipeline, typhoon_ids: list, output_file: str = None):
    """
    批量預測
    
    Args:
        pipeline: 預測流程
        typhoon_ids: 颱風 ID 列表
        output_file: 輸出 JSON 文件（可選）
    """
    print(f"\n🚀 執行批量預測... ({len(typhoon_ids)} 個颱風)")
    
    results = pipeline.predict_batch(typhoon_ids, k=5)
    
    print(f"✓ 完成: {len(results)} 個預測成功\n")
    
    # 保存結果
    if output_file:
        output_data = [r.to_dict() for r in results]
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        print(f"✓ 結果已保存: {output_file}")
    
    return results


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='災害影響預測系統')
    parser.add_argument('--typhoon-data', type=str, default='data/raw/typhoon.csv',
                        help='颱風數據路徑')
    parser.add_argument('--impact-data', type=str, default='data/raw/impact.csv',
                        help='災害影響數據路徑')
    parser.add_argument('--typhoon-id', type=str, default=None,
                        help='特定颱風 ID（若指定，則只預測該颱風）')
    parser.add_argument('--output', type=str, default='results/predictions.json',
                        help='結果輸出文件')
    parser.add_argument('--k', type=int, default=5,
                        help='相似颱風數量')
    parser.add_argument('--similarity', type=str, default='knn',
                        choices=['knn'],
                        help='相似度計算方法')
    parser.add_argument('--model', type=str, default='analog',
                        choices=['analog'],
                        help='預測模型')
    parser.add_argument('--aggregation', type=str, default='weighted_mean',
                        choices=['mean', 'weighted_mean', 'majority_vote', 'max'],
                        help='聚合方式')
    
    args = parser.parse_args()
    
    print("🌀 災害影響預測系統")
    print("="*60)
    
    # 初始化模型
    print("\n📦 初始化模型...")
    
    similarity_model = KNNSimilarity(metric='euclidean')
    prediction_model = AnalogModel(aggregation_method=args.aggregation)
    
    # 初始化流程
    pipeline = DisasterImpactPipeline(
        similarity_model=similarity_model,
        prediction_model=prediction_model,
        feature_extractor=TyphoonFeatureExtractor(),
        impact_mapper=ImpactMapper()
    )
    
    # 加載數據
    print(f"📂 加載數據...")
    try:
        pipeline.initialize(args.typhoon_data, args.impact_data)
    except FileNotFoundError as e:
        print(f"✗ 錯誤: {e}")
        print(f"\n💡 請先運行:")
        print(f"   python scripts/build_dataset.py")
        return
    
    # 執行預測
    if args.typhoon_id:
        # 單個預測
        result = run_single_prediction(pipeline, args.typhoon_id, verbose=True)
        
        # 保存結果
        if result and args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, 'w') as f:
                json.dump(result.to_dict(), f, indent=2, default=str)
    
    else:
        # 批量預測
        all_typhoon_ids = pipeline.reference_typhoons
        
        results = run_batch_prediction(pipeline, all_typhoon_ids, args.output)
        
        # 簡要統計
        print(f"\n📊 預測統計:")
        print(f"  - 總計: {len(results)} 個")
        
        if results:
            avg_confidence = {}
            for result in results:
                for impact_type, pred in result.predictions.items():
                    if impact_type not in avg_confidence:
                        avg_confidence[impact_type] = []
                    avg_confidence[impact_type].append(pred.get('confidence', 0))
            
            print(f"\n  平均信心度:")
            for impact_type, confidences in avg_confidence.items():
                avg = sum(confidences) / len(confidences)
                print(f"    - {impact_type}: {avg:.2%}")
    
    print(f"\n✅ 預測完成！")


if __name__ == '__main__':
    main()
