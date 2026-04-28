"""
簡單的端到端測試
驗證整個系統是否正常運作
"""

import sys
from pathlib import Path

# 添加 src 到路徑
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_imports():
    """測試所有模組的導入"""
    print("🧪 測試模組導入...")
    
    try:
        from src.data import DataLoader
        print("  ✓ DataLoader")
        
        from src.features import TyphoonFeatureExtractor
        print("  ✓ TyphoonFeatureExtractor")
        
        from src.similarity import KNNSimilarity
        print("  ✓ KNNSimilarity")
        
        from src.models import AnalogModel
        print("  ✓ AnalogModel")
        
        from src.impact import ImpactMapper
        print("  ✓ ImpactMapper")
        
        from src.pipeline import DisasterImpactPipeline
        print("  ✓ DisasterImpactPipeline")
        
        return True
    except Exception as e:
        print(f"  ✗ 導入失敗: {e}")
        return False


def test_feature_extraction():
    """測試特徵提取"""
    print("\n🧪 測試特徵提取...")
    
    import numpy as np
    import pandas as pd
    from src.features import TyphoonFeatureExtractor
    
    try:
        extractor = TyphoonFeatureExtractor()
        
        # 創建測試數據
        trajectory = pd.DataFrame({
            'typhoon_id': ['TEST_001'] * 5,
            'lat': [15, 16, 17, 18, 19],
            'lon': [110, 112, 114, 116, 118],
            'max_wind': [100, 120, 150, 140, 130],
            'central_pressure': [980, 960, 950, 960, 970]
        })
        
        features = extractor.extract(trajectory)
        print(f"  ✓ 特徵提取成功")
        print(f"    - 距台灣: {features.distance_to_taiwan:.1f} km")
        print(f"    - 最大風速: {features.max_wind:.1f} km/h")
        print(f"    - 移動速度: {features.speed:.1f} km/h")
        
        return True
    except Exception as e:
        print(f"  ✗ 特徵提取失敗: {e}")
        return False


def test_similarity():
    """測試相似度計算"""
    print("\n🧪 測試相似度計算...")
    
    import numpy as np
    from src.similarity import KNNSimilarity
    
    try:
        # 創建測試數據
        reference = np.random.randn(10, 8)  # 10 個樣本，8 個特徵
        query = reference[0]  # 查詢與第一個樣本相同
        
        similarity = KNNSimilarity(metric='euclidean')
        similarity.fit(reference)
        
        indices, distances = similarity.find_similar(query, k=3)
        
        print(f"  ✓ 相似度計算成功")
        print(f"    - 找到的相似樣本: {len(indices)}")
        print(f"    - 最相似的距離: {distances[0]:.4f}")
        
        return True
    except Exception as e:
        print(f"  ✗ 相似度計算失敗: {e}")
        return False


def test_analog_model():
    """測試類比模型"""
    print("\n🧪 測試類比模型...")
    
    import numpy as np
    from src.models import AnalogModel
    
    try:
        model = AnalogModel(aggregation_method='weighted_mean')
        
        # 測試預測
        indices = [0, 1, 2]
        distances = [1.5, 2.0, 2.5]
        labels = np.array([2, 3, 1, 0, 1, 2, 3, 2, 1, 0])
        
        result = model.predict(indices, distances, labels)
        
        print(f"  ✓ 類比預測成功")
        print(f"    - 預測值: {result['prediction']:.2f}")
        print(f"    - 信心度: {result['confidence']:.2%}")
        
        return True
    except Exception as e:
        print(f"  ✗ 類比預測失敗: {e}")
        return False


def test_impact_mapper():
    """測試災害映射"""
    print("\n🧪 測試災害映射...")
    
    import numpy as np
    import pandas as pd
    from src.impact import ImpactMapper
    
    try:
        mapper = ImpactMapper()
        
        # 創建測試數據
        impact_df = pd.DataFrame({
            'typhoon_id': ['T001', 'T002', 'T003', 'T001'],
            'impact_type': ['flooding', 'damage', 'flooding', 'blackout'],
            'severity': [2, 3, 4, 1]
        })
        
        labels = mapper.create_binary_label(impact_df, 'flooding')
        
        print(f"  ✓ 災害映射成功")
        print(f"    - 生成的標籤: {labels.tolist()}")
        
        return True
    except Exception as e:
        print(f"  ✗ 災害映射失敗: {e}")
        return False


def main():
    """運行所有測試"""
    print("="*60)
    print("🔬 Disaster Impact Engine - 系統測試")
    print("="*60)
    
    tests = [
        test_imports,
        test_feature_extraction,
        test_similarity,
        test_analog_model,
        test_impact_mapper,
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"✗ 測試異常: {e}")
            results.append(False)
    
    # 總結
    print("\n" + "="*60)
    print(f"📊 測試結果: {sum(results)}/{len(results)} 通過")
    
    if all(results):
        print("\n✅ 所有測試通過！系統準備就緒。")
        print("\n📝 下一步:")
        print("   1. python scripts/build_dataset.py  # 生成測試數據")
        print("   2. python scripts/run_prediction.py  # 運行預測")
    else:
        print("\n❌ 部分測試失敗，請檢查錯誤。")
    
    print("="*60)


if __name__ == '__main__':
    main()
