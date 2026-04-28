"""
執行資料預分析 (EDA)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import DataLoader
from src.features.typhoon import TyphoonFeatureExtractor
from src.analysis.eda import TyphoonEDA
from src.visualization.plots import TyphoonVisualizer


def main():
    print("=" * 60)
    print("📊 颱風資料探索性分析 (EDA)")
    print("=" * 60)

    # 載入資料
    loader = DataLoader("data/processed")
    loader.load()

    # EDA
    eda = TyphoonEDA(loader)
    eda.print_summary()

    # 提取特徵
    print("\n🔧 提取特徵...")
    extractor = TyphoonFeatureExtractor()
    features = extractor.extract_all(loader)

    # 視覺化
    viz = TyphoonVisualizer("outputs/analysis")
    viz.generate_all_analysis_plots(loader, features)

    print("\n✅ EDA 分析完成！圖表已儲存至 outputs/analysis/")


if __name__ == "__main__":
    main()
