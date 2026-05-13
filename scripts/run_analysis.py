"""
執行資料預分析 (EDA)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.typhoon.loader import DataLoader
from src.features.typhoon.extractor import TyphoonFeatureExtractor
from src.analysis.typhoon.eda import TyphoonEDA
from src.analysis.typhoon.rainfall import RainfallAnalyzer
from src.visualization.typhoon.plots import TyphoonVisualizer


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

    # 視覺化 — 路徑分析圖
    viz = TyphoonVisualizer("experiments/analysis")
    viz.generate_all_analysis_plots(loader, features)

    # 視覺化 — 降水 EDA 圖
    print("\n🌧️ 載入降水資料...")
    rainfall = RainfallAnalyzer()
    rainfall.load()
    viz.generate_all_rainfall_eda_plots(rainfall._records, loader, features)

    print("\n✅ EDA 分析完成！圖表已儲存至 experiments/analysis/")


if __name__ == "__main__":
    main()
