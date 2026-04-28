#!/usr/bin/env python3
"""
快速啟動指南 - 互動式項目演示
"""

import sys
from pathlib import Path

def main():
    print("=" * 70)
    print("🌀 災害影響預測引擎 - Disaster Impact Engine")
    print("=" * 70)
    print()
    print("📋 項目概述:")
    print("-" * 70)
    print()
    print("這是一個輕量但完整的颱風類比災害預測系統。")
    print("基於歷史颱風數據進行相似度匹配，預測災害影響。")
    print()
    print("🎯 核心流程:")
    print("-" * 70)
    print("""
    原始颱風數據
         ↓
    特徵提取（8 個特徵）
         ↓
    相似度計算（KNN）
         ↓
    找到 K 個最相似颱風
         ↓
    類比預測（加權平均）
         ↓
    災害預測結果
    """)
    
    print("📂 項目結構:")
    print("-" * 70)
    print("""
    src/                        # 核心模組
    ├── data/                   # 數據加載
    ├── features/               # 特徵提取
    ├── similarity/             # 相似度計算
    ├── models/                 # 預測模型
    ├── impact/                 # 災害標籤
    └── pipeline/               # 完整流程
    
    scripts/                    # 可執行腳本
    ├── build_dataset.py        # 數據構建
    └── run_prediction.py       # 預測運行
    
    data/                       # 數據目錄
    ├── raw/                    # 原始數據
    └── processed/              # 處理後數據
    """)
    
    print("🚀 快速開始:")
    print("-" * 70)
    print("""
    1. 安裝依賴:
       pip install -r requirements.txt
    
    2. 生成測試數據:
       python scripts/build_dataset.py --num-typhoons 50
    
    3. 運行預測:
       python scripts/run_prediction.py
    
    4. 查看結果:
       cat results/predictions.json
    """)
    
    print("📚 文檔:")
    print("-" * 70)
    print("""
    - README.md               完整使用指南
    - QUICKSTART.md           快速開始指南
    - PROJECT_SUMMARY.md      項目總結
    - IMPLEMENTATION_REPORT.md 完成報告
    """)
    
    print("🔧 核心模組:")
    print("-" * 70)
    print("""
    DataLoader                    讀取颱風/災害數據
    TyphoonFeatureExtractor       提取 8 個特徵
    KNNSimilarity                 K-Nearest Neighbors 相似度
    AnalogModel                   類比預測模型
    ImpactMapper                  災害標籤映射
    DisasterImpactPipeline        完整預測流程
    """)
    
    print("✨ 特色:")
    print("-" * 70)
    print("""
    ✓ 輕量設計 - 最小化依賴，快速上手
    ✓ 完整實現 - 所有功能都已實現
    ✓ 高度可擴展 - 易於替換和增強
    ✓ 文檔完善 - 詳細的代碼注釋和指南
    ✓ 開箱即用 - 包含示例數據和腳本
    ✓ 模塊化設計 - 清晰的職責分離
    """)
    
    print("=" * 70)
    print("✅ 項目已完成，準備就緒！")
    print("=" * 70)
    print()
    print("建議下一步:")
    print("  1. 閱讀 README.md 了解詳細信息")
    print("  2. 運行 python test_system.py 驗證環境")
    print("  3. 執行 python scripts/build_dataset.py 生成測試數據")
    print("  4. 運行 python scripts/run_prediction.py 進行預測")
    print()

if __name__ == '__main__':
    main()
