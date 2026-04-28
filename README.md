# 🌀 Disaster Impact Engine — 颱風類比災害預測系統

基於**歷史類比方法**，透過颱風軌跡特徵和 DTW 路徑比對，找出最相似的歷史颱風並預測侵臺路徑分類。

## 核心流程

```
原始資料 (xlsx + IBTrACS JSON)
    ↓  scripts/build_dataset.py
處理後資料 (data/processed/)
    ↓  scripts/run_analysis.py
EDA 分析 + 視覺化
    ↓  scripts/run_prediction.py
特徵提取 → 相似度計算 (KNN / DTW / Combined) → 預測 + 評估 + 視覺化
```

## 方法概述

1. **空間標準化**：以台灣為原點 (23.7°N, 121°E)，計算極座標 (r, θ)
2. **時間對齊**：透過 DTW 比對 500km 影響窗口內的多維序列
3. **多維特徵**：11 維特徵向量（最近距離、方位角、風速、氣壓、移動速度、降雨代理等）
4. **混合評分**：Combined Score = α × 特徵距離 + (1−α) × DTW 距離

## 評估結果 (Leave-One-Out)

| 方法 | 總準確率 | 說明 |
|------|---------|------|
| KNN | 52.7% | 僅用 11 維特徵向量 |
| Combined (α=0.5) | **60.9%** | KNN + DTW 混合 |

## 目錄結構

```
disaster-impact-engine/
├── src/
│   ├── data/loader.py              # 資料載入（JSON → TyphoonRecord）
│   ├── features/typhoon.py         # 特徵工程（11 維向量 + 影響窗口序列）
│   ├── similarity/
│   │   ├── base.py                 # 抽象介面
│   │   ├── knn.py                  # KNN（標準化歐氏距離）
│   │   ├── dtw.py                  # DTW（多維路徑比對）
│   │   └── combined.py             # 混合相似度
│   ├── models/analog.py            # 加權投票預測
│   ├── impact/mapping.py           # 路徑分類說明對照
│   ├── analysis/eda.py             # 資料探索性分析
│   ├── visualization/plots.py      # 所有圖表（分析 + 預測）
│   └── pipeline/predict.py         # 完整 pipeline + LOO 評估
├── scripts/
│   ├── build_dataset.py            # xlsx + ibtracs → JSON
│   ├── run_analysis.py             # EDA + 視覺化
│   └── run_prediction.py           # 預測 + 評估 + 視覺化
├── data/
│   ├── raw/                        # 原始資料
│   └── processed/                  # 處理後 JSON
├── outputs/                        # 圖表與結果
└── docs/strategy.md                # 方法策略文件
```

## 使用方式

### 安裝依賴

```bash
pip install pandas numpy scikit-learn matplotlib seaborn dtaidistance openpyxl
```

### 1. 構建資料集

```bash
python scripts/build_dataset.py
```

從 `data/raw/typhoon_information_overview.xlsx` 篩選 207 筆有侵臺路徑分類且有 IBTrACS 匹配的颱風，輸出至 `data/processed/`。

### 2. 探索性分析

```bash
python scripts/run_analysis.py
```

產出分布、軌跡、生成位置、強度、特徵相關性等圖表至 `outputs/analysis/`。

### 3. 預測與評估

```bash
# KNN 方法
python scripts/run_prediction.py --method knn --k 5

# Combined（KNN + DTW）
python scripts/run_prediction.py --method combined --alpha 0.5 --k 5

# 預測單一颱風
python scripts/run_prediction.py --typhoon-id 202411 --method combined
```

結果（混淆矩陣、各類準確率、範例軌跡圖）儲存至 `outputs/predictions/`。

## 資料來源

- 中央氣象署颱風資料庫（颱風總覽 xlsx）
- IBTrACS（6 小時間距路徑強度資料）
- 侵臺路徑分類（1–9 類 + 特殊）：CWA 官方分類
