# 🌀 Disaster Impact Engine — 颱風類比災害預測系統

基於**歷史類比方法**，透過颱風軌跡特徵和 DTW 路徑比對，找出最相似的歷史颱風並預測侵臺路徑分類。

## 快速開始

### 環境需求

- Python ≥ 3.10
- pip

### 安裝

```bash
# 建立虛擬環境（建議）
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或 .venv\Scripts\activate  # Windows

# 安裝依賴
pip install -r requirements.txt

# 或使用 pip install -e .（開發模式）
pip install -e .
```

### 啟動 Web 服務

```bash
# 啟動 Flask 開發伺服器
python web/app.py

# 服務啟動後打開瀏覽器：
# http://localhost:5000
```

Web 介面提供：
- **首頁**：系統概述與方法說明
- **資料分析**：歷史颱風 EDA 圖表
- **預測結果**：各版本預測實驗結果
- **線上預測**：輸入颱風路徑即時預測

### 完整工作流程

```bash
# 1. 構建資料集（從原始 xlsx + IBTrACS → JSON）
python scripts/build_dataset.py

# 2. 探索性分析（統計 + 視覺化）
python scripts/run_analysis.py

# 3. 執行預測（多種方法）
python scripts/run_prediction.py --method combined --k 5
python scripts/run_prediction.py --method rule_based --k 5

# 4. 啟動 Web 介面
python web/app.py
```

## 系統架構

```
原始資料 (xlsx + IBTrACS JSON)
    ↓  scripts/build_dataset.py
處理後資料 (data/processed/)
    ↓  scripts/run_analysis.py
EDA 分析 + 視覺化
    ↓  scripts/run_prediction.py
特徵提取 → 相似度計算 → 預測 + 評估 + 視覺化
    ↓  web/app.py
Web 前端展示 + 線上預測 API
```

## 兩種預測方法

### 方法一：類比相似度預測法 (Analog Similarity Method)

結合 DTW 路徑對齊 + KNN 特徵距離 + Rule-Based 前置分類的混合方法。

**核心流程：**
1. 空間標準化（修正座標 + 極座標）
2. Impact Window 提取（r < 300km）
3. DTW 路徑相似度（環形θ + Sakoe-Chiba）
4. KNN 特徵距離（11維向量）
5. Combined Score（0.6×DTW + 0.4×KNN + 同類折扣）

詳見 [`docs/analog_similarity_method.md`](docs/analog_similarity_method.md)

### 方法二：規則式路徑分類法 (Rule-Based Classification)

基於 CWA 官方分類定義，透過幾何特徵自動判斷路徑類型。

**核心流程：**
1. 雙層 Impact Window（core=200km, context=400km）
2. 向量平均方向計算
3. 登陸判斷（data > bbox > distance 三級優先）
4. 優先順序分類（type6 → 特殊 → 無影響 → 登陸 → 海面）

詳見 [`docs/rule_based_method.md`](docs/rule_based_method.md)

## 使用方式

### 預測指令

```bash
# Combined 方法（最佳）
python scripts/run_prediction.py --method combined --alpha 0.4 --k 5

# Rule-Based 方法
python scripts/run_prediction.py --method rule_based --k 5

# KNN 方法
python scripts/run_prediction.py --method knn --k 5

# DTW 方法
python scripts/run_prediction.py --method dtw --k 5

# 預測單一颱風
python scripts/run_prediction.py --typhoon-id 202411 --method combined
```

### API 使用

```bash
# 啟動服務
python web/app.py

# POST 預測 API
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "track": [
      {"latitude": 15.0, "longitude": 135.0, "wind_kt": 25, "pressure_mb": 1002},
      {"latitude": 18.0, "longitude": 129.0, "wind_kt": 50, "pressure_mb": 980},
      {"latitude": 22.5, "longitude": 122.0, "wind_kt": 80, "pressure_mb": 950}
    ],
    "method": "combined",
    "k": 5
  }'
```

## 目錄結構

```
disaster-impact-engine/
├── src/
│   ├── data/loader.py              # 資料載入（JSON → TyphoonRecord）
│   ├── features/typhoon.py         # 特徵工程 v2（修正座標 + 300km window）
│   ├── similarity/
│   │   ├── base.py                 # 抽象介面
│   │   ├── knn.py                  # KNN（標準化歐氏距離）
│   │   ├── dtw.py                  # DTW v2（環形距離 + 物理標準化 + Sakoe-Chiba）
│   │   ├── combined.py             # Combined v2（Rule-Based filter + DTW + KNN）
│   │   ├── rule_based.py           # 規則式分類 v2（雙層window + 向量平均）
│   │   └── baseline.py             # 隨機基線（對照組）
│   ├── models/analog.py            # 加權投票預測
│   ├── impact/mapping.py           # 路徑分類說明對照
│   ├── analysis/eda.py             # 資料探索性分析
│   ├── visualization/plots.py      # 所有圖表
│   └── pipeline/predict.py         # 完整 pipeline + LOO 評估
├── scripts/
│   ├── build_dataset.py            # xlsx + IBTrACS → JSON
│   ├── run_analysis.py             # EDA + 視覺化
│   ├── run_prediction.py           # 預測 + 評估
│   └── run_all_predictions.py      # 批量執行多組預測
├── web/
│   ├── app.py                      # Flask Web 應用
│   ├── templates/                  # HTML 模板
│   └── static/                     # CSS/JS 靜態檔案
├── data/
│   ├── raw/                        # 原始資料
│   └── processed/                  # 處理後 JSON
├── outputs/
│   ├── analysis/                   # EDA 圖表
│   └── predictions/                # 預測結果（按版本）
├── docs/
│   ├── analog_similarity_method.md # 類比相似度預測法文件
│   └── rule_based_method.md        # 規則式分類法文件
├── requirements.txt
└── pyproject.toml
```

## 開發指南

### 開發環境設定

```bash
# 1. Clone
git clone <repo-url>
cd disaster-impact-engine

# 2. 建立虛擬環境
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. 安裝開發依賴
pip install -e .
pip install flask

# 4. 確認資料集存在
ls data/processed/typhoons_with_tracks.json

# 5. 啟動開發伺服器（支援 hot-reload）
python web/app.py
```

### 新增預測方法

1. 在 `src/similarity/` 新增模組，繼承 `SimilarityBase`
2. 實作 `fit()`, `find_similar()`, `compute_distance()`
3. 在 `src/pipeline/predict.py` 註冊方法
4. 在 `scripts/run_prediction.py` 加入 choices

### 測試流程

```bash
# 執行完整評估（所有方法）
python scripts/run_all_predictions.py

# 單一方法測試
python scripts/run_prediction.py --method combined --k 5
```

## 資料來源

- **中央氣象署颱風資料庫**：颱風總覽（1958-2025）
- **IBTrACS**：6 小時間距路徑強度資料
- **侵臺路徑分類**：CWA 官方 1–9 類 + 特殊

## 技術細節

### v2 改進摘要

| 項目 | v1 | v2 |
|------|-----|-----|
| 座標轉換 | dx = lon - 121 | dx = (lon-121) × cos(lat) |
| Impact Window | 500km | 300km + 距離加權 |
| DTW θ距離 | 歐氏距離 | 環形距離 min(\|Δθ\|, 2π-\|Δθ\|) |
| DTW 標準化 | 統計標準化 | 物理標準化 (r/300, θ/π, wind/100) |
| DTW 限制 | 無 | Sakoe-Chiba band (30%) |
| Rain proxy | wind / distance | wind × cos(θ-θ_normal) / distance |
| 規則方向 | dlon < -0.5 | arctan2 角度判斷 |
| 規則 window | 500km 單層 | 200km core + 400km context |
| 規則位置 | mean_lat | closest_lat |
| Combined | α×KNN + (1-α)×DTW | + Rule-Based soft filter + 同類折扣 |
