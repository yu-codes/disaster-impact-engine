# 🌀 Disaster Impact Engine — 多災害類比預測系統

基於**歷史類比方法**的多災害影響分析平台。目前支援颱風路徑分類預測與降水分析，架構設計可擴展至其他災害類型。

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
- **首頁**：系統概述、多災害模組切換
- **資料分析**：颱風路徑分析 + 降水分析（分頁切換）
- **預測結果**：各版本實驗結果（含降水機率分布）
- **線上預測**：輸入颱風路徑即時預測

### 完整工作流程

```bash
# 1. 構建資料集（從原始 xlsx + IBTrACS → JSON）
python scripts/build_dataset.py

# 2. 探索性分析（統計 + 視覺化）
python scripts/run_analysis.py

# 3. 執行實驗（推薦方式）
python experiments/exp001_combined_rrf.py
python experiments/exp002_rule_based.py

# 或使用通用 runner
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

## 侵臺颱風路徑分類（CWA 定義）

中央氣象署將侵臺颱風路徑分為 9 類：

| 分類 | 描述 | 資料筆數 |
|------|------|---------|
| 1 | 通過台灣北部海面向西或西北西進行者 | 23 |
| 2 | 通過台灣北部向西或西北進行者（含登陸北部） | 29 |
| 3 | 通過台灣中部向西進行者（含登陸中部） | 30 |
| 4 | 通過台灣南部向西進行者（含登陸南部） | 21 |
| 5 | 通過台灣南部海面向西進行者 | 30 |
| 6 | 沿台灣東岸或東部海面北上者 | 30 |
| 7 | 通過台灣南部海面向東或東北進行者 | 11 |
| 8 | 通過台灣南部海面向北或北北西進行者 | 6 |
| 9 | 特殊路徑或對台灣有影響但無侵襲者 | 18 |

> 評估時僅使用分類 1-9（共 198 筆），排除「特殊」標記的 9 筆。

## 兩種預測方法（獨立運作）

### Combined RRF（KNN + DTW + Rule-Based 排名融合）

結合 KNN 特徵距離與 DTW 時序路徑對齊，使用 Reciprocal Rank Fusion 融合排名。

**Pipeline：** 極座標轉換 → Impact Window (500km) → KNN 排名 + DTW 排名 + Rule 排名 → RRF 融合 → Top-K 投票

**核心參數：** α=0.13, rule_weight=0.25, rrf_k=60, pool_size_factor=10, Sakoe-Chiba 30%

**準確率：** 72.2%（198 筆 LOO，類別 1-9）

### Rule-Based Classification（幾何規則分類）

基於 CWA 官方路徑分類定義，透過軌跡幾何特徵（接近方向、穿越判斷、登陸地點）直接判定路徑類型。

**Pipeline：** Haversine 距離 → 最近點 → 接近方向 → 穿越判斷 → 登陸解析 → 規則分類

**準確率：** 75.8%（198 筆 LOO，類別 1-9）

### 降水分析模組

基於類比結果分析臺南、高雄兩站的事件降水量，生成降水機率分布與損失指標。

- **資料**：440 筆颱風事件降水記錄，207 筆與颱風索引配對
- **指標**：MAE、RMSE（實際降水 vs 類比降水均值）
- **輸出**：散佈圖、箱型圖、誤差分布圖、分類別降水統計

## 使用方式

### 新增預測配置流程

1. 在 `configs/experiments/` 建立新的 YAML 配置檔
2. 在 `pipelines/` 建立對應的 pipeline 腳本（或使用通用 runner）
3. 執行預測

```bash
# 方法一：使用獨立 pipeline 腳本
python pipelines/combined_rrf.py
python pipelines/rule_based.py

# 方法二：使用通用 runner + 指定配置檔
python scripts/run_prediction.py --config configs/experiments/combined_rrf.yaml
python scripts/run_prediction.py --config configs/experiments/rule_based.yaml

# 方法三：傳統參數模式（向下相容）
python scripts/run_prediction.py --method combined --alpha 0.13 --k 5
python scripts/run_prediction.py --method rule_based --k 5
```

### 新增預設配置流程

1. 複製現有配置為模板：
   ```bash
   cp configs/experiments/combined_rrf.yaml configs/experiments/my_experiment.yaml
   ```

2. 編輯配置參數：
   ```yaml
   name: "my_experiment"
   method: "combined"
   parameters:
     alpha: 0.15
     rule_weight: 0.3
     k: 7
     ...
   ```

3. 建立對應 pipeline（可選）：
   ```bash
   cp pipelines/combined_rrf.py pipelines/my_experiment.py
   # 修改 DEFAULT_CONFIG 路徑
   ```

4. 執行預測：
   ```bash
   python scripts/run_prediction.py --config configs/experiments/my_experiment.yaml
   ```

5. 結果自動存至 `outputs/predictions/{timestamp}_{name}/`

### 預測指令

```bash
# 使用 YAML 設定檔（推薦）
python scripts/run_prediction.py --config configs/experiments/combined_rrf.yaml
python scripts/run_prediction.py --config configs/experiments/rule_based.yaml

# 傳統參數模式（向下相容）
python scripts/run_prediction.py --method combined --alpha 0.2 --k 5
python scripts/run_prediction.py --method rule_based --k 5
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
│   ├── features/typhoon.py         # 特徵工程（極座標 + Impact Window）
│   ├── similarity/
│   │   ├── base.py                 # 抽象介面
│   │   ├── knn.py                  # KNN（標準化歐氏距離）
│   │   ├── dtw.py                  # DTW（環形距離 + Sakoe-Chiba）
│   │   ├── combined.py             # Combined RRF（KNN + DTW + Rule 排名融合）
│   │   ├── rule_based.py           # 幾何規則分類（CWA 定義）
│   │   └── baseline.py             # 隨機基線（對照組）
│   ├── hazards/                    # 多災害模組（可擴展）
│   │   └── typhoon/
│   │       ├── __init__.py         # 颱風模組匯出
│   │       └── rainfall.py         # 降水分析（MAE/RMSE + 機率分布）
│   ├── models/analog.py            # 加權投票預測
│   ├── impact/mapping.py           # 路徑分類說明對照
│   ├── analysis/eda.py             # 資料探索性分析
│   ├── visualization/plots.py      # 所有圖表
│   └── pipeline/predict.py         # Config-driven pipeline + LOO 評估
├── experiments/                    # 可重現實驗腳本
│   ├── exp001_combined_rrf.py      # Combined RRF + 降水分析
│   └── exp002_rule_based.py        # Rule-Based + 降水分析
├── pipelines/                      # 獨立預測 pipeline 腳本
│   ├── combined_rrf.py             # Combined RRF 執行入口
│   └── rule_based.py              # Rule-Based 執行入口
├── scripts/
│   ├── build_dataset.py            # xlsx + IBTrACS → JSON
│   ├── run_analysis.py             # EDA + 視覺化
│   ├── run_prediction.py           # 通用預測 runner（指定 config）
│   └── run_all_predictions.py      # 批量執行多組預測
├── configs/experiments/            # YAML 實驗設定檔
│   ├── combined_rrf.yaml
│   └── rule_based.yaml
├── docs/                           # 方法詳細文件
│   ├── combined_rrf.md             # Combined RRF 方法文件（含 Mermaid）
│   └── rule_based_classification.md # Rule-Based 方法文件（含 Mermaid）
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
- **颱風事件雨量**：臺南、高雄兩站事件降水量（440 筆）

## 技術細節

### 設計原則

- **Config-Driven**：每次實驗由 YAML 設定檔驅動，結果資料夾內含完整 config 供重現
- **方法獨立**：Combined RRF 與 Rule-Based 完全解耦，各自獨立運作
- **評估彈性**：`src/evaluation/metrics.py` 提供 METRIC_REGISTRY，可擴充降水 loss 等指標
- **類別 1-9**：評估排除「特殊」分類（9 筆），僅對 198 筆明確路徑颱風做 LOO
