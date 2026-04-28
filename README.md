# 🌀 Disaster Impact Engine - 颱風類比災害預測系統

輕量但完整的颱風災害預測模組。基於**歷史類比方法**，快速預測颱風可能帶來的災害。

---

## 📋 快速概述

### 核心思想
```
找相似的歷史颱風 → 參考其災害記錄 → 預測新颱風的災害
```

### 完整流程
```
颱風軌跡數據
    ↓
特徵提取（距離、風速、速度等）
    ↓
相似度計算（KNN）
    ↓
找到 K 個最相似颱風
    ↓
類比預測（平均/加權平均）
    ↓
災害預測結果
```

---

## 📂 目錄結構

```
disaster-impact-engine/
│
├── src/                               # 核心模組
│   ├── data/
│   │   └── loader.py                 # 數據加載器
│   │
│   ├── features/
│   │   └── typhoon.py                # 特徵提取器
│   │
│   ├── similarity/
│   │   ├── base.py                   # 抽象介面
│   │   └── knn.py                    # KNN 實現
│   │
│   ├── models/
│   │   ├── base.py                   # 模型介面
│   │   └── analog.py                 # 類比模型
│   │
│   ├── impact/
│   │   └── mapping.py                # 災害標籤定義
│   │
│   └── pipeline/
│       └── predict.py                # 完整流程串聯
│
├── scripts/
│   ├── build_dataset.py              # 數據構建
│   └── run_prediction.py             # 預測運行
│
├── data/
│   ├── raw/                          # 原始數據
│   └── processed/                    # 處理後數據
│
├── notebooks/                        # Jupyter 筆記本（可選）
│
├── README.md                         # 本文件
└── QUICKSTART.md                     # 快速開始指南

```

---

## 🧩 各模組說明

### 1️⃣ `src/data/loader.py`

**職責**: 讀取颱風和災害數據

```python
from src.data import DataLoader

loader = DataLoader()
typhoon_df = loader.load_typhoon_data('data/raw/typhoon.csv')
impact_df = loader.load_impact_data('data/raw/impact.csv')
```

**預期數據格式**:
- 颱風軌跡：typhoon_id, date, lat, lon, max_wind, central_pressure
- 災害記錄：typhoon_id, impact_type, severity

---

### 2️⃣ `src/features/typhoon.py`

**職責**: 特徵工程（將原始數據轉換為模型可用的特徵）

提取的特徵：
- 距台灣距離
- 方位角
- 最大風速
- 移動速度
- 中心氣壓
- 預計到達時間
- 風速變化率
- 氣壓變化率

```python
from src.features import TyphoonFeatureExtractor

extractor = TyphoonFeatureExtractor()
features = extractor.extract(typhoon_trajectory)
# 返回 TyphoonFeatures 物件
```

---

### 3️⃣ `src/similarity/knn.py`

**職責**: 找相似颱風

使用 K-Nearest Neighbors，支援：
- 歐式距離
- 餘弦相似度

```python
from src.similarity import KNNSimilarity

similarity = KNNSimilarity(metric='euclidean')
similarity.fit(reference_features)
indices, distances = similarity.find_similar(query_vector, k=5)
```

**可替換**: 未來可以改為 DTW、FAISS 等

---

### 4️⃣ `src/models/analog.py`

**職責**: 基於相似颱風進行預測

支援多種聚合方式：
- `mean`: 簡單平均
- `weighted_mean`: 距離加權平均（推薦）
- `majority_vote`: 多數投票（分類）
- `max`: 取最大值（極端預測）

```python
from src.models import AnalogModel

model = AnalogModel(aggregation_method='weighted_mean')
result = model.predict(similar_indices, similar_distances, impact_labels)
# 返回 {prediction, confidence, details}
```

**可替換**: 未來可以改為 ML 模型（隨機森林、神經網絡等）

---

### 5️⃣ `src/impact/mapping.py`

**職責**: 災害數據標籤化

支援的災害類型：
- flooding（淹水）
- blackout（停電）
- damage（損害）
- landslide（山崩）
- wind_damage（風災）

支援的標籤格式：
- 二元（0/1）
- 嚴重程度（0-4）
- 複合標籤

```python
from src.impact import ImpactMapper

mapper = ImpactMapper()
labels = mapper.create_severity_label(impact_df, 'flooding')
```

---

### 6️⃣ `src/pipeline/predict.py`

**職責**: 串聯整個預測流程

核心類：`DisasterImpactPipeline`

```python
from src.pipeline import DisasterImpactPipeline
from src.similarity import KNNSimilarity
from src.models import AnalogModel

pipeline = DisasterImpactPipeline(
    similarity_model=KNNSimilarity(),
    prediction_model=AnalogModel(),
)

pipeline.initialize('data/raw/typhoon.csv', 'data/raw/impact.csv')

result = pipeline.predict('TYPHOON_001', k=5)
# 返回 PredictionResult 物件
```

---

## 🚀 使用示例

### 安裝依賴

```bash
pip install pandas numpy scikit-learn
```

### 1. 構建數據集

```bash
python scripts/build_dataset.py --num-typhoons 50
```

輸出：
- `data/raw/typhoon.csv` - 颱風軌跡
- `data/raw/impact.csv` - 災害記錄

### 2. 運行預測

#### 批量預測
```bash
python scripts/run_prediction.py
```

#### 單個颱風預測
```bash
python scripts/run_prediction.py --typhoon-id TYPHOON_001
```

#### 自定義選項
```bash
python scripts/run_prediction.py \
    --typhoon-data data/raw/typhoon.csv \
    --impact-data data/raw/impact.csv \
    --aggregation weighted_mean \
    --output results/predictions.json
```

---

## 🔄 可擴展性

### 替換相似度方法

目前使用 KNN，未來可替換為其他方法（無需改動其他代碼）：

```python
# 現在：KNN
pipeline = DisasterImpactPipeline(
    similarity_model=KNNSimilarity(),
    prediction_model=AnalogModel()
)

# 未來：DTW（只需改這一行）
# pipeline = DisasterImpactPipeline(
#     similarity_model=DTWSimilarity(),  # 自己實現
#     prediction_model=AnalogModel()
# )
```

### 替換預測模型

目前使用類比模型，未來可替換為 ML 模型：

```python
# 現在：類比模型
pipeline = DisasterImpactPipeline(
    similarity_model=KNNSimilarity(),
    prediction_model=AnalogModel()
)

# 未來：ML 模型（只需改這一行）
# pipeline = DisasterImpactPipeline(
#     similarity_model=KNNSimilarity(),
#     prediction_model=MLModel(...)  # 自己實現
# )
```

### 添加新的特徵

在 `src/features/typhoon.py` 中添加即可，不影響其他模組。

---

## 📊 輸出格式

### 預測結果結構

```json
{
  "typhoon_id": "TYPHOON_001",
  "features": {
    "distance_to_taiwan": 500.5,
    "azimuth": 120.3,
    "max_wind": 150.0,
    "speed": 25.5
  },
  "similar_typhoons": ["TYPHOON_010", "TYPHOON_015", "TYPHOON_020"],
  "similar_distances": [12.3, 15.6, 18.2],
  "predictions": {
    "flooding": {
      "prediction": 2.3,
      "confidence": 0.85,
      "num_analogs": 5
    },
    "damage": {
      "prediction": 1.8,
      "confidence": 0.72,
      "num_analogs": 5
    }
  }
}
```

---

## 🛠️ API 參考

### DataLoader

```python
loader.load_typhoon_data(filepath)    # 加載颱風數據
loader.load_impact_data(filepath)     # 加載災害數據
loader.get_typhoon_by_id(typhoon_id)  # 取特定颱風軌跡
loader.get_impact_by_typhoon(id)      # 取特定颱風的災害
```

### TyphoonFeatureExtractor

```python
extractor = TyphoonFeatureExtractor(use_normalization=True)
features = extractor.extract(trajectory)              # 單個提取
batch = extractor.extract_batch(typhoon_data)        # 批量提取
normalized = extractor.normalize_features(features)  # 標準化
```

### KNNSimilarity

```python
sim = KNNSimilarity(metric='euclidean')
sim.fit(reference_vectors)
indices, distances = sim.find_similar(query_vector, k=5)
similarity_score = sim.compute_similarity(vec1, vec2)
```

### AnalogModel

```python
model = AnalogModel(aggregation_method='weighted_mean')
result = model.predict(indices, distances, labels)
batch_results = model.predict_batch(indices_list, distances_list, labels)
```

### DisasterImpactPipeline

```python
pipeline.initialize(typhoon_path, impact_path)
result = pipeline.predict(typhoon_id, k=5)
results = pipeline.predict_batch(typhoon_ids, k=5)
summary = pipeline.get_prediction_summary(result)
```

---

## 🔍 故障排除

### Q: 數據文件未找到
**A**: 先運行 `python scripts/build_dataset.py`

### Q: 模型準確率不高
**A**: 檢查：
1. 訓練數據是否充足
2. 特徵是否相關
3. 相似颱風數量（k 值）是否合適
4. 是否有異常值

### Q: 如何使用自己的數據
**A**: 確保數據格式符合預期，然後：
```python
pipeline.initialize('your_typhoon.csv', 'your_impact.csv')
```

---

## 📚 進階用法

### 自定義特徵

在 `TyphoonFeatureExtractor` 中添加新方法：

```python
class CustomExtractor(TyphoonFeatureExtractor):
    def extract(self, trajectory):
        features = super().extract(trajectory)
        # 添加自定義特徵
        features.custom_feature = self._compute_custom(trajectory)
        return features
```

### 自定義相似度

繼承 `SimilarityBase` 實現自己的相似度計算：

```python
from src.similarity import SimilarityBase

class DTWSimilarity(SimilarityBase):
    def fit(self, reference_vectors, labels=None):
        # 實現 DTW 擬合
        pass
    
    def find_similar(self, query_vector, k=5):
        # 實現 DTW 搜索
        pass
```

---

## 📝 許可證

MIT License

---

## 👤 作者

Disaster Impact Engine Team

---

## 🤝 貢獻

歡迎提交 Pull Requests 或報告 Issues。

---

**最後更新**: 2026-04-28
