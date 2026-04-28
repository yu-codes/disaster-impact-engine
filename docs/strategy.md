# 一、核心觀念（先講清楚）
- 起點不同
- 時間不同（速度不同）
- 最終要對應「災害相似性」
你要找的不是：

> ❌ 路徑長得像

而是：

> ✅ **「對台灣產生類似影響的動態行為」**
> 

---

# 二、整體策略（四層架構）

```
1. 空間標準化（解決起點問題）
2. 時間對齊（解決速度問題）
3. 多維相似度（不只路徑）
4. 影響導向加權（連到災害）
```

---

# 三、Step 1：空間標準化（這是關鍵）

你不能直接用經緯度比。

---

## ✔ 方法：Relative Position（強烈建議）

以台灣為中心（例如 23.7N, 121E）：

```python
dx = lon - 121.0
dy = lat - 23.7
```

👉 每個颱風都變成「相對台灣的軌跡」

---

## ✔ 再做一層轉換（更穩）

轉極座標：

```python
r = sqrt(dx^2 + dy^2)      # 距離
theta = atan2(dy, dx)      # 方位
```

👉 解決問題：

- 起點不同 ✔
- 地理位置影響 ✔（這很重要）

---

# 四、Step 2：時間對齊（你最該認真做的）

這裡才用 Dynamic Time Warping（DTW）

但你要**改用 impact-aware DTW**

---

## ❗ 重點：不要用原始整條路徑

---

## ✔ 做「Impact Window」

只取：

```
距離台灣 < 500 km 的時間段
```

👉 這段才和災害相關

---

## ✔ 再做 DTW 對齊

但 distance function 改成：

```python
d = w1*(Δr)^2 + w2*(Δtheta)^2 + w3*(Δwind)^2 + w4*(Δpressure)^2
```

---

👉 這樣 DTW 比的是：

- 距離變化
- 接近方向
- 強度演化

---

# 五、Step 3：加入「動態特徵」（比 DTW 更重要）

光靠 DTW 還不夠

---

## ✔ 你要額外抽這些 feature：

### 1️⃣ 最接近距離

```
min_distance_to_taiwan
```

---

### 2️⃣ 通過哪一側

```
mean_theta（南 / 西南 / 東）
```

---

### 3️⃣ 移動速度

```python
speed = distance / time
```

👉 慢 → 災害大

---

### 4️⃣ 強度曲線

```
max_wind
intensification_rate
```

---

### 5️⃣ 是否登陸 / 擦邊

---

# 六、Step 4：最終 Similarity Score（核心公式）

這是你整個系統的靈魂

---

## ✔ 建議公式

```
Similarity =
  w1 * DTW_path +
  w2 * Distance_feature +
  w3 * Angle_feature +
  w4 * Intensity_feature +
  w5 * Speed_feature
```

---

## ✔ 更進階（強烈建議）

加入「災害 proxy」

```
w6 * Rain_proxy
```

---

# 七、Rain Proxy（讓你接近災害）

你目前沒有降雨資料也沒關係

---

## ✔ 簡化版：

```python
rain_proxy = wind_speed / distance_to_taiwan
```

👉 越近 + 越強 → 雨越大

---

## ✔ 進階版：

```python
rain_proxy = wind_direction × terrain_factor
```

👉 台灣非常吃這個

---

# 八、完整演算法流程（可直接寫 code）

---

## Step 1：預處理

```python
for typhoon:
    convert_to_relative_coordinates()
    extract_impact_window()
```

---

## Step 2：特徵抽取

```python
features = [
    min_distance,
    mean_angle,
    max_wind,
    speed,
    rain_proxy
]
```

---

## Step 3：相似度計算

```python
dtw_score = DTW(track1, track2)
feature_score = euclidean(f1, f2)

final_score = alpha * dtw_score + beta * feature_score
```

---

## Step 4：取 Top-K

```python
similar_typhoons = top_k(final_score)
```

---

# 九、關鍵優化（這會讓你贏別人）

---

## 1️⃣ 分群再比（非常重要）

先分：

- 登陸型
- 擦邊型
- 西南氣流型

👉 再做 similarity

---

## 2️⃣ 加入「經過台灣的時間點」

例如：

```
距離最近時的 timestamp
```

👉 對應雨量 timing

---

## 3️⃣ 過濾無效颱風

```
min_distance > 800km → 直接丟掉
```

---

# 十、用你這筆資料的實際觀察

這個 BILLIE (1959)：

- 從低緯度西進再北轉
- 接近台灣東側（約 120~122E）
- 強度逐漸增強到 ~90kt
- 有明顯「北轉 + 掃過」型態

👉 它會被歸類為：

```
北轉掃過型（高風險降雨）
```

---

# 十一、一句話講透你的策略

> **用「相對台灣的動態行為」取代「絕對路徑形狀」，再用多維特徵補足物理影響。**
> 

---

# 十二、你下一步該做什麼（很具體）

我建議你現在直接實作：

1. `relative_coordinate()`
2. `extract_impact_window()`
3. `compute_basic_features()`
4. `knn + feature distance`（包含 DTW版本）

👉 先跑出第一版結果

接著實作以下：

- 寫 **完整 Python similarity module（含 DTW + feature）**
- 或幫你定義 **最合理的權重（w1~w6）**
- 或幫你把這整套變成 **可評估模型（含 accuracy 指標）**
