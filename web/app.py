"""
颱風類比預測系統 — Web 前端

端點:
  GET  /                         → 首頁
  GET  /analysis                 → 分析圖表頁面
  GET  /predictions              → 預測結果列表（按版本）
  GET  /predictions/<run_id>     → 特定版本的預測結果
  POST /api/predict              → 自訂颱風預測 API
  GET  /api/runs                 → 列出所有預測版本
  GET  /api/runs/<run_id>        → 取得特定版本的結果摘要
  GET  /outputs/<path>           → 靜態檔案服務
"""

import sys
import json
import uuid
import traceback
from pathlib import Path
from datetime import datetime

from flask import Flask, render_template, jsonify, request, send_from_directory, abort

sys.path.insert(0, str(Path(__file__).parent.parent))

BASE_DIR = Path(__file__).parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
ANALYSIS_DIR = OUTPUTS_DIR / "analysis"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "web" / "templates"),
    static_folder=str(BASE_DIR / "web" / "static"),
)


# === 靜態檔案服務（outputs 目錄）===
@app.route("/outputs/<path:filepath>")
def serve_output(filepath):
    return send_from_directory(str(OUTPUTS_DIR), filepath)


# === 頁面路由 ===
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/methods")
def methods_page():
    return render_template("methods.html")


@app.route("/analysis")
def analysis_page():
    # 收集 analysis 目錄下的所有圖片
    images = []
    if ANALYSIS_DIR.exists():
        for img in sorted(ANALYSIS_DIR.glob("*.png")):
            images.append(
                {
                    "name": img.stem,
                    "url": f"/outputs/analysis/{img.name}",
                }
            )
    return render_template("analysis.html", images=images)


@app.route("/predictions")
def predictions_page():
    runs = _list_runs()
    return render_template("predictions.html", runs=runs)


@app.route("/predictions/<run_id>")
def prediction_detail(run_id):
    run_dir = PREDICTIONS_DIR / run_id
    if not run_dir.is_dir():
        abort(404)

    meta = _load_run_meta(run_dir)
    images = []
    for img in sorted(run_dir.glob("*.png")):
        images.append(
            {
                "name": img.stem,
                "url": f"/outputs/predictions/{run_id}/{img.name}",
            }
        )
    return render_template(
        "prediction_detail.html", run_id=run_id, meta=meta, images=images
    )


@app.route("/predict")
def predict_page():
    return render_template("predict.html")


# === API 路由 ===
@app.route("/api/runs")
def api_list_runs():
    return jsonify(_list_runs())


@app.route("/api/runs/<run_id>")
def api_run_detail(run_id):
    run_dir = PREDICTIONS_DIR / run_id
    if not run_dir.is_dir():
        return jsonify({"error": "Run not found"}), 404

    meta = _load_run_meta(run_dir)

    # 載入 evaluation_summary
    summary_path = run_dir / "evaluation_summary.json"
    summary = {}
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

    images = [
        {"name": img.stem, "url": f"/outputs/predictions/{run_id}/{img.name}"}
        for img in sorted(run_dir.glob("*.png"))
    ]

    return jsonify({"meta": meta, "summary": summary, "images": images})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    接收 JSON 格式的颱風路徑資料，找最相似颱風。

    Expected JSON:
    {
        "track": [
            {"latitude": 15.0, "longitude": 130.0, "wind_kt": 30, "pressure_mb": 1000},
            ...
        ],
        "method": "combined",   // optional, default "combined"
        "k": 5,                 // optional, default 5
        "alpha": 0.5            // optional, default 0.5
    }
    """
    try:
        data = request.get_json()
        if not data or "track" not in data:
            return jsonify({"error": "Missing 'track' field in JSON"}), 400

        track_points = data["track"]
        if len(track_points) < 2:
            return jsonify({"error": "Track must have at least 2 points"}), 400

        method = data.get("method", "combined")
        k = int(data.get("k", 5))
        alpha = float(data.get("alpha", 0.5))

        # 延遲匯入避免啟動時載入資料
        from src.pipeline.predict import DisasterImpactPipeline
        from src.features.typhoon import TyphoonFeatureExtractor, TyphoonFeatures
        from src.visualization.plots import TyphoonVisualizer
        from src.data.loader import TyphoonRecord
        import pandas as pd
        import numpy as np

        # 初始化 pipeline
        pipeline = DisasterImpactPipeline(similarity_method=method, alpha=alpha)
        pipeline.initialize("data/processed")

        # 建立查詢軌跡 DataFrame
        track_df = pd.DataFrame(track_points)
        for col in ["latitude", "longitude"]:
            if col not in track_df.columns:
                return jsonify({"error": f"Missing '{col}' in track points"}), 400
        if "wind_kt" not in track_df.columns:
            track_df["wind_kt"] = None
        if "pressure_mb" not in track_df.columns:
            track_df["pressure_mb"] = None
        if "timestamp_utc" not in track_df.columns:
            track_df["timestamp_utc"] = pd.date_range(
                "2000-01-01", periods=len(track_df), freq="6h"
            )

        # 提取特徵
        extractor = TyphoonFeatureExtractor()
        query_features = extractor.extract(
            typhoon_id="query",
            track=track_df,
        )

        # 找相似颱風
        query_vec = query_features.to_feature_vector()

        if method in ("knn", "combined"):
            from src.similarity.knn import KNNSimilarity

            knn = pipeline.similarity if method == "knn" else pipeline.similarity.knn
            sim_result = knn.find_similar_by_vector(query_vec, k=k)
        else:
            # 暫用 KNN fallback
            from src.similarity.knn import KNNSimilarity

            knn_sim = KNNSimilarity()
            knn_sim.fit(pipeline.features)
            sim_result = knn_sim.find_similar_by_vector(query_vec, k=k)

        # 組裝結果
        similar_info = []
        for tid, dist, score in zip(
            sim_result.similar_ids, sim_result.distances, sim_result.scores
        ):
            rec = pipeline.loader.get(tid)
            similar_info.append(
                {
                    "typhoon_id": tid,
                    "name_zh": rec.name_zh,
                    "name_en": rec.name_en,
                    "year": rec.year,
                    "category": rec.taiwan_track_category,
                    "distance": round(dist, 4),
                    "score": round(score, 4),
                }
            )

        # 用 AnalogModel 預測
        pred = pipeline.model.predict(
            query_id="query",
            similar_ids=sim_result.similar_ids,
            distances=sim_result.distances,
        )

        # 生成圖表
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_query"
        fig_dir = PREDICTIONS_DIR / run_id
        fig_dir.mkdir(parents=True, exist_ok=True)

        viz = TyphoonVisualizer(str(fig_dir))

        # 建立查詢的 TyphoonRecord（用於繪圖）
        query_rec = TyphoonRecord(
            typhoon_id="query",
            year=0,
            name_zh="查詢颱風",
            name_en="Query",
            taiwan_track_category=pred.get("predicted_category", "?"),
            birth_lon=float(track_df["longitude"].iloc[0]),
            birth_lat=float(track_df["latitude"].iloc[0]),
            max_sustained_wind_ms=None,
            min_pressure=None,
            max_intensity_class=None,
            landfall_location=None,
            movement_summary=None,
            disaster_summary=None,
            track=track_df,
        )
        similar_recs = [pipeline.loader.get(tid) for tid in sim_result.similar_ids]
        viz.plot_prediction_example(
            query_rec,
            similar_recs,
            pred.get("predicted_category", "?"),
            pred.get("confidence", 0.0),
        )

        # 圖片 URL
        images = [
            {"name": img.stem, "url": f"/outputs/predictions/{run_id}/{img.name}"}
            for img in sorted(fig_dir.glob("*.png"))
        ]

        return jsonify(
            {
                "predicted_category": pred.get("predicted_category"),
                "confidence": round(pred.get("confidence", 0.0), 4),
                "category_votes": {
                    k: round(v, 4) for k, v in pred.get("category_votes", {}).items()
                },
                "similar_typhoons": similar_info,
                "images": images,
                "run_id": run_id,
            }
        )

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# === 輔助函式 ===
def _list_runs() -> list[dict]:
    """列出所有預測版本（按時間倒序）"""
    runs = []
    if not PREDICTIONS_DIR.exists():
        return runs

    for d in sorted(PREDICTIONS_DIR.iterdir(), reverse=True):
        if d.is_dir():
            meta = _load_run_meta(d)
            runs.append(
                {
                    "run_id": d.name,
                    "meta": meta,
                }
            )
    return runs


def _load_run_meta(run_dir: Path) -> dict:
    meta_path = run_dir / "run_meta.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    # 嘗試從 evaluation_summary.json 讀取
    summary_path = run_dir / "evaluation_summary.json"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"note": "No metadata found"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
