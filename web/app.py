"""
災害類比預測系統 — Web 前端

端點:
  GET  /                         → 首頁（系統架構 + 資料集概覽）
  GET  /analysis                 → 資料分析（原始資料 EDA）
  GET  /methods                  → 方法說明
  GET  /predictions              → 預測結果列表
  GET  /predictions/<path>       → 特定實驗的預測結果
  GET  /predict                  → 即時預測
  POST /api/predict              → 預測 API
  GET  /api/runs                 → 列出所有預測版本
"""

import sys
import json
import traceback
from pathlib import Path
from datetime import datetime

from flask import Flask, render_template, jsonify, request, send_from_directory, abort

sys.path.insert(0, str(Path(__file__).parent.parent))

BASE_DIR = Path(__file__).parent.parent
EXPERIMENTS_DIR = BASE_DIR / "experiments"
ALL_CASES_DIR = EXPERIMENTS_DIR / "all_cases"
SINGLE_CASE_DIR = EXPERIMENTS_DIR / "single_case"
ANALYSIS_DIR = EXPERIMENTS_DIR / "analysis"

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "web" / "templates"),
    static_folder=str(BASE_DIR / "web" / "static"),
)


# === 靜態檔案服務 ===
@app.route("/experiments/<path:filepath>")
def serve_experiment(filepath):
    return send_from_directory(str(EXPERIMENTS_DIR), filepath)


@app.route("/outputs/<path:filepath>")
def serve_output(filepath):
    return send_from_directory(str(BASE_DIR / "outputs"), filepath)


# === 頁面路由 ===
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/methods")
def methods_page():
    return render_template("typhoon/methods.html")


@app.route("/analysis")
def analysis_page():
    track_images = []
    rainfall_images = []
    if ANALYSIS_DIR.exists():
        for img in sorted(ANALYSIS_DIR.glob("*.png")):
            entry = {"name": img.stem, "url": f"/experiments/analysis/{img.name}"}
            if "rainfall" in img.stem:
                rainfall_images.append(entry)
            else:
                track_images.append(entry)
    return render_template(
        "typhoon/analysis.html",
        track_images=track_images,
        rainfall_images=rainfall_images,
    )


@app.route("/predictions")
def predictions_page():
    runs = _list_runs()
    return render_template("typhoon/predictions.html", runs=runs)


@app.route("/predictions/<path:run_path>")
def prediction_detail(run_path):
    run_dir = EXPERIMENTS_DIR / run_path
    if not run_dir.is_dir():
        abort(404)

    meta = _load_run_meta(run_dir)

    images = []
    rainfall_images = []
    for img in sorted(run_dir.glob("*.png")):
        entry = {
            "name": img.stem,
            "url": f"/experiments/{run_path}/{img.name}",
        }
        if "rainfall" in img.stem:
            rainfall_images.append(entry)
        else:
            images.append(entry)

    rainfall_data = None
    rainfall_path = run_dir / "rainfall_analysis.json"
    if rainfall_path.exists():
        with open(rainfall_path, "r", encoding="utf-8") as f:
            rainfall_data = json.load(f)

    return render_template(
        "typhoon/prediction_detail.html",
        run_path=run_path,
        meta=meta,
        images=images,
        rainfall_images=rainfall_images,
        rainfall_data=rainfall_data,
    )


@app.route("/predict")
def predict_page():
    return render_template("typhoon/predict.html")


# === API 路由 ===
@app.route("/api/runs")
def api_list_runs():
    return jsonify(_list_runs())


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json()
        if not data or "track" not in data:
            return jsonify({"error": "Missing 'track' field in JSON"}), 400

        track_points = data["track"]
        if len(track_points) < 2:
            return jsonify({"error": "Track must have at least 2 points"}), 400

        method = data.get("method", "combined")
        k = int(data.get("k", 5))

        # 方法專屬參數
        params = {}
        if method == "combined":
            params["alpha"] = float(data.get("alpha", 0.13))
            params["rule_weight"] = float(data.get("rule_weight", 0.25))
            params["rrf_k"] = int(data.get("rrf_k", 60))

        from src.pipeline.typhoon.predict import DisasterImpactPipeline
        from src.features.typhoon.extractor import TyphoonFeatureExtractor
        from src.data.typhoon.loader import TyphoonRecord
        from src.analysis.typhoon.rainfall import RainfallAnalyzer
        from src.visualization.typhoon.plots import TyphoonVisualizer
        import pandas as pd
        import numpy as np

        # 建立 pipeline config
        config = {
            "method": method,
            "parameters": {"k": k, **params},
            "evaluation": {"categories": ["1", "2", "3", "4", "5", "6", "7", "8", "9"]},
        }
        pipeline = DisasterImpactPipeline(config=config)
        pipeline.initialize("data/processed")

        # 建立 track DataFrame
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

        # === 根據方法執行預測 ===
        if method == "rule_based":
            # Rule-Based: 直接使用幾何規則分類
            from src.similarity.typhoon.rule_based import classify_typhoon_by_rules

            rule_result = classify_typhoon_by_rules(track_df, landfall_location=None)
            predicted_cat = rule_result["predicted_category"]
            confidence = rule_result["confidence"]
            category_votes = {predicted_cat: 1.0}

            # 仍取得相似颱風供參考（用 KNN）
            extractor = TyphoonFeatureExtractor()
            query_features = extractor.extract(typhoon_id="query", track=track_df)
            from src.similarity.typhoon.knn import KNNSimilarity

            knn_sim = KNNSimilarity()
            knn_sim.fit(pipeline.features)
            sim_result = knn_sim.find_similar_by_vector(
                query_features.to_feature_vector(), k=k
            )
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
        else:
            # Combined RRF: 使用完整融合管道
            extractor = TyphoonFeatureExtractor()
            query_features = extractor.extract(typhoon_id="query", track=track_df)
            query_vec = query_features.to_feature_vector()

            sim_result = pipeline.similarity.find_similar_by_vector(
                query_vec, k=k, query_features=query_features
            )

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

            pred = pipeline.model.predict(
                query_id="query",
                similar_ids=sim_result.similar_ids,
                distances=sim_result.distances,
            )
            predicted_cat = pred.get("predicted_category")
            confidence = pred.get("confidence", 0.0)
            category_votes = pred.get("category_votes", {})

        # === 生成圖表 ===
        case_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        case_dir = SINGLE_CASE_DIR / case_timestamp
        case_dir.mkdir(parents=True, exist_ok=True)

        query_rec = TyphoonRecord(
            typhoon_id="query",
            year=2026,
            name_zh="查詢颱風",
            name_en="QUERY",
            taiwan_track_category=str(predicted_cat or "?"),
            birth_lon=None,
            birth_lat=None,
            max_sustained_wind_ms=None,
            min_pressure=None,
            max_intensity_class=None,
            landfall_location=None,
            movement_summary=None,
            disaster_summary=None,
            track=track_df,
        )
        similar_recs = []
        for si in similar_info:
            try:
                similar_recs.append(pipeline.loader.get(si["typhoon_id"]))
            except KeyError:
                pass

        viz = TyphoonVisualizer(str(case_dir))
        if similar_recs:
            viz.plot_prediction_example(
                query_rec,
                similar_recs,
                str(predicted_cat or ""),
                confidence,
            )

        # === 降水分析 ===
        rainfall_result = None
        try:
            rainfall = RainfallAnalyzer()
            rainfall.load()
            analog_ids = [si["typhoon_id"] for si in similar_info]

            analog_rainfalls = []
            for aid in analog_ids:
                rec_rain = rainfall.get_rainfall(aid)
                if rec_rain:
                    analog_rainfalls.append(
                        {
                            "typhoon_id": aid,
                            "臺南": rec_rain.tainan_mm,
                            "高雄": rec_rain.kaohsiung_mm,
                        }
                    )

            if analog_rainfalls:
                rainfall_result = {
                    "analog_count": len(analog_rainfalls),
                    "stations": {},
                }
                for station in ["臺南", "高雄"]:
                    vals = [
                        ar[station]
                        for ar in analog_rainfalls
                        if ar.get(station) is not None
                    ]
                    if vals:
                        rainfall_result["stations"][station] = {
                            "mean": round(float(np.mean(vals)), 1),
                            "median": round(float(np.median(vals)), 1),
                            "min": round(float(min(vals)), 1),
                            "max": round(float(max(vals)), 1),
                            "values": [round(v, 1) for v in vals],
                        }
                _plot_single_case_loss(analog_rainfalls, case_dir)
        except Exception:
            pass

        # 收集圖片 URL
        chart_urls = []
        for img in sorted(case_dir.glob("*.png")):
            chart_urls.append(f"/experiments/single_case/{case_timestamp}/{img.name}")

        return jsonify(
            {
                "method": method,
                "predicted_category": predicted_cat,
                "confidence": round(confidence, 4),
                "category_votes": {k_: round(v, 4) for k_, v in category_votes.items()},
                "similar_typhoons": similar_info,
                "charts": chart_urls,
                "rainfall": rainfall_result,
            }
        )
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# === 輔助函式 ===
def _list_runs() -> list[dict]:
    runs = []
    if not ALL_CASES_DIR.exists():
        return runs

    for exp_dir in sorted(ALL_CASES_DIR.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name.startswith(("_", ".")):
            continue
        pred_dir = exp_dir / "predictions"
        if not pred_dir.exists() or not pred_dir.is_dir():
            continue
        # 一個 exp 就是一組結果，predictions/ 底下直接就是結果檔案
        meta = _load_run_meta(pred_dir)
        run_path = f"all_cases/{exp_dir.name}/predictions"
        runs.append(
            {
                "run_id": exp_dir.name,
                "experiment": exp_dir.name,
                "run_path": run_path,
                "meta": meta,
            }
        )

    runs.sort(key=lambda r: r["run_id"], reverse=True)
    return runs


def _load_run_meta(run_dir: Path) -> dict:
    meta_path = run_dir / "run_meta.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    summary_path = run_dir / "evaluation_summary.json"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"note": "No metadata found"}


def _plot_single_case_loss(analog_rainfalls: list[dict], output_dir: Path):
    """為即時預測生成損失分布圖（以降水量為損失代理指標）"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    import numpy as np

    # 中文字型
    for font in ["Microsoft JhengHei", "Microsoft YaHei", "SimHei"]:
        try:
            fm.findfont(font, fallback_to_default=False)
            plt.rcParams["font.sans-serif"] = [font]
            plt.rcParams["axes.unicode_minus"] = False
            break
        except Exception:
            continue

    stations = ["臺南", "高雄"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, station in enumerate(stations):
        ax = axes[idx]
        vals = [ar[station] for ar in analog_rainfalls if ar.get(station) is not None]
        if vals:
            bp = ax.boxplot([vals], patch_artist=True, widths=0.5)
            bp["boxes"][0].set_facecolor("#377eb8")
            bp["boxes"][0].set_alpha(0.6)
            ax.scatter(
                [1] * len(vals), vals, alpha=0.5, s=40, color="#e41a1c", zorder=5
            )
            ax.set_ylabel("損失量 (mm)")
            ax.set_title(f"{station} — 類比颱風損失分布 (n={len(vals)})")
            ax.set_xticklabels(["類比颱風"])
            ax.grid(True, alpha=0.3, axis="y")
        else:
            ax.text(
                0.5, 0.5, "無資料", ha="center", va="center", transform=ax.transAxes
            )
            ax.set_title(f"{station}")

    fig.suptitle("類比颱風事件損失量分布", fontsize=13)
    fig.tight_layout()
    fig.savefig(
        output_dir / "loss_distribution.png",
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
