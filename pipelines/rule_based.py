"""
Pipeline: Rule-Based Classification (CWA 幾何規則分類)

使用方式：
  python pipelines/rule_based.py
  python pipelines/rule_based.py --config configs/experiments/rule_based.yaml

預設使用 configs/experiments/rule_based.yaml 配置
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_prediction import load_config, run_evaluation
from src.pipeline.predict import DisasterImpactPipeline

DEFAULT_CONFIG = "configs/experiments/rule_based.yaml"


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Rule-Based 預測 Pipeline")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    parser.add_argument("--processed-dir", type=str, default="data/processed")
    parser.add_argument("--output-dir", type=str, default="outputs/predictions")
    args = parser.parse_args()

    config_path = args.config
    config = load_config(config_path)

    print("=" * 60)
    print("🌀 Rule-Based Classification Pipeline")
    print(f"   配置: {config_path}")
    print(f"   參數: {config.get('parameters', {})}")
    print("=" * 60)

    pipeline = DisasterImpactPipeline(config=config)
    pipeline.initialize(args.processed_dir)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_evaluation(pipeline, config, output_dir, config_path)


if __name__ == "__main__":
    main()
