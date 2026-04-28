"""
一次跑三組預測結果：baseline, combined, rule_based
"""

import sys
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    scripts_dir = Path(__file__).parent
    methods = [
        ("baseline", {"--method": "baseline", "--k": "5"}),
        ("combined", {"--method": "combined", "--alpha": "0.5", "--k": "5"}),
        ("rule_based", {"--method": "rule_based", "--k": "5"}),
    ]

    for name, params in methods:
        print(f"\n{'='*60}")
        print(f"🌀 執行 {name} 預測...")
        print(f"{'='*60}")

        cmd = [sys.executable, str(scripts_dir / "run_prediction.py")]
        for k, v in params.items():
            cmd.extend([k, v])

        result = subprocess.run(cmd, cwd=str(scripts_dir.parent))
        if result.returncode != 0:
            print(f"❌ {name} 執行失敗 (exit code {result.returncode})")
        else:
            print(f"✅ {name} 完成")

    print(f"\n{'='*60}")
    print("🎉 三組預測全部完成！可執行 python web/app.py 啟動前端檢視")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
