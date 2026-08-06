from __future__ import annotations
import argparse
import sys
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Industrial ASO Pipeline
from airfoil_discovery.pipeline import AirfoilDiscoveryPipeline

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the production-grade ASO pipeline.")
    parser.add_argument("--config", default="config/default.yaml", help="Path to YAML config file.")
    parser.add_argument("--iterations", type=int, default=None, help="Override optimization iterations.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override per-iteration CFD batch size.")
    return parser.parse_args()

def main() -> None:
    import os
    import time

    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    telemetry_path = PROJECT_ROOT / "data" / "logs" / "telemetry_events.jsonl"
    telemetry_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("AIRFOIL_TELEMETRY_PATH", str(telemetry_path))
    os.environ.setdefault("AIRFOIL_RUN_ID", f"run_{int(time.time())}")

    print("Initializing Aerospace-Grade ASO Campaign...")
    pipeline = AirfoilDiscoveryPipeline.from_config(args.config)
    status = pipeline.run(iterations=args.iterations, batch_size=args.batch_size)
    if status != "completed":
        print(f"ASO Optimization campaign terminated with status: {status}")
        raise SystemExit(1)
    print("ASO Optimization campaign finished successfully.")

if __name__ == "__main__":
    main()
