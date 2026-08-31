"""Run fixed Phase 0/1 training, recording epoch zero and every completed epoch."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from landscape_exp.config import ConfigError, load_config


def main() -> int:
    """Set only this process's CUDA contract before importing any torch module."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--batch-size", type=int, choices=(64, 256, 1024), help="Effective optimizer batch size; microbatch stays 64")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--name")
    parser.add_argument("--resume-from", type=Path, help="Exact completed epoch directory; no implicit latest")
    args = parser.parse_args()
    last_completed: Path | None = None
    try:
        loaded = load_config(args.config, project_root=args.project_root,
                             batch_size=args.batch_size, seed=args.seed, name=args.name)
        existing = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if existing is not None and existing != ":4096:8":
            raise ValueError("Existing CUBLAS_WORKSPACE_CONFIG differs from :4096:8; no override performed")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        from landscape_exp.train import run_experiment
        from landscape_exp.logging_utils import EpochMetrics

        def report(path: Path, metrics: EpochMetrics) -> None:
            nonlocal last_completed
            last_completed = path
            print(json.dumps({"status": "epoch_completed", "checkpoint": str(path),
                              **metrics.record()}, ensure_ascii=False, allow_nan=False), flush=True)

        resume = args.resume_from
        if resume is not None and not resume.is_absolute():
            resume = loaded.project_root / resume
        final = run_experiment(loaded, resume_from=resume, on_complete=report)
        print(json.dumps({"status": "training_completed", "phase": loaded.config.experiment.phase,
                          "run_id": loaded.config.run_id, "last_completed_epoch": str(final),
                          "stop_after_epoch": loaded.config.end_epoch,
                          "schedule_epochs": loaded.config.training.epochs}, ensure_ascii=False), flush=True)
    except KeyboardInterrupt:
        print(f"Training interrupted. Last reported completed epoch: {last_completed}. Partial files are preserved.",
              file=sys.stderr)
        return 130
    except (ConfigError, OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"Training failed: {error}\nLast reported completed epoch: {last_completed}. "
              "Existing artifacts were not replaced or deleted; resume only from a complete.json epoch.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
