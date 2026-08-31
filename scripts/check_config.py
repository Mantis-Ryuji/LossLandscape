"""Validate configuration only; --prepare-run explicitly creates config records."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from landscape_exp.config import ConfigError, load_config, prepare_run


def main() -> int:
    """Report configuration validation without importing any ML framework."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--batch-size", type=int, choices=(64, 256, 1024), help="Effective optimizer batch size")
    parser.add_argument("--seed", type=int, choices=(0, 1, 2))
    parser.add_argument("--name", help="Experiment-series name; does not overwrite an old series")
    parser.add_argument("--prepare-run", action="store_true", help="Create config records in a new run directory; never train")
    args = parser.parse_args()
    try:
        loaded = load_config(
            args.config,
            project_root=args.project_root,
            batch_size=args.batch_size,
            seed=args.seed,
            name=args.name,
        )
        config = loaded.config
        if args.prepare_run:
            prepare_run(loaded)
        summary = {
            "status": "configuration_prepared" if args.prepare_run else "configuration_valid",
            "phase": config.experiment.phase,
            "run_id": config.run_id,
            "run_directory": str(config.run_directory),
            "schedule_epochs": config.training.epochs,
            "stop_after_epoch": config.end_epoch,
            "analysis_points_per_run": config.end_epoch + 1,
            "effective_batch_size": config.training.batch_size,
            "microbatch_size": config.training.microbatch_size,
            "accumulation_steps": config.training.accumulation_steps,
            "optimizer_steps_per_epoch": (config.split.train_size + config.training.batch_size - 1) // config.training.batch_size,
            "effective_sha256": loaded.effective_sha256,
            "created_artifacts": bool(args.prepare_run),
            "scope": "Configuration only. Data, model, GPU and training are not checked.",
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except (ConfigError, OSError) as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
