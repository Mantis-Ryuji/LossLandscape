"""Compute immutable train/validation loss surfaces for one completed PCA."""

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--projection", type=Path, required=True)
    args = parser.parse_args()
    try:
        loaded = load_config(args.config, project_root=args.project_root)
        existing = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if existing is not None and existing != ":4096:8":
            raise ValueError(
                "Existing CUBLAS_WORKSPACE_CONFIG differs from :4096:8; no override performed"
            )
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        from landscape_exp.loss_surface import compute_loss_surfaces

        def report(record: dict[str, object]) -> None:
            if record.get("status") == "surface_grid_point_completed":
                index, count = record.get("index"), record.get("grid_points")
                if type(index) is int and type(count) is int and index % 10 != 0 and index + 1 != count:
                    return
            print(json.dumps(record, ensure_ascii=False, allow_nan=False), flush=True)

        result = compute_loss_surfaces(loaded, args.projection, progress=report)
        print(json.dumps({
            "status": "loss_surfaces_ready",
            "projection_id": result.projection_id,
            "directory": str(result.directory),
            "grid_points_per_background": result.grid_points,
            "train_samples": result.train_samples,
            "validation_samples": result.validation_samples,
            "shared_loss_minimum": result.loss_minimum,
            "shared_loss_maximum": result.loss_maximum,
        }, ensure_ascii=False, allow_nan=False), flush=True)
    except KeyboardInterrupt:
        print(
            "Loss-surface evaluation interrupted. Partial new artifacts are preserved and must not be reused.",
            file=sys.stderr,
        )
        return 130
    except (ConfigError, OSError, ValueError, RuntimeError, ImportError) as error:
        print(
            f"Loss-surface evaluation failed: {error}. Partial new artifacts are preserved; "
            "existing artifacts were not changed.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
