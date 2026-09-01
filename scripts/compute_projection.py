"""Compute one immutable common PCA from explicitly selected segment branches."""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--comparison-scope", required=True)
    parser.add_argument(
        "--segment", type=Path, action="append", required=True,
        help="Explicit final segment directory; repeat in the intended run order",
    )
    args = parser.parse_args()
    try:
        loaded = load_config(args.config, project_root=args.project_root)
        from landscape_exp.projection import compute_projection

        def report(record: dict[str, object]) -> None:
            if record.get("status") == "checkpoint_extracted":
                index, count = record.get("index"), record.get("checkpoint_count")
                if type(index) is int and type(count) is int and index % 10 != 0 and index + 1 != count:
                    return
            print(json.dumps(record, ensure_ascii=False, allow_nan=False), flush=True)

        result = compute_projection(
            loaded, args.segment, comparison_scope=args.comparison_scope, progress=report,
        )
        print(json.dumps({
            "status": "projection_ready",
            "projection_id": result.projection_id,
            "directory": str(result.directory),
            "work_directory": str(result.work_directory),
            "sample_count": result.sample_count,
            "parameter_count": result.parameter_count,
            "effective_rank": result.effective_rank,
            "explained_variance_ratio": list(result.explained_variance_ratio),
        }, ensure_ascii=False, allow_nan=False), flush=True)
    except KeyboardInterrupt:
        print("Projection interrupted. Partial new artifacts are preserved and must not be reused.", file=sys.stderr)
        return 130
    except (ConfigError, OSError, ValueError, RuntimeError, ImportError) as error:
        print(
            f"Projection failed: {error}. Partial new artifacts are preserved; existing artifacts were not changed.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
