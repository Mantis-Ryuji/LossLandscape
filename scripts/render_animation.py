"""Render paired train/validation trajectory GIFs from completed saved artifacts."""

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
    parser.add_argument("--projection", type=Path, required=True)
    parser.add_argument(
        "--name",
        required=True,
        help="Lowercase output stem; produces <name>.gif and <name>_val.gif",
    )
    args = parser.parse_args()
    try:
        loaded = load_config(args.config, project_root=args.project_root)
        from landscape_exp.animation import render_animation_pair

        def report(record: dict[str, object]) -> None:
            if record.get("status") == "animation_frame_rendered":
                index, count = record.get("frame_index"), record.get("frame_count")
                if type(index) is int and type(count) is int and index % 10 != 0 and index + 1 != count:
                    return
            print(json.dumps(record, ensure_ascii=False, allow_nan=False), flush=True)

        result = render_animation_pair(
            loaded,
            args.projection,
            animation_name=args.name,
            progress=report,
        )
        print(json.dumps({
            "status": "animation_pair_ready",
            "animation_id": result.animation_id,
            "projection_id": result.projection_id,
            "directory": str(result.directory),
            "train_gif": str(result.train_path),
            "validation_gif": str(result.validation_path),
            "frame_count": result.frame_count,
            "width": result.width,
            "height": result.height,
            "colors": result.colors,
            "train_size_bytes": result.train_size_bytes,
            "validation_size_bytes": result.validation_size_bytes,
            "max_size_bytes": 3_000_000,
            "model_evaluation_performed": False,
        }, ensure_ascii=False, allow_nan=False), flush=True)
    except KeyboardInterrupt:
        print(
            "Animation rendering interrupted. No completed artifact was published; any partial new "
            "publication is preserved and existing artifacts were not changed.",
            file=sys.stderr,
        )
        return 130
    except (ConfigError, OSError, ValueError, RuntimeError, ImportError) as error:
        print(
            f"Animation rendering failed: {error}. Any partial new publication is preserved; "
            "existing artifacts were not changed.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
