"""Create/verify the shared ConvNeXt V2 scratch theta_0; never download weights."""

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
    """Create once, or verify existing records without replacing any files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--verify-only", action="store_true", help="Never download or create initial records")
    args = parser.parse_args()
    try:
        loaded = load_config(args.config, project_root=args.project_root)
        from landscape_exp.models import load_initial_checkpoint, prepare_initial_checkpoint

        path = loaded.config.paths.init_checkpoint
        existed = path.exists() or path.with_suffix(".json").exists()
        initial = (load_initial_checkpoint(loaded.config) if args.verify_only
                   else prepare_initial_checkpoint(loaded))
        print(json.dumps({
            "status": "initial_checkpoint_verified" if existed else "initial_checkpoint_created",
            "path": str(initial.checkpoint_path), "sha256": initial.checkpoint_sha256,
            "parameter_count": initial.metadata["parameter_count"],
            "initialization": initial.metadata["initialization"],
            "model": initial.metadata["model"], "runtime": initial.metadata["runtime"],
            "preprocessing": initial.metadata["data_config"],
            "device": "cpu", "parameter_dtype": "torch.float32",
            "dataset_used": False, "training_started": False,
            "pretrained_fetch_requested": False,
        }, ensure_ascii=False, indent=2))
    except (ConfigError, OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"Initial checkpoint preparation failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
