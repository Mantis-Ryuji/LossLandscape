"""Prepare shared split indices from existing CIFAR-10 data; never download it."""

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
    """Read the official train partition, then create or verify small split files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=REPOSITORY_ROOT)
    args = parser.parse_args()
    try:
        config = load_config(args.config, project_root=args.project_root).config
        from landscape_exp.data import load_cifar10_training, prepare_cifar10_splits, split_path

        source = load_cifar10_training(config.paths.dataset_root)
        path = split_path(config)
        existed = path.exists() or path.with_suffix(".json").exists()
        indices = prepare_cifar10_splits(config, source)
        print(json.dumps({
            "status": "split_verified" if existed else "split_created",
            "path": str(path), "train": len(indices.train),
            "validation": len(indices.validation),
            "train_subset": len(indices.train_subset),
            "validation_subset": len(indices.validation_subset),
            "labels_sha256": indices.labels_sha256,
            "official_test_used": False, "downloaded": False,
        }, ensure_ascii=False, indent=2))
    except (ConfigError, OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"Split preparation failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
