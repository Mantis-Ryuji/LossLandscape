"""One row per completed epoch; JSON checkpoints remain the source of truth."""

from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class EpochMetrics:
    run_id: str
    segment_id: str
    epoch: int
    global_step: int
    batch_size: int
    seed: int
    train_loss: float | None
    train_accuracy: float | None
    train_subset_loss: float
    train_subset_accuracy: float
    val_loss: float
    val_accuracy: float
    gradient_norm: float | None
    learning_rate: float | None
    learning_rate_next: float | None
    parameter_displacement: float
    train_samples: int
    train_subset_samples: int
    val_samples: int
    train_seconds: float
    evaluation_seconds: float
    checkpoint_seconds: float = 0.0
    epoch_seconds: float = 0.0
    peak_allocated_bytes: int = 0
    peak_reserved_bytes: int = 0

    def record(self) -> dict[str, object]:
        """Reject misleading epoch-zero values and nonfinite serialized metrics."""
        result = asdict(self)
        for name, value in result.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        for name in ("epoch", "global_step", "train_samples", "train_subset_samples", "val_samples"):
            if type(result[name]) is not int or result[name] < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("train_accuracy", "train_subset_accuracy", "val_accuracy"):
            value = result[name]
            if value is not None and not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        online = (self.train_loss, self.train_accuracy, self.gradient_norm, self.learning_rate)
        if self.epoch == 0:
            if self.global_step != 0 or self.train_samples != 0 or any(value is not None for value in online):
                raise ValueError("Epoch zero has no online metrics, completed updates or used LR")
        elif any(value is None for value in online) or self.train_samples == 0:
            raise ValueError("Completed training epochs require online metrics and sample counts")
        return result


def create_metrics_csv(path: Path) -> None:
    """Create one new segment table, never truncating an existing file."""
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[field.name for field in fields(EpochMetrics)])
        writer.writeheader()


def append_completed_metrics(path: Path, metrics: EpochMetrics) -> None:
    """Append only after complete.json has been published for this epoch."""
    if not path.is_file():
        raise ValueError("Segment CSV must already exist")
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[field.name for field in fields(EpochMetrics)])
        writer.writerow(metrics.record())
