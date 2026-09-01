"""Render immutable trajectory GIFs from completed projection and surface artifacts."""

from __future__ import annotations

import math
import os
import re
import shutil
import sys
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence, cast

import matplotlib


matplotlib.use("Agg")

import numpy as np
import PIL
from matplotlib import pyplot as plt
from matplotlib import patheffects
from matplotlib.table import Table
from PIL import Image, ImageColor

from .checkpoints import file_hash, read_json, write_json
from .config import LoadedConfig


ProgressCallback = Callable[[dict[str, object]], None]
_PROJECTION_FILES = {
    "mean.npy", "pc1.npy", "pc2.npy", "coordinates.npy", "residuals.npy",
    "eigenvalues.npy", "explained_variance_ratio.npy", "metadata.json",
}
_SURFACE_FILES = {
    "x_values.npy", "y_values.npy", "train_loss.npy", "train_accuracy.npy",
    "validation_loss.npy", "validation_accuracy.npy", "color_levels.npy",
    "checkpoint_metrics.json", "metadata.json",
}
_ANIMATION_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,99}$")
_WINDOWS_RESERVED = {
    "con", "prn", "aux", "nul", *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
}
_MAX_GIF_BYTES = 3_000_000
_FRAME_DURATION_MS = 200
_FINAL_HOLD_MS = 1_000
_PROFILES = ((960, 128), (960, 64), (800, 64), (640, 64))
_LINE_STYLES = ("-", "--", ":")
_MARKERS = ("o", "s", "^")
# Okabe-Ito colors selected to remain distinct from one another.  The double
# black/white halo applied by the renderer supplies contrast across both the
# dark and bright regions of the shared viridis-like contour background.
_BATCH_COLORS = {
    64: "#D55E00",  # vermillion
    256: "#56B4E9",  # sky blue
    1024: "#CC79A7",  # reddish purple
}
_ADDITIONAL_BATCH_COLORS = (
    "#009E73",  # bluish green
    "#E69F00",  # orange
    "#0072B2",  # blue
)
_HALO_OUTER = "#000000"
_HALO_INNER = "#FFFFFF"


@dataclass(frozen=True)
class CompletedArtifact:
    directory: Path
    complete: dict[str, object]
    metadata: dict[str, object]
    complete_sha256: str
    metadata_sha256: str


@dataclass(frozen=True)
class RunTrajectory:
    run_id: str
    batch_size: int
    seed: int
    epochs: tuple[int, ...]
    records: tuple[dict[str, object], ...]
    coordinates: np.ndarray
    residuals: np.ndarray
    color: str
    line_style: str
    marker: str

    @property
    def label(self) -> str:
        return f"B{self.batch_size} seed{self.seed}"


@dataclass(frozen=True)
class AnimationInputs:
    projection: CompletedArtifact
    surfaces: CompletedArtifact
    x_values: np.ndarray
    y_values: np.ndarray
    losses: Mapping[str, np.ndarray]
    color_levels: np.ndarray
    explained_variance_ratio: tuple[float, float]
    epochs: tuple[int, ...]
    runs: tuple[RunTrajectory, ...]
    train_background_samples: int
    validation_background_samples: int
    train_metric_samples: int
    validation_metric_samples: int


@dataclass(frozen=True)
class AnimationResult:
    animation_id: str
    projection_id: str
    directory: Path
    train_path: Path
    validation_path: Path
    frame_count: int
    width: int
    height: int
    colors: int
    train_size_bytes: int
    validation_size_bytes: int


def _notify(callback: ProgressCallback | None, **values: object) -> None:
    if callback is not None:
        callback(values)


def _manifest_record(manifest: Mapping[str, object], name: str) -> Mapping[str, object]:
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("Completion manifest has no file records")
    record = files.get(name)
    if not isinstance(record, dict):
        raise ValueError(f"Completion manifest lacks file record: {name}")
    size, digest = record.get("size_bytes"), record.get("sha256")
    if type(size) is not int or size <= 0 or not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"Invalid completed file record: {name}")
    return cast(Mapping[str, object], record)


def _load_completed_artifact(
    directory: Path,
    *,
    parent: Path,
    completed_kind: str,
    metadata_kind: str,
    expected_files: set[str],
) -> CompletedArtifact:
    """Verify every manifest-listed byte before reading metadata or NumPy payloads."""
    parent = parent.resolve()
    directory = directory.resolve()
    if directory.parent != parent:
        raise ValueError(f"Artifact must be a direct child of {parent}")
    complete_path = directory / "complete.json"
    if not directory.is_dir() or not complete_path.is_file() or complete_path.resolve() != complete_path:
        raise ValueError(f"Artifact has no direct completion manifest: {directory}")
    complete = read_json(complete_path)
    if complete.get("schema_version") != 1 or complete.get("kind") != completed_kind:
        raise ValueError(f"Unsupported completion schema: {completed_kind}")
    projection_id = complete.get("projection_id")
    if not isinstance(projection_id, str) or projection_id != directory.name:
        raise ValueError("Artifact path and projection identity disagree")
    files = complete.get("files")
    if not isinstance(files, dict) or set(files) != expected_files:
        raise ValueError(f"Completed {completed_kind} file set differs from schema")
    for name in sorted(expected_files):
        path = directory / name
        record = _manifest_record(complete, name)
        if not path.is_file() or path.resolve() != path:
            raise ValueError(f"Completed artifact file is missing or redirected: {name}")
        if path.stat().st_size != record["size_bytes"] or file_hash(path) != record["sha256"]:
            raise ValueError(f"Completed artifact file size/hash mismatch: {name}")
    metadata_record = _manifest_record(complete, "metadata.json")
    if complete.get("metadata_sha256") != metadata_record["sha256"]:
        raise ValueError("Completion marker and metadata hash disagree")
    metadata = read_json(directory / "metadata.json")
    if (
        metadata.get("schema_version") != 1
        or metadata.get("kind") != metadata_kind
        or metadata.get("projection_id") != projection_id
    ):
        raise ValueError(f"Unsupported or mismatched metadata schema: {metadata_kind}")
    return CompletedArtifact(
        directory=directory,
        complete=complete,
        metadata=metadata,
        complete_sha256=file_hash(complete_path),
        metadata_sha256=cast(str, metadata_record["sha256"]),
    )


def _load_array(
    artifact: CompletedArtifact,
    key: str,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    arrays = artifact.metadata.get("arrays")
    declaration = arrays.get(key) if isinstance(arrays, dict) else None
    expected = {"path": f"{key}.npy", "shape": list(expected_shape), "dtype": "float64"}
    if declaration != expected:
        raise ValueError(f"Array declaration differs from schema: {key}")
    # Animation inputs are only coordinates, residuals, ratios and 21x21 grids.
    # Load them eagerly so NumPy closes the .npy handle immediately; a read-only
    # memmap keeps the file locked on Windows and prevents temporary artifacts
    # from being removed after rendering or tests.
    array = np.load(artifact.directory / f"{key}.npy", allow_pickle=False)
    if array.dtype != np.dtype(np.float64) or array.shape != expected_shape:
        raise ValueError(f"Array dtype/shape differs from declaration: {key}")
    if not np.isfinite(array).all():
        raise ValueError(f"Animation source array contains nonfinite values: {key}")
    return array


def _finite_number(value: object, name: str, *, minimum: float | None = None) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError(f"Animation metric is not finite: {name}")
    result = float(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"Animation metric is below its valid range: {name}")
    return result


def _validated_metric_record(record: object, index: int) -> dict[str, object]:
    if not isinstance(record, dict) or record.get("index") != index:
        raise ValueError("Checkpoint metric order differs from projection order")
    metrics = record.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("Checkpoint animation record lacks metrics")
    for key in ("run_id", "segment_id", "epoch", "global_step"):
        if metrics.get(key) != record.get(key):
            raise ValueError("Checkpoint and metric identities disagree")
    if not isinstance(record.get("run_id"), str) or not isinstance(record.get("segment_id"), str):
        raise ValueError("Checkpoint metric identity is invalid")
    if type(record.get("epoch")) is not int or type(record.get("global_step")) is not int:
        raise ValueError("Checkpoint metric epoch or step is invalid")
    batch, seed = metrics.get("batch_size"), metrics.get("seed")
    if type(batch) is not int or batch <= 0 or type(seed) is not int or seed < 0:
        raise ValueError("Checkpoint batch size or seed is invalid")
    for key in ("train_subset_loss", "train_subset_accuracy", "val_loss", "val_accuracy"):
        value = _finite_number(metrics.get(key), key)
        if key.endswith("accuracy") and not 0 <= value <= 1:
            raise ValueError(f"Checkpoint accuracy is outside [0, 1]: {key}")
    for key in ("train_subset_samples", "val_samples"):
        if type(metrics.get(key)) is not int or cast(int, metrics[key]) <= 0:
            raise ValueError(f"Checkpoint sample count is invalid: {key}")
    epoch = cast(int, record["epoch"])
    for key in ("learning_rate", "gradient_norm"):
        value = metrics.get(key)
        if epoch == 0:
            if value is not None:
                raise ValueError(f"Epoch-zero {key} must be undefined")
        else:
            _finite_number(value, key, minimum=0.0)
    return cast(dict[str, object], record)


def _styles(
    records: Sequence[dict[str, object]], run_order: Sequence[str],
) -> dict[str, tuple[str, str, str]]:
    batches: list[int] = []
    seeds: list[int] = []
    first_by_run: dict[str, dict[str, object]] = {}
    for record in records:
        first_by_run.setdefault(cast(str, record["run_id"]), record)
    if set(first_by_run) != set(run_order):
        raise ValueError("Checkpoint run identities differ from run_order")
    for run_id in run_order:
        metrics = cast(dict[str, object], first_by_run[run_id]["metrics"])
        batch, seed = cast(int, metrics["batch_size"]), cast(int, metrics["seed"])
        if batch not in batches:
            batches.append(batch)
        if seed not in seeds:
            seeds.append(seed)
    unknown_batches = sorted(batch for batch in batches if batch not in _BATCH_COLORS)
    if len(unknown_batches) > len(_ADDITIONAL_BATCH_COLORS):
        raise ValueError("Animation has more batch conditions than the fixed palette supports")
    batch_colors = {
        **_BATCH_COLORS,
        **dict(zip(unknown_batches, _ADDITIONAL_BATCH_COLORS)),
    }
    return {
        run_id: (
            batch_colors[cast(int, cast(dict[str, object], first_by_run[run_id]["metrics"])["batch_size"])],
            _LINE_STYLES[seeds.index(cast(int, cast(dict[str, object], first_by_run[run_id]["metrics"])["seed"])) % len(_LINE_STYLES)],
            _MARKERS[seeds.index(cast(int, cast(dict[str, object], first_by_run[run_id]["metrics"])["seed"])) % len(_MARKERS)],
        )
        for run_id in run_order
    }


def _build_runs(
    projection: CompletedArtifact,
    coordinates: np.ndarray,
    residuals: np.ndarray,
    metric_records: Sequence[dict[str, object]],
) -> tuple[tuple[int, ...], tuple[RunTrajectory, ...]]:
    metadata = projection.metadata
    run_order, common_epochs = metadata.get("run_order"), metadata.get("common_epochs")
    if (
        not isinstance(run_order, list)
        or not run_order
        or len(set(run_order)) != len(run_order)
        or any(not isinstance(item, str) or not item for item in run_order)
    ):
        raise ValueError("Projection run_order is invalid")
    if (
        not isinstance(common_epochs, list)
        or not common_epochs
        or any(type(epoch) is not int for epoch in common_epochs)
        or common_epochs[0] != 0
        or any(right <= left for left, right in zip(common_epochs, common_epochs[1:]))
    ):
        raise ValueError("Projection common_epochs is invalid")
    epochs = tuple(cast(list[int], common_epochs))
    expected_count = len(run_order) * len(epochs)
    if len(metric_records) != expected_count or len(coordinates) != expected_count:
        raise ValueError("Run/epoch grid differs from projection sample count")
    styles = _styles(metric_records, cast(list[str], run_order))
    runs: list[RunTrajectory] = []
    for run_index, run_id in enumerate(cast(list[str], run_order)):
        start, stop = run_index * len(epochs), (run_index + 1) * len(epochs)
        records = tuple(metric_records[start:stop])
        if tuple(record["run_id"] for record in records) != (run_id,) * len(epochs):
            raise ValueError("Checkpoint records are not contiguous in run_order")
        if tuple(record["epoch"] for record in records) != epochs:
            raise ValueError("Checkpoint records do not cover the common epoch axis")
        first = cast(dict[str, object], records[0]["metrics"])
        if any(
            cast(dict[str, object], record["metrics"])["batch_size"] != first["batch_size"]
            or cast(dict[str, object], record["metrics"])["seed"] != first["seed"]
            for record in records
        ):
            raise ValueError("Batch size or seed changes within one trajectory")
        color, line_style, marker = styles[run_id]
        runs.append(RunTrajectory(
            run_id=run_id,
            batch_size=cast(int, first["batch_size"]),
            seed=cast(int, first["seed"]),
            epochs=epochs,
            records=records,
            coordinates=coordinates[start:stop],
            residuals=residuals[start:stop],
            color=color,
            line_style=line_style,
            marker=marker,
        ))
    return epochs, tuple(runs)


def load_animation_inputs(output_root: Path, projection_directory: Path) -> AnimationInputs:
    """Load only completed, mutually consistent saved arrays and JSON records."""
    output_root = output_root.resolve()
    projection = _load_completed_artifact(
        projection_directory,
        parent=output_root / "projections",
        completed_kind="completed_projection",
        metadata_kind="common_pca_projection",
        expected_files=_PROJECTION_FILES,
    )
    projection_id = projection.directory.name
    surfaces = _load_completed_artifact(
        output_root / "surfaces" / projection_id,
        parent=output_root / "surfaces",
        completed_kind="completed_loss_surfaces",
        metadata_kind="common_loss_surfaces",
        expected_files=_SURFACE_FILES,
    )
    source_projection = surfaces.metadata.get("projection")
    if not isinstance(source_projection, dict) or (
        source_projection.get("complete_sha256") != projection.complete_sha256
        or source_projection.get("metadata_sha256") != projection.metadata_sha256
        or source_projection.get("sample_count") != projection.metadata.get("sample_count")
    ):
        raise ValueError("Loss surfaces do not identify the completed projection bytes")
    for key in ("config_source_sha256", "effective_config_sha256"):
        if surfaces.metadata.get(key) != projection.metadata.get(key):
            raise ValueError(f"Projection and loss-surface {key} differ")

    sample_count = projection.metadata.get("sample_count")
    if type(sample_count) is not int or sample_count < 3:
        raise ValueError("Projection sample_count is invalid")
    coordinates = _load_array(projection, "coordinates", (sample_count, 2))
    residuals = _load_array(projection, "residuals", (sample_count,))
    ratios = _load_array(projection, "explained_variance_ratio", (2,))
    if np.any(residuals < 0) or np.any(ratios < 0) or float(ratios.sum()) > 1.0 + 1e-12:
        raise ValueError("Projection residuals or explained-variance ratios are invalid")
    declared_ratios = projection.metadata.get("explained_variance_ratio")
    if not isinstance(declared_ratios, list) or not np.array_equal(
        np.asarray(declared_ratios, dtype=np.float64), ratios,
    ):
        raise ValueError("Projection ratio array and metadata disagree")

    grid = surfaces.metadata.get("grid")
    if not isinstance(grid, dict) or not isinstance(grid.get("shape"), list):
        raise ValueError("Surface grid metadata is invalid")
    shape = tuple(grid["shape"])
    if len(shape) != 2 or any(type(value) is not int or value < 2 for value in shape):
        raise ValueError("Surface grid shape is invalid")
    y_count, x_count = cast(tuple[int, int], shape)
    x_values = _load_array(surfaces, "x_values", (x_count,))
    y_values = _load_array(surfaces, "y_values", (y_count,))
    if np.any(np.diff(x_values) <= 0) or np.any(np.diff(y_values) <= 0):
        raise ValueError("Surface axes must be strictly increasing")
    losses = {
        split: _load_array(surfaces, f"{split}_loss", (y_count, x_count))
        for split in ("train", "validation")
    }
    levels = _load_array(surfaces, "color_levels", (21,))
    color = surfaces.metadata.get("color_scale")
    if (
        not isinstance(color, dict)
        or color.get("intervals") != 20
        or color.get("shared_by") != ["train", "validation"]
        or np.any(np.diff(levels) <= 0)
        or min(float(value.min()) for value in losses.values()) < float(levels[0])
        or max(float(value.max()) for value in losses.values()) > float(levels[-1])
    ):
        raise ValueError("Shared surface color scale is invalid")
    evaluation = surfaces.metadata.get("evaluation")
    if not isinstance(evaluation, dict) or any((
        evaluation.get("parameter_dtype") != "torch.float32",
        evaluation.get("input_dtype") != "torch.float32",
        evaluation.get("amp") is not False,
        evaluation.get("tf32") is not False,
    )):
        raise ValueError("Surface numerical conditions differ from the animation contract")

    metrics_document = read_json(surfaces.directory / "checkpoint_metrics.json")
    records_value = metrics_document.get("records")
    if (
        metrics_document.get("schema_version") != 1
        or metrics_document.get("kind") != "actual_checkpoint_metrics"
        or metrics_document.get("projection_id") != projection_id
        or metrics_document.get("train_scope")
        != "fixed train subset with evaluation preprocessing"
        or metrics_document.get("validation_scope") != "full validation split"
        or not isinstance(records_value, list)
        or len(records_value) != sample_count
    ):
        raise ValueError("Actual checkpoint metric document is invalid")
    metric_records = tuple(
        _validated_metric_record(record, index) for index, record in enumerate(records_value)
    )
    checkpoints = projection.metadata.get("checkpoints")
    if not isinstance(checkpoints, list) or len(checkpoints) != sample_count:
        raise ValueError("Projection checkpoint declarations are invalid")
    for record, checkpoint in zip(metric_records, checkpoints):
        if not isinstance(checkpoint, dict) or record != {
            "index": checkpoint.get("index"),
            "run_id": checkpoint.get("run_id"),
            "segment_id": checkpoint.get("segment_id"),
            "epoch": checkpoint.get("epoch"),
            "global_step": checkpoint.get("global_step"),
            "metrics": checkpoint.get("metrics"),
        }:
            raise ValueError("Saved actual metrics differ from projection checkpoint records")
    epochs, runs = _build_runs(projection, coordinates, residuals, metric_records)

    subsets = surfaces.metadata.get("subsets")
    train_subset = subsets.get("train") if isinstance(subsets, dict) else None
    validation_subset = subsets.get("validation") if isinstance(subsets, dict) else None
    if not isinstance(train_subset, dict) or not isinstance(validation_subset, dict):
        raise ValueError("Surface subset provenance is invalid")
    train_background = train_subset.get("samples")
    validation_background = validation_subset.get("samples")
    first_metrics = cast(dict[str, object], runs[0].records[0]["metrics"])
    train_metric_samples, validation_metric_samples = (
        first_metrics.get("train_subset_samples"), first_metrics.get("val_samples")
    )
    if any(
        type(value) is not int or value <= 0
        for value in (
            train_background, validation_background, train_metric_samples,
            validation_metric_samples,
        )
    ):
        raise ValueError("Animation source sample counts are invalid")
    for run in runs:
        for record in run.records:
            metrics = cast(dict[str, object], record["metrics"])
            if (
                metrics["train_subset_samples"] != train_metric_samples
                or metrics["val_samples"] != validation_metric_samples
            ):
                raise ValueError("Actual checkpoint sample scope changes between frames")
    return AnimationInputs(
        projection=projection,
        surfaces=surfaces,
        x_values=x_values,
        y_values=y_values,
        losses=losses,
        color_levels=levels,
        explained_variance_ratio=(float(ratios[0]), float(ratios[1])),
        epochs=epochs,
        runs=runs,
        train_background_samples=cast(int, train_background),
        validation_background_samples=cast(int, validation_background),
        train_metric_samples=cast(int, train_metric_samples),
        validation_metric_samples=cast(int, validation_metric_samples),
    )


def _format_number(value: object, *, scientific: bool = False) -> str:
    if value is None:
        return "N/A"
    number = float(cast(float, value))
    return f"{number:.2e}" if scientific else f"{number:.3f}"


def _trajectory_dimensions(scale: float) -> dict[str, float]:
    return {
        "line_core": max(0.9, 1.5 * scale),
        "line_white_halo": max(1.6, 2.5 * scale),
        "line_black_halo": max(2.3, 3.5 * scale),
        "current_marker_size": max(4.0, 7.0 * scale),
        "current_marker_edge": max(0.6, 0.9 * scale),
        "current_marker_black_halo": max(1.8, 2.8 * scale),
    }


class _FrameRenderer:
    def __init__(self, inputs: AnimationInputs, split: str, width: int) -> None:
        if split not in ("train", "validation"):
            raise ValueError(f"Unsupported animation background: {split}")
        self.inputs, self.split, self.width = inputs, split, width
        self.height = round(width * 2 / 3)
        scale = width / 960
        trajectory_dimensions = _trajectory_dimensions(scale)
        run_count = len(inputs.runs)
        ratios = [3.55, 1.25] if run_count <= 3 else [2.65, 2.0]
        self.figure = plt.figure(figsize=(width / 100, self.height / 100), dpi=100)
        grid = self.figure.add_gridspec(2, 1, height_ratios=ratios, hspace=0.32)
        self.axis = self.figure.add_subplot(grid[0, 0])
        self.metrics_axis = self.figure.add_subplot(grid[1, 0])
        self.metrics_axis.axis("off")
        contour = self.axis.contourf(
            inputs.x_values,
            inputs.y_values,
            inputs.losses[split],
            levels=inputs.color_levels,
        )
        self.axis.contour(
            inputs.x_values,
            inputs.y_values,
            inputs.losses[split],
            levels=inputs.color_levels,
            colors=plt.rcParams["text.color"],
            linewidths=max(0.15, 0.3 * scale),
            alpha=0.18,
        )
        colorbar = self.figure.colorbar(contour, ax=self.axis, pad=0.018, aspect=25)
        colorbar.set_label("Plane cross-entropy loss", fontsize=max(6, 8 * scale))
        colorbar.ax.tick_params(labelsize=max(5, 7 * scale))
        ratio_1, ratio_2 = inputs.explained_variance_ratio
        self.axis.set_xlabel(f"PC1 ({ratio_1 * 100:.2f}% variance)", fontsize=max(6, 9 * scale))
        self.axis.set_ylabel(f"PC2 ({ratio_2 * 100:.2f}% variance)", fontsize=max(6, 9 * scale))
        self.axis.set_xlim(float(inputs.x_values[0]), float(inputs.x_values[-1]))
        self.axis.set_ylim(float(inputs.y_values[0]), float(inputs.y_values[-1]))
        self.axis.tick_params(labelsize=max(5, 7.5 * scale))
        self.axis.grid(False)
        self.lines = []
        self.points = []
        for run in inputs.runs:
            line, = self.axis.plot(
                [], [], color=run.color, linestyle=run.line_style,
                linewidth=trajectory_dimensions["line_core"], alpha=1.0,
            )
            line.set_path_effects([
                patheffects.Stroke(
                    linewidth=trajectory_dimensions["line_black_halo"],
                    foreground=_HALO_OUTER,
                ),
                patheffects.Stroke(
                    linewidth=trajectory_dimensions["line_white_halo"],
                    foreground=_HALO_INNER,
                ),
                patheffects.Normal(),
            ])
            point, = self.axis.plot(
                [], [], color=run.color, marker=run.marker, linestyle="none",
                markersize=trajectory_dimensions["current_marker_size"],
                markeredgecolor=_HALO_INNER,
                markeredgewidth=trajectory_dimensions["current_marker_edge"], label=run.label,
            )
            point.set_path_effects([
                patheffects.Stroke(
                    linewidth=trajectory_dimensions["current_marker_black_halo"],
                    foreground=_HALO_OUTER,
                ),
                patheffects.Normal(),
            ])
            self.lines.append(line)
            self.points.append(point)
        self.table = self._create_table(scale)
        self.scope_text = self.metrics_axis.text(
            0.0, 0.99,
            (
                f"Actual checkpoint metrics — train: fixed eval subset n={inputs.train_metric_samples:,}; "
                f"validation: full split n={inputs.validation_metric_samples:,}; "
                "grad: epoch-mean global L2; residual: distance outside PC1/PC2"
            ),
            transform=self.metrics_axis.transAxes,
            ha="left", va="top", fontsize=max(5, 7.5 * scale),
        )
        self.figure.subplots_adjust(left=0.09, right=0.92, top=0.93, bottom=0.055)

    def _create_table(self, scale: float) -> Table:
        columns = ("Run", "Step", "LR", "Train loss / acc", "Full-val loss / acc", "Grad L2", "Residual")
        widths = (0.13, 0.09, 0.10, 0.19, 0.19, 0.13, 0.13)
        rows = [["" for _ in columns] for _ in self.inputs.runs]
        table = self.metrics_axis.table(
            cellText=rows,
            colLabels=columns,
            colWidths=widths,
            cellLoc="center",
            bbox=(0.0, 0.02, 1.0, 0.76),
        )
        table.auto_set_font_size(False)
        font_size = (7.5 if len(rows) <= 3 else 6.5) * scale
        table.set_fontsize(max(4.7, font_size))
        for (row, _), cell in table.get_celld().items():
            cell.set_linewidth(0.35)
            if row == 0:
                cell.set_facecolor(plt.rcParams["grid.color"])
                cell.get_text().set_weight("bold")
        for row, run in enumerate(self.inputs.runs, start=1):
            table[(row, 0)].get_text().set_color(run.color)
            table[(row, 0)].get_text().set_weight("bold")
        return table

    def render(self, frame_index: int) -> Image.Image:
        epoch = self.inputs.epochs[frame_index]
        background_samples = (
            self.inputs.train_background_samples
            if self.split == "train"
            else self.inputs.validation_background_samples
        )
        label = "train fixed subset" if self.split == "train" else "validation fixed subset"
        captured = sum(self.inputs.explained_variance_ratio) * 100
        self.axis.set_title(
            f"Epoch {epoch} | background: {label} (n={background_samples:,}), FP32 plane loss\n"
            f"PC1+PC2 capture {captured:.2f}% — saved epoch points joined; no within-epoch path implied",
            fontsize=max(6.5, 9.5 * self.width / 960),
        )
        for row, (run, line, point) in enumerate(
            zip(self.inputs.runs, self.lines, self.points), start=1,
        ):
            history = run.coordinates[:frame_index + 1]
            line.set_data(history[:, 0], history[:, 1])
            point.set_data([history[-1, 0]], [history[-1, 1]])
            record = run.records[frame_index]
            metrics = cast(dict[str, object], record["metrics"])
            values = (
                run.label,
                f"{cast(int, record['global_step']):,}",
                _format_number(metrics["learning_rate"], scientific=True),
                f"{_format_number(metrics['train_subset_loss'])} / {float(cast(float, metrics['train_subset_accuracy'])) * 100:.1f}%",
                f"{_format_number(metrics['val_loss'])} / {float(cast(float, metrics['val_accuracy'])) * 100:.1f}%",
                _format_number(metrics["gradient_norm"], scientific=True),
                f"{float(run.residuals[frame_index]):.3g}",
            )
            for column, value in enumerate(values):
                self.table[(row, column)].get_text().set_text(value)
        self.figure.canvas.draw()
        rgba = np.asarray(self.figure.canvas.buffer_rgba()).copy()
        image = Image.fromarray(rgba, mode="RGBA").convert("RGB")
        if image.size != (self.width, self.height):
            raise RuntimeError(
                f"Rendered frame size {image.size} differs from {(self.width, self.height)}"
            )
        return image

    def close(self) -> None:
        plt.close(self.figure)


def _fixed_palette_frames(
    frames: Sequence[Image.Image],
    colors: int,
    *,
    required_colors: Sequence[str],
) -> list[Image.Image]:
    if not frames:
        raise ValueError("Cannot encode an empty animation")
    unique_required = tuple(dict.fromkeys(required_colors))
    if len(unique_required) > colors:
        raise ValueError("Required trajectory colors exceed the GIF palette capacity")
    palette = frames[0].quantize(
        colors=colors,
        method=Image.Quantize.MEDIANCUT,
        dither=Image.Dither.NONE,
    )
    palette_values = palette.getpalette()
    if palette_values is None or len(palette_values) < colors * 3:
        raise RuntimeError("Adaptive GIF palette is incomplete")
    for index, color in enumerate(unique_required):
        palette_values[index * 3:(index + 1) * 3] = ImageColor.getrgb(color)
    palette.putpalette(palette_values)
    return [
        frame.quantize(palette=palette, dither=Image.Dither.NONE)
        for frame in frames
    ]


def _trajectory_palette_colors(inputs: AnimationInputs) -> tuple[str, ...]:
    return tuple(dict.fromkeys((
        *(run.color for run in inputs.runs),
        _HALO_OUTER,
        _HALO_INNER,
    )))


def _save_gif(
    inputs: AnimationInputs,
    split: str,
    path: Path,
    *,
    width: int,
    colors: int,
    progress: ProgressCallback | None = None,
) -> tuple[int, int]:
    if type(width) is not int or width <= 0 or colors not in (64, 128):
        raise ValueError("GIF width and fixed-palette color count are invalid")
    renderer = _FrameRenderer(inputs, split, width)
    frames: list[Image.Image] = []
    quantized: list[Image.Image] = []
    required_colors = _trajectory_palette_colors(inputs)
    try:
        for index in range(len(inputs.epochs)):
            frames.append(renderer.render(index))
            _notify(
                progress,
                status="animation_frame_rendered",
                split=split,
                frame_index=index,
                frame_count=len(inputs.epochs),
                epoch=inputs.epochs[index],
                width=width,
                colors=colors,
            )
        quantized = _fixed_palette_frames(
            frames,
            colors,
            required_colors=required_colors,
        )
        durations = [_FRAME_DURATION_MS] * len(quantized)
        durations[-1] += _FINAL_HOLD_MS
        palette_values = quantized[0].getpalette()
        if palette_values is None:
            raise RuntimeError("Quantized GIF frame has no fixed palette")
        quantized[0].save(
            path,
            format="GIF",
            save_all=True,
            append_images=quantized[1:],
            duration=durations,
            loop=0,
            disposal=1,
            optimize=True,
            include_color_table=False,
            palette=bytes(palette_values[:colors * 3]),
        )
    finally:
        renderer.close()
        for frame in (*frames, *quantized):
            frame.close()
    height = round(width * 2 / 3)
    _validate_gif(
        path,
        frame_count=len(inputs.epochs),
        size=(width, height),
        required_colors=required_colors,
    )
    return path.stat().st_size, height


def _validate_gif(
    path: Path,
    *,
    frame_count: int,
    size: tuple[int, int],
    required_colors: Sequence[str] = (),
) -> None:
    with Image.open(path) as animation:
        if animation.format != "GIF" or animation.size != size:
            raise ValueError(f"Encoded GIF format or size is invalid: {path.name}")
        if getattr(animation, "n_frames", 1) != frame_count:
            raise ValueError(f"Encoded GIF dropped epoch frames: {path.name}")
        palette = animation.getpalette()
        palette_colors = {
            tuple(palette[index:index + 3])
            for index in range(0, len(palette), 3)
        } if palette is not None else set()
        required_rgb = {ImageColor.getrgb(color) for color in required_colors}
        if not required_rgb.issubset(palette_colors):
            raise ValueError(f"Encoded GIF dropped required trajectory colors: {path.name}")
        durations: list[int] = []
        for index in range(frame_count):
            animation.seek(index)
            durations.append(int(animation.info.get("duration", 0)))
        expected = [_FRAME_DURATION_MS] * frame_count
        expected[-1] += _FINAL_HOLD_MS
        if durations != expected:
            raise ValueError(f"Encoded GIF frame durations differ from the contract: {path.name}")


def _new_animation_id(animation_name: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{animation_name[:32]}_{stamp}_{uuid.uuid4().hex}"


def _animation_destination(output_root: Path, animation_id: str, *, create: bool) -> Path:
    parent = (output_root / "animations").resolve()
    destination = (parent / animation_id).resolve()
    if destination.parent != parent:
        raise ValueError("Animation identity escapes the animation artifact parent")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite animation artifact: {destination}")
    if create:
        parent.mkdir(parents=True, exist_ok=True)
        if parent.resolve() != parent:
            raise ValueError("Animation artifact parent changed while it was prepared")
        destination.mkdir(exist_ok=False)
    return destination


def _validate_animation_name(name: str) -> str:
    if not isinstance(name, str) or not _ANIMATION_NAME.fullmatch(name):
        raise ValueError(
            "animation_name must contain only lowercase letters, digits, '_' or '-'"
        )
    if name.rstrip(" .").lower() in _WINDOWS_RESERVED:
        raise ValueError("animation_name is reserved by Windows")
    return name


def _copy_new(source: Path, destination: Path) -> None:
    with source.open("rb") as input_handle, destination.open("xb") as output_handle:
        shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)
        output_handle.flush()
        os.fsync(output_handle.fileno())


def render_animation_pair(
    loaded: LoadedConfig,
    projection_directory: Path,
    *,
    animation_name: str,
    progress: ProgressCallback | None = None,
) -> AnimationResult:
    """Render the train/validation GIF pair without loading data, models, or checkpoints."""
    animation_name = _validate_animation_name(animation_name)
    projection_path = projection_directory
    if not projection_path.is_absolute():
        projection_path = loaded.project_root / projection_path
    inputs = load_animation_inputs(loaded.config.paths.output_root, projection_path)
    if (
        inputs.projection.metadata.get("config_source_sha256") != loaded.source_sha256
        or inputs.projection.metadata.get("effective_config_sha256") != loaded.effective_sha256
    ):
        raise ValueError("Projection configuration differs from this rendering request")
    projection_id = inputs.projection.directory.name
    animation_id = _new_animation_id(animation_name)
    destination = _animation_destination(
        loaded.config.paths.output_root, animation_id, create=False,
    )
    attempts: list[dict[str, object]] = []
    accepted: tuple[Path, Path, int, int, int, int, int] | None = None
    with tempfile.TemporaryDirectory(prefix="loss-landscape-gif-") as temporary:
        temporary_root = Path(temporary)
        for width, colors in _PROFILES:
            _notify(
                progress,
                status="animation_profile_started",
                width=width,
                height=round(width * 2 / 3),
                colors=colors,
            )
            train_candidate = temporary_root / f"train_{width}_{colors}.gif"
            validation_candidate = temporary_root / f"validation_{width}_{colors}.gif"
            train_size, height = _save_gif(
                inputs, "train", train_candidate, width=width, colors=colors, progress=progress,
            )
            validation_size, validation_height = _save_gif(
                inputs, "validation", validation_candidate,
                width=width, colors=colors, progress=progress,
            )
            if validation_height != height:
                raise RuntimeError("Paired GIF dimensions differ")
            fits = train_size <= _MAX_GIF_BYTES and validation_size <= _MAX_GIF_BYTES
            attempts.append({
                "width": width,
                "height": height,
                "colors": colors,
                "train_size_bytes": train_size,
                "validation_size_bytes": validation_size,
                "both_within_limit": fits,
            })
            _notify(
                progress,
                status="animation_profile_measured",
                **attempts[-1],
                max_size_bytes=_MAX_GIF_BYTES,
            )
            if fits:
                accepted = (
                    train_candidate, validation_candidate, width, height, colors,
                    train_size, validation_size,
                )
                break
        if accepted is None:
            summary = "; ".join(
                f"{item['width']}x{item['height']}/{item['colors']} colors: "
                f"train={item['train_size_bytes']}, validation={item['validation_size_bytes']} bytes"
                for item in attempts
            )
            raise ValueError(
                f"GIF pair exceeds {_MAX_GIF_BYTES} bytes after every fixed fallback: {summary}"
            )
        train_candidate, validation_candidate, width, height, colors, train_size, validation_size = accepted
        _animation_destination(loaded.config.paths.output_root, animation_id, create=True)
        train_path = destination / f"{animation_name}.gif"
        validation_path = destination / f"{animation_name}_val.gif"
        _copy_new(train_candidate, train_path)
        _copy_new(validation_candidate, validation_path)
        if train_path.stat().st_size != train_size or validation_path.stat().st_size != validation_size:
            raise OSError("Published GIF size differs from the verified candidate")
        required_colors = _trajectory_palette_colors(inputs)
        _validate_gif(
            train_path,
            frame_count=len(inputs.epochs),
            size=(width, height),
            required_colors=required_colors,
        )
        _validate_gif(
            validation_path,
            frame_count=len(inputs.epochs),
            size=(width, height),
            required_colors=required_colors,
        )

    implementation_paths = (
        loaded.project_root / "src" / "landscape_exp" / "animation.py",
        loaded.project_root / "scripts" / "render_animation.py",
    )
    implementation_files = {
        path.relative_to(loaded.project_root).as_posix(): file_hash(path)
        for path in implementation_paths
    }
    outputs = {
        train_path.name: {"size_bytes": train_path.stat().st_size, "sha256": file_hash(train_path)},
        validation_path.name: {
            "size_bytes": validation_path.stat().st_size,
            "sha256": file_hash(validation_path),
        },
    }
    write_json(destination / "metadata.json", {
        "schema_version": 1,
        "kind": "trajectory_animation_pair",
        "animation_id": animation_id,
        "projection_id": projection_id,
        "animation_name": animation_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_source": str(loaded.source_path),
        "config_source_sha256": loaded.source_sha256,
        "effective_config_sha256": loaded.effective_sha256,
        "implementation_files": implementation_files,
        "runtime": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
            "pillow": PIL.__version__,
        },
        "sources": {
            "projection": {
                "directory": str(inputs.projection.directory),
                "complete_sha256": inputs.projection.complete_sha256,
                "metadata_sha256": inputs.projection.metadata_sha256,
            },
            "loss_surfaces": {
                "directory": str(inputs.surfaces.directory),
                "complete_sha256": inputs.surfaces.complete_sha256,
                "metadata_sha256": inputs.surfaces.metadata_sha256,
            },
            "model_evaluation_performed": False,
            "checkpoint_deserialization_performed": False,
        },
        "frames": {
            "epochs": list(inputs.epochs),
            "count": len(inputs.epochs),
            "fps": 5,
            "duration_ms": _FRAME_DURATION_MS,
            "final_additional_hold_ms": _FINAL_HOLD_MS,
            "interpolation_or_smoothing": False,
        },
        "visual_contract": {
            "fixed_axes": {
                "x_min": float(inputs.x_values[0]),
                "x_max": float(inputs.x_values[-1]),
                "y_min": float(inputs.y_values[0]),
                "y_max": float(inputs.y_values[-1]),
            },
            "shared_color_levels": [float(value) for value in inputs.color_levels],
            "paired_trajectory_and_time_axis": True,
            "plane_backgrounds": {
                "train_samples": inputs.train_background_samples,
                "validation_samples": inputs.validation_background_samples,
                "parameter_and_input_dtype": "float32",
            },
            "actual_checkpoint_metrics": {
                "train_subset_samples": inputs.train_metric_samples,
                "validation_full_split_samples": inputs.validation_metric_samples,
            },
            "explained_variance_ratio": list(inputs.explained_variance_ratio),
            "residual_definition": "saved high-dimensional distance outside the PC1/PC2 plane",
        },
        "runs": [{
            "run_id": run.run_id,
            "batch_size": run.batch_size,
            "seed": run.seed,
            "color": run.color,
            "line_style": run.line_style,
            "marker": run.marker,
        } for run in inputs.runs],
        "trajectory_style": {
            "batch_color_palette": "Okabe-Ito high-contrast subset",
            "double_halo": "black outer stroke and white inner stroke",
            "accepted_profile_dimensions_points": _trajectory_dimensions(width / 960),
            "palette_colors_reserved": [
                *_trajectory_palette_colors(inputs),
            ],
            "seed_encoding": "line style and current-point marker",
        },
        "encoding": {
            "format": "GIF",
            "max_size_bytes": _MAX_GIF_BYTES,
            "fixed_palette_per_gif": True,
            "differential_frames": True,
            "accepted_profile": {"width": width, "height": height, "colors": colors},
            "attempts": attempts,
        },
        "outputs": outputs,
    })
    files = {
        **outputs,
        "metadata.json": {
            "size_bytes": (destination / "metadata.json").stat().st_size,
            "sha256": file_hash(destination / "metadata.json"),
        },
    }
    write_json(destination / "complete.json", {
        "schema_version": 1,
        "kind": "completed_trajectory_animation_pair",
        "animation_id": animation_id,
        "projection_id": projection_id,
        "animation_name": animation_name,
        "metadata_sha256": files["metadata.json"]["sha256"],
        "files": files,
    })
    _notify(
        progress,
        status="animation_pair_completed",
        animation_id=animation_id,
        projection_id=projection_id,
        directory=str(destination),
        train_path=str(train_path),
        validation_path=str(validation_path),
    )
    return AnimationResult(
        animation_id=animation_id,
        projection_id=projection_id,
        directory=destination,
        train_path=train_path,
        validation_path=validation_path,
        frame_count=len(inputs.epochs),
        width=width,
        height=height,
        colors=colors,
        train_size_bytes=train_size,
        validation_size_bytes=validation_size,
    )
