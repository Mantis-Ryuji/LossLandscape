"""Evaluate and publish immutable train/validation loss surfaces.

The shared PCA artifact defines the plane.  Both fixed CIFAR-10 subsets are
evaluated at every grid point after one parameter assignment, while the full
validation metrics remain the measurements stored with the real checkpoints.
"""

from __future__ import annotations

import hashlib
import math
import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch import nn

from .checkpoints import file_hash, read_json, write_json
from .config import LoadedConfig
from .data import (
    SplitSpec, build_dataset_views, load_cifar10_training, load_split_indices,
    make_loader, split_path,
)
from .evaluate import EvaluationResult, evaluation_context, prepare_batch, classification_loss
from .landscape import (
    ParameterLayout, assign_parameter_vector, build_parameter_layout,
    flatten_parameters, validate_model_state,
)
from .models import load_initial_checkpoint
from .projection import load_analysis_checkpoint
from .seeds import LoaderGenerators, preserve_random_state


ProgressCallback = Callable[[dict[str, object]], None]
_PROJECTION_ARRAY_FILES = (
    "mean.npy", "pc1.npy", "pc2.npy", "coordinates.npy", "residuals.npy",
    "eigenvalues.npy", "explained_variance_ratio.npy",
)
_SURFACE_ARRAY_FILES = (
    "x_values.npy", "y_values.npy", "train_loss.npy", "train_accuracy.npy",
    "validation_loss.npy", "validation_accuracy.npy", "color_levels.npy",
)


@dataclass(frozen=True)
class ProjectionArtifact:
    directory: Path
    projection_id: str
    metadata: dict[str, object]
    complete_sha256: str
    metadata_sha256: str
    mean: np.ndarray
    pc1: np.ndarray
    pc2: np.ndarray
    coordinates: np.ndarray


@dataclass(frozen=True)
class LossSurface:
    x_values: np.ndarray
    y_values: np.ndarray
    loss: np.ndarray
    accuracy: np.ndarray
    samples: int


@dataclass(frozen=True)
class ColorScale:
    raw_minimum: float
    raw_maximum: float
    display_minimum: float
    display_maximum: float
    levels: np.ndarray
    intervals: int


@dataclass(frozen=True)
class SurfaceResult:
    projection_id: str
    directory: Path
    grid_points: int
    train_samples: int
    validation_samples: int
    loss_minimum: float
    loss_maximum: float


def _notify(callback: ProgressCallback | None, **values: object) -> None:
    if callback is not None:
        callback(values)


def _file_record(manifest: Mapping[str, object], name: str) -> dict[str, object]:
    files = manifest.get("files")
    expected = set(_PROJECTION_ARRAY_FILES) | {"metadata.json"}
    if not isinstance(files, dict) or set(files) != expected:
        raise ValueError("Completed projection file manifest differs from schema")
    record = files.get(name)
    if not isinstance(record, dict):
        raise ValueError(f"Missing projection file record: {name}")
    size, digest = record.get("size_bytes"), record.get("sha256")
    if type(size) is not int or size <= 0 or not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"Invalid projection file record: {name}")
    return cast(dict[str, object], record)


def _load_projection_array(
    directory: Path, metadata: Mapping[str, object], name: str,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    arrays = metadata.get("arrays")
    if not isinstance(arrays, dict):
        raise ValueError("Projection metadata lacks array declarations")
    declaration = arrays.get(name)
    if not isinstance(declaration, dict) or declaration != {
        "path": f"{name}.npy", "shape": list(expected_shape), "dtype": "float64",
    }:
        raise ValueError(f"Projection array declaration differs: {name}")
    array = np.load(directory / f"{name}.npy", mmap_mode="r", allow_pickle=False)
    if array.dtype != np.dtype(np.float64) or array.shape != expected_shape:
        raise ValueError(f"Projection array dtype/shape differs: {name}")
    return array


def load_projection_artifact(
    directory: Path, *, projections_root: Path | None = None,
) -> ProjectionArtifact:
    """Verify every completed projection file before opening any NumPy array."""
    declared = directory
    directory = declared.resolve()
    if declared.is_absolute() and declared != directory:
        raise ValueError("Projection directory resolves through a redirected path")
    if projections_root is not None:
        root = projections_root.resolve()
        if directory.parent != root:
            raise ValueError("Projection must be a direct child of output_root/projections")
    complete_path = directory / "complete.json"
    if not directory.is_dir() or not complete_path.is_file() or complete_path.resolve() != complete_path:
        raise ValueError(f"Projection has no direct completion manifest: {directory}")
    complete = read_json(complete_path)
    if complete.get("schema_version") != 1 or complete.get("kind") != "completed_projection":
        raise ValueError("Unsupported completed projection schema")
    identifier = complete.get("projection_id")
    if not isinstance(identifier, str) or identifier != directory.name:
        raise ValueError("Projection path and completed identity disagree")

    for name in (*_PROJECTION_ARRAY_FILES, "metadata.json"):
        path = directory / name
        record = _file_record(complete, name)
        if not path.is_file() or path.resolve() != path:
            raise ValueError(f"Projection file is missing or redirected: {name}")
        if path.stat().st_size != record["size_bytes"] or file_hash(path) != record["sha256"]:
            raise ValueError(f"Projection file size/hash mismatch: {name}")
    metadata_record = _file_record(complete, "metadata.json")
    if complete.get("metadata_sha256") != metadata_record["sha256"]:
        raise ValueError("Projection metadata hash differs from the completion marker")
    metadata = read_json(directory / "metadata.json")
    if (
        metadata.get("schema_version") != 1
        or metadata.get("kind") != "common_pca_projection"
        or metadata.get("projection_id") != identifier
    ):
        raise ValueError("Projection metadata identity differs from completion")
    samples, parameters = metadata.get("sample_count"), metadata.get("parameter_count")
    if type(samples) is not int or samples < 3 or type(parameters) is not int or parameters < 2:
        raise ValueError("Projection dimensions are invalid")
    mean = _load_projection_array(directory, metadata, "mean", (parameters,))
    pc1 = _load_projection_array(directory, metadata, "pc1", (parameters,))
    pc2 = _load_projection_array(directory, metadata, "pc2", (parameters,))
    coordinates = _load_projection_array(directory, metadata, "coordinates", (samples, 2))
    if not np.isfinite(coordinates).all():
        raise ValueError("Projection coordinates are nonfinite")
    return ProjectionArtifact(
        directory=directory,
        projection_id=identifier,
        metadata=metadata,
        complete_sha256=file_hash(complete_path),
        metadata_sha256=cast(str, metadata_record["sha256"]),
        mean=mean,
        pc1=pc1,
        pc2=pc2,
        coordinates=coordinates,
    )


def common_grid(
    coordinates: np.ndarray, *, grid_size: int, margin_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return equal-spaced axes covering all trajectories plus fixed margins."""
    if (
        not isinstance(coordinates, np.ndarray)
        or coordinates.ndim != 2
        or coordinates.shape[1] != 2
        or coordinates.dtype != np.float64
        or not np.isfinite(coordinates).all()
    ):
        raise ValueError("Coordinates must be a finite [samples, 2] float64 array")
    if type(grid_size) is not int or grid_size < 2:
        raise ValueError("grid_size must be an integer of at least two")
    if type(margin_ratio) not in (int, float) or not math.isfinite(margin_ratio) or margin_ratio <= 0:
        raise ValueError("margin_ratio must be a finite positive number")
    axes: list[np.ndarray] = []
    for component in range(2):
        minimum = float(coordinates[:, component].min())
        maximum = float(coordinates[:, component].max())
        width = maximum - minimum
        if width == 0:
            center = (minimum + maximum) * 0.5
            minimum, maximum = center - 1e-6, center + 1e-6
        else:
            minimum -= float(margin_ratio) * width
            maximum += float(margin_ratio) * width
        axes.append(np.linspace(minimum, maximum, grid_size, dtype=np.float64))
    return axes[0], axes[1]


def _validated_axis(values: np.ndarray, name: str) -> np.ndarray:
    if (
        not isinstance(values, np.ndarray)
        or values.dtype != np.float64
        or values.ndim != 1
        or len(values) < 2
        or not np.isfinite(values).all()
        or np.any(np.diff(values) <= 0)
    ):
        raise ValueError(f"{name} must be a strictly increasing finite float64 axis")
    return values


def _validated_plane_vector(value: torch.Tensor, name: str, expected: int) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.device.type != "cpu"
        or value.dtype != torch.float64
        or value.ndim != 1
        or value.numel() != expected
        or not torch.isfinite(value).all().item()
    ):
        raise ValueError(f"{name} must be a finite CPU float64 parameter vector")
    return value


def _evaluate_assigned_model(
    model: nn.Module, batches: Iterable[object], device: torch.device,
) -> EvaluationResult:
    loss_sum, correct, samples = 0.0, 0, 0
    for batch in batches:
        images, labels = prepare_batch(batch, device)
        logits = model(images)
        loss = classification_loss(logits, labels)
        count = len(labels)
        loss_sum += float(loss.item()) * count
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        samples += count
        del images, labels, logits, loss
    if samples == 0:
        raise ValueError("Cannot evaluate an empty landscape subset")
    return EvaluationResult(loss_sum / samples, correct / samples, samples)


def _evaluate_surface_pair(
    model: nn.Module,
    reference_vector: torch.Tensor,
    direction_1: torch.Tensor,
    direction_2: torch.Tensor,
    x_values: np.ndarray,
    y_values: np.ndarray,
    batches_by_split: Mapping[str, Iterable[object]],
    device: torch.device,
    *,
    progress: ProgressCallback | None = None,
) -> dict[str, LossSurface]:
    split_names = tuple(batches_by_split)
    if not split_names or any(not isinstance(name, str) or not name for name in split_names):
        raise ValueError("At least one named surface batch sequence is required")
    if any(iter(value) is value for value in batches_by_split.values()):
        raise ValueError("Landscape batches must be reiterable, not one-shot iterators")
    x_values, y_values = _validated_axis(x_values, "x_values"), _validated_axis(y_values, "y_values")
    layout = build_parameter_layout(model)
    size = layout.total_numel
    reference_vector = _validated_plane_vector(reference_vector, "reference_vector", size)
    direction_1 = _validated_plane_vector(direction_1, "direction_1", size)
    direction_2 = _validated_plane_vector(direction_2, "direction_2", size)
    original = flatten_parameters(
        {name: parameter.detach().to(device="cpu") for name, parameter in model.named_parameters()},
        layout.parameter_names,
    )
    losses = {
        name: np.empty((len(y_values), len(x_values)), dtype=np.float64)
        for name in batches_by_split
    }
    accuracies = {name: np.empty_like(losses[name]) for name in batches_by_split}
    sample_counts: dict[str, int] = {}
    total = len(x_values) * len(y_values)
    completed = 0
    try:
        with evaluation_context(model, device), torch.inference_mode():
            for y_index, y_value in enumerate(y_values):
                for x_index, x_value in enumerate(x_values):
                    point = torch.add(reference_vector, direction_1, alpha=float(x_value))
                    point.add_(direction_2, alpha=float(y_value))
                    point_fp32 = point.to(dtype=torch.float32)
                    assign_parameter_vector(model, point_fp32, layout.parameters)
                    split_results: dict[str, EvaluationResult] = {}
                    for split_name, batches in batches_by_split.items():
                        result = _evaluate_assigned_model(model, batches, device)
                        previous = sample_counts.setdefault(split_name, result.samples)
                        if result.samples != previous:
                            raise ValueError(f"{split_name} sample count changed between grid points")
                        losses[split_name][y_index, x_index] = result.loss
                        accuracies[split_name][y_index, x_index] = result.accuracy
                        split_results[split_name] = result
                    progress_record: dict[str, object] = {
                        "status": "surface_grid_point_completed",
                        "index": completed,
                        "grid_points": total,
                        "x_index": x_index,
                        "y_index": y_index,
                        "x": float(x_value),
                        "y": float(y_value),
                    }
                    progress_record.update({
                        f"{name}_loss": result.loss for name, result in split_results.items()
                    })
                    _notify(progress, **progress_record)
                    completed += 1
                    del point, point_fp32
    finally:
        assign_parameter_vector(model, original, layout.parameters)
    results = {
        name: LossSurface(
            x_values=x_values.copy(), y_values=y_values.copy(),
            loss=losses[name], accuracy=accuracies[name], samples=sample_counts[name],
        )
        for name in split_names
    }
    for name, surface in results.items():
        if not np.isfinite(surface.loss).all() or not np.isfinite(surface.accuracy).all():
            raise ValueError(f"{name} loss surface is nonfinite")
        if np.any(surface.accuracy < 0) or np.any(surface.accuracy > 1):
            raise ValueError(f"{name} accuracy surface is outside [0, 1]")
    return results


def evaluate_loss_surface(
    model: nn.Module,
    reference_vector: torch.Tensor,
    direction_1: torch.Tensor,
    direction_2: torch.Tensor,
    x_values: np.ndarray,
    y_values: np.ndarray,
    dataloader: Iterable[object],
    device: torch.device,
) -> LossSurface:
    """Evaluate one reusable batch sequence with FP32 parameters on a PCA plane."""
    result = _evaluate_surface_pair(
        model, reference_vector, direction_1, direction_2, x_values, y_values,
        {"surface": dataloader}, device,
    )
    return result["surface"]


def common_color_scale(
    train_loss: np.ndarray, validation_loss: np.ndarray, *, intervals: int = 20,
) -> ColorScale:
    if type(intervals) is not int or intervals <= 0:
        raise ValueError("Color-scale intervals must be a positive integer")
    arrays = (train_loss, validation_loss)
    if any(
        not isinstance(array, np.ndarray)
        or array.dtype != np.float64
        or array.ndim != 2
        or not np.isfinite(array).all()
        for array in arrays
    ):
        raise ValueError("Loss grids must be finite two-dimensional float64 arrays")
    raw_minimum = min(float(array.min()) for array in arrays)
    raw_maximum = max(float(array.max()) for array in arrays)
    display_minimum, display_maximum = raw_minimum, raw_maximum
    if raw_minimum == raw_maximum:
        half_width = max(1e-12, abs(raw_minimum) * 1e-12)
        display_minimum -= half_width
        display_maximum += half_width
    levels = np.linspace(display_minimum, display_maximum, intervals + 1, dtype=np.float64)
    return ColorScale(
        raw_minimum, raw_maximum, display_minimum, display_maximum, levels, intervals,
    )


def _cache_batches(
    loaders: Mapping[str, Iterable[object]], generators: LoaderGenerators,
) -> tuple[dict[str, tuple[object, ...]], dict[str, int]]:
    cached: dict[str, tuple[object, ...]] = {}
    byte_counts: dict[str, int] = {}
    with preserve_random_state(generators):
        for name in ("train", "validation"):
            batches = tuple(loaders[name])
            if not batches:
                raise ValueError(f"{name} landscape loader is empty")
            cached[name] = batches
            total = 0
            for batch in batches:
                if not isinstance(batch, (tuple, list)) or len(batch) != 2:
                    raise ValueError(f"{name} landscape loader returned an invalid batch")
                for tensor in batch:
                    if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu":
                        raise ValueError("Cached landscape batches must contain CPU tensors")
                    total += tensor.numel() * tensor.element_size()
            byte_counts[name] = total
    return cached, byte_counts


def _subset_sha256(indices: np.ndarray) -> str:
    if indices.dtype != np.dtype(np.int64) or indices.ndim != 1:
        raise ValueError("Subset indices must be a one-dimensional int64 array")
    return hashlib.sha256(indices.astype("<i8", copy=False).tobytes()).hexdigest()


def _checkpoint_metric_records(metadata: Mapping[str, object]) -> list[dict[str, object]]:
    checkpoints = metadata.get("checkpoints")
    sample_count = metadata.get("sample_count")
    if not isinstance(checkpoints, list) or type(sample_count) is not int or len(checkpoints) != sample_count:
        raise ValueError("Projection checkpoint records differ from sample_count")
    result: list[dict[str, object]] = []
    for index, checkpoint in enumerate(checkpoints):
        if not isinstance(checkpoint, dict) or checkpoint.get("index") != index:
            raise ValueError("Projection checkpoint order is invalid")
        metrics = checkpoint.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError("Projection checkpoint lacks real-model metrics")
        identities = {
            "run_id": checkpoint.get("run_id"),
            "segment_id": checkpoint.get("segment_id"),
            "epoch": checkpoint.get("epoch"),
            "global_step": checkpoint.get("global_step"),
        }
        if any(metrics.get(key) != value for key, value in identities.items()):
            raise ValueError("Projection checkpoint metrics have a different identity")
        for key in ("train_subset_loss", "train_subset_accuracy", "val_loss", "val_accuracy"):
            value = metrics.get(key)
            if type(value) not in (int, float) or not math.isfinite(float(value)):
                raise ValueError(f"Projection checkpoint metric is invalid: {key}")
        for key in ("train_subset_accuracy", "val_accuracy"):
            if not 0 <= float(metrics[key]) <= 1:
                raise ValueError(f"Projection checkpoint accuracy is outside [0, 1]: {key}")
        result.append({"index": index, **identities, "metrics": metrics})
    return result


def _cuda_evaluation_environment() -> tuple[torch.device, dict[str, object]]:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError(
            "Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before importing torch via "
            "scripts/compute_loss_surfaces.py"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Phase 0/1 loss surfaces; no CPU fallback")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    properties = torch.cuda.get_device_properties(device)
    return device, {
        "device": str(device),
        "gpu_name": properties.name,
        "gpu_uuid": str(getattr(properties, "uuid", "unavailable")),
        "total_memory": properties.total_memory,
        "capability": list(torch.cuda.get_device_capability(device)),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "parameter_dtype": "torch.float32",
        "input_dtype": "torch.float32",
        "amp": False,
        "tf32": False,
        "deterministic_algorithms": True,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
    }


def _surface_destination(output_root: Path, projection_id: str, *, create: bool) -> Path:
    parent = output_root / "surfaces"
    resolved_parent = parent.resolve()
    if parent.is_absolute() and parent != resolved_parent:
        raise ValueError("Surface artifact parent resolves through a redirected path")
    destination = (resolved_parent / projection_id).resolve()
    if destination.parent != resolved_parent:
        raise ValueError("Projection identity escapes the surface artifact parent")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite loss surfaces: {destination}")
    if create:
        resolved_parent.mkdir(parents=True, exist_ok=True)
        if resolved_parent.resolve() != resolved_parent:
            raise ValueError("Surface artifact parent changed while it was prepared")
        destination.mkdir(exist_ok=False)
    return destination


def _save_array_new(path: Path, array: np.ndarray) -> None:
    if array.dtype != np.dtype(np.float64) or not np.isfinite(array).all():
        raise ValueError(f"Surface output must be finite float64: {path.name}")
    with path.open("xb") as handle:
        np.save(handle, array, allow_pickle=False)


def _epoch_zero_state(
    artifact: ProjectionArtifact, expected_layout: ParameterLayout,
    reference_state: Mapping[str, torch.Tensor],
) -> None:
    checkpoints = artifact.metadata.get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints or not isinstance(checkpoints[0], dict):
        raise ValueError("Projection has no first checkpoint record")
    record = checkpoints[0]
    if record.get("epoch") != 0:
        raise ValueError("Projection must begin with an epoch-zero checkpoint")
    path_value = record.get("analysis_path")
    run_id, contract = record.get("run_id"), record.get("contract_sha256")
    if not isinstance(path_value, str) or not isinstance(run_id, str) or not isinstance(contract, str):
        raise ValueError("Projection epoch-zero source identity is invalid")
    checkpoint, state = load_analysis_checkpoint(
        Path(path_value).parent,
        expected_run_id=run_id,
        expected_contract_sha256=contract,
    )
    if checkpoint.epoch != 0:
        raise ValueError("Projection epoch-zero source changed")
    validate_model_state(state, expected_layout, reference_state)
    initial_vector = flatten_parameters(reference_state, expected_layout.parameter_names)
    saved_vector = flatten_parameters(state, expected_layout.parameter_names)
    if not torch.equal(initial_vector, saved_vector):
        raise ValueError("Current theta_0 differs from the projection epoch-zero checkpoint")


def compute_loss_surfaces(
    loaded: LoadedConfig, projection_directory: Path, *,
    progress: ProgressCallback | None = None,
) -> SurfaceResult:
    """Compute both fixed-subset backgrounds and publish them as one artifact."""
    config = loaded.config
    projection = projection_directory
    if not projection.is_absolute():
        projection = loaded.project_root / projection
    artifact = load_projection_artifact(
        projection, projections_root=config.paths.output_root / "projections",
    )
    if artifact.metadata.get("effective_config_sha256") != loaded.effective_sha256:
        raise ValueError("Projection effective configuration differs from this request")
    if artifact.metadata.get("config_source_sha256") != loaded.source_sha256:
        raise ValueError("Projection source YAML differs from this request")
    destination = _surface_destination(config.paths.output_root, artifact.projection_id, create=False)
    checkpoint_metrics = _checkpoint_metric_records(artifact.metadata)
    for record in checkpoint_metrics:
        metrics = cast(dict[str, object], record["metrics"])
        if metrics.get("train_subset_samples") != config.landscape.subset_size:
            raise ValueError("Real checkpoint train-subset sample count differs from the surface subset")
        if metrics.get("val_samples") != config.split.val_size:
            raise ValueError("Real checkpoint validation metric is not for the full validation split")

    initial = load_initial_checkpoint(config)
    layout = build_parameter_layout(initial.model)
    if (
        layout.parameter_spec_sha256 != artifact.metadata.get("parameter_spec_sha256")
        or layout.total_numel != artifact.metadata.get("parameter_count")
    ):
        raise ValueError("Projection parameter layout differs from the current theta_0 model")
    reference_state = {
        name: tensor.detach().cpu().clone() for name, tensor in initial.model.state_dict().items()
    }
    _epoch_zero_state(artifact, layout, reference_state)
    runs = artifact.metadata.get("runs")
    if not isinstance(runs, list) or not runs or any(
        not isinstance(run, dict) or run.get("preprocessing") != initial.metadata.get("preprocessing")
        for run in runs
    ):
        raise ValueError("Projection preprocessing differs from the current initial model")

    source = load_cifar10_training(config.paths.dataset_root)
    indices_path = split_path(config)
    indices = load_split_indices(indices_path, source.targets, SplitSpec.from_config(config))
    views = build_dataset_views(source, indices, initial.preprocessing)
    # Pin-memory iteration may initialize CUDA in the parent process.  Establish
    # the device first so the RNG snapshot taken while caching batches has the
    # same CPU/CUDA stream set when it is restored.
    device, numerical = _cuda_evaluation_environment()
    generators = LoaderGenerators.from_seed(config.landscape.subset_seed)
    loaders = {
        "train": make_loader(
            views.train_subset, role="train_subset", batch_size=config.evaluation.batch_size,
            num_workers=config.evaluation.num_workers,
            pin_memory=config.reproducibility.pin_memory,
            generators=generators,
        ),
        "validation": make_loader(
            views.validation_subset, role="validation_subset", batch_size=config.evaluation.batch_size,
            num_workers=config.evaluation.num_workers,
            pin_memory=config.reproducibility.pin_memory,
            generators=generators,
        ),
    }
    cached_batches, cached_bytes = _cache_batches(loaders, generators)
    if any(len(getattr(indices, f"{name}_subset")) != config.landscape.subset_size for name in loaders):
        raise ValueError("Landscape subset count differs from the fixed configuration")

    x_values, y_values = common_grid(
        np.asarray(artifact.coordinates),
        grid_size=config.landscape.grid_size,
        margin_ratio=config.landscape.margin_ratio,
    )
    model = initial.model.to(device=device)
    mean = torch.tensor(np.asarray(artifact.mean), dtype=torch.float64, device="cpu")
    pc1 = torch.tensor(np.asarray(artifact.pc1), dtype=torch.float64, device="cpu")
    pc2 = torch.tensor(np.asarray(artifact.pc2), dtype=torch.float64, device="cpu")
    destination = _surface_destination(config.paths.output_root, artifact.projection_id, create=True)
    surfaces = _evaluate_surface_pair(
        model, mean, pc1, pc2, x_values, y_values, cached_batches, device,
        progress=progress,
    )
    train, validation = surfaces["train"], surfaces["validation"]
    if train.samples != config.landscape.subset_size or validation.samples != config.landscape.subset_size:
        raise ValueError("Evaluated landscape sample count differs from the fixed subset size")
    color = common_color_scale(train.loss, validation.loss, intervals=20)

    arrays = {
        "x_values.npy": x_values,
        "y_values.npy": y_values,
        "train_loss.npy": train.loss,
        "train_accuracy.npy": train.accuracy,
        "validation_loss.npy": validation.loss,
        "validation_accuracy.npy": validation.accuracy,
        "color_levels.npy": color.levels,
    }
    for name, array in arrays.items():
        _save_array_new(destination / name, array)
    write_json(destination / "checkpoint_metrics.json", {
        "schema_version": 1,
        "kind": "actual_checkpoint_metrics",
        "projection_id": artifact.projection_id,
        "train_scope": "fixed train subset with evaluation preprocessing",
        "validation_scope": "full validation split",
        "records": checkpoint_metrics,
    })
    split_metadata_path = indices_path.with_suffix(".json")
    implementation_files = {
        path.relative_to(loaded.project_root).as_posix(): file_hash(path)
        for path in (
            loaded.project_root / "src" / "landscape_exp" / "loss_surface.py",
            loaded.project_root / "src" / "landscape_exp" / "landscape.py",
            loaded.project_root / "src" / "landscape_exp" / "evaluate.py",
            loaded.project_root / "src" / "landscape_exp" / "data.py",
            loaded.project_root / "scripts" / "compute_loss_surfaces.py",
        )
    }
    metadata = {
        "schema_version": 1,
        "kind": "common_loss_surfaces",
        "projection_id": artifact.projection_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_source": str(loaded.source_path),
        "config_source_sha256": loaded.source_sha256,
        "effective_config_sha256": loaded.effective_sha256,
        "implementation_files": implementation_files,
        "runtime": initial.metadata["runtime"],
        "projection": {
            "directory": str(artifact.directory),
            "complete_sha256": artifact.complete_sha256,
            "metadata_sha256": artifact.metadata_sha256,
            "parameter_spec_sha256": layout.parameter_spec_sha256,
            "sample_count": artifact.metadata["sample_count"],
            "explained_variance_ratio": artifact.metadata["explained_variance_ratio"],
        },
        "grid": {
            "shape": [len(y_values), len(x_values)],
            "array_order": "surface[y_index, x_index]",
            "grid_points": len(x_values) * len(y_values),
            "margin_ratio": config.landscape.margin_ratio,
            "range_rule": "trajectory axis min/max plus ten percent of nonzero width; zero width uses center +/- 1e-6",
            "x_min": float(x_values[0]),
            "x_max": float(x_values[-1]),
            "y_min": float(y_values[0]),
            "y_max": float(y_values[-1]),
        },
        "subsets": {
            "split_path": str(indices_path),
            "split_sha256": file_hash(indices_path),
            "split_metadata_path": str(split_metadata_path),
            "split_metadata_sha256": file_hash(split_metadata_path),
            "labels_sha256": indices.labels_sha256,
            "algorithm": "stratified_pcg64_v1",
            "subset_seed": config.landscape.subset_seed,
            "per_class": config.landscape.subset_size // config.model.num_classes,
            "train": {
                "source_split": "train",
                "array_reference": "train_subset in the recorded split NPZ",
                "samples": train.samples,
                "indices_sha256": _subset_sha256(indices.train_subset),
            },
            "validation": {
                "source_split": "validation",
                "array_reference": "validation_subset in the recorded split NPZ",
                "samples": validation.samples,
                "indices_sha256": _subset_sha256(indices.validation_subset),
            },
            "official_test_used": False,
        },
        "evaluation": {
            "loss": "per-image cross entropy sample mean",
            "accuracy": "correct predictions divided by samples",
            "model_mode": "eval",
            "gradient_enabled": False,
            "inference_mode": True,
            "parameter_dtype": "torch.float32",
            "input_dtype": "torch.float32",
            "amp": False,
            "tf32": False,
            "batch_size": config.evaluation.batch_size,
            "num_workers": config.evaluation.num_workers,
            "preprocessing": initial.metadata["preprocessing"],
            "cached_preprocessed_batches": True,
            "cached_bytes": cached_bytes,
            "numerical": numerical,
        },
        "actual_checkpoint_metrics": {
            "path": "checkpoint_metrics.json",
            "source": "real high-dimensional checkpoints copied from verified projection metadata",
            "validation_scope": "full validation split, not the validation background subset",
            "record_count": len(checkpoint_metrics),
        },
        "color_scale": {
            "intervals": color.intervals,
            "raw_minimum": color.raw_minimum,
            "raw_maximum": color.raw_maximum,
            "display_minimum": color.display_minimum,
            "display_maximum": color.display_maximum,
            "constant_loss_display_expansion": color.raw_minimum == color.raw_maximum,
            "shared_by": ["train", "validation"],
        },
        "arrays": {
            "x_values": {"path": "x_values.npy", "shape": [len(x_values)], "dtype": "float64"},
            "y_values": {"path": "y_values.npy", "shape": [len(y_values)], "dtype": "float64"},
            "train_loss": {"path": "train_loss.npy", "shape": list(train.loss.shape), "dtype": "float64"},
            "train_accuracy": {"path": "train_accuracy.npy", "shape": list(train.accuracy.shape), "dtype": "float64"},
            "validation_loss": {"path": "validation_loss.npy", "shape": list(validation.loss.shape), "dtype": "float64"},
            "validation_accuracy": {"path": "validation_accuracy.npy", "shape": list(validation.accuracy.shape), "dtype": "float64"},
            "color_levels": {"path": "color_levels.npy", "shape": [len(color.levels)], "dtype": "float64"},
        },
    }
    write_json(destination / "metadata.json", metadata)
    files = {}
    for name in (*_SURFACE_ARRAY_FILES, "checkpoint_metrics.json", "metadata.json"):
        path = destination / name
        files[name] = {"size_bytes": path.stat().st_size, "sha256": file_hash(path)}
    write_json(destination / "complete.json", {
        "schema_version": 1,
        "kind": "completed_loss_surfaces",
        "projection_id": artifact.projection_id,
        "metadata_sha256": files["metadata.json"]["sha256"],
        "files": files,
    })
    _notify(
        progress, status="loss_surfaces_completed", projection_id=artifact.projection_id,
        directory=str(destination),
    )
    return SurfaceResult(
        projection_id=artifact.projection_id,
        directory=destination,
        grid_points=len(x_values) * len(y_values),
        train_samples=train.samples,
        validation_samples=validation.samples,
        loss_minimum=color.raw_minimum,
        loss_maximum=color.raw_maximum,
    )
