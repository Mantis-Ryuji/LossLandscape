"""Checkpoint trajectory extraction and exact blocked common PCA."""

from __future__ import annotations

import math
import re
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import numpy as np
import torch

from .checkpoints import file_hash, read_json, write_json
from .config import LoadedConfig, config_dict
from .landscape import ParameterLayout, build_parameter_layout, flatten_model_state
from .models import load_initial_checkpoint


ProgressCallback = Callable[[dict[str, object]], None]
_ARRAY_FILES = (
    "mean.npy", "pc1.npy", "pc2.npy", "coordinates.npy", "residuals.npy",
    "eigenvalues.npy", "explained_variance_ratio.npy",
)


@dataclass(frozen=True)
class AnalysisCheckpoint:
    directory: Path
    run_id: str
    segment_id: str
    epoch: int
    global_step: int
    contract_sha256: str
    complete_sha256: str
    analysis_sha256: str
    analysis_size_bytes: int
    metrics: dict[str, object]

    def record(self, index: int) -> dict[str, object]:
        return {
            "index": index,
            "run_id": self.run_id,
            "segment_id": self.segment_id,
            "epoch": self.epoch,
            "global_step": self.global_step,
            "contract_sha256": self.contract_sha256,
            "complete_sha256": self.complete_sha256,
            "analysis_path": str(self.directory / "analysis.pt"),
            "analysis_sha256": self.analysis_sha256,
            "analysis_size_bytes": self.analysis_size_bytes,
            "metrics": self.metrics,
        }


@dataclass(frozen=True)
class PCAResult:
    sample_count: int
    parameter_count: int
    effective_rank: int
    rank_threshold: float
    eigenvalues: tuple[float, ...]
    explained_variance_ratio: tuple[float, float]
    orthonormality_max_error: float
    reconstruction_identity_max_error: float


@dataclass(frozen=True)
class ProjectionResult:
    projection_id: str
    directory: Path
    work_directory: Path
    sample_count: int
    parameter_count: int
    explained_variance_ratio: tuple[float, float]
    effective_rank: int


def _notify(callback: ProgressCallback | None, **values: object) -> None:
    if callback is not None:
        callback(values)


def _notify_block(
    callback: ProgressCallback | None, status: str, start: int, stop: int,
    total: int, block_parameters: int,
) -> None:
    block_index = start // block_parameters
    if block_index % 128 == 0 or stop == total:
        _notify(
            callback, status=status, parameter_start=start,
            parameter_stop=stop, parameter_count=total,
        )


def _small_record(path: Path) -> dict[str, object]:
    value = read_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"Expected object metadata: {path}")
    return value


def _file_record(manifest: Mapping[str, object], name: str) -> dict[str, object]:
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != {
        "analysis.pt", "resume.pt", "metrics.json", "metadata.json",
    }:
        raise ValueError("Completed epoch file manifest differs from schema")
    record = files.get(name)
    if not isinstance(record, dict):
        raise ValueError(f"Missing file record: {name}")
    size, digest = record.get("size_bytes"), record.get("sha256")
    if type(size) is not int or size <= 0 or not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"Invalid file record: {name}")
    return cast(dict[str, object], record)


def load_analysis_checkpoint(
    directory: Path, *, expected_run_id: str, expected_contract_sha256: str,
) -> tuple[AnalysisCheckpoint, dict[str, torch.Tensor]]:
    """Verify one completed epoch fully before loading its analysis state."""
    directory = directory.resolve()
    manifest_path = directory / "complete.json"
    if not manifest_path.is_file() or manifest_path.resolve() != manifest_path:
        raise ValueError(f"Epoch has no direct completion manifest: {directory}")
    manifest = _small_record(manifest_path)
    if manifest.get("schema_version") != 1 or manifest.get("kind") != "completed_epoch":
        raise ValueError("Unsupported completed epoch schema")
    epoch, step = manifest.get("epoch"), manifest.get("global_step")
    if type(epoch) is not int or epoch < 0 or type(step) is not int or step < 0:
        raise ValueError("Invalid completed epoch identity")
    if directory.name != f"epoch_{epoch:04d}" or directory.parent.name != "epochs":
        raise ValueError("Epoch path and completed identity disagree")
    if manifest.get("run_id") != expected_run_id or manifest.get("segment_id") != directory.parents[1].name:
        raise ValueError("Completed epoch belongs to a different run or segment")
    if manifest.get("contract_sha256") != expected_contract_sha256:
        raise ValueError("Completed epoch contract differs from its selected lineage")

    # A completion marker covers all four files. Verify each recorded artifact,
    # even though projection deserializes only analysis.pt.
    records: dict[str, dict[str, object]] = {}
    for name in ("analysis.pt", "resume.pt", "metrics.json", "metadata.json"):
        record = _file_record(manifest, name)
        path = directory / name
        if path.resolve() != path or not path.is_file():
            raise ValueError(f"Missing or redirected completed epoch file: {name}")
        if path.stat().st_size != record["size_bytes"] or file_hash(path) != record["sha256"]:
            raise ValueError(f"Completed epoch file size/hash mismatch: {name}")
        records[name] = record

    metrics = _small_record(directory / "metrics.json")
    metadata = _small_record(directory / "metadata.json")
    identity = (epoch, step, expected_contract_sha256)
    if (metrics.get("epoch"), metrics.get("global_step"), metrics.get("run_id")) != (
        epoch, step, expected_run_id,
    ):
        raise ValueError("Metrics identity differs from completed epoch")
    if metrics.get("segment_id") != directory.parents[1].name:
        raise ValueError("Metrics segment differs from completed epoch")
    if (metadata.get("epoch"), metadata.get("global_step"), metadata.get("contract_sha256")) != identity:
        raise ValueError("Analysis metadata differs from completed epoch")
    if metadata.get("analysis_parameter_dtype") != "torch.float32":
        raise ValueError("Analysis metadata does not declare FP32 parameters")

    with (directory / "analysis.pt").open("rb") as handle:
        payload = torch.load(handle, map_location="cpu", weights_only=True)
    expected_keys = {"schema_version", "kind", "epoch", "global_step", "contract_sha256", "model_state"}
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise ValueError("Unsupported analysis checkpoint payload")
    if (payload.get("schema_version"), payload.get("kind")) != (1, "analysis"):
        raise ValueError("Unsupported analysis checkpoint identity")
    if (payload.get("epoch"), payload.get("global_step"), payload.get("contract_sha256")) != identity:
        raise ValueError("Analysis checkpoint identity differs from completed epoch")
    state = payload.get("model_state")
    if not isinstance(state, dict) or any(not isinstance(name, str) for name in state):
        raise ValueError("Analysis model state must be a string-keyed mapping")
    checkpoint = AnalysisCheckpoint(
        directory=directory,
        run_id=expected_run_id,
        segment_id=directory.parents[1].name,
        epoch=epoch,
        global_step=step,
        contract_sha256=expected_contract_sha256,
        complete_sha256=file_hash(manifest_path),
        analysis_sha256=cast(str, records["analysis.pt"]["sha256"]),
        analysis_size_bytes=cast(int, records["analysis.pt"]["size_bytes"]),
        metrics=metrics,
    )
    return checkpoint, cast(dict[str, torch.Tensor], state)


def resolve_analysis_lineage(segment_directory: Path, runs_root: Path) -> tuple[str, str, list[Path]]:
    """Resolve one explicit segment branch without guessing a latest run."""
    declared_segment = segment_directory
    segment_directory, runs_root = declared_segment.resolve(), runs_root.resolve()
    if declared_segment.is_absolute() and declared_segment != segment_directory:
        raise ValueError("Selected segment resolves through a redirected path")
    if not segment_directory.is_dir() or segment_directory.parent.name != "segments":
        raise ValueError(f"Expected a segment directory: {segment_directory}")
    if not segment_directory.is_relative_to(runs_root):
        raise ValueError("Selected segment must stay inside output_root/runs")
    seen: set[Path] = set()
    selected_run_id: str | None = None
    selected_contract: str | None = None

    def visit(directory: Path, limit: int | None) -> list[Path]:
        nonlocal selected_run_id, selected_contract
        directory = directory.resolve()
        if directory in seen or directory.parent.name != "segments":
            raise ValueError("Invalid or cyclic segment lineage")
        seen.add(directory)
        info = _small_record(directory / "segment.json")
        run_id, contract = info.get("run_id"), info.get("contract_sha256")
        if info.get("schema_version") != 1 or info.get("segment_id") != directory.name:
            raise ValueError("Unsupported segment identity")
        if not isinstance(run_id, str) or not run_id or not isinstance(contract, str) or len(contract) != 64:
            raise ValueError("Invalid segment run/contract identity")
        expected_root = (runs_root / Path(run_id)).resolve()
        if directory.parent.parent != expected_root or not expected_root.is_relative_to(runs_root):
            raise ValueError("Segment path and run_id disagree")
        if selected_run_id is None:
            selected_run_id, selected_contract = run_id, contract
        elif run_id != selected_run_id or contract != selected_contract:
            raise ValueError("Segment lineage changes run or contract")

        result: list[Path] = []
        parent = info.get("parent")
        if parent is not None:
            if not isinstance(parent, dict) or not isinstance(parent.get("epoch_directory"), str):
                raise ValueError("Invalid segment parent")
            declared_parent_epoch = Path(parent["epoch_directory"])
            parent_epoch = declared_parent_epoch.resolve()
            if declared_parent_epoch.is_absolute() and declared_parent_epoch != parent_epoch:
                raise ValueError("Segment parent resolves through a redirected path")
            if not parent_epoch.is_relative_to(expected_root) or parent_epoch.parent.name != "epochs":
                raise ValueError("Segment parent escapes its run")
            parent_number = parent.get("epoch")
            if type(parent_number) is not int or parent_number < 0 or parent_epoch.name != f"epoch_{parent_number:04d}":
                raise ValueError("Invalid segment parent epoch")
            complete = parent_epoch / "complete.json"
            if not complete.is_file() or file_hash(complete) != parent.get("complete_sha256"):
                raise ValueError("Segment parent completion marker changed")
            result = visit(parent_epoch.parents[1], parent_number)

        epochs = directory / "epochs"
        if not epochs.is_dir() or epochs.resolve() != epochs:
            raise ValueError("Segment has no direct epochs directory")
        for path in sorted(epochs.iterdir()):
            match = re.fullmatch(r"epoch_(\d{4})", path.name)
            if not match or not path.is_dir() or not (path / "complete.json").is_file():
                continue
            epoch = int(match[1])
            if limit is not None and epoch > limit:
                continue
            if epoch != len(result):
                raise ValueError("Completed analysis lineage has a gap or duplicate")
            resolved_path = path.resolve()
            if resolved_path != path:
                raise ValueError("Completed epoch resolves through a redirected path")
            result.append(resolved_path)
        if limit is not None and len(result) != limit + 1:
            raise ValueError("Parent epoch is absent from its recorded lineage")
        return result

    paths = visit(segment_directory, None)
    if not paths or selected_run_id is None or selected_contract is None:
        raise ValueError("Selected segment has no completed analysis checkpoints")
    return selected_run_id, selected_contract, paths


def _new_memmap(path: Path, *, dtype: str, shape: tuple[int, ...]) -> np.memmap:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite array: {path}")
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def _save_array_new(path: Path, value: np.ndarray) -> None:
    with path.open("xb") as handle:
        np.save(handle, value, allow_pickle=False)


def compute_blocked_pca(
    weights: np.ndarray, destination: Path, *, block_parameters: int,
    progress: ProgressCallback | None = None,
) -> PCAResult:
    """Compute centered PCA through a FP64 Gram matrix in parameter blocks."""
    if not isinstance(weights, np.ndarray) or weights.ndim != 2 or weights.dtype != np.float32:
        raise ValueError("Weight matrix must be a two-dimensional FP32 NumPy array")
    samples, parameters = weights.shape
    if samples < 3 or parameters < 2:
        raise ValueError("Two-dimensional PCA requires at least three samples and two parameters")
    if type(block_parameters) is not int or block_parameters <= 0:
        raise ValueError("block_parameters must be a positive integer")
    destination = destination.resolve()
    if not destination.is_dir():
        raise ValueError("PCA destination must be a prepared directory")
    for name in _ARRAY_FILES:
        if (destination / name).exists():
            raise FileExistsError(f"Refusing to overwrite PCA output: {destination / name}")

    mean = _new_memmap(destination / "mean.npy", dtype="float64", shape=(parameters,))
    gram = np.zeros((samples, samples), dtype=np.float64)
    for start in range(0, parameters, block_parameters):
        stop = min(start + block_parameters, parameters)
        block = np.asarray(weights[:, start:stop], dtype=np.float64)
        block_mean = block.mean(axis=0, dtype=np.float64)
        mean[start:stop] = block_mean
        block -= block_mean
        gram += block @ block.T
        _notify_block(progress, "pca_gram_block", start, stop, parameters, block_parameters)
    mean.flush()
    gram = (gram + gram.T) * 0.5
    if not np.isfinite(gram).all():
        raise ValueError("Centered Gram matrix is nonfinite")
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    largest = float(eigenvalues[0])
    trace = float(np.trace(gram))
    if not math.isfinite(largest) or not math.isfinite(trace) or largest <= 0 or trace <= 0:
        raise ValueError("Weight trajectory has no finite positive variance")
    rank_threshold = 1e-10 * largest
    effective_rank = int(np.count_nonzero(eigenvalues > rank_threshold))
    if effective_rank < 2:
        raise ValueError("Centered weight trajectory has effective rank below two")
    selected_values = eigenvalues[:2]
    selected_vectors = eigenvectors[:, :2]
    explained = selected_values / trace

    pc1 = _new_memmap(destination / "pc1.npy", dtype="float64", shape=(parameters,))
    pc2 = _new_memmap(destination / "pc2.npy", dtype="float64", shape=(parameters,))
    maxima = np.full(2, -1.0, dtype=np.float64)
    signs = np.ones(2, dtype=np.float64)
    scale = np.sqrt(selected_values)
    for start in range(0, parameters, block_parameters):
        stop = min(start + block_parameters, parameters)
        block = np.asarray(weights[:, start:stop], dtype=np.float64)
        block -= np.asarray(mean[start:stop])
        components = (block.T @ selected_vectors) / scale
        pc1[start:stop], pc2[start:stop] = components[:, 0], components[:, 1]
        for component in range(2):
            index = int(np.argmax(np.abs(components[:, component])))
            magnitude = float(abs(components[index, component]))
            if magnitude > maxima[component]:
                maxima[component] = magnitude
                signs[component] = 1.0 if components[index, component] >= 0 else -1.0
        _notify_block(progress, "pca_basis_block", start, stop, parameters, block_parameters)
    if signs[0] < 0:
        for start in range(0, parameters, block_parameters):
            pc1[start:min(start + block_parameters, parameters)] *= -1
    if signs[1] < 0:
        for start in range(0, parameters, block_parameters):
            pc2[start:min(start + block_parameters, parameters)] *= -1
    pc1.flush()
    pc2.flush()

    coordinates = np.zeros((samples, 2), dtype=np.float64)
    centered_norm_squared = np.zeros(samples, dtype=np.float64)
    orthonormality = np.zeros((2, 2), dtype=np.float64)
    for start in range(0, parameters, block_parameters):
        stop = min(start + block_parameters, parameters)
        block = np.asarray(weights[:, start:stop], dtype=np.float64)
        block -= np.asarray(mean[start:stop])
        basis = np.column_stack((np.asarray(pc1[start:stop]), np.asarray(pc2[start:stop])))
        coordinates += block @ basis
        centered_norm_squared += np.einsum("ij,ij->i", block, block)
        orthonormality += basis.T @ basis
    orthonormality_error = float(np.max(np.abs(orthonormality - np.eye(2))))
    if not math.isfinite(orthonormality_error) or orthonormality_error > 1e-8:
        raise ValueError("Computed PCA directions are not orthonormal")

    residual_squared = np.zeros(samples, dtype=np.float64)
    for start in range(0, parameters, block_parameters):
        stop = min(start + block_parameters, parameters)
        block = np.asarray(weights[:, start:stop], dtype=np.float64)
        block -= np.asarray(mean[start:stop])
        basis = np.column_stack((np.asarray(pc1[start:stop]), np.asarray(pc2[start:stop])))
        delta = block - coordinates @ basis.T
        residual_squared += np.einsum("ij,ij->i", delta, delta)
        _notify_block(progress, "pca_residual_block", start, stop, parameters, block_parameters)
    identity_residual = centered_norm_squared - np.einsum("ij,ij->i", coordinates, coordinates)
    consistency = float(np.max(np.abs(residual_squared - identity_residual)))
    consistency_scale = max(1.0, float(np.max(centered_norm_squared)))
    if not math.isfinite(consistency) or consistency > 1e-7 * consistency_scale:
        raise ValueError("Blocked PCA reconstruction identity check failed")
    residuals = np.sqrt(np.maximum(residual_squared, 0.0))
    if not np.isfinite(coordinates).all() or not np.isfinite(residuals).all():
        raise ValueError("PCA coordinates or residuals are nonfinite")

    _save_array_new(destination / "coordinates.npy", coordinates)
    _save_array_new(destination / "residuals.npy", residuals)
    _save_array_new(destination / "eigenvalues.npy", eigenvalues.astype(np.float64, copy=False))
    _save_array_new(destination / "explained_variance_ratio.npy", explained.astype(np.float64, copy=False))
    del mean, pc1, pc2
    return PCAResult(
        sample_count=samples,
        parameter_count=parameters,
        effective_rank=effective_rank,
        rank_threshold=rank_threshold,
        eigenvalues=tuple(float(value) for value in eigenvalues),
        explained_variance_ratio=(float(explained[0]), float(explained[1])),
        orthonormality_max_error=orthonormality_error,
        reconstruction_identity_max_error=consistency,
    )


def _compatibility_record(config: Mapping[str, object]) -> dict[str, object]:
    try:
        experiment = dict(cast(dict[str, object], config["experiment"]))
        training = dict(cast(dict[str, object], config["training"]))
        paths = cast(dict[str, object], config["paths"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Run configuration lacks the projection compatibility schema") from error
    experiment.pop("name", None)
    experiment.pop("seed", None)
    training.pop("batch_size", None)
    return {
        "schema_version": config.get("schema_version"),
        "experiment": experiment,
        "paths": {"dataset_root": paths.get("dataset_root"), "init_checkpoint": paths.get("init_checkpoint")},
        "model": config.get("model"),
        "training_except_effective_batch": training,
        "augmentation": config.get("augmentation"),
        "split": config.get("split"),
        "reproducibility": config.get("reproducibility"),
        "evaluation": config.get("evaluation"),
        "checkpoint": config.get("checkpoint"),
        "projection": config.get("projection"),
        "landscape": config.get("landscape"),
        "logging": config.get("logging"),
        "phase1": config.get("phase1"),
    }


def _run_context(
    segment: Path, run_id: str, contract_sha256: str,
    expected_compatibility: dict[str, object],
) -> dict[str, object]:
    run_root = segment.parent.parent
    run_config = _small_record(run_root / "config.json")
    if _compatibility_record(run_config) != expected_compatibility:
        raise ValueError(f"Selected run is incompatible with the projection config: {run_id}")
    root_environment = _small_record(run_root / "environment.json")
    segment_environment = _small_record(segment / "environment.json")
    if root_environment != segment_environment:
        raise ValueError(f"Run and segment environments differ: {run_id}")
    if root_environment.get("contract_sha256") != contract_sha256:
        raise ValueError(f"Run environment and segment contract differ: {run_id}")
    for key in ("runtime", "numerical", "preprocessing", "sources"):
        if not isinstance(root_environment.get(key), dict):
            raise ValueError(f"Run environment lacks {key}: {run_id}")
    if not isinstance(root_environment.get("batching"), dict):
        raise ValueError(f"Run environment lacks batching: {run_id}")
    sources = cast(dict[str, object], root_environment["sources"])
    return {
        "run_id": run_id,
        "segment_id": segment.name,
        "segment_path": str(segment),
        "contract_sha256": contract_sha256,
        "effective_batch_size": cast(dict[str, object], root_environment["batching"]).get("effective_batch_size"),
        "runtime": root_environment["runtime"],
        "numerical": root_environment["numerical"],
        "preprocessing": root_environment["preprocessing"],
        "source_sha256": sources.get("sha256"),
        "git_commit": sources.get("git_commit"),
        "git_dirty": sources.get("git_dirty"),
    }


def _projection_id(experiment_name: str, comparison_scope: str) -> str:
    if re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", comparison_scope) is None:
        raise ValueError("comparison_scope must use 1-64 lowercase letters, digits, '_' or '-'")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{experiment_name}_{comparison_scope}_{timestamp}_{uuid.uuid4().hex}"


def _prepared_output_directory(parent: Path, identifier: str) -> Path:
    declared_parent = parent
    parent = declared_parent.resolve()
    if declared_parent.is_absolute() and declared_parent != parent:
        raise ValueError(f"Artifact parent resolves through a redirected path: {declared_parent}")
    parent.mkdir(parents=True, exist_ok=True)
    if parent.resolve() != parent:
        raise ValueError(f"Artifact parent changed while it was prepared: {parent}")
    destination = (parent / identifier).resolve()
    if destination.parent != parent:
        raise ValueError("Artifact identifier escapes its parent")
    destination.mkdir(exist_ok=False)
    return destination


def compute_projection(
    loaded: LoadedConfig, segment_directories: Sequence[Path], *, comparison_scope: str,
    progress: ProgressCallback | None = None,
) -> ProjectionResult:
    """Extract selected trajectories and publish one immutable common PCA."""
    if not segment_directories:
        raise ValueError("At least one explicit segment is required")
    config = loaded.config
    runs_root = (config.paths.output_root / "runs").resolve()
    expected_compatibility = _compatibility_record(config_dict(config))
    lineages: list[tuple[str, str, list[Path]]] = []
    run_contexts: list[dict[str, object]] = []
    seen_runs: set[str] = set()
    common_epochs: tuple[int, ...] | None = None
    for source in segment_directories:
        segment = source if source.is_absolute() else loaded.project_root / source
        run_id, contract, paths = resolve_analysis_lineage(segment, runs_root)
        if run_id in seen_runs:
            raise ValueError(f"A run may appear only once in a projection: {run_id}")
        epochs = tuple(int(path.name.removeprefix("epoch_")) for path in paths)
        if common_epochs is None:
            common_epochs = epochs
        elif epochs != common_epochs:
            raise ValueError("All comparison trajectories must use the same completed epochs")
        seen_runs.add(run_id)
        lineages.append((run_id, contract, paths))
        run_contexts.append(_run_context(paths[-1].parents[1], run_id, contract, expected_compatibility))
    if common_epochs is None or len(common_epochs) < 3:
        raise ValueError("Projection requires at least three common epoch points")
    first_environment = run_contexts[0]
    for context in run_contexts[1:]:
        for key in ("runtime", "numerical", "preprocessing"):
            if context[key] != first_environment[key]:
                raise ValueError(f"Selected runs differ in {key}")

    initial = load_initial_checkpoint(config)
    initial_sha256 = initial.checkpoint_sha256
    layout: ParameterLayout = build_parameter_layout(initial.model)
    if layout.parameter_spec_sha256 != initial.metadata.get("parameter_spec_sha256"):
        raise ValueError("Runtime parameter specification differs from theta_0 metadata")
    reference_state = {
        name: tensor.detach().cpu().clone() for name, tensor in initial.model.state_dict().items()
    }
    reference_vector = flatten_model_state(reference_state, layout, reference_state)
    del initial

    identifier = _projection_id(config.experiment.name, comparison_scope)
    destination = _prepared_output_directory(config.paths.output_root / "projections", identifier)
    work = _prepared_output_directory(config.paths.scratch_root, identifier)
    checkpoint_count = sum(len(paths) for _, _, paths in lineages)
    matrix_path = work / "weights.npy"
    matrix = _new_memmap(
        matrix_path, dtype="float32", shape=(checkpoint_count, layout.total_numel),
    )
    checkpoint_records: list[AnalysisCheckpoint] = []
    row = 0
    for run_id, contract, paths in lineages:
        for path in paths:
            checkpoint, state = load_analysis_checkpoint(
                path, expected_run_id=run_id, expected_contract_sha256=contract,
            )
            if checkpoint.epoch == 0:
                metadata = _small_record(path / "metadata.json")
                declared_initial = metadata.get("initial_checkpoint")
                if not isinstance(declared_initial, dict) or declared_initial.get("sha256") != initial_sha256:
                    raise ValueError(f"Epoch zero does not declare the shared theta_0: {run_id}")
            vector = flatten_model_state(state, layout, reference_state)
            if checkpoint.epoch == 0 and not torch.equal(vector, reference_vector):
                raise ValueError(f"Epoch zero parameters differ from shared theta_0: {run_id}")
            matrix[row, :] = vector.numpy()
            checkpoint_records.append(checkpoint)
            row += 1
            _notify(
                progress, status="checkpoint_extracted", index=row - 1,
                checkpoint_count=checkpoint_count, run_id=run_id,
                epoch=checkpoint.epoch, global_step=checkpoint.global_step,
            )
            del state, vector
    matrix.flush()
    del reference_state, reference_vector
    pca = compute_blocked_pca(
        matrix, destination, block_parameters=config.projection.block_parameters, progress=progress,
    )
    del matrix

    implementation_files = {
        path.relative_to(loaded.project_root).as_posix(): file_hash(path)
        for path in (
            loaded.project_root / "src" / "landscape_exp" / "landscape.py",
            loaded.project_root / "src" / "landscape_exp" / "projection.py",
            loaded.project_root / "scripts" / "compute_projection.py",
        )
    }
    metadata = {
        "schema_version": 1,
        "kind": "common_pca_projection",
        "projection_id": identifier,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "comparison_scope": comparison_scope,
        "config_source": str(loaded.source_path),
        "config_source_sha256": loaded.source_sha256,
        "effective_config_sha256": loaded.effective_sha256,
        "implementation_files": implementation_files,
        "runtime": {"numpy": str(np.__version__), "torch": str(torch.__version__)},
        "solver": config.projection.solver,
        "compute_dtype": "float64",
        "weight_matrix_dtype": "float32",
        "block_parameters": config.projection.block_parameters,
        "centering": "common mean across every ordered run/epoch checkpoint",
        "rank_threshold_rule": "1e-10 * largest_eigenvalue",
        "axis_sign_rule": "largest absolute parameter coefficient positive; earliest parameter breaks ties",
        "parameter_spec": [item.record() for item in layout.parameters],
        "parameter_spec_sha256": layout.parameter_spec_sha256,
        "parameter_count": layout.total_numel,
        "buffer_names": list(layout.buffer_names),
        "buffers": "excluded from PCA; every checkpoint verified equal to theta_0",
        "run_order": [context["run_id"] for context in run_contexts],
        "runs": run_contexts,
        "common_epochs": list(common_epochs),
        "checkpoints": [item.record(index) for index, item in enumerate(checkpoint_records)],
        "sample_count": pca.sample_count,
        "eigenvalues": list(pca.eigenvalues),
        "explained_variance_ratio": list(pca.explained_variance_ratio),
        "effective_rank": pca.effective_rank,
        "rank_threshold": pca.rank_threshold,
        "orthonormality_max_error": pca.orthonormality_max_error,
        "reconstruction_identity_max_error": pca.reconstruction_identity_max_error,
        "arrays": {
            "mean": {"path": "mean.npy", "shape": [layout.total_numel], "dtype": "float64"},
            "pc1": {"path": "pc1.npy", "shape": [layout.total_numel], "dtype": "float64"},
            "pc2": {"path": "pc2.npy", "shape": [layout.total_numel], "dtype": "float64"},
            "coordinates": {"path": "coordinates.npy", "shape": [pca.sample_count, 2], "dtype": "float64"},
            "residuals": {"path": "residuals.npy", "shape": [pca.sample_count], "dtype": "float64"},
            "eigenvalues": {"path": "eigenvalues.npy", "shape": [pca.sample_count], "dtype": "float64"},
            "explained_variance_ratio": {"path": "explained_variance_ratio.npy", "shape": [2], "dtype": "float64"},
        },
        "work_matrix": {
            "path": str(matrix_path), "shape": [pca.sample_count, layout.total_numel],
            "dtype": "float32", "size_bytes": matrix_path.stat().st_size,
            "retained": True,
        },
    }
    write_json(destination / "metadata.json", metadata)
    files = {}
    for name in (*_ARRAY_FILES, "metadata.json"):
        path = destination / name
        files[name] = {"size_bytes": path.stat().st_size, "sha256": file_hash(path)}
    write_json(destination / "complete.json", {
        "schema_version": 1,
        "kind": "completed_projection",
        "projection_id": identifier,
        "metadata_sha256": files["metadata.json"]["sha256"],
        "files": files,
    })
    _notify(progress, status="projection_completed", projection_id=identifier, directory=str(destination))
    return ProjectionResult(
        projection_id=identifier,
        directory=destination,
        work_directory=work,
        sample_count=pca.sample_count,
        parameter_count=pca.parameter_count,
        explained_variance_ratio=pca.explained_variance_ratio,
        effective_rank=pca.effective_rank,
    )
