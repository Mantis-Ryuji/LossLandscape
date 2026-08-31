"""Immutable epoch records and explicit lineage for training resumption."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import torch

from .logging_utils import EpochMetrics, append_completed_metrics, create_metrics_csv


def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def record_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def file_hash(path: Path) -> str:
    """Hash a user-requested artifact incrementally, never loading it all at once."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, object]:
    if path.stat().st_size > 1024 * 1024:
        raise ValueError(f"Metadata is unexpectedly large: {path}")
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"Expected object metadata: {path}")
    canonical_json(value)
    return value


def write_json(path: Path, value: object) -> None:
    """Write a new file only; all serialization must finish before publication."""
    content = canonical_json(value)
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def cpu_tree(value: object) -> object:
    """Detach optimizer/RNG trees into weights_only-compatible CPU containers."""
    if isinstance(value, torch.Tensor):
        result = value.detach().to(device="cpu").clone()
        if (result.is_floating_point() or result.is_complex()) and not torch.isfinite(result).all().item():
            raise ValueError("Nonfinite state cannot be checkpointed")
        return result
    if value is None or type(value) in (str, bool, int, float):
        if type(value) is float:
            canonical_json(value)
        return value
    if isinstance(value, dict):
        if any(type(key) not in (str, int) for key in value):
            raise ValueError("Checkpoint mapping keys must be strings or integers")
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        items = [cpu_tree(item) for item in value]
        return tuple(items) if isinstance(value, tuple) else items
    raise ValueError(f"Unsupported checkpoint object: {type(value).__name__}")


@dataclass(frozen=True)
class CompletedEpoch:
    directory: Path
    epoch: int
    global_step: int
    manifest_sha256: str
    metrics: dict[str, object]
    resume: dict[str, object]


@dataclass
class Segment:
    directory: Path
    run_id: str
    contract: dict[str, object]
    contract_sha256: str
    next_epoch: int

    @property
    def segment_id(self) -> str:
        return self.directory.name


def create_segment(
    run_directory: Path, contract: dict[str, object], parent: CompletedEpoch | None = None,
) -> Segment:
    """Start a fresh segment; a parent is an exact completed epoch, never latest."""
    root = run_directory.resolve()
    if not root.is_dir() or not isinstance(contract.get("run_id"), str):
        raise ValueError("A prepared run directory and run_id are required")
    contract_copy = cast(dict[str, object], json.loads(canonical_json(contract)))
    digest = record_hash(contract_copy)
    if parent is not None:
        if parent.directory.parents[3] != root or parent.resume.get("contract") != contract_copy:
            raise ValueError("Resume parent must belong to this run and identical contract")
        if file_hash(parent.directory / "complete.json") != parent.manifest_sha256:
            raise ValueError("Resume parent changed after validation")
    segments = root / "segments"
    if segments.resolve() != segments:
        raise ValueError("Segment directory must not resolve through a symlink")
    segments.mkdir(exist_ok=True)
    identifier = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid.uuid4().hex
    destination = segments / identifier
    destination.mkdir(exist_ok=False)
    (destination / "epochs").mkdir()
    write_json(destination / "segment.json", {
        "schema_version": 1, "run_id": contract_copy["run_id"], "segment_id": identifier,
        "contract_sha256": digest,
        "parent": None if parent is None else {
            "epoch_directory": str(parent.directory), "epoch": parent.epoch,
            "complete_sha256": parent.manifest_sha256,
        },
    })
    create_metrics_csv(destination / "metrics.csv")
    return Segment(destination, str(contract_copy["run_id"]), contract_copy, digest,
                   0 if parent is None else parent.epoch + 1)


def _write_torch(path: Path, value: object) -> None:
    with path.open("xb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())


def save_epoch(
    segment: Segment, metrics: EpochMetrics, *, model_state: dict[str, torch.Tensor],
    optimizer_state: object, scheduler_state: dict[str, object], rng_state: object,
    loader_state: object, epoch_started: float, checkpoint_started: float,
) -> tuple[Path, EpochMetrics]:
    """Publish all epoch files together, then append the derived CSV row.

    A same-directory hard link publishes complete.json atomically without replace.
    The filesystem must support hard links (such as NTFS). Failure leaves partial
    files for inspection; they are never repaired or reused by a later writer.
    """
    metrics.record()
    if metrics.epoch != segment.next_epoch or metrics.segment_id != segment.segment_id or metrics.run_id != segment.run_id:
        raise ValueError("Epoch order or segment identity differs from this writer")
    if record_hash(segment.contract) != segment.contract_sha256:
        raise ValueError("Run contract changed while the segment was active")
    destination = segment.directory / "epochs" / f"epoch_{metrics.epoch:04d}"
    destination.mkdir(exist_ok=False)
    identity = {"schema_version": 1, "epoch": metrics.epoch, "global_step": metrics.global_step}
    _write_torch(destination / "analysis.pt", {
        **identity, "kind": "analysis", "contract_sha256": segment.contract_sha256,
        "model_state": model_state,
    })
    _write_torch(destination / "resume.pt", {
        **identity, "kind": "resume", "contract": segment.contract,
        "model_state": model_state, "optimizer_state": optimizer_state,
        "scheduler_state": scheduler_state, "rng_state": rng_state,
        "loader_state": loader_state, "scaler_state": None,
    })
    files = {
        name: {"size_bytes": (destination / name).stat().st_size, "sha256": file_hash(destination / name)}
        for name in ("analysis.pt", "resume.pt")
    }
    # These times include CPU snapshots, serialization, fsync and weight hashes.
    # Final small JSON/manifest publication and CSV append are excluded explicitly.
    now = time.perf_counter()
    finished = replace(metrics, checkpoint_seconds=now - checkpoint_started, epoch_seconds=now - epoch_started)
    write_json(destination / "metrics.json", finished.record())
    write_json(destination / "metadata.json", {
        **identity, "contract_sha256": segment.contract_sha256,
        "timing_scope": "through weight hashes; excludes final JSON/manifest/CSV publication",
        "initial_checkpoint": segment.contract.get("initial_checkpoint"),
        "analysis_parameter_dtype": "torch.float32", "scaler_used": False,
    })
    for name in ("metrics.json", "metadata.json"):
        files[name] = {"size_bytes": (destination / name).stat().st_size, "sha256": file_hash(destination / name)}
    temporary = destination / "complete.json.tmp"
    write_json(temporary, {
        **identity, "kind": "completed_epoch", "run_id": segment.run_id,
        "segment_id": segment.segment_id, "contract_sha256": segment.contract_sha256,
        "files": files,
    })
    os.link(temporary, destination / "complete.json")
    temporary.unlink()
    segment.next_epoch += 1
    append_completed_metrics(segment.directory / "metrics.csv", finished)
    return destination, finished


def load_completed_epoch(directory: Path, contract: dict[str, object]) -> CompletedEpoch:
    """Verify every recorded file before loading trusted resume tensors on CPU."""
    directory = directory.resolve()
    manifest_path = directory / "complete.json"
    if not manifest_path.is_file():
        raise ValueError(f"Epoch has no completion manifest: {directory}")
    manifest = read_json(manifest_path)
    if type(manifest.get("schema_version")) is not int or manifest.get("schema_version") != 1 or manifest.get("kind") != "completed_epoch":
        raise ValueError("Unsupported completed epoch schema")
    epoch, step = manifest.get("epoch"), manifest.get("global_step")
    if type(epoch) is not int or epoch < 0 or type(step) is not int or step < 0:
        raise ValueError("Invalid completed epoch/step")
    if directory.name != f"epoch_{epoch:04d}" or directory.parent.name != "epochs":
        raise ValueError("Epoch directory and recorded identity disagree")
    if manifest.get("segment_id") != directory.parent.parent.name or manifest.get("run_id") != contract.get("run_id"):
        raise ValueError("Completed epoch belongs to a different run/segment")
    if manifest.get("contract_sha256") != record_hash(contract):
        raise ValueError("Run configuration, runtime, source or data identity changed")
    records = manifest.get("files")
    expected = {"analysis.pt", "resume.pt", "metrics.json", "metadata.json"}
    if not isinstance(records, dict) or set(records) != expected:
        raise ValueError("Completed epoch file manifest differs from the schema")
    for name, record in records.items():
        path = directory / name
        if path.resolve() != path or not path.is_file() or not isinstance(record, dict):
            raise ValueError(f"Missing or redirected epoch file: {name}")
        if path.stat().st_size != record.get("size_bytes") or file_hash(path) != record.get("sha256"):
            raise ValueError(f"Epoch file size/hash mismatch: {name}")
    metrics = read_json(directory / "metrics.json")
    metadata = read_json(directory / "metadata.json")
    try:
        EpochMetrics(**metrics).record()
    except TypeError as error:
        raise ValueError("Epoch metrics differ from the declared schema") from error
    if metrics.get("epoch") != epoch or metrics.get("global_step") != step:
        raise ValueError("Metric identity differs from its completion manifest")
    if metrics.get("run_id") != contract["run_id"] or metrics.get("segment_id") != manifest["segment_id"]:
        raise ValueError("Metric run/segment identity differs")
    if metadata.get("contract_sha256") != record_hash(contract) or metadata.get("epoch") != epoch or metadata.get("global_step") != step:
        raise ValueError("Epoch metadata contract differs")
    with (directory / "resume.pt").open("rb") as handle:
        resume = torch.load(handle, map_location="cpu", weights_only=True)
    resume_keys = {"schema_version", "kind", "epoch", "global_step", "contract", "model_state",
                   "optimizer_state", "scheduler_state", "rng_state", "loader_state", "scaler_state"}
    if not isinstance(resume, dict) or set(resume) != resume_keys or resume.get("schema_version") != 1 or resume.get("kind") != "resume":
        raise ValueError("Unsupported resume checkpoint")
    if resume.get("epoch") != epoch or resume.get("global_step") != step or resume.get("contract") != contract:
        raise ValueError("Resume payload identity differs from its completion manifest")
    return CompletedEpoch(directory, epoch, step, file_hash(manifest_path), metrics, resume)


def completed_lineage(
    segment_directory: Path, contract: dict[str, object], *, through_epoch: int | None = None,
) -> list[Path]:
    """Resolve a selected branch, cutting each ancestor at its explicit parent epoch."""
    seen: set[Path] = set()
    run_root = segment_directory.resolve().parents[1]

    def visit(directory: Path, limit: int | None) -> list[Path]:
        directory = directory.resolve()
        if directory in seen or directory.parent != run_root / "segments":
            raise ValueError("Invalid or cyclic segment lineage")
        seen.add(directory)
        info = read_json(directory / "segment.json")
        if info.get("contract_sha256") != record_hash(contract) or info.get("segment_id") != directory.name:
            raise ValueError("Segment lineage contract/identity differs")
        result: list[Path] = []
        parent = info.get("parent")
        if parent is not None:
            if not isinstance(parent, dict) or not isinstance(parent.get("epoch_directory"), str):
                raise ValueError("Invalid segment parent")
            ancestor = load_completed_epoch(Path(parent["epoch_directory"]), contract)
            if ancestor.epoch != parent.get("epoch") or ancestor.manifest_sha256 != parent.get("complete_sha256"):
                raise ValueError("Segment parent changed")
            result = visit(ancestor.directory.parents[1], ancestor.epoch)
        for path in sorted((directory / "epochs").iterdir()):
            match = re.fullmatch(r"epoch_(\d{4})", path.name)
            if not match or not path.is_dir() or (limit is not None and int(match[1]) > limit):
                continue
            if not (path / "complete.json").is_file():
                continue
            completed = load_completed_epoch(path, contract)
            if completed.epoch != len(result):
                raise ValueError("Completed epoch lineage has a gap or duplicate")
            result.append(path)
        if limit is not None and len(result) != limit + 1:
            raise ValueError("Parent epoch is absent from its recorded lineage")
        return result

    return visit(segment_directory, through_epoch)
