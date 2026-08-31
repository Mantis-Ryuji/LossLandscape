"""Create one shared CPU initial model, then restore it without downloading.

Phase 0/1 always use scratch initialization and never fetch pretrained weights.
Importing this module never creates a model. Existing artifacts are never replaced.
Load only trusted checkpoints produced by this project, using weights_only=True.
"""

from __future__ import annotations

import hashlib
import json
import platform
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, TypedDict, cast

import numpy as np
import PIL
import timm
import torch
import torchvision
from torch import nn

from .config import ExperimentConfig, LoadedConfig, config_dict, validate_config
from .data import Preprocessing, build_preprocessing
from .seeds import preserve_random_state, seed_global


class TensorSpec(TypedDict):
    name: str
    shape: list[int]
    numel: int
    dtype: str


@dataclass(frozen=True)
class InitialModel:
    """An exact initial model in CPU FP32/eval mode, with its provenance."""

    model: nn.Module
    preprocessing: Preprocessing
    checkpoint_path: Path
    checkpoint_sha256: str
    metadata: dict[str, object]


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _plain_mapping(value: object, label: str) -> dict[str, object]:
    """Normalize tuples to lists and reject non-JSON provenance values."""
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a string-keyed mapping")
    try:
        return cast(dict[str, object], json.loads(_json_bytes(value)))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must contain finite JSON values") from error


def _runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(), "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__), "timm": str(timm.__version__),
        "numpy": str(np.__version__), "pillow": str(PIL.__version__),
    }


def _tensor_specs(values: Iterable[tuple[str, torch.Tensor]]) -> list[TensorSpec]:
    return [
        {"name": name, "shape": list(tensor.shape), "numel": tensor.numel(),
         "dtype": str(tensor.dtype)}
        for name, tensor in values
    ]


def _model_layout(model: nn.Module, num_classes: int) -> dict[str, object]:
    """Describe ordered parameters separately from persistent state buffers."""
    get_classifier = getattr(model, "get_classifier", None)
    if not callable(get_classifier):
        raise ValueError("Model does not expose a classifier")
    head = get_classifier()
    if not isinstance(head, nn.Linear) or head.out_features != num_classes:
        raise ValueError("Expected a Linear classifier with the configured class count")
    parameters = dict(model.named_parameters())
    if not parameters:
        raise ValueError("Model has no parameters")
    for name, parameter in parameters.items():
        if parameter.device.type != "cpu" or parameter.dtype != torch.float32 or not parameter.requires_grad:
            raise ValueError(f"{name}: initial parameters must be trainable CPU FP32 tensors")
    state = model.state_dict()
    for name, tensor in state.items():
        if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu":
            raise ValueError(f"{name}: initial state must contain only CPU tensors")
        if (tensor.is_floating_point() or tensor.is_complex()) and not torch.isfinite(tensor).all().item():
            raise ValueError(f"{name}: nonfinite initial model state")
    parameter_spec = _tensor_specs(parameters.items())
    return {
        "parameter_spec": parameter_spec,
        "parameter_spec_sha256": hashlib.sha256(_json_bytes(parameter_spec)).hexdigest(),
        "buffer_spec": _tensor_specs((name, tensor) for name, tensor in state.items() if name not in parameters),
        "state_spec": _tensor_specs(state.items()),
        "parameter_count": sum(parameter.numel() for parameter in parameters.values()),
    }


def _construct_model(config: ExperimentConfig) -> tuple[nn.Module, Preprocessing, dict[str, object]]:
    """Initialize the entire model with timm defaults; never load external weights."""
    if torch.get_default_dtype() != torch.float32:
        raise ValueError("Initial model construction requires the default torch dtype to be FP32")
    with preserve_random_state(), torch.device("cpu"):
        seed_global(config.model.init_seed)
        # A single construction initializes backbone and head together. Do not
        # reset the head afterward: that would bypass ConvNeXt's model-wide init.
        model = timm.create_model(
            config.model.name, pretrained=False, num_classes=config.model.num_classes,
        )
        model.requires_grad_(True)
        model.eval()
        _model_layout(model, config.model.num_classes)
        data_config = _plain_mapping(timm.data.resolve_model_data_config(model), "model data config")
        preprocessing = build_preprocessing(data_config)
    return model, preprocessing, data_config


def _artifact_paths(config: ExperimentConfig) -> tuple[Path, Path]:
    """Recheck immutable destinations before model construction or file I/O."""
    validate_config(config)
    path = config.paths.init_checkpoint
    if not path.is_absolute() or path.suffix != ".pt":
        raise ValueError("Initial checkpoint must have an absolute .pt path")
    if path.resolve() != path or path.is_relative_to(config.paths.dataset_root.resolve()):
        raise ValueError("Initial checkpoint resolves into raw data or through a changed symlink")
    metadata_path = path.with_suffix(".json")
    if metadata_path.resolve() != metadata_path:
        raise ValueError("Initial metadata must not resolve through a symlink")
    return path, metadata_path


def _hash_stream(handle: BinaryIO) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _identity() -> dict[str, object]:
    return {"schema_version": 2, "kind": "initial_model", "epoch": 0, "global_step": 0}


def _initialization(config: ExperimentConfig) -> dict[str, object]:
    return {
        "mode": config.model.initialization, "seed": config.model.init_seed,
        "method": "timm.create_model(pretrained=False,num_classes=10); model default initialization",
        "pretrained": False,
    }


def create_initial_checkpoint(loaded: LoadedConfig) -> InitialModel:
    """Save a shared scratch backbone and head; no downloads, dataset or GPU."""
    config = loaded.config
    path, metadata_path = _artifact_paths(config)
    if path.exists() or metadata_path.exists():
        raise FileExistsError(f"Refusing to overwrite initial model records: {path}")
    model, preprocessing, data_config = _construct_model(config)
    metadata: dict[str, object] = {
        **_identity(), "model": asdict(config.model),
        "initialization": _initialization(config),
        "data_config": data_config, "preprocessing": preprocessing.metadata,
        **_model_layout(model, config.model.num_classes),
        # timm may attach a default pretrained_cfg even with pretrained=False.
        # Its download URL is not evidence that any pretrained weight was used.
        "pretrained_reference": None,
        "runtime": _runtime_versions(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "creation_config": config_dict(config),
        "source_sha256": loaded.source_sha256,
        "effective_sha256": loaded.effective_sha256,
    }
    _json_bytes(metadata)
    state = {name: tensor.detach().clone().contiguous() for name, tensor in model.state_dict().items()}
    _artifact_paths(config)
    # Do not create even an empty artifact directory if model loading fails.
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or metadata_path.exists():
        raise FileExistsError(f"Initial records appeared during model construction: {path}")
    with path.open("xb") as handle:
        torch.save({"metadata": metadata, "model_state": state}, handle)
    with path.open("rb") as handle:
        checkpoint_sha256 = _hash_stream(handle)
    complete = {
        **metadata, "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_size_bytes": path.stat().st_size,
    }
    # The JSON is the final marker. A partial marker is rejected, never repaired.
    with metadata_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(complete, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    return InitialModel(model, preprocessing, path, checkpoint_sha256, complete)


def _read_metadata(path: Path, config: ExperimentConfig) -> dict[str, object]:
    if path.stat().st_size > 1024 * 1024:
        raise ValueError("Initial metadata exceeds the expected small JSON size")
    with path.open(encoding="utf-8") as handle:
        metadata = _plain_mapping(json.load(handle), "initial metadata")
    if _json_bytes({key: metadata.get(key) for key in _identity()}) != _json_bytes(_identity()):
        raise ValueError("Unsupported initial model identity/schema")
    if metadata.get("model") != asdict(config.model):
        raise ValueError("Initial model configuration differs from the requested model")
    if metadata.get("initialization") != _initialization(config) or metadata.get("pretrained_reference") is not None:
        raise ValueError("Initial model initialization differs from the scratch contract")
    if metadata.get("runtime") != _runtime_versions():
        raise ValueError("Runtime versions differ from initial model creation; do not silently reuse it")
    expected_hash = metadata.get("checkpoint_sha256")
    expected_size = metadata.get("checkpoint_size_bytes")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError("Missing or invalid initial checkpoint hash")
    if type(expected_size) is not int or expected_size <= 0:
        raise ValueError("Missing or invalid initial checkpoint size")
    return metadata


def _validate_saved_state(value: object, model: nn.Module) -> dict[str, torch.Tensor]:
    """Reject dtype/shape/key drift before load_state_dict can cast tensors."""
    expected = model.state_dict()
    if not isinstance(value, dict) or list(value) != list(expected):
        raise ValueError("Saved model state keys/order differ from the model")
    for name, current in expected.items():
        tensor = value[name]
        if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu":
            raise ValueError(f"{name}: saved state must be a CPU tensor")
        if tensor.dtype != current.dtype or tensor.shape != current.shape:
            raise ValueError(f"{name}: saved state dtype/shape differs from the model")
        if (tensor.is_floating_point() or tensor.is_complex()) and not torch.isfinite(tensor).all().item():
            raise ValueError(f"{name}: saved state is nonfinite")
    return cast(dict[str, torch.Tensor], value)


def load_initial_checkpoint(config: ExperimentConfig) -> InitialModel:
    """Restore theta_0 on CPU without downloading or changing caller RNG streams.

    Batch size and run seed need not match creation_config: all runs deliberately
    share this artifact. Model, runtime, preprocessing and tensor layout must match.
    """
    path, metadata_path = _artifact_paths(config)
    if not path.is_file() or not metadata_path.is_file():
        raise ValueError(f"Incomplete initial records: both {path.name} and {metadata_path.name} are required")
    metadata = _read_metadata(metadata_path, config)
    with path.open("rb") as handle:
        handle.seek(0, 2)
        if handle.tell() != metadata["checkpoint_size_bytes"]:
            raise ValueError("Initial checkpoint size does not match completed metadata")
        handle.seek(0)
        checkpoint_sha256 = _hash_stream(handle)
        if checkpoint_sha256 != metadata["checkpoint_sha256"]:
            raise ValueError("Initial checkpoint hash does not match completed metadata")
        handle.seek(0)
        payload = torch.load(handle, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {"metadata", "model_state"}:
        raise ValueError("Unsupported initial checkpoint payload")
    embedded = {key: value for key, value in metadata.items()
                if key not in ("checkpoint_sha256", "checkpoint_size_bytes")}
    payload_metadata = _plain_mapping(payload["metadata"], "embedded initial metadata")
    if _json_bytes(payload_metadata) != _json_bytes(embedded):
        raise ValueError("Initial checkpoint metadata differs from its completion record")
    model, preprocessing, data_config = _construct_model(config)
    if metadata.get("data_config") != data_config or metadata.get("preprocessing") != preprocessing.metadata:
        raise ValueError("Model preprocessing differs from initial model creation")
    for key, value in _model_layout(model, config.model.num_classes).items():
        if metadata.get(key) != value:
            raise ValueError(f"Initial model layout differs: {key}")
    state = _validate_saved_state(payload["model_state"], model)
    model.load_state_dict(state, strict=True)
    return InitialModel(model, preprocessing, path, checkpoint_sha256, metadata)


def prepare_initial_checkpoint(loaded: LoadedConfig) -> InitialModel:
    """Reuse complete initial records; never regenerate missing/corrupt records."""
    path, metadata_path = _artifact_paths(loaded.config)
    if path.exists() or metadata_path.exists():
        return load_initial_checkpoint(loaded.config)
    return create_initial_checkpoint(loaded)
