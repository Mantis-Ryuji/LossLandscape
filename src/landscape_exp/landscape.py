"""Strict parameter-vector utilities for loss-landscape analysis.

Parameters follow ``theta_0.named_parameters()`` order. Persistent state that
is not a parameter is treated as a shared buffer: it is excluded from vectors
and must remain exactly equal to theta_0 before a checkpoint can be analyzed.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ParameterSpec:
    """The immutable identity of one flattened model parameter."""

    name: str
    shape: tuple[int, ...]
    numel: int
    dtype: str

    def record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "numel": self.numel,
            "dtype": self.dtype,
        }


@dataclass(frozen=True)
class ParameterLayout:
    """Ordered parameters plus the complete parameter/buffer state boundary."""

    parameters: tuple[ParameterSpec, ...]
    state_names: tuple[str, ...]
    buffer_names: tuple[str, ...]
    parameter_spec_sha256: str

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.parameters)

    @property
    def total_numel(self) -> int:
        return sum(item.numel for item in self.parameters)


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def parameter_spec_sha256(parameter_spec: Sequence[ParameterSpec]) -> str:
    """Hash the same ordered JSON records stored with the initial model."""
    specs = _validated_spec(parameter_spec)
    return hashlib.sha256(_canonical_json([item.record() for item in specs])).hexdigest()


def _validated_spec(parameter_spec: Sequence[ParameterSpec]) -> tuple[ParameterSpec, ...]:
    if isinstance(parameter_spec, (str, bytes)):
        raise ValueError("Parameter specification must be a sequence of ParameterSpec values")
    specs = tuple(parameter_spec)
    if not specs or any(not isinstance(item, ParameterSpec) for item in specs):
        raise ValueError("Parameter specification must contain ParameterSpec values")
    names: set[str] = set()
    for item in specs:
        if not item.name or item.name in names:
            raise ValueError("Parameter specification names must be nonempty and unique")
        if any(type(size) is not int or size < 0 for size in item.shape):
            raise ValueError(f"Invalid parameter shape in specification: {item.name}")
        expected_numel = 1
        for size in item.shape:
            expected_numel *= size
        if type(item.numel) is not int or item.numel <= 0 or item.numel != expected_numel:
            raise ValueError(f"Invalid parameter numel in specification: {item.name}")
        if item.dtype != "torch.float32":
            raise ValueError(f"Analysis parameters must be FP32: {item.name}")
        names.add(item.name)
    return specs


def build_parameter_layout(model: nn.Module) -> ParameterLayout:
    """Capture theta_0 parameter order and distinguish persistent buffers."""
    parameters = tuple(model.named_parameters())
    if not parameters:
        raise ValueError("Model has no parameters")
    state = model.state_dict()
    parameter_names = tuple(name for name, _ in parameters)
    if len(set(parameter_names)) != len(parameter_names):
        raise ValueError("Model parameter names must be unique")
    if any(name not in state for name in parameter_names):
        raise ValueError("Every named parameter must appear in model state")
    specs = tuple(
        ParameterSpec(name, tuple(parameter.shape), parameter.numel(), str(parameter.dtype))
        for name, parameter in parameters
    )
    specs = _validated_spec(specs)
    state_names = tuple(state)
    buffer_names = tuple(name for name in state_names if name not in set(parameter_names))
    if set(state_names) != set(parameter_names) | set(buffer_names):
        raise ValueError("Model state cannot be separated into parameters and buffers")
    return ParameterLayout(
        parameters=specs,
        state_names=state_names,
        buffer_names=buffer_names,
        parameter_spec_sha256=parameter_spec_sha256(specs),
    )


def flatten_parameters(
    state_dict: Mapping[str, torch.Tensor], parameter_names: Sequence[str],
) -> torch.Tensor:
    """Flatten named CPU FP32 parameters in the explicitly supplied order."""
    if not isinstance(state_dict, Mapping):
        raise ValueError("Model state must be a mapping")
    if isinstance(parameter_names, (str, bytes)):
        raise ValueError("Parameter names must be a sequence")
    names = tuple(parameter_names)
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("Parameter names must be nonempty strings")
    if len(set(names)) != len(names):
        raise ValueError("Parameter names must be unique")
    pieces: list[torch.Tensor] = []
    for name in names:
        tensor = state_dict.get(name)
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Missing parameter tensor: {name}")
        if tensor.device.type != "cpu" or tensor.dtype != torch.float32:
            raise ValueError(f"Analysis parameter must be a CPU FP32 tensor: {name}")
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"Analysis parameter is nonfinite: {name}")
        pieces.append(tensor.detach().reshape(-1))
    return torch.cat(pieces).contiguous()


def validate_model_state(
    state_dict: Mapping[str, torch.Tensor], layout: ParameterLayout,
    reference_state: Mapping[str, torch.Tensor],
) -> None:
    """Validate an analysis state and enforce theta_0's shared buffers."""
    if not isinstance(state_dict, Mapping) or not isinstance(reference_state, Mapping):
        raise ValueError("Model and reference states must be mappings")
    specs = _validated_spec(layout.parameters)
    if layout.parameter_spec_sha256 != parameter_spec_sha256(specs):
        raise ValueError("Parameter specification hash differs from its records")
    if tuple(state_dict) != layout.state_names or tuple(reference_state) != layout.state_names:
        raise ValueError("Model state keys/order differ from theta_0 layout")
    if layout.parameter_names != tuple(item.name for item in specs):
        raise ValueError("Parameter order differs from theta_0 layout")
    expected_buffers = tuple(name for name in layout.state_names if name not in set(layout.parameter_names))
    if layout.buffer_names != expected_buffers:
        raise ValueError("Buffer names/order differ from theta_0 layout")

    by_name = {item.name: item for item in specs}
    for name in layout.state_names:
        tensor, reference = state_dict[name], reference_state[name]
        if not isinstance(tensor, torch.Tensor) or not isinstance(reference, torch.Tensor):
            raise ValueError(f"Model state must contain tensors: {name}")
        if tensor.device.type != "cpu" or reference.device.type != "cpu":
            raise ValueError(f"Analysis state must contain CPU tensors: {name}")
        if tensor.shape != reference.shape or tensor.dtype != reference.dtype:
            raise ValueError(f"State shape/dtype differs from theta_0: {name}")
        if tensor.is_floating_point() or tensor.is_complex():
            if not torch.isfinite(tensor).all().item():
                raise ValueError(f"Analysis state is nonfinite: {name}")
            if not torch.isfinite(reference).all().item():
                raise ValueError(f"Theta_0 state is nonfinite: {name}")
        spec = by_name.get(name)
        if spec is not None:
            if tuple(tensor.shape) != spec.shape or tensor.numel() != spec.numel or str(tensor.dtype) != spec.dtype:
                raise ValueError(f"Parameter differs from theta_0 specification: {name}")
        elif not torch.equal(tensor, reference):
            raise ValueError(f"Buffer changed from theta_0: {name}")


def flatten_model_state(
    state_dict: Mapping[str, torch.Tensor], layout: ParameterLayout,
    reference_state: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Validate a checkpoint against theta_0, then flatten only parameters."""
    validate_model_state(state_dict, layout, reference_state)
    vector = flatten_parameters(state_dict, layout.parameter_names)
    if vector.numel() != layout.total_numel:
        raise ValueError("Flattened vector length differs from theta_0 layout")
    return vector


def assign_parameter_vector(
    model: nn.Module, vector: torch.Tensor, parameter_spec: Sequence[ParameterSpec],
) -> None:
    """Assign one FP32 vector without modifying buffers or parameter metadata."""
    specs = _validated_spec(parameter_spec)
    if not isinstance(vector, torch.Tensor) or vector.ndim != 1:
        raise ValueError("Parameter vector must be a one-dimensional tensor")
    if vector.dtype != torch.float32:
        raise ValueError("Parameter vector must be FP32")
    if not torch.isfinite(vector).all().item():
        raise ValueError("Parameter vector is nonfinite")
    parameters = tuple(model.named_parameters())
    if tuple(name for name, _ in parameters) != tuple(item.name for item in specs):
        raise ValueError("Model parameter names/order differ from the specification")
    if vector.numel() != sum(item.numel for item in specs):
        raise ValueError("Parameter vector length differs from the specification")

    # Complete structural validation before the first in-place write.
    for (name, parameter), spec in zip(parameters, specs):
        if tuple(parameter.shape) != spec.shape or parameter.numel() != spec.numel:
            raise ValueError(f"Model parameter shape differs from the specification: {name}")
        if str(parameter.dtype) != spec.dtype:
            raise ValueError(f"Model parameter dtype differs from the specification: {name}")

    offset = 0
    with torch.no_grad():
        for (_, parameter), spec in zip(parameters, specs):
            values = vector[offset:offset + spec.numel].reshape(spec.shape)
            parameter.copy_(values.to(device=parameter.device))
            offset += spec.numel
