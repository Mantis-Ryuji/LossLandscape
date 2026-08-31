"""Independent random streams and serializable epoch-boundary RNG snapshots."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, fields
from typing import TypedDict, cast

import numpy as np
import torch


PythonRandomState = tuple[int, tuple[int, ...], float | None]


class NumpyRandomState(TypedDict):
    algorithm: str
    keys: list[int]
    position: int
    has_gauss: int
    cached_gaussian: float


class RandomState(TypedDict):
    schema_version: int
    python: PythonRandomState
    numpy: NumpyRandomState
    torch_cpu: torch.Tensor
    torch_cuda: list[torch.Tensor] | None


def derive_seed(seed: int, namespace: str) -> int:
    """Derive a stable seed without Python's process-dependent hash function."""
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    if not isinstance(namespace, str) or not namespace or "|" in namespace:
        raise ValueError("namespace must be nonempty and must not contain '|'")
    payload = f"losslandscape-v1|{seed}|{namespace}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 2**63


def seed_global(seed: int) -> None:
    """Seed CPU RNGs and already initialized CUDA devices, without starting CUDA.

    GPU training must initialize its device before calling this function. Using
    a separate CPU generator avoids queuing an untracked lazy CUDA seed in a
    CPU-only context such as initial-head creation or a unit test.
    """
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    random.seed(seed)
    np.random.seed(seed % 2**32)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    torch.set_rng_state(generator.get_state())
    if torch.cuda.is_initialized():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    """Initialize worker-local Python/NumPy RNGs from the DataLoader torch seed."""
    if type(worker_id) is not int or worker_id < 0:
        raise ValueError("worker_id must be a nonnegative integer")
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def capture_random_state() -> RandomState:
    """Capture primitive containers and tensors, suitable for weights_only files."""
    algorithm, keys, position, has_gauss, cached = np.random.get_state()
    return {
        "schema_version": 1,
        "python": cast(PythonRandomState, random.getstate()),
        "numpy": {
            "algorithm": str(algorithm),
            "keys": [int(key) for key in keys],
            "position": int(position),
            "has_gauss": int(has_gauss),
            "cached_gaussian": float(cached),
        },
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": (
            [state.clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_initialized() else None
        ),
    }


def _validate_cpu_tensor(state: torch.Tensor) -> None:
    """Validate CPU generator state without changing an active RNG stream."""
    if not isinstance(state, torch.Tensor) or state.device.type != "cpu":
        raise ValueError("RNG state must be a CPU tensor")
    if state.dtype != torch.uint8 or state.ndim != 1:
        raise ValueError("RNG state must be a one-dimensional uint8 tensor")
    torch.Generator(device="cpu").set_state(state)


def restore_random_state(state: RandomState) -> None:
    """Restore saved streams after constructing the model and before iteration."""
    if state["schema_version"] != 1:
        raise ValueError("Unsupported RNG state schema")
    cuda_states = state["torch_cuda"]
    if (cuda_states is not None) != torch.cuda.is_initialized():
        raise ValueError("CUDA initialization changed since capture; initialize devices before restoring")
    if cuda_states is not None:
        if len(cuda_states) != torch.cuda.device_count():
            raise ValueError("CUDA device count differs from the saved RNG state")
        for tensor in cuda_states:
            if tensor.device.type != "cpu" or tensor.dtype != torch.uint8 or tensor.ndim != 1:
                raise ValueError("Invalid CUDA RNG state tensor")
    _validate_cpu_tensor(state["torch_cpu"])
    random.Random().setstate(state["python"])
    numpy_state = state["numpy"]
    if numpy_state["algorithm"] != "MT19937":
        raise ValueError("Unsupported NumPy global RNG algorithm")
    keys = numpy_state["keys"]
    if len(keys) != 624 or any(type(key) is not int or not 0 <= key < 2**32 for key in keys):
        raise ValueError("Invalid NumPy MT19937 state keys")
    restored_numpy = (
        numpy_state["algorithm"], np.asarray(keys, dtype=np.uint32),
        numpy_state["position"], numpy_state["has_gauss"], numpy_state["cached_gaussian"],
    )
    # Validate on a local RNG before mutating process-global streams.
    np.random.RandomState(0).set_state(restored_numpy)
    random.setstate(state["python"])
    np.random.set_state(restored_numpy)
    torch.set_rng_state(state["torch_cpu"])
    if cuda_states is not None:
        torch.cuda.set_rng_state_all(cuda_states)


@dataclass(frozen=True)
class LoaderGenerators:
    """Keep shuffling independent of worker creation and all evaluation loaders."""

    train_order: torch.Generator
    train_workers: torch.Generator
    validation_workers: torch.Generator
    train_subset_workers: torch.Generator
    validation_subset_workers: torch.Generator

    @classmethod
    def from_seed(cls, seed: int) -> LoaderGenerators:
        """Use the same sampling stream for every batch size within a seed."""
        generators = {
            field.name: torch.Generator(device="cpu").manual_seed(derive_seed(seed, field.name))
            for field in fields(cls)
        }
        return cls(**generators)

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Copy state so subsequent iterator creation cannot change the snapshot."""
        return {
            field.name: getattr(self, field.name).get_state().clone()
            for field in fields(self)
        }

    def load_state_dict(self, state: Mapping[str, torch.Tensor]) -> None:
        """Require a complete matching set of streams before restoring any."""
        if set(state) != {field.name for field in fields(self)}:
            raise ValueError("Loader RNG state keys do not match the five declared streams")
        for tensor in state.values():
            _validate_cpu_tensor(tensor)
        for name, tensor in state.items():
            getattr(self, name).set_state(tensor)


@contextmanager
def preserve_random_state(generators: LoaderGenerators | None = None) -> Iterator[None]:
    """Isolate evaluation, including failure paths, from subsequent training."""
    state = capture_random_state()
    loader_state = None if generators is None else generators.state_dict()
    try:
        yield
    finally:
        restore_random_state(state)
        if generators is not None and loader_state is not None:
            generators.load_state_dict(loader_state)
