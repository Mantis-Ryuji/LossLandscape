"""Sample-weighted classification evaluation isolated from training state."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .seeds import LoaderGenerators, preserve_random_state


@dataclass(frozen=True)
class EvaluationResult:
    loss: float
    accuracy: float
    samples: int


def synchronize(device: torch.device) -> None:
    """Include queued CUDA work in stage timings, without initializing CUDA on CPU."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def check_model_precision(model: nn.Module, device: torch.device) -> None:
    """Require FP32 parameters on the selected device; never silently convert."""
    for name, parameter in model.named_parameters():
        if parameter.dtype != torch.float32 or parameter.device != device:
            raise ValueError(f"{name}: expected FP32 parameters on {device}")


def prepare_batch(batch: object, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate the classification boundary before transferring a batch."""
    if not isinstance(batch, (tuple, list)) or len(batch) != 2:
        raise ValueError("Expected an (images, labels) batch")
    images, labels = batch
    if not isinstance(images, torch.Tensor) or images.dtype != torch.float32 or images.ndim < 2:
        raise ValueError("Images must be a batched FP32 tensor")
    if not isinstance(labels, torch.Tensor) or labels.dtype != torch.int64 or labels.ndim != 1:
        raise ValueError("Labels must be a one-dimensional int64 tensor")
    if len(labels) == 0 or len(images) != len(labels):
        raise ValueError("Images and labels must have the same nonzero batch size")
    return images.to(device, non_blocking=True), labels.to(device, non_blocking=True)


def classification_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute FP32 cross entropy and reject undefined/nonfinite updates."""
    if not isinstance(logits, torch.Tensor) or logits.ndim != 2 or logits.shape[0] != len(labels):
        raise ValueError("Model output must be [batch, classes] logits")
    if logits.shape[1] < 2 or not logits.is_floating_point():
        raise ValueError("Expected floating logits for at least two classes")
    loss = F.cross_entropy(logits.float(), labels, reduction="mean")
    if not torch.isfinite(loss).item():
        raise ValueError("Nonfinite classification loss; no completed epoch may be saved")
    return loss


@contextmanager
def evaluation_context(
    model: nn.Module, device: torch.device, generators: LoaderGenerators | None = None,
) -> Iterator[None]:
    """Restore RNGs, every module's mode, and numerical flags even on failure."""
    check_model_precision(model, device)
    modes = [(module, module.training) for module in model.modules()]
    flags = (
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.get_float32_matmul_precision(),
    )
    with preserve_random_state(generators):
        try:
            model.eval()
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True)
            with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
                yield
        finally:
            for module, training in modes:
                module.training = training
            torch.backends.cuda.matmul.allow_tf32 = flags[0]
            torch.backends.cudnn.allow_tf32 = flags[1]
            torch.backends.cudnn.benchmark = flags[2]
            torch.backends.cudnn.deterministic = flags[3]
            torch.use_deterministic_algorithms(flags[4], warn_only=flags[5])
            torch.set_float32_matmul_precision(flags[6])


def evaluate(
    model: nn.Module, batches: Iterable[object], *, device: torch.device,
    generators: LoaderGenerators | None = None,
) -> EvaluationResult:
    """Evaluate the actual model, keeping short final batches correctly weighted."""
    loss_sum, correct, samples = 0.0, 0, 0
    with evaluation_context(model, device, generators):
        for batch in batches:
            images, labels = prepare_batch(batch, device)
            logits = model(images)
            loss = classification_loss(logits, labels)
            count = len(labels)
            loss_sum += float(loss.item()) * count
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            samples += count
    if samples == 0:
        raise ValueError("Cannot evaluate an empty dataset")
    return EvaluationResult(loss_sum / samples, correct / samples, samples)
