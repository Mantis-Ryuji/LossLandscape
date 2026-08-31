"""Epoch-based AdamW training, real-checkpoint evaluation and exact resumption.

The low-level loop accepts small CPU fixtures. The production entry point uses
only the validated Phase 0/1 configuration, existing CIFAR-10/splits/theta_0,
and CUDA bf16 training; it never downloads data or changes batch size on failure.
"""

from __future__ import annotations

import math
import os
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import torch
from torch import nn
from torch.utils.data import DataLoader

from .checkpoints import (
    CompletedEpoch, Segment, canonical_json, completed_lineage, cpu_tree, create_segment,
    file_hash, load_completed_epoch, read_json, record_hash, save_epoch, write_json,
)
from .config import LoadedConfig, config_dict, prepare_run
from .data import (
    SplitSpec, build_dataset_views, load_cifar10_training, load_split_indices, make_loader, split_path,
)
from .evaluate import check_model_precision, classification_loss, evaluate, prepare_batch, synchronize
from .logging_utils import EpochMetrics
from .models import load_initial_checkpoint
from .seeds import (
    LoaderGenerators, RandomState, capture_random_state, preserve_random_state,
    restore_random_state, seed_global,
)


@dataclass
class EpochSchedule:
    epochs: int
    warmup_epochs: int
    steps_per_epoch: int
    base_lr: float
    completed_updates: int = 0
    scheduler: str = "cosine"

    def __post_init__(self) -> None:
        if any(type(value) is not int for value in (self.epochs, self.warmup_epochs, self.steps_per_epoch, self.completed_updates)):
            raise ValueError("Schedule counts must be integers")
        if self.epochs <= 0 or self.steps_per_epoch <= 0 or not 0 <= self.warmup_epochs < self.epochs:
            raise ValueError("Invalid total/warmup epochs or steps per epoch")
        if type(self.base_lr) not in (float, int) or not math.isfinite(self.base_lr) or self.base_lr <= 0:
            raise ValueError("base_lr must be finite and positive")
        if self.scheduler not in ("constant", "cosine"):
            raise ValueError("Unknown LR scheduler")
        if self.scheduler == "constant" and self.warmup_epochs != 0:
            raise ValueError("Constant LR requires warmup_epochs=0")
        if not 0 <= self.completed_updates <= self.total_updates:
            raise ValueError("Completed update count is outside the schedule")

    @property
    def total_updates(self) -> int:
        return self.epochs * self.steps_per_epoch

    def rate(self, update: int) -> float:
        """Return the LR used before a one-based optimizer update."""
        if type(update) is not int or not 1 <= update <= self.total_updates:
            raise ValueError("Optimizer update must be in [1, total_updates]")
        if self.scheduler == "constant":
            return self.base_lr
        warmup = self.warmup_epochs * self.steps_per_epoch
        if update <= warmup:
            return self.base_lr if update == warmup else self.base_lr * update / warmup
        return self.base_lr * (1 + math.cos(math.pi * (update - warmup) / (self.total_updates - warmup))) / 2

    @property
    def last_lr(self) -> float | None:
        return None if self.completed_updates == 0 else self.rate(self.completed_updates)

    @property
    def next_lr(self) -> float | None:
        return None if self.completed_updates == self.total_updates else self.rate(self.completed_updates + 1)

    def apply_next(self, optimizer: torch.optim.Optimizer) -> None:
        rate = self.next_lr
        if rate is None:
            raise ValueError("The full LR schedule is already complete")
        for group in optimizer.param_groups:
            group["lr"] = rate

    def state_dict(self) -> dict[str, object]:
        return {"schema_version": 2, **asdict(self)}

    def validate_state(self, state: object, global_step: int) -> None:
        expected = {**self.state_dict(), "completed_updates": global_step}
        if not isinstance(state, dict) or canonical_json(state) != canonical_json(expected):
            raise ValueError("Resume LR schedule differs from the full training schedule")
        if not 0 <= global_step <= self.total_updates:
            raise ValueError("Resume step is outside the full LR schedule")


@dataclass(frozen=True)
class TrainingLoaders:
    train: DataLoader
    train_subset: DataLoader
    validation: DataLoader


@dataclass(frozen=True)
class OnlineResult:
    loss: float
    accuracy: float
    gradient_norm: float
    samples: int
    seconds: float


def make_optimizer(model: nn.Module, schedule: EpochSchedule, weight_decay: float) -> torch.optim.AdamW:
    """Use one uniform group; preserve AdamW defaults explicitly in its state."""
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    if any(not parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("Full fine-tuning requires all parameters to be trainable")
    return torch.optim.AdamW(model.parameters(), lr=schedule.base_lr, weight_decay=weight_decay,
                             betas=(0.9, 0.999), eps=1e-8)


def gradient_l2(model: nn.Module) -> float:
    """Measure the unmodified global gradient norm before optimizer.step()."""
    norms = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            raise ValueError(f"Missing gradient for full-finetuning parameter: {name}")
        norms.append(torch.linalg.vector_norm(parameter.grad.detach(), dtype=torch.float64))
    if not norms:
        raise ValueError("Model has no gradients")
    result = float(torch.linalg.vector_norm(torch.stack(norms)).item())
    if not math.isfinite(result):
        raise ValueError("Nonfinite gradient norm; optimizer update aborted")
    return result


def _batch_layout(loader: DataLoader, accumulation_steps: int) -> tuple[int, int, int]:
    """Validate fixed microbatches covering one full, non-dropping epoch."""
    if type(accumulation_steps) is not int or accumulation_steps <= 0:
        raise ValueError("accumulation_steps must be a positive integer")
    microbatch = loader.batch_size
    if type(microbatch) is not int or microbatch <= 0 or loader.drop_last:
        raise ValueError("Training requires a fixed microbatch size and drop_last=False")
    samples = len(loader.dataset)
    if samples <= 0 or len(loader) != math.ceil(samples / microbatch):
        raise ValueError("Training loader must cover the declared full dataset")
    effective_batch = microbatch * accumulation_steps
    return microbatch, effective_batch, math.ceil(samples / effective_batch)


def train_one_epoch(
    model: nn.Module, loader: DataLoader, optimizer: torch.optim.AdamW, schedule: EpochSchedule,
    *, device: torch.device, use_bf16: bool, accumulation_steps: int = 1,
) -> OnlineResult:
    """Average each effective batch before one AdamW update, including the tail."""
    check_model_precision(model, device)
    if use_bf16 and device.type != "cuda":
        raise ValueError("Production bf16 training requires CUDA; CPU fixtures use FP32")
    microbatch, effective_batch, expected_updates = _batch_layout(loader, accumulation_steps)
    if expected_updates != schedule.steps_per_epoch or schedule.completed_updates % schedule.steps_per_epoch:
        raise ValueError("Training must start at an epoch boundary with the original loader size")
    epoch_samples = len(loader.dataset)
    synchronize(device)
    started = time.perf_counter()
    model.train()
    loss_sum, correct, samples, norm_sum, updates = 0.0, 0, 0, 0.0, 0
    group_samples = 0
    for microstep, batch in enumerate(loader):
        images, labels = prepare_batch(batch, device)
        count = len(labels)
        if count != min(microbatch, epoch_samples - samples):
            raise ValueError("Training microbatch size differs from the declared full epoch")
        if microstep % accumulation_steps == 0:
            optimizer.zero_grad(set_to_none=True)
            schedule.apply_next(optimizer)
            group_samples = min(effective_batch, epoch_samples - samples)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            logits = model(images)
        with torch.autocast(device_type=device.type, enabled=False):
            loss = classification_loss(logits, labels)
        # A short final group is normalized by its actual sample count, not
        # by the nominal accumulation count or by the number of microbatches.
        backward_loss = loss if count == group_samples else loss * (count / group_samples)
        backward_loss.backward()
        loss_sum += float(loss.item()) * count
        correct += int((logits.detach().argmax(dim=1) == labels).sum().item())
        samples += count
        if (microstep + 1) % accumulation_steps == 0 or samples == epoch_samples:
            norm_sum += gradient_l2(model)
            optimizer.step()
            schedule.completed_updates += 1
            updates += 1
        # Do not retain this microbatch's graph/input during the next transfer.
        del images, labels, logits, loss, backward_loss
    if updates != schedule.steps_per_epoch or samples != len(loader.dataset):
        raise ValueError("Training iterator did not cover the declared full epoch")
    synchronize(device)
    return OnlineResult(loss_sum / samples, correct / samples, norm_sum / updates,
                        samples, time.perf_counter() - started)


def model_snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    """Capture FP32 parameters and buffers with their original dtypes on CPU."""
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise ValueError("Analysis/resume model parameters must remain FP32")
    return cast(dict[str, torch.Tensor], cpu_tree(dict(model.state_dict())))


def displacement(
    state: dict[str, torch.Tensor], reference: dict[str, torch.Tensor], parameter_names: list[str],
) -> float:
    """Compute parameter L2 distance in FP64 and reject changing shared buffers."""
    if list(state) != list(reference):
        raise ValueError("Model state order differs from theta_0")
    parameters = set(parameter_names)
    squared = 0.0
    for name, tensor in state.items():
        baseline = reference[name]
        if tensor.shape != baseline.shape or tensor.dtype != baseline.dtype:
            raise ValueError(f"State shape/dtype differs from theta_0: {name}")
        if name not in parameters:
            if not torch.equal(tensor, baseline):
                raise ValueError(f"Buffer changed from theta_0: {name}; shared-buffer landscape is invalid")
            continue
        current_flat, reference_flat = tensor.reshape(-1), baseline.reshape(-1)
        for offset in range(0, tensor.numel(), 65536):
            delta = current_flat[offset:offset + 65536].double() - reference_flat[offset:offset + 65536].double()
            squared += float(torch.dot(delta, delta).item())
    result = math.sqrt(squared)
    if not math.isfinite(result):
        raise ValueError("Nonfinite displacement from theta_0")
    return result


def _validate_optimizer_state(
    optimizer: torch.optim.AdamW, state: object, schedule: EpochSchedule, step: int,
) -> dict[str, object]:
    if not isinstance(state, dict) or set(state) != {"state", "param_groups"}:
        raise ValueError("Invalid AdamW resume state")
    current_groups = optimizer.state_dict()["param_groups"]
    groups = state["param_groups"]
    if not isinstance(groups, list) or len(groups) != 1 or len(current_groups) != 1:
        raise ValueError("Resume requires the original single AdamW group")
    expected_group = {**current_groups[0], "lr": schedule.base_lr if step == 0 else schedule.rate(step)}
    if groups[0] != expected_group:
        raise ValueError("AdamW hyperparameters, parameter order or last-used LR changed")
    values = state["state"]
    ids = current_groups[0]["params"]
    if not isinstance(values, dict) or set(values) != (set() if step == 0 else set(ids)):
        raise ValueError("AdamW state does not cover the completed updates")
    for identifier, parameter in zip(ids, optimizer.param_groups[0]["params"]):
        if step == 0:
            continue
        entry = values[identifier]
        if not isinstance(entry, dict) or set(entry) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("AdamW moment state differs from the declared optimizer")
        counter = entry["step"]
        if not isinstance(counter, torch.Tensor) or counter.numel() != 1 or counter.item() != step:
            raise ValueError("AdamW step count differs from the completed epoch")
        for key in ("exp_avg", "exp_avg_sq"):
            tensor = entry[key]
            if not isinstance(tensor, torch.Tensor) or tensor.dtype != parameter.dtype or tensor.shape != parameter.shape:
                raise ValueError("AdamW moment shape/dtype differs from its parameter")
            if tensor.device.type != "cpu" or not torch.isfinite(tensor).all().item():
                raise ValueError("AdamW moments must be finite CPU tensors in the saved state")
        if (entry["exp_avg_sq"] < 0).any().item():
            raise ValueError("AdamW second moment cannot be negative")
    return state


def restore_training_state(
    completed: CompletedEpoch, model: nn.Module, optimizer: torch.optim.AdamW,
    schedule: EpochSchedule, generators: LoaderGenerators,
) -> None:
    """Construct everything first; restore RNGs last, immediately before iteration."""
    saved = completed.resume
    step = completed.global_step
    if step != completed.epoch * schedule.steps_per_epoch or saved.get("scaler_state") is not None:
        raise ValueError("Resume epoch/update/scaler state differs from this bf16 contract")
    schedule.validate_state(saved.get("scheduler_state"), step)
    state = saved.get("model_state")
    current = model.state_dict()
    if not isinstance(state, dict) or list(state) != list(current):
        raise ValueError("Resume model keys/order differ")
    for name, tensor in state.items():
        if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu":
            raise ValueError("Resume model state must contain CPU tensors")
        if tensor.shape != current[name].shape or tensor.dtype != current[name].dtype:
            raise ValueError(f"Resume model shape/dtype differs: {name}")
        if tensor.is_floating_point() and not torch.isfinite(tensor).all().item():
            raise ValueError(f"Nonfinite resume model state: {name}")
    optimizer_state = _validate_optimizer_state(optimizer, saved.get("optimizer_state"), schedule, step)
    model.load_state_dict(state, strict=True)
    optimizer.load_state_dict(optimizer_state)
    schedule.completed_updates = step
    generators.load_state_dict(cast(dict[str, torch.Tensor], saved["loader_state"]))
    restore_random_state(cast(RandomState, saved["rng_state"]))


def run_segment(
    model: nn.Module, loaders: TrainingLoaders, optimizer: torch.optim.AdamW,
    schedule: EpochSchedule, generators: LoaderGenerators, reference: dict[str, torch.Tensor],
    segment: Segment, *, end_epoch: int, seed: int, device: torch.device, use_bf16: bool,
    parent: CompletedEpoch | None = None,
    on_complete: Callable[[Path, EpochMetrics], None] | None = None,
    accumulation_steps: int = 1,
) -> Path:
    """Record epoch zero and every completed epoch; stop only at an epoch boundary."""
    if type(end_epoch) is not int or not 1 <= end_epoch <= schedule.epochs:
        raise ValueError("Stopping epoch must stay within the full schedule")
    check_model_precision(model, device)
    _, effective_batch, steps = _batch_layout(loaders.train, accumulation_steps)
    if steps != schedule.steps_per_epoch:
        raise ValueError("Schedule must count effective-batch optimizer updates")
    if parent is not None and parent.metrics.get("batch_size") != effective_batch:
        raise ValueError("Resume effective batch size changed")
    expected_next = 0 if parent is None else parent.epoch + 1
    if segment.next_epoch != expected_next or expected_next > end_epoch:
        raise ValueError("Segment has no remaining epochs or does not match its resume parent")
    if parent is None:
        if schedule.completed_updates != 0:
            raise ValueError("A fresh segment must start at epoch zero")
        seed_global(seed)
    else:
        restore_training_state(parent, model, optimizer, schedule, generators)
    parameter_names = [name for name, _ in model.named_parameters()]
    last_path = segment.directory
    for epoch in range(expected_next, end_epoch + 1):
        synchronize(device)
        started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        online = None if epoch == 0 else train_one_epoch(
            model, loaders.train, optimizer, schedule, device=device, use_bf16=use_bf16,
            accumulation_steps=accumulation_steps,
        )
        evaluation_started = time.perf_counter()
        train_subset = evaluate(model, loaders.train_subset, device=device, generators=generators)
        validation = evaluate(model, loaders.validation, device=device, generators=generators)
        synchronize(device)
        evaluation_seconds = time.perf_counter() - evaluation_started
        checkpoint_started = time.perf_counter()
        state = model_snapshot(model)
        distance = displacement(state, reference, parameter_names)
        metrics = EpochMetrics(
            run_id=segment.run_id, segment_id=segment.segment_id, epoch=epoch,
            global_step=schedule.completed_updates, batch_size=effective_batch, seed=seed,
            train_loss=None if online is None else online.loss,
            train_accuracy=None if online is None else online.accuracy,
            train_subset_loss=train_subset.loss, train_subset_accuracy=train_subset.accuracy,
            val_loss=validation.loss, val_accuracy=validation.accuracy,
            gradient_norm=None if online is None else online.gradient_norm,
            learning_rate=schedule.last_lr, learning_rate_next=schedule.next_lr,
            parameter_displacement=distance, train_samples=0 if online is None else online.samples,
            train_subset_samples=train_subset.samples, val_samples=validation.samples,
            train_seconds=0.0 if online is None else online.seconds, evaluation_seconds=evaluation_seconds,
            peak_allocated_bytes=0 if device.type == "cpu" else torch.cuda.max_memory_allocated(device),
            peak_reserved_bytes=0 if device.type == "cpu" else torch.cuda.max_memory_reserved(device),
        )
        last_path, metrics = save_epoch(
            segment, metrics, model_state=state, optimizer_state=cpu_tree(optimizer.state_dict()),
            scheduler_state=schedule.state_dict(), rng_state=capture_random_state(),
            loader_state=generators.state_dict(), epoch_started=started, checkpoint_started=checkpoint_started,
        )
        if on_complete is not None:
            with preserve_random_state(generators):
                on_complete(last_path, metrics)
    return last_path


def _source_identity(root: Path) -> dict[str, object]:
    """Record only project Python sources, never scanning data/model directories."""
    files = sorted(path for folder in (root / "src", root / "scripts") for path in folder.rglob("*.py"))
    hashes = {path.relative_to(root).as_posix(): file_hash(path) for path in files}
    def git(*arguments: str) -> str | None:
        try:
            result = subprocess.run(["git", *arguments], cwd=root, capture_output=True, text=True, timeout=10, check=False)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return result.stdout.strip() if result.returncode == 0 else None
    status = git("status", "--porcelain", "--untracked-files=normal")
    return {"files": hashes, "sha256": record_hash(hashes), "git_commit": git("rev-parse", "HEAD"),
            "git_dirty": None if status is None else bool(status)}


def _cuda_environment() -> tuple[torch.device, dict[str, object]]:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before importing torch via scripts/run_train.py")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the configured Phase 0/1 run; no CPU fallback")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    if not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError("The selected CUDA device must support native bf16")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    properties = torch.cuda.get_device_properties(device)
    return device, {
        "device": str(device), "gpu_name": properties.name, "gpu_uuid": str(getattr(properties, "uuid", "unavailable")),
        "total_memory": properties.total_memory, "capability": list(torch.cuda.get_device_capability(device)),
        "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "parameter_dtype": "torch.float32", "training_autocast": "torch.bfloat16",
        "evaluation_dtype": "torch.float32", "tf32": False, "deterministic_algorithms": True,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
    }


def _prepared_run(loaded: LoadedConfig, *, resuming: bool) -> Path:
    root = loaded.config.run_directory
    if not root.exists():
        if resuming:
            raise ValueError("Cannot resume a run without its original configuration records")
        return prepare_run(loaded)
    required = (root / "source.yaml", root / "config.json", root / "prepared.json")
    if not all(path.is_file() and path.resolve() == path for path in required):
        raise ValueError("Existing run is incompletely prepared; do not overwrite or repair it")
    prepared = read_json(root / "prepared.json")
    if prepared.get("source_sha256") != file_hash(root / "source.yaml"):
        raise ValueError("Original run YAML hash differs from its record")
    if prepared.get("effective_sha256") != file_hash(root / "config.json") or read_json(root / "config.json") != config_dict(loaded.config):
        raise ValueError("Existing run configuration differs from this request")
    if prepared.get("run_id") != loaded.config.run_id:
        raise ValueError("Prepared run identity differs")
    segments = root / "segments"
    if not resuming and segments.exists() and any(segments.iterdir()):
        raise ValueError("This run already has a segment; use an explicit --resume-from completed epoch")
    return root


def run_experiment(
    loaded: LoadedConfig, *, resume_from: Path | None = None,
    on_complete: Callable[[Path, EpochMetrics], None] | None = None,
) -> Path:
    """Run the fixed real-data experiment; all preparation must already exist."""
    config = loaded.config
    initial = load_initial_checkpoint(config)
    source = load_cifar10_training(config.paths.dataset_root)
    indices_path = split_path(config)
    indices = load_split_indices(indices_path, source.targets, SplitSpec.from_config(config))
    views = build_dataset_views(source, indices, initial.preprocessing)
    generators = LoaderGenerators.from_seed(config.experiment.seed)
    loaders = TrainingLoaders(
        train=make_loader(views.train, role="train", batch_size=config.training.microbatch_size,
                          num_workers=config.training.num_workers, pin_memory=True, generators=generators),
        train_subset=make_loader(views.train_subset, role="train_subset", batch_size=config.evaluation.batch_size,
                                 num_workers=config.evaluation.num_workers, pin_memory=True, generators=generators),
        validation=make_loader(views.validation, role="validation", batch_size=config.evaluation.batch_size,
                               num_workers=config.evaluation.num_workers, pin_memory=True, generators=generators),
    )
    device, numerical = _cuda_environment()
    sources = _source_identity(loaded.project_root)
    microbatch, effective_batch, steps = _batch_layout(loaders.train, config.training.accumulation_steps)
    batching = {"effective_batch_size": effective_batch, "microbatch_size": microbatch,
                "accumulation_steps": config.training.accumulation_steps, "optimizer_steps_per_epoch": steps}
    contract: dict[str, object] = {
        "schema_version": 1, "run_id": config.run_id, "config": config_dict(config),
        "effective_sha256": loaded.effective_sha256,
        "initial_checkpoint": {"path": str(initial.checkpoint_path), "sha256": initial.checkpoint_sha256},
        "split": {"path": str(indices_path), "sha256": file_hash(indices_path),
                  "metadata_sha256": file_hash(indices_path.with_suffix(".json"))},
        "parameter_spec_sha256": initial.metadata["parameter_spec_sha256"],
        "runtime": initial.metadata["runtime"], "numerical": numerical,
        "source_sha256": sources["sha256"],
        "batching": batching,
    }
    root = _prepared_run(loaded, resuming=resume_from is not None)
    parent = None
    if resume_from is not None:
        path = resume_from.resolve()
        if len(path.parents) < 4 or path.parents[3] != root:
            raise ValueError("--resume-from must name a completed epoch directory inside this run")
        parent = load_completed_epoch(path, contract)
        if parent.epoch >= config.end_epoch:
            raise ValueError("This checkpoint has already reached the configured stopping epoch")
        if path not in completed_lineage(path.parents[1], contract, through_epoch=parent.epoch):
            raise ValueError("Requested parent is absent from the completed branch")
    environment_path = root / "environment.json"
    environment = {"schema_version": 1, "runtime": initial.metadata["runtime"], "numerical": numerical,
                   "preprocessing": initial.metadata["preprocessing"], "sources": sources,
                   "contract_sha256": record_hash(contract), "batching": batching}
    if environment_path.exists():
        if read_json(environment_path).get("contract_sha256") != record_hash(contract):
            raise ValueError("Existing run environment/identity differs; it will not be overwritten")
    elif parent is not None:
        raise ValueError("Resume run is missing its original environment record")
    else:
        write_json(environment_path, environment)
    reference = model_snapshot(initial.model)
    model = initial.model.to(device=device)
    schedule = EpochSchedule(config.training.epochs, config.training.warmup_epochs, steps,
                             config.training.learning_rate, scheduler=config.training.scheduler)
    optimizer = make_optimizer(model, schedule, config.training.weight_decay)
    segment = create_segment(root, contract, parent)
    write_json(segment.directory / "environment.json", environment)
    return run_segment(
        model, loaders, optimizer, schedule, generators, reference, segment,
        end_epoch=config.end_epoch, seed=config.experiment.seed, device=device, use_bf16=True,
        parent=parent, on_complete=on_complete,
        accumulation_steps=config.training.accumulation_steps,
    )
