"""Load the Phase 0/1 contract without importing training or numerical code.

Loading is read-only. Only ``prepare_run`` creates files, exclusively in a new
run directory; a failed or existing run is never overwritten or cleaned up.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, fields, is_dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from types import UnionType
from typing import TypeVar, cast, get_args, get_origin, get_type_hints


class ConfigError(ValueError):
    """An input does not satisfy the declared experiment contract."""


@dataclass(frozen=True)
class Experiment:
    name: str
    phase: str
    seed: int


@dataclass(frozen=True)
class Paths:
    dataset_root: Path
    output_root: Path
    init_checkpoint: Path
    scratch_root: Path


@dataclass(frozen=True)
class Model:
    name: str
    initialization: str
    num_classes: int
    image_size: int
    full_finetune: bool
    init_seed: int


@dataclass(frozen=True)
class Training:
    epochs: int
    stop_after_epoch: int | None
    batch_size: int
    microbatch_size: int
    num_workers: int
    learning_rate: float
    weight_decay: float
    optimizer: str
    scheduler: str
    warmup_epochs: int
    amp_dtype: str
    gradient_clip_norm: float | None
    checkpoint_interval_epochs: int
    drop_last: bool

    @property
    def accumulation_steps(self) -> int:
        """Derive the full-group count from the validated effective batch size."""
        return self.batch_size // self.microbatch_size


@dataclass(frozen=True)
class Augmentation:
    random_resized_crop: bool
    horizontal_flip: bool
    mixup: bool
    cutmix: bool
    randaugment: bool


@dataclass(frozen=True)
class Split:
    train_size: int
    val_size: int
    split_seed: int


@dataclass(frozen=True)
class Reproducibility:
    deterministic_algorithms: bool
    persistent_workers: bool
    pin_memory: bool


@dataclass(frozen=True)
class Evaluation:
    batch_size: int
    num_workers: int
    dtype: str
    amp: bool
    tf32: bool


@dataclass(frozen=True)
class Checkpoint:
    analysis_format: str
    parameter_dtype: str
    save_resume: bool
    keep_all: bool


@dataclass(frozen=True)
class Projection:
    solver: str
    compute_dtype: str
    block_parameters: int


@dataclass(frozen=True)
class Landscape:
    subset_size: int
    subset_seed: int
    grid_size: int
    margin_ratio: float


@dataclass(frozen=True)
class Logging:
    format: str
    train_reduction: str
    gradient_reduction: str


@dataclass(frozen=True)
class Phase1:
    batch_sizes: tuple[int, ...]
    seeds: tuple[int, ...]
    same_learning_rate: bool


@dataclass(frozen=True)
class Animation:
    fps: int
    format: str
    max_file_size_mb: float
    width: int
    height: int
    min_width: int
    palette_colors: int
    final_hold_ms: int
    show_learning_rate: bool
    show_train_subset_metrics: bool
    show_validation_loss: bool
    show_validation_accuracy: bool
    show_gradient_norm: bool


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: int
    experiment: Experiment
    paths: Paths
    model: Model
    training: Training
    augmentation: Augmentation
    split: Split
    reproducibility: Reproducibility
    evaluation: Evaluation
    checkpoint: Checkpoint
    projection: Projection
    landscape: Landscape
    logging: Logging
    phase1: Phase1
    animation: Animation

    @property
    def run_id(self) -> str:
        """Return a stable identity independent of the process working directory."""
        return (
            f"{self.experiment.name}/"
            f"b{self.training.batch_size}_seed{self.experiment.seed}"
        )

    @property
    def end_epoch(self) -> int:
        """Return the stopping point without changing the LR schedule horizon."""
        stop = self.training.stop_after_epoch
        return self.training.epochs if stop is None else stop

    @property
    def run_directory(self) -> Path:
        """Return the resolved destination and reject existing symlink escapes."""
        parent = (self.paths.output_root / "runs").resolve()
        destination = (parent / self.run_id).resolve()
        if not destination.is_relative_to(parent):
            raise ConfigError("run_directory escapes output_root/runs")
        if not parent.is_relative_to(self.paths.output_root):
            raise ConfigError("output_root/runs resolves outside output_root")
        return destination


@dataclass(frozen=True)
class LoadedConfig:
    config: ExperimentConfig
    project_root: Path
    source_path: Path
    source_bytes: bytes

    @property
    def source_sha256(self) -> str:
        """Hash the exact bytes that will be preserved as source.yaml."""
        return hashlib.sha256(self.source_bytes).hexdigest()

    @property
    def effective_sha256(self) -> str:
        """Hash the canonical effective configuration, including CLI overrides."""
        return hashlib.sha256(_json_bytes(config_dict(self.config))).hexdigest()


ConfigSection = TypeVar("ConfigSection")
_MAX_CONFIG_BYTES = 1_048_576


def _decode_section(
    section_type: type[ConfigSection], value: object, location: str
) -> ConfigSection:
    """Decode only our statically declared dataclasses; reject schema drift."""
    if not isinstance(value, dict) or any(not isinstance(k, str) for k in value):
        raise ConfigError(f"{location}: expected a mapping with string keys")
    hints = get_type_hints(section_type)
    expected = set(hints)
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ConfigError(f"{location}: missing keys={missing}; unknown keys={unknown}")
    decoded = {
        key: _decode_value(annotation, value[key], f"{location}.{key}")
        for key, annotation in hints.items()
    }
    return section_type(**decoded)


def _decode_value(annotation: object, value: object, location: str) -> object:
    """Check scalar types strictly, including Python's bool/int distinction."""
    origin = get_origin(annotation)
    if origin is UnionType:
        members = get_args(annotation)
        if value is None and type(None) in members:
            return None
        remaining = tuple(item for item in members if item is not type(None))
        if len(remaining) == 1:
            return _decode_value(remaining[0], value, location)
    if isinstance(annotation, type) and is_dataclass(annotation):
        return _decode_section(annotation, value, location)
    if origin is tuple:
        if not isinstance(value, list):
            raise ConfigError(f"{location}: expected a YAML list")
        element_type, _ = get_args(annotation)
        return tuple(
            _decode_value(element_type, item, f"{location}[{index}]")
            for index, item in enumerate(value)
        )
    if annotation is Path:
        if type(value) is not str or not value.strip():
            raise ConfigError(f"{location}: expected a nonempty path string")
        return Path(value)
    if annotation is float:
        if type(value) not in (int, float):
            raise ConfigError(f"{location}: expected a finite number")
        try:
            number = float(value)
        except (ValueError, OverflowError) as error:
            raise ConfigError(f"{location}: number is out of range") from error
        if not math.isfinite(number):
            raise ConfigError(f"{location}: expected a finite number")
        return number
    if annotation in (str, int, bool) and type(value) is annotation:
        return value
    raise ConfigError(f"{location}: expected {annotation}, got {type(value).__name__}")


def _yaml_mapping(source: bytes) -> object:
    """Use SafeLoader, with duplicate and non-string mapping keys rejected."""
    try:
        import yaml
    except ImportError as error:
        raise ConfigError(
            "PyYAML is required. Use the documented torch_env; "
            "this command does not install packages."
        ) from error

    class UniqueKeyLoader(yaml.SafeLoader):
        def construct_mapping(
            self, node: yaml.MappingNode, deep: bool = False
        ) -> dict[str, object]:
            if not isinstance(node, yaml.MappingNode):
                raise ConfigError("YAML: expected a mapping")
            result: dict[str, object] = {}
            for key_node, value_node in node.value:
                key = self.construct_object(key_node, deep=deep)
                if not isinstance(key, str):
                    raise ConfigError("YAML: mapping keys must be strings")
                if key in result:
                    raise ConfigError(f"YAML: duplicate key {key!r}")
                result[key] = self.construct_object(value_node, deep=deep)
            return result

    try:
        return yaml.load(source.decode("utf-8-sig"), Loader=UniqueKeyLoader)
    except (UnicodeError, yaml.YAMLError, RecursionError) as error:
        raise ConfigError(f"Invalid UTF-8 YAML: {error}") from error


def _require(condition: bool, message: str) -> None:
    """Raise a user-facing validation error instead of relying on assertions."""
    if not condition:
        raise ConfigError(message)


def validate_config(config: ExperimentConfig) -> None:
    """Validate numeric ranges and the accepted Phase 0/1 experiment boundary."""
    e, t, a = config.experiment, config.training, config.animation
    _require(config.schema_version == 3, "schema_version must be 3 (scratch gradient accumulation contract)")
    _require(e.phase in ("phase0", "phase1"), "experiment.phase must be phase0 or phase1")
    _require(
        re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", e.name) is not None,
        "experiment.name: use 1-64 lowercase letters, digits, '_' or '-'",
    )
    _require(
        re.fullmatch(r"(?:con|prn|aux|nul|com[0-9]|lpt[0-9])", e.name) is None,
        "experiment.name is a reserved Windows filename",
    )
    _require(config.phase1.batch_sizes == (64, 256, 1024), "phase1.batch_sizes must be [64, 256, 1024]")
    _require(config.phase1.seeds == (0, 1, 2), "phase1.seeds must be [0, 1, 2]")
    _require(e.seed in config.phase1.seeds, "experiment.seed must be 0, 1 or 2")
    _require(t.batch_size in config.phase1.batch_sizes, "training.batch_size must be 64, 256 or 1024 (effective batch)")
    _require(t.microbatch_size == 64, "training.microbatch_size must be 64")
    _require(t.epochs == 100, "training.epochs must retain the 100-epoch scratch schedule")
    _require(t.warmup_epochs == 0, "Phase 0/1 constant LR requires training.warmup_epochs=0")
    _require(t.learning_rate == 1e-3, "Phase 0/1 requires the shared fixed learning_rate=1e-3")
    _require(t.weight_decay >= 0, "training.weight_decay must be nonnegative")
    _require(0 <= t.num_workers <= 4, "training.num_workers must be between 0 and 4")
    if e.phase == "phase0":
        _require(e.seed == 0, "Phase 0 requires seed 0")
        _require(t.stop_after_epoch == 5, "Phase 0 requires stop_after_epoch=5")
    else:
        _require(t.stop_after_epoch is None, "Phase 1 requires stop_after_epoch=null")
    _require(config.evaluation.batch_size == 64, "evaluation.batch_size must be 64")
    _require(0 <= config.evaluation.num_workers <= 4, "evaluation.num_workers must be between 0 and 4")
    _require(1 <= config.projection.block_parameters <= 65536, "projection.block_parameters must be between 1 and 65536")
    _require(0 < config.landscape.margin_ratio <= 1, "landscape.margin_ratio must be in (0, 1]")
    for name, seed in (
        ("split.split_seed", config.split.split_seed),
        ("landscape.subset_seed", config.landscape.subset_seed),
    ):
        _require(0 <= seed < 2**32, f"{name} must be in [0, 2**32)")
    _require(0 < a.max_file_size_mb <= 3.0, "animation.max_file_size_mb must be in (0, 3]")
    _require(a.fps > 0 and 100 % a.fps == 0, "animation.fps must divide 100 for GIF timing")
    _require(a.width >= a.min_width >= 640 and a.width <= 960, "animation widths must satisfy 640 <= min_width <= width <= 960")
    _require(400 <= a.height <= 640, "animation.height must be between 400 and 640")
    _require(a.palette_colors in (64, 128), "animation.palette_colors must be 64 or 128")
    _require(a.final_hold_ms >= 0 and a.final_hold_ms % 10 == 0, "animation.final_hold_ms must be a nonnegative multiple of 10")

    fixed: tuple[tuple[str, object, object], ...] = (
        ("model.name", config.model.name, "convnextv2_tiny"),
        ("model.initialization", config.model.initialization, "scratch"),
        ("model.num_classes", config.model.num_classes, 10),
        ("model.image_size", config.model.image_size, 224),
        ("model.full_finetune", config.model.full_finetune, True),
        ("model.init_seed", config.model.init_seed, 0),
        ("training.optimizer", t.optimizer, "adamw"),
        ("training.scheduler", t.scheduler, "constant"),
        ("training.amp_dtype", t.amp_dtype, "bf16"),
        ("training.gradient_clip_norm", t.gradient_clip_norm, None),
        ("training.checkpoint_interval_epochs", t.checkpoint_interval_epochs, 1),
        ("training.drop_last", t.drop_last, False),
        ("augmentation.random_resized_crop", config.augmentation.random_resized_crop, True),
        ("augmentation.horizontal_flip", config.augmentation.horizontal_flip, True),
        ("augmentation.mixup", config.augmentation.mixup, False),
        ("augmentation.cutmix", config.augmentation.cutmix, False),
        ("augmentation.randaugment", config.augmentation.randaugment, False),
        ("split.train_size", config.split.train_size, 45000),
        ("split.val_size", config.split.val_size, 5000),
        ("reproducibility.deterministic_algorithms", config.reproducibility.deterministic_algorithms, True),
        ("reproducibility.persistent_workers", config.reproducibility.persistent_workers, False),
        ("reproducibility.pin_memory", config.reproducibility.pin_memory, True),
        ("evaluation.dtype", config.evaluation.dtype, "float32"),
        ("evaluation.amp", config.evaluation.amp, False),
        ("evaluation.tf32", config.evaluation.tf32, False),
        ("checkpoint.analysis_format", config.checkpoint.analysis_format, "pt"),
        ("checkpoint.parameter_dtype", config.checkpoint.parameter_dtype, "float32"),
        ("checkpoint.save_resume", config.checkpoint.save_resume, True),
        ("checkpoint.keep_all", config.checkpoint.keep_all, True),
        ("projection.solver", config.projection.solver, "gram_eigh"),
        ("projection.compute_dtype", config.projection.compute_dtype, "float64"),
        ("landscape.subset_size", config.landscape.subset_size, 1000),
        ("landscape.grid_size", config.landscape.grid_size, 21),
        ("logging.format", config.logging.format, "csv"),
        ("logging.train_reduction", config.logging.train_reduction, "sample_mean"),
        ("logging.gradient_reduction", config.logging.gradient_reduction, "update_mean"),
        ("phase1.same_learning_rate", config.phase1.same_learning_rate, True),
        ("animation.format", a.format, "gif"),
        ("animation.show_learning_rate", a.show_learning_rate, True),
        ("animation.show_train_subset_metrics", a.show_train_subset_metrics, True),
        ("animation.show_validation_loss", a.show_validation_loss, True),
        ("animation.show_validation_accuracy", a.show_validation_accuracy, True),
        ("animation.show_gradient_norm", a.show_gradient_norm, True),
    )
    for name, actual, expected in fixed:
        _require(actual == expected, f"{name} must be {expected!r} for Phase 0/1")


def _resolve_path(value: Path, root: Path) -> Path:
    """Resolve against the repository, without shell or environment expansion."""
    if value.drive and not value.is_absolute():
        raise ConfigError(f"Drive-relative paths are unsupported: {value}")
    try:
        return (value if value.is_absolute() else root / value).resolve()
    except (OSError, ValueError, RuntimeError) as error:
        raise ConfigError(f"Invalid or unresolvable path: {value!s}") from error


def _resolve_config_paths(config: ExperimentConfig, root: Path) -> ExperimentConfig:
    """Protect raw data from artifact destinations, including existing symlinks."""
    p = config.paths
    resolved = Paths(
        dataset_root=_resolve_path(p.dataset_root, root),
        output_root=_resolve_path(p.output_root, root),
        init_checkpoint=_resolve_path(p.init_checkpoint, root),
        scratch_root=_resolve_path(p.scratch_root, root),
    )
    for label, destination in (
        ("output_root", resolved.output_root),
        ("scratch_root", resolved.scratch_root),
    ):
        if destination.is_relative_to(resolved.dataset_root) or resolved.dataset_root.is_relative_to(destination):
            raise ConfigError(f"paths.{label} must not overlap paths.dataset_root")
    if resolved.init_checkpoint.is_relative_to(resolved.dataset_root):
        raise ConfigError("paths.init_checkpoint must be outside raw data")
    result = replace(config, paths=resolved)
    # Check before any writer is reachable; directories need not exist yet.
    result.run_directory
    return result


def load_config(
    path: str | Path,
    *,
    project_root: str | Path | None = None,
    batch_size: int | None = None,
    seed: int | None = None,
    name: str | None = None,
) -> LoadedConfig:
    """Read and validate YAML without creating artifacts or loading a model.

    Relative input and artifact paths are relative to ``project_root``. The
    default is the repository containing this module, never the current cwd.
    Overrides are restricted to batch, seed and experiment-series name.
    """
    try:
        root = (
            Path(project_root).resolve()
            if project_root is not None
            else Path(__file__).resolve().parents[2]
        )
    except (OSError, ValueError, RuntimeError) as error:
        raise ConfigError("project_root is invalid or cannot be resolved") from error
    if not root.is_dir():
        raise ConfigError(f"project_root must be an existing directory: {root}")
    source_path = _resolve_path(Path(path), root)
    with source_path.open("rb") as handle:
        source = handle.read(_MAX_CONFIG_BYTES + 1)
    if len(source) > _MAX_CONFIG_BYTES:
        raise ConfigError("Configuration exceeds the 1 MiB input limit")
    config = _decode_section(ExperimentConfig, _yaml_mapping(source), "config")
    for label, value, kind in (
        ("batch_size", batch_size, int), ("seed", seed, int), ("name", name, str)
    ):
        if value is not None and type(value) is not kind:
            raise ConfigError(f"Override {label} has an invalid type")
    experiment = replace(
        config.experiment,
        seed=config.experiment.seed if seed is None else seed,
        name=config.experiment.name if name is None else name,
    )
    training = replace(
        config.training,
        batch_size=config.training.batch_size if batch_size is None else batch_size,
    )
    config = replace(config, experiment=experiment, training=training)
    validate_config(config)
    config = _resolve_config_paths(config, root)
    return LoadedConfig(config, root, source_path, source)


def _json_value(value: object) -> object:
    """Convert our immutable types to JSON-compatible values explicitly."""
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise TypeError(f"Unsupported configuration value: {type(value).__name__}")


def config_dict(config: ExperimentConfig) -> dict[str, object]:
    """Return the complete effective configuration for persistence and review."""
    return cast(dict[str, object], _json_value(config))


def _json_bytes(value: dict[str, object]) -> bytes:
    """Use one canonical representation for fingerprints and saved files."""
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def prepare_run(loaded: LoadedConfig) -> Path:
    """Create configuration records in a new run, refusing all existing runs.

    This does not prepare data, weights or a resumable training checkpoint.
    Failure leaves the incomplete directory for inspection; no cleanup occurs.
    """
    config = loaded.config
    validate_config(config)
    # Re-resolve immediately before writing so changed parent symlinks are checked.
    checked = _resolve_config_paths(config, loaded.project_root)
    if checked.paths != config.paths:
        raise ConfigError("Artifact paths changed since configuration loading")
    destination = checked.run_directory
    destination.mkdir(parents=True, exist_ok=False)
    records: tuple[tuple[str, bytes], ...] = (
        ("source.yaml", loaded.source_bytes),
        ("config.json", _json_bytes(config_dict(config))),
        (
            "prepared.json",
            _json_bytes({
                "schema_version": config.schema_version,
                "status": "configuration_prepared",
                "run_id": config.run_id,
                "source_path": str(loaded.source_path),
                "project_root": str(loaded.project_root),
                "source_sha256": loaded.source_sha256,
                "effective_sha256": loaded.effective_sha256,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            }),
        ),
    )
    for filename, content in records:
        with (destination / filename).open("xb") as handle:
            handle.write(content)
    return destination
