"""CIFAR-10 train-only views, immutable split records and deterministic evaluation.

No data is downloaded on import or by these functions. Dataset reads, split
creation and DataLoader iteration occur only when the user invokes them.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, TypeVar, cast

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torchvision import transforms
from torchvision.datasets import CIFAR10

from .config import ExperimentConfig
from .seeds import LoaderGenerators, derive_seed, seed_worker


IndexArray = NDArray[np.int64]
LoaderRole = Literal["train", "validation", "train_subset", "validation_subset"]
ImageTransform = Callable[[Image.Image], torch.Tensor]
Sample = TypeVar("Sample")
_INDEX_NAMES = ("train", "validation", "train_subset", "validation_subset")


@dataclass(frozen=True)
class SplitSpec:
    num_classes: int
    train_per_class: int
    validation_per_class: int
    subset_per_class: int
    split_seed: int
    subset_seed: int

    @classmethod
    def from_config(cls, config: ExperimentConfig) -> SplitSpec:
        """Translate the fixed Phase 0/1 counts to per-class sampling counts."""
        classes = config.model.num_classes
        sizes = (config.split.train_size, config.split.val_size, config.landscape.subset_size)
        if classes <= 0 or any(size % classes for size in sizes):
            raise ValueError("Split and subset sizes must be divisible by the class count")
        return cls(classes, *(size // classes for size in sizes),
                   config.split.split_seed, config.landscape.subset_seed)

    @property
    def dataset_size(self) -> int:
        """Return the expected source length, excluding the official test set."""
        return self.num_classes * (self.train_per_class + self.validation_per_class)

    def validate(self) -> None:
        """Reject impossible sampling requests instead of silently undersampling."""
        counts = (self.num_classes, self.train_per_class, self.validation_per_class, self.subset_per_class)
        if any(type(count) is not int or count <= 0 for count in counts):
            raise ValueError("Class and per-class counts must be positive integers")
        if self.subset_per_class > min(self.train_per_class, self.validation_per_class):
            raise ValueError("Each landscape subset must fit inside its own split")
        for seed in (self.split_seed, self.subset_seed):
            if type(seed) is not int or not 0 <= seed < 2**32:
                raise ValueError("Split seeds must be integers in [0, 2**32)")


@dataclass(frozen=True)
class SplitIndices:
    spec: SplitSpec
    labels_sha256: str
    train: IndexArray
    validation: IndexArray
    train_subset: IndexArray
    validation_subset: IndexArray


def _labels_array(targets: Sequence[int], spec: SplitSpec) -> IndexArray:
    spec.validate()
    labels = np.asarray(targets)
    if labels.ndim != 1 or labels.dtype.kind not in "iu":
        raise ValueError("Targets must be a one-dimensional integer label sequence")
    if len(labels) != spec.dataset_size:
        raise ValueError(f"Expected {spec.dataset_size} source labels, got {len(labels)}")
    if np.any(labels < 0) or np.any(labels >= spec.num_classes):
        raise ValueError("Targets must use contiguous class IDs starting at zero")
    expected = spec.train_per_class + spec.validation_per_class
    if not np.all(np.bincount(labels.astype(np.int64), minlength=spec.num_classes) == expected):
        raise ValueError(f"Every source class must contain exactly {expected} samples")
    return labels.astype(np.int64, copy=False)


def _labels_hash(labels: IndexArray) -> str:
    return hashlib.sha256(labels.astype("<i8", copy=False).tobytes()).hexdigest()


def _fixed_indices(parts: Sequence[IndexArray]) -> IndexArray:
    indices = np.sort(np.concatenate(parts)).astype(np.int64, copy=False)
    indices.setflags(write=False)
    return indices


def create_split_indices(targets: Sequence[int], spec: SplitSpec) -> SplitIndices:
    """Stratify once, then sample each landscape subset strictly within its split.

    The count parameters permit small unit fixtures; production counts come
    exclusively from the validated Phase 0/1 configuration.
    """
    labels = _labels_array(targets, spec)
    parts: dict[str, list[IndexArray]] = {name: [] for name in _INDEX_NAMES}
    for class_id in range(spec.num_classes):
        class_indices = np.flatnonzero(labels == class_id).astype(np.int64)
        split_rng = np.random.Generator(np.random.PCG64(derive_seed(spec.split_seed, f"split/class/{class_id}")))
        shuffled = split_rng.permutation(class_indices)
        validation = shuffled[:spec.validation_per_class]
        train = shuffled[spec.validation_per_class:]
        parts["validation"].append(validation)
        parts["train"].append(train)
        for split_name, indices in (("train", train), ("validation", validation)):
            namespace = f"subset/{split_name}/class/{class_id}"
            rng = np.random.Generator(np.random.PCG64(derive_seed(spec.subset_seed, namespace)))
            parts[f"{split_name}_subset"].append(rng.permutation(indices)[:spec.subset_per_class])
    result = SplitIndices(spec, _labels_hash(labels), **{
        name: _fixed_indices(parts[name]) for name in _INDEX_NAMES
    })
    validate_split_indices(result, targets)
    return result


def validate_split_indices(indices: SplitIndices, targets: Sequence[int]) -> None:
    """Check provenance, class balance, coverage, disjointness and subset bounds."""
    labels = _labels_array(targets, indices.spec)
    if indices.labels_sha256 != _labels_hash(labels):
        raise ValueError("Source label order differs from the saved split")
    spec = indices.spec
    counts = (spec.train_per_class, spec.validation_per_class,
              spec.subset_per_class, spec.subset_per_class)
    for name, count in zip(_INDEX_NAMES, counts):
        values = getattr(indices, name)
        if values.dtype != np.dtype(np.int64) or values.ndim != 1:
            raise ValueError(f"{name}: expected a one-dimensional int64 array")
        if len(values) != count * spec.num_classes:
            raise ValueError(f"{name}: incorrect number of indices")
        if np.any(values < 0) or np.any(values >= len(labels)) or np.any(np.diff(values) <= 0):
            raise ValueError(f"{name}: indices must be unique, sorted and within the source")
        if not np.all(np.bincount(labels[values], minlength=spec.num_classes) == count):
            raise ValueError(f"{name}: class counts differ from the declared split")
    if np.intersect1d(indices.train, indices.validation).size:
        raise ValueError("Train and validation must be disjoint")
    for name in ("train", "validation"):
        if not np.all(np.isin(getattr(indices, f"{name}_subset"), getattr(indices, name))):
            raise ValueError(f"{name}_subset contains indices from another split")


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_split_indices(path: Path, indices: SplitIndices, targets: Sequence[int]) -> None:
    """Write new indices and their final metadata marker; never replace a file."""
    validate_split_indices(indices, targets)
    if path.suffix != ".npz":
        raise ValueError("Split index path must end in .npz")
    metadata_path = path.with_suffix(".json")
    if path.exists() or metadata_path.exists():
        raise FileExistsError(f"Refusing to overwrite split records: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez(handle, **{name: getattr(indices, name) for name in _INDEX_NAMES})
    metadata = {
        "schema_version": 1,
        "algorithm": "stratified_pcg64_v1",
        "spec": asdict(indices.spec),
        "labels_sha256": indices.labels_sha256,
        "indices_sha256": _file_hash(path),
        "numpy_version": np.__version__,
    }
    # A partial/missing JSON is not a valid completed split. Leave it for inspection.
    with metadata_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(metadata, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def load_split_indices(path: Path, targets: Sequence[int], spec: SplitSpec) -> SplitIndices:
    """Reuse only complete records for exactly the requested source and spec."""
    spec.validate()
    metadata_path = path.with_suffix(".json")
    if not path.is_file() or not metadata_path.is_file():
        raise ValueError(f"Incomplete split records: both {path.name} and {metadata_path.name} are required")
    if path.stat().st_size > 8 * 1024 * 1024 or metadata_path.stat().st_size > 65536:
        raise ValueError("Split records exceed the expected small metadata size")
    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict) or metadata.get("schema_version") != 1:
        raise ValueError("Unsupported split metadata schema")
    if metadata.get("algorithm") != "stratified_pcg64_v1" or metadata.get("spec") != asdict(spec):
        raise ValueError("Saved split configuration does not match this experiment")
    if metadata.get("indices_sha256") != _file_hash(path):
        raise ValueError("Split file hash does not match its completed metadata")
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != set(_INDEX_NAMES):
            raise ValueError("Unexpected arrays in the split file")
        arrays = {name: saved[name].copy() for name in _INDEX_NAMES}
    for values in arrays.values():
        values.setflags(write=False)
    result = SplitIndices(spec, metadata.get("labels_sha256", ""), **arrays)
    validate_split_indices(result, targets)
    return result


def load_cifar10_training(root: Path) -> CIFAR10:
    """Read only the official training partition, without automatic downloading."""
    return CIFAR10(root=str(root), train=True, transform=None, download=False)


def split_path(config: ExperimentConfig) -> Path:
    """Use one shared split/subset record across all runs, not one per run seed."""
    filename = (
        f"cifar10_train_val_seed{config.split.split_seed}"
        f"_subset{config.landscape.subset_seed}.npz"
    )
    parent = (config.paths.output_root / "splits").resolve()
    if not parent.is_relative_to(config.paths.output_root):
        raise ValueError("Split destination escapes output_root")
    return parent / filename


def prepare_cifar10_splits(config: ExperimentConfig, source: CIFAR10) -> SplitIndices:
    """Create once or verify existing shared indices; reject official test data."""
    if source.train is not True:
        raise ValueError("Phase 0/1 must not use the official CIFAR-10 test partition")
    spec = SplitSpec.from_config(config)
    path = split_path(config)
    if path.exists() or path.with_suffix(".json").exists():
        return load_split_indices(path, source.targets, spec)
    result = create_split_indices(source.targets, spec)
    save_split_indices(path, result, source.targets)
    return result


@dataclass(frozen=True)
class Preprocessing:
    train: ImageTransform
    evaluation: ImageTransform
    metadata: dict[str, object]


def _triple(value: object, name: str) -> tuple[float, float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError(f"{name} must contain exactly three channel values")
    if any(type(item) not in (int, float) or not math.isfinite(item) for item in value):
        raise ValueError(f"{name} must contain finite numbers")
    return cast(tuple[float, float, float], tuple(float(item) for item in value))


def build_preprocessing(data_config: Mapping[str, object]) -> Preprocessing:
    """Consume the model-resolved timm config without constructing a model.

    Training has only the accepted crop/flip operations. Evaluation delegates
    resize/crop behavior to timm, with every relevant resolved value supplied.
    """
    from timm.data import create_transform

    required = {"input_size", "interpolation", "mean", "std", "crop_pct", "crop_mode"}
    if set(data_config) != required:
        raise ValueError(f"Expected resolved model data keys: {sorted(required)}")
    size = data_config["input_size"]
    if not isinstance(size, (tuple, list)) or tuple(size) != (3, 224, 224):
        raise ValueError("Phase 0/1 requires model input_size=(3, 224, 224)")
    mean, std = _triple(data_config["mean"], "mean"), _triple(data_config["std"], "std")
    if any(value <= 0 for value in std):
        raise ValueError("All normalization standard deviations must be positive")
    modes = {"bicubic": transforms.InterpolationMode.BICUBIC,
             "bilinear": transforms.InterpolationMode.BILINEAR}
    interpolation = data_config["interpolation"]
    if not isinstance(interpolation, str) or interpolation not in modes:
        raise ValueError("Expected deterministic bicubic or bilinear model interpolation")
    crop_pct = data_config["crop_pct"]
    crop_mode = data_config["crop_mode"]
    if type(crop_pct) not in (int, float) or not math.isfinite(crop_pct) or not 0 < crop_pct <= 1:
        raise ValueError("crop_pct must be a finite number in (0, 1]")
    if crop_mode not in ("center", "squash", "border"):
        raise ValueError("Unsupported model evaluation crop mode")
    train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(3 / 4, 4 / 3),
                                     interpolation=modes[interpolation], antialias=True),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    evaluation = create_transform(
        input_size=(3, 224, 224), is_training=False,
        interpolation=interpolation, mean=mean, std=std,
        crop_pct=float(crop_pct), crop_mode=str(crop_mode),
        use_prefetcher=False, normalize=True,
    )
    metadata: dict[str, object] = {
        "input_size": [3, 224, 224], "interpolation": interpolation,
        "mean": list(mean), "std": list(std),
        "crop_pct": float(crop_pct), "crop_mode": crop_mode,
        "resize_size": math.floor(224 / crop_pct),
        "train_scale": [0.08, 1.0], "train_ratio": [3 / 4, 4 / 3],
        "horizontal_flip_probability": 0.5, "antialias": True,
        "train_transform": repr(train), "evaluation_transform": repr(evaluation),
    }
    return Preprocessing(cast(ImageTransform, train), cast(ImageTransform, evaluation), metadata)


class ImageView(Dataset[tuple[torch.Tensor, int]]):
    """Apply a view-specific transform while sharing one immutable image source."""

    def __init__(self, source: CIFAR10, indices: IndexArray, transform: ImageTransform) -> None:
        if source.train is not True or source.transform is not None or source.target_transform is not None:
            raise ValueError("Expected an untransformed official CIFAR-10 training source")
        if indices.dtype != np.dtype(np.int64) or indices.ndim != 1:
            raise ValueError("View indices must be a one-dimensional int64 array")
        if np.any(indices < 0) or np.any(indices >= len(source)):
            raise ValueError("View indices must stay within the training source")
        self.source = source
        self.indices = indices.copy()
        self.indices.setflags(write=False)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image, label = self.source[int(self.indices[index])]
        return self.transform(image), int(label)


@dataclass(frozen=True)
class DatasetViews:
    train: ImageView
    validation: ImageView
    train_subset: ImageView
    validation_subset: ImageView


def build_dataset_views(source: CIFAR10, indices: SplitIndices, preprocessing: Preprocessing) -> DatasetViews:
    """Keep evaluation augmentation separate; no official-test view is exposed."""
    if source.train is not True or source.transform is not None or source.target_transform is not None:
        raise ValueError("Expected an untransformed official CIFAR-10 training source")
    validate_split_indices(indices, source.targets)
    return DatasetViews(
        train=ImageView(source, indices.train, preprocessing.train),
        validation=ImageView(source, indices.validation, preprocessing.evaluation),
        train_subset=ImageView(source, indices.train_subset, preprocessing.evaluation),
        validation_subset=ImageView(source, indices.validation_subset, preprocessing.evaluation),
    )


def make_loader(
    dataset: Dataset[Sample], *, role: LoaderRole, batch_size: int,
    num_workers: int, pin_memory: bool, generators: LoaderGenerators,
) -> DataLoader[Sample]:
    """Create an epoch-restartable loader without consuming the global torch RNG."""
    if role not in _INDEX_NAMES:
        raise ValueError("Unsupported loader role; official test is excluded")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if type(num_workers) is not int or not 0 <= num_workers <= 4:
        raise ValueError("num_workers must be an integer between 0 and 4")
    sampler = RandomSampler(dataset, generator=generators.train_order) if role == "train" else None
    return DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, shuffle=False,
        num_workers=num_workers, drop_last=False, pin_memory=pin_memory,
        persistent_workers=False, worker_init_fn=seed_worker,
        generator=getattr(generators, f"{role}_workers"),
    )
