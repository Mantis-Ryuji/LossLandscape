"""Tiny CPU fixtures for stratification, preprocessing and epoch-boundary loaders."""

from __future__ import annotations

import random
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from landscape_exp.config import load_config
from landscape_exp.data import (
    SplitSpec, build_dataset_views, build_preprocessing, create_split_indices,
    load_cifar10_training, load_split_indices, make_loader,
    prepare_cifar10_splits, save_split_indices, validate_split_indices,
)
from landscape_exp.seeds import (
    LoaderGenerators, capture_random_state, preserve_random_state,
    restore_random_state, seed_global,
)


class IndexDataset(Dataset[int]):
    def __len__(self) -> int:
        return 23

    def __getitem__(self, index: int) -> int:
        return index


class RandomDataset(Dataset[tuple[int, float, float, float]]):
    """Top-level class so Windows spawn workers can reconstruct the fixture."""

    def __len__(self) -> int:
        return 9

    def __getitem__(self, index: int) -> tuple[int, float, float, float]:
        return index, random.random(), float(np.random.random()), float(torch.rand(()))


class TinyCifar(CIFAR10):
    """Duck-compatible image source without calling the downloading constructor."""

    def __init__(self, targets: list[int]) -> None:
        self.targets = targets
        self.train = True
        self.transform = None
        self.target_transform = None

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[Image.Image, int]:
        return Image.new("RGB", (32, 32), (128, 64, 32)), self.targets[index]


class DataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = SplitSpec(2, 8, 2, 2, 20260831, 20260831)
        self.targets = [0] * 10 + [1] * 10
        self.previous = capture_random_state()
        self.addCleanup(restore_random_state, self.previous)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "split.npz"

    def test_class_balance_and_no_split_leakage(self) -> None:
        split = create_split_indices(self.targets, self.spec)
        self.assertEqual(len(split.train), 16)
        self.assertEqual(len(split.validation), 4)
        self.assertEqual(set(split.train) | set(split.validation), set(range(20)))
        self.assertFalse(set(split.train) & set(split.validation))
        for name in ("train", "validation"):
            self.assertTrue(set(getattr(split, f"{name}_subset")) <= set(getattr(split, name)))
            counts = np.bincount(np.asarray(self.targets)[getattr(split, f"{name}_subset")])
            np.testing.assert_array_equal(counts, [2, 2])
        self.assertFalse(split.train.flags.writeable)

    def test_split_does_not_consume_global_randomness(self) -> None:
        seed_global(7)
        before = capture_random_state()
        first = create_split_indices(self.targets, self.spec)
        after = (random.random(), np.random.random(), torch.rand(1))
        restore_random_state(before)
        self.assertEqual(after[0], random.random())
        self.assertEqual(after[1], np.random.random())
        torch.testing.assert_close(after[2], torch.rand(1), rtol=0, atol=0)
        second = create_split_indices(self.targets, self.spec)
        np.testing.assert_array_equal(first.train, second.train)
        np.testing.assert_array_equal(first.validation_subset, second.validation_subset)

    def test_impossible_counts_or_noninteger_labels_fail(self) -> None:
        for spec, targets in (
            (replace(self.spec, subset_per_class=3), self.targets),
            (self.spec, [0.0] * 10 + [1.0] * 10),
            (self.spec, [0] * 11 + [1] * 9),
        ):
            with self.subTest(spec=spec, targets=targets):
                with self.assertRaises(ValueError):
                    create_split_indices(targets, spec)

    def test_roundtrip_reuse_and_existing_file_protection(self) -> None:
        split = create_split_indices(self.targets, self.spec)
        save_split_indices(self.path, split, self.targets)
        original = self.path.read_bytes()
        restored = load_split_indices(self.path, self.targets, self.spec)
        np.testing.assert_array_equal(restored.train, split.train)
        with self.assertRaises(FileExistsError):
            save_split_indices(self.path, split, self.targets)
        self.assertEqual(self.path.read_bytes(), original)
        with self.assertRaises(ValueError):
            load_split_indices(self.path, self.targets[::-1], self.spec)
        with self.assertRaises(ValueError):
            load_split_indices(self.path, self.targets, replace(self.spec, subset_seed=3))

    def test_partial_and_corrupted_splits_are_not_reused(self) -> None:
        split = create_split_indices(self.targets, self.spec)
        save_split_indices(self.path, split, self.targets)
        with self.path.open("ab") as handle:
            handle.write(b"corrupt")
        with self.assertRaisesRegex(ValueError, "hash"):
            load_split_indices(self.path, self.targets, self.spec)
        partial = Path(self.temp.name) / "partial.npz"
        partial.write_bytes(b"unfinished")
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            load_split_indices(partial, self.targets, self.spec)
        with self.assertRaises(FileExistsError):
            save_split_indices(partial, split, self.targets)
        self.assertEqual(partial.read_bytes(), b"unfinished")

    def test_duplicate_or_cross_split_indices_are_rejected(self) -> None:
        split = create_split_indices(self.targets, self.spec)
        bad_train = split.train.copy()
        bad_train[1] = bad_train[0]
        with self.assertRaises(ValueError):
            validate_split_indices(replace(split, train=bad_train), self.targets)
        with self.assertRaises(ValueError):
            validate_split_indices(replace(split, train_subset=split.validation_subset), self.targets)

    def test_official_test_and_automatic_download_are_excluded(self) -> None:
        source = TinyCifar(self.targets)
        source.train = False
        config = load_config("configs/phase0.yaml").config
        with self.assertRaisesRegex(ValueError, "official.*test"):
            prepare_cifar10_splits(config, source)
        with patch("landscape_exp.data.CIFAR10") as constructor:
            load_cifar10_training(Path(self.temp.name))
            constructor.assert_called_once_with(
                root=str(Path(self.temp.name)), train=True, transform=None, download=False
            )

    def test_evaluation_preprocessing_is_repeatable_and_normalized(self) -> None:
        processing = build_preprocessing({
            "input_size": (3, 224, 224), "interpolation": "bicubic",
            "mean": (0.5, 0.5, 0.5), "std": (0.25, 0.25, 0.25),
            "crop_pct": 0.875, "crop_mode": "center",
        })
        source = TinyCifar(self.targets)
        split = create_split_indices(self.targets, self.spec)
        views = build_dataset_views(source, split, processing)
        first, label = views.validation[0]
        second, _ = views.validation[0]
        self.assertEqual(first.shape, (3, 224, 224))
        self.assertEqual(first.dtype, torch.float32)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        expected = (torch.tensor([128, 64, 32]) / 255 - 0.5) / 0.25
        torch.testing.assert_close(first[:, 0, 0], expected)
        self.assertEqual(label, self.targets[int(split.validation[0])])
        self.assertIs(views.train.source, views.validation_subset.source)
        self.assertIs(views.validation.transform, views.train_subset.transform)
        self.assertEqual(processing.metadata["resize_size"], 256)
        source.train = False
        with self.assertRaises(ValueError):
            build_dataset_views(source, split, processing)

    def test_batch_size_does_not_change_order_or_drop_the_tail(self) -> None:
        orders = []
        for batch_size in (3, 7):
            loader = make_loader(IndexDataset(), role="train", batch_size=batch_size,
                                 num_workers=0, pin_memory=False, generators=LoaderGenerators.from_seed(0))
            orders.append([index for batch in loader for index in batch.tolist()])
        self.assertEqual(orders[0], orders[1])
        self.assertEqual(sorted(orders[0]), list(range(23)))

    def test_evaluation_iteration_preserves_the_next_training_order(self) -> None:
        generators = LoaderGenerators.from_seed(0)
        saved = generators.state_dict()
        with preserve_random_state(generators):
            evaluation = make_loader(RandomDataset(), role="validation", batch_size=3,
                                     num_workers=0, pin_memory=False, generators=generators)
            list(evaluation)
        for key, tensor in saved.items():
            torch.testing.assert_close(generators.state_dict()[key], tensor, rtol=0, atol=0)

    def test_worker_restart_reproduces_next_epoch_order_and_augmentation(self) -> None:
        generators = LoaderGenerators.from_seed(2)

        def epoch_values(streams: LoaderGenerators) -> list[list[list[float]]]:
            loader = make_loader(RandomDataset(), role="train", batch_size=4,
                                 num_workers=1, pin_memory=False, generators=streams)
            return [[column.tolist() for column in batch] for batch in loader]

        epoch_values(generators)
        saved = generators.state_dict()
        expected = epoch_values(generators)
        restored = LoaderGenerators.from_seed(0)
        restored.load_state_dict(saved)
        self.assertEqual(epoch_values(restored), expected)


if __name__ == "__main__":
    unittest.main()
