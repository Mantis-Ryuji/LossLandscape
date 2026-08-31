"""CPU-only RNG isolation and state restoration tests; no model or data download."""

from __future__ import annotations

import io
import random
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from landscape_exp.seeds import (
    LoaderGenerators, capture_random_state, derive_seed, preserve_random_state,
    restore_random_state, seed_global,
)


class SeedTests(unittest.TestCase):
    def setUp(self) -> None:
        self.previous = capture_random_state()
        self.addCleanup(restore_random_state, self.previous)
        seed_global(17)

    def test_namespaces_are_stable_and_separated(self) -> None:
        self.assertEqual(derive_seed(0, "train_order"), derive_seed(0, "train_order"))
        self.assertNotEqual(derive_seed(0, "train_order"), derive_seed(0, "train_workers"))
        self.assertNotEqual(derive_seed(0, "train_order"), derive_seed(1, "train_order"))
        self.assertLess(derive_seed(0, "train_order"), 2**63)
        with self.assertRaises(ValueError):
            derive_seed(True, "train_order")

    def test_restore_reproduces_python_numpy_and_torch(self) -> None:
        random.gauss(0, 1)
        np.random.normal()
        state = capture_random_state()
        expected_python = (random.random(), random.gauss(0, 1))
        expected_numpy = np.random.normal(size=6)
        expected_torch = torch.rand(6)
        restore_random_state(state)
        self.assertEqual((random.random(), random.gauss(0, 1)), expected_python)
        np.testing.assert_array_equal(np.random.normal(size=6), expected_numpy)
        torch.testing.assert_close(torch.rand(6), expected_torch, rtol=0, atol=0)

    def test_snapshot_supports_weights_only_loading(self) -> None:
        state = capture_random_state()
        buffer = io.BytesIO()
        torch.save(state, buffer)
        expected = torch.rand(4)
        buffer.seek(0)
        restored = torch.load(buffer, map_location="cpu", weights_only=True)
        restore_random_state(restored)
        torch.testing.assert_close(torch.rand(4), expected, rtol=0, atol=0)

    def test_evaluation_exception_restores_all_streams(self) -> None:
        generators = LoaderGenerators.from_seed(2)
        saved = capture_random_state()
        loader_saved = generators.state_dict()
        with self.assertRaisesRegex(RuntimeError, "evaluation failed"):
            with preserve_random_state(generators):
                random.random()
                np.random.random(5)
                torch.rand(5)
                torch.randperm(17, generator=generators.train_order)
                torch.rand(5, generator=generators.validation_workers)
                raise RuntimeError("evaluation failed")
        observed = (random.random(), float(np.random.random()), torch.rand(5))
        restore_random_state(saved)
        self.assertEqual(observed[0], random.random())
        self.assertEqual(observed[1], float(np.random.random()))
        torch.testing.assert_close(observed[2], torch.rand(5), rtol=0, atol=0)
        for key, value in loader_saved.items():
            torch.testing.assert_close(generators.state_dict()[key], value, rtol=0, atol=0)

    def test_loader_state_rejects_missing_stream_before_mutation(self) -> None:
        generators = LoaderGenerators.from_seed(0)
        before = generators.state_dict()
        broken = dict(before)
        del broken["validation_workers"]
        with self.assertRaises(ValueError):
            generators.load_state_dict(broken)
        for name, value in before.items():
            torch.testing.assert_close(generators.state_dict()[name], value, rtol=0, atol=0)

    def test_cpu_capture_does_not_initialize_cuda(self) -> None:
        with patch("torch.cuda.is_initialized", return_value=False), patch(
            "torch.cuda.get_rng_state_all", side_effect=AssertionError("CUDA must stay untouched")
        ):
            self.assertIsNone(capture_random_state()["torch_cuda"])


if __name__ == "__main__":
    unittest.main()
