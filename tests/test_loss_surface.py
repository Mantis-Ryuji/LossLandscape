"""Small CPU tests for shared-grid loss-surface evaluation and provenance."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from landscape_exp.checkpoints import file_hash, write_json
from landscape_exp.loss_surface import (
    common_color_scale, common_grid, evaluate_loss_surface, load_projection_artifact,
)


class PlaneClassifier(nn.Module):
    def __init__(self, *, fail: bool = False) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2, bias=False)
        self.fail = fail

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if self.fail:
            raise RuntimeError("fixture forward failure")
        return self.linear(values)


class LossSurfaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.points = torch.tensor([
            [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 1.0],
        ], dtype=torch.float32)
        self.labels = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
        loader = DataLoader(TensorDataset(self.points, self.labels), batch_size=3, shuffle=False)
        self.batches = tuple(loader)
        self.reference = torch.tensor([1.0, 0.0, 0.0, 1.0], dtype=torch.float64)
        self.direction_1 = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        self.direction_2 = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64)
        self.x_values = np.array([-0.5, 0.5], dtype=np.float64)
        self.y_values = np.array([-0.25, 0.25], dtype=np.float64)

    def test_common_grid_uses_fixed_margin_and_degenerate_fallback(self) -> None:
        coordinates = np.array([
            [-2.0, 4.0], [3.0, 4.0], [1.0, 4.0],
        ], dtype=np.float64)
        x_values, y_values = common_grid(coordinates, grid_size=5, margin_ratio=0.1)
        np.testing.assert_allclose(x_values, np.linspace(-2.5, 3.5, 5), rtol=0, atol=0)
        np.testing.assert_allclose(y_values, np.linspace(4.0 - 1e-6, 4.0 + 1e-6, 5), rtol=0, atol=0)

    def test_surface_matches_direct_fp32_evaluation_and_restores_model(self) -> None:
        model = PlaneClassifier()
        with torch.no_grad():
            model.linear.weight.copy_(self.reference.float().reshape(2, 2))
        model.train()
        before = model.linear.weight.detach().clone()
        surface = evaluate_loss_surface(
            model, self.reference, self.direction_1, self.direction_2,
            self.x_values, self.y_values, self.batches, torch.device("cpu"),
        )
        expected_loss = np.empty((2, 2), dtype=np.float64)
        expected_accuracy = np.empty((2, 2), dtype=np.float64)
        for y_index, y_value in enumerate(self.y_values):
            for x_index, x_value in enumerate(self.x_values):
                vector = (
                    self.reference
                    + float(x_value) * self.direction_1
                    + float(y_value) * self.direction_2
                ).float()
                logits = F.linear(self.points, vector.reshape(2, 2))
                expected_loss[y_index, x_index] = float(F.cross_entropy(logits, self.labels).item())
                expected_accuracy[y_index, x_index] = float(
                    (logits.argmax(dim=1) == self.labels).float().mean().item()
                )
        np.testing.assert_allclose(surface.loss, expected_loss, rtol=0, atol=1e-7)
        np.testing.assert_allclose(surface.accuracy, expected_accuracy, rtol=0, atol=0)
        self.assertEqual(surface.samples, len(self.labels))
        torch.testing.assert_close(model.linear.weight, before, rtol=0, atol=0)
        self.assertTrue(model.training)

    def test_surface_failure_restores_original_parameters_and_mode(self) -> None:
        model = PlaneClassifier(fail=True)
        before = model.linear.weight.detach().clone()
        model.train()
        with self.assertRaisesRegex(RuntimeError, "fixture forward failure"):
            evaluate_loss_surface(
                model, self.reference, self.direction_1, self.direction_2,
                self.x_values, self.y_values, self.batches, torch.device("cpu"),
            )
        torch.testing.assert_close(model.linear.weight, before, rtol=0, atol=0)
        self.assertTrue(model.training)

    def test_common_color_scale_covers_both_grids_and_constant_loss(self) -> None:
        train = np.array([[3.0, 1.0], [2.0, 4.0]], dtype=np.float64)
        validation = np.array([[0.5, 5.0], [1.5, 2.5]], dtype=np.float64)
        scale = common_color_scale(train, validation)
        self.assertEqual(scale.intervals, 20)
        self.assertEqual(len(scale.levels), 21)
        self.assertEqual(scale.raw_minimum, 0.5)
        self.assertEqual(scale.raw_maximum, 5.0)
        self.assertEqual(scale.levels[0], 0.5)
        self.assertEqual(scale.levels[-1], 5.0)

        constant = common_color_scale(
            np.full((2, 2), 2.0, dtype=np.float64),
            np.full((2, 2), 2.0, dtype=np.float64),
        )
        self.assertEqual(constant.raw_minimum, constant.raw_maximum)
        self.assertLess(constant.display_minimum, 2.0)
        self.assertGreater(constant.display_maximum, 2.0)

    def projection_fixture(self, name: str) -> Path:
        directory = self.root / name
        directory.mkdir()
        arrays = {
            "mean": np.arange(4, dtype=np.float64),
            "pc1": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            "pc2": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64),
            "coordinates": np.array([[0.0, 0.0], [1.0, -1.0], [2.0, 1.0]], dtype=np.float64),
            "residuals": np.zeros(3, dtype=np.float64),
            "eigenvalues": np.array([2.0, 1.0, 0.0], dtype=np.float64),
            "explained_variance_ratio": np.array([2 / 3, 1 / 3], dtype=np.float64),
        }
        for key, value in arrays.items():
            with (directory / f"{key}.npy").open("xb") as handle:
                np.save(handle, value, allow_pickle=False)
        write_json(directory / "metadata.json", {
            "schema_version": 1,
            "kind": "common_pca_projection",
            "projection_id": name,
            "sample_count": 3,
            "parameter_count": 4,
            "arrays": {
                key: {"path": f"{key}.npy", "shape": list(value.shape), "dtype": "float64"}
                for key, value in arrays.items()
            },
        })
        files = {
            path.name: {"size_bytes": path.stat().st_size, "sha256": file_hash(path)}
            for path in directory.iterdir()
        }
        write_json(directory / "complete.json", {
            "schema_version": 1,
            "kind": "completed_projection",
            "projection_id": name,
            "metadata_sha256": files["metadata.json"]["sha256"],
            "files": files,
        })
        return directory

    def test_projection_artifact_verifies_every_hash_before_array_loading(self) -> None:
        directory = self.projection_fixture("projection_fixture")
        with (directory / "mean.npy").open("ab") as handle:
            handle.write(b"corrupt")
        with patch(
            "landscape_exp.loss_surface.np.load",
            side_effect=AssertionError("arrays must not open before every hash is verified"),
        ):
            with self.assertRaisesRegex(ValueError, "size/hash mismatch: mean.npy"):
                load_projection_artifact(directory)


if __name__ == "__main__":
    unittest.main()
