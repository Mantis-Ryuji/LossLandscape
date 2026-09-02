"""Small CPU tests for the blocked common-PCA implementation."""

from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from landscape_exp.checkpoints import file_hash, write_json
from landscape_exp.projection import (
    _compatibility_record,
    compute_blocked_pca,
    load_analysis_checkpoint,
)


class ProjectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        mean = np.array([2.0, -1.0, 0.5, 4.0, -3.0, 1.0, 2.5], dtype=np.float64)
        direction_a = np.array([1.0, 0.0, 2.0, -1.0, 0.5, 0.0, 1.0], dtype=np.float64)
        direction_b = np.array([0.0, 1.0, -0.5, 0.0, 1.0, 2.0, -1.0], dtype=np.float64)
        first = np.array([-3.0, -1.0, 0.0, 1.0, 3.0, 0.0], dtype=np.float64)
        second = np.array([0.5, -1.0, 2.0, -1.5, 0.0, 0.0], dtype=np.float64)
        self.weights = (
            mean + first[:, None] * direction_a + second[:, None] * direction_b
        ).astype(np.float32)

    def compute(self, name: str, block: int):
        destination = self.root / name
        destination.mkdir()
        result = compute_blocked_pca(self.weights, destination, block_parameters=block)
        arrays = {
            key: np.load(destination / f"{key}.npy", allow_pickle=False)
            for key in ("mean", "pc1", "pc2", "coordinates", "residuals", "eigenvalues", "explained_variance_ratio")
        }
        return result, arrays

    def test_known_rank_two_trajectory_roundtrips_through_common_plane(self) -> None:
        result, arrays = self.compute("known", 3)
        basis = np.column_stack((arrays["pc1"], arrays["pc2"]))
        reconstructed = arrays["mean"] + arrays["coordinates"] @ basis.T
        np.testing.assert_allclose(reconstructed, self.weights.astype(np.float64), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(basis.T @ basis, np.eye(2), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(arrays["residuals"], 0.0, rtol=0, atol=1e-12)
        self.assertEqual(result.effective_rank, 2)
        self.assertAlmostEqual(sum(result.explained_variance_ratio), 1.0, places=12)
        for component in (arrays["pc1"], arrays["pc2"]):
            self.assertGreaterEqual(component[int(np.argmax(np.abs(component)))], 0.0)

    def test_block_size_does_not_change_the_defined_projection(self) -> None:
        first, arrays_a = self.compute("block2", 2)
        second, arrays_b = self.compute("block7", 7)
        self.assertEqual(first.effective_rank, second.effective_rank)
        for name in arrays_a:
            np.testing.assert_allclose(arrays_a[name], arrays_b[name], rtol=1e-11, atol=1e-11)

    def test_eigenvalues_and_coordinates_match_the_centered_gram_matrix(self) -> None:
        _, arrays = self.compute("gram", 4)
        centered = self.weights.astype(np.float64) - self.weights.astype(np.float64).mean(axis=0)
        expected = np.linalg.eigvalsh(centered @ centered.T)[::-1]
        np.testing.assert_allclose(arrays["eigenvalues"], expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(
            arrays["coordinates"] @ arrays["coordinates"].T,
            centered @ centered.T,
            rtol=1e-11,
            atol=1e-11,
        )

    def test_projection_compatibility_ignores_absolute_storage_paths(self) -> None:
        original = {
            "schema_version": 3,
            "experiment": {"name": "phase1", "phase": "phase1", "seed": 0},
            "paths": {
                "dataset_root": "C:/workspace/data",
                "output_root": "C:/workspace/artifacts",
                "init_checkpoint": "C:/workspace/artifacts/init/theta_0.pt",
                "scratch_root": "C:/workspace/artifacts/work",
            },
            "model": {"name": "fixture"},
            "training": {"batch_size": 64, "learning_rate": 1.0e-3},
            "augmentation": {"horizontal_flip": True},
            "split": {"split_seed": 1},
            "reproducibility": {"deterministic_algorithms": True},
            "evaluation": {"dtype": "float32"},
            "checkpoint": {"parameter_dtype": "float32"},
            "projection": {"solver": "gram_eigh"},
            "landscape": {"grid_size": 21},
            "logging": {"format": "csv"},
            "phase1": {"same_learning_rate": True},
        }
        relocated = copy.deepcopy(original)
        relocated["paths"] = {
            "dataset_root": "D:/LossLandscape/data",
            "output_root": "D:/LossLandscape/artifacts",
            "init_checkpoint": "D:/LossLandscape/artifacts/init/theta_0.pt",
            "scratch_root": "D:/LossLandscape/artifacts/work",
        }
        self.assertEqual(_compatibility_record(original), _compatibility_record(relocated))

        changed_recipe = copy.deepcopy(relocated)
        changed_recipe["training"] = {"batch_size": 64, "learning_rate": 2.0e-3}
        self.assertNotEqual(_compatibility_record(original), _compatibility_record(changed_recipe))

    def test_rank_one_and_existing_outputs_are_rejected(self) -> None:
        rank_one = np.arange(15, dtype=np.float32).reshape(5, 3)
        destination = self.root / "rank1"
        destination.mkdir()
        with self.assertRaisesRegex(ValueError, "rank below two"):
            compute_blocked_pca(rank_one, destination, block_parameters=2)

        existing = self.root / "existing"
        existing.mkdir()
        (existing / "mean.npy").write_bytes(b"do not replace")
        with self.assertRaisesRegex(FileExistsError, "overwrite"):
            compute_blocked_pca(self.weights, existing, block_parameters=2)
        self.assertEqual((existing / "mean.npy").read_bytes(), b"do not replace")

    def completed_epoch(self) -> Path:
        run_id, segment_id = "fixture/b64_seed0", "segment_fixture"
        directory = self.root / "run" / "segments" / segment_id / "epochs" / "epoch_0000"
        directory.mkdir(parents=True)
        torch.save({
            "schema_version": 1, "kind": "analysis", "epoch": 0, "global_step": 0,
            "contract_sha256": "a" * 64,
            "model_state": {"weight": torch.tensor([1.0, 2.0])},
        }, directory / "analysis.pt")
        (directory / "resume.pt").write_bytes(b"verified but not deserialized")
        write_json(directory / "metrics.json", {
            "epoch": 0, "global_step": 0, "run_id": run_id, "segment_id": segment_id,
        })
        write_json(directory / "metadata.json", {
            "epoch": 0, "global_step": 0, "contract_sha256": "a" * 64,
            "analysis_parameter_dtype": "torch.float32",
        })
        files = {
            name: {"size_bytes": (directory / name).stat().st_size, "sha256": file_hash(directory / name)}
            for name in ("analysis.pt", "resume.pt", "metrics.json", "metadata.json")
        }
        write_json(directory / "complete.json", {
            "schema_version": 1, "kind": "completed_epoch", "epoch": 0, "global_step": 0,
            "run_id": run_id, "segment_id": segment_id, "contract_sha256": "a" * 64,
            "files": files,
        })
        return directory

    def test_completed_analysis_is_verified_before_weights_only_loading(self) -> None:
        directory = self.completed_epoch()
        record, state = load_analysis_checkpoint(
            directory, expected_run_id="fixture/b64_seed0", expected_contract_sha256="a" * 64,
        )
        self.assertEqual(record.epoch, 0)
        self.assertEqual(record.analysis_sha256, file_hash(directory / "analysis.pt"))
        torch.testing.assert_close(state["weight"], torch.tensor([1.0, 2.0]), rtol=0, atol=0)

        (directory / "resume.pt").write_bytes(b"changed after completion")
        with self.assertRaisesRegex(ValueError, "size/hash mismatch: resume.pt"):
            load_analysis_checkpoint(
                directory, expected_run_id="fixture/b64_seed0", expected_contract_sha256="a" * 64,
            )


if __name__ == "__main__":
    unittest.main()
