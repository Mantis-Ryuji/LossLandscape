"""Small CPU tests for saved-artifact trajectory animation rendering."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from landscape_exp.animation import (
    _HALO_INNER,
    _HALO_OUTER,
    _FrameRenderer,
    _new_animation_id,
    _save_gif,
    _select_runs,
    _styles,
    _trajectory_dimensions,
    _validate_animation_name,
    load_animation_inputs,
    render_animation_pair,
)
from landscape_exp.checkpoints import file_hash, read_json, write_json
from landscape_exp.config import load_config


class AnimationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.config_source = self.root / "phase1.yaml"
        self.config_source.write_bytes((REPOSITORY_ROOT / "configs/phase1.yaml").read_bytes())
        self.loaded = load_config(self.config_source, project_root=self.root)
        self.output = self.root / "artifacts"
        (self.output / "projections").mkdir(parents=True)
        (self.output / "surfaces").mkdir()
        self.projection = self._projection_fixture("fixture_projection")
        self.surface = self._surface_fixture("fixture_projection")

    @staticmethod
    def _metrics(
        run_id: str, segment_id: str, batch: int, seed: int, epoch: int,
    ) -> dict[str, object]:
        return {
            "run_id": run_id,
            "segment_id": segment_id,
            "epoch": epoch,
            "global_step": epoch * {64: 8, 256: 2, 1024: 1}[batch],
            "batch_size": batch,
            "seed": seed,
            "learning_rate": None if epoch == 0 else 1e-3,
            "gradient_norm": None if epoch == 0 else 2.0 / epoch,
            "train_subset_loss": 2.4 - 0.2 * epoch - batch / 4096,
            "train_subset_accuracy": 0.1 + 0.1 * epoch,
            "train_subset_samples": 1000,
            "val_loss": 2.5 - 0.2 * epoch - batch / 4096,
            "val_accuracy": 0.09 + 0.1 * epoch,
            "val_samples": 5000,
        }

    def _records(self) -> list[dict[str, object]]:
        records: list[dict[str, object]] = []
        for run_id, segment_id, batch, seed in self._run_specs():
            for epoch in range(3):
                metrics = self._metrics(run_id, segment_id, batch, seed, epoch)
                records.append({
                    "index": len(records),
                    "run_id": run_id,
                    "segment_id": segment_id,
                    "epoch": epoch,
                    "global_step": metrics["global_step"],
                    "metrics": metrics,
                })
        return records

    @staticmethod
    def _run_specs() -> list[tuple[str, str, int, int]]:
        return [
            (
                f"fixture/b{batch}_seed{seed}",
                f"segment_{batch}_seed{seed}",
                batch,
                seed,
            )
            for seed in range(3)
            for batch in (64, 256, 1024)
        ]

    @staticmethod
    def _save_arrays(directory: Path, arrays: dict[str, np.ndarray]) -> None:
        for name, array in arrays.items():
            with (directory / f"{name}.npy").open("xb") as handle:
                np.save(handle, array, allow_pickle=False)

    @staticmethod
    def _publish(directory: Path, kind: str, projection_id: str) -> None:
        files = {
            path.name: {"size_bytes": path.stat().st_size, "sha256": file_hash(path)}
            for path in directory.iterdir()
        }
        write_json(directory / "complete.json", {
            "schema_version": 1,
            "kind": kind,
            "projection_id": projection_id,
            "metadata_sha256": files["metadata.json"]["sha256"],
            "files": files,
        })

    def _projection_fixture(self, name: str) -> Path:
        directory = self.output / "projections" / name
        directory.mkdir()
        coordinates = np.array([
            point
            for run_index, _ in enumerate(self._run_specs())
            for point in (
                [-1.0, 0.0],
                [-0.3 + 0.04 * run_index, 0.45 - 0.06 * run_index],
                [0.45 + 0.04 * run_index, 0.75 - 0.10 * run_index],
            )
        ], dtype=np.float64)
        residuals = np.array([
            value
            for run_index, _ in enumerate(self._run_specs())
            for value in (0.0, 0.1 + 0.01 * run_index, 0.2 + 0.01 * run_index)
        ], dtype=np.float64)
        arrays = {
            "mean": np.arange(4, dtype=np.float64),
            "pc1": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            "pc2": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64),
            "coordinates": coordinates,
            "residuals": residuals,
            "eigenvalues": np.linspace(4.0, 0.0, len(coordinates), dtype=np.float64),
            "explained_variance_ratio": np.array([0.5, 0.25], dtype=np.float64),
        }
        self._save_arrays(directory, arrays)
        records = self._records()
        write_json(directory / "metadata.json", {
            "schema_version": 1,
            "kind": "common_pca_projection",
            "projection_id": name,
            "config_source_sha256": self.loaded.source_sha256,
            "effective_config_sha256": self.loaded.effective_sha256,
            "sample_count": len(records),
            "parameter_count": 4,
            "run_order": [run_id for run_id, _, _, _ in self._run_specs()],
            "common_epochs": [0, 1, 2],
            "checkpoints": records,
            "explained_variance_ratio": [0.5, 0.25],
            "arrays": {
                key: {"path": f"{key}.npy", "shape": list(value.shape), "dtype": "float64"}
                for key, value in arrays.items()
            },
        })
        self._publish(directory, "completed_projection", name)
        return directory

    def _surface_fixture(self, name: str) -> Path:
        directory = self.output / "surfaces" / name
        directory.mkdir()
        x_values = np.array([-1.5, 1.5], dtype=np.float64)
        y_values = np.array([-1.2, 1.2], dtype=np.float64)
        arrays = {
            "x_values": x_values,
            "y_values": y_values,
            "train_loss": np.array([[1.0, 1.6], [1.2, 2.0]], dtype=np.float64),
            "train_accuracy": np.array([[0.6, 0.4], [0.5, 0.3]], dtype=np.float64),
            "validation_loss": np.array([[1.1, 1.7], [1.3, 1.9]], dtype=np.float64),
            "validation_accuracy": np.array([[0.55, 0.35], [0.45, 0.25]], dtype=np.float64),
            "color_levels": np.linspace(1.0, 2.0, 21, dtype=np.float64),
        }
        self._save_arrays(directory, arrays)
        records = self._records()
        write_json(directory / "checkpoint_metrics.json", {
            "schema_version": 1,
            "kind": "actual_checkpoint_metrics",
            "projection_id": name,
            "train_scope": "fixed train subset with evaluation preprocessing",
            "validation_scope": "full validation split",
            "records": records,
        })
        projection_complete = self.projection / "complete.json"
        projection_manifest = read_json(projection_complete)
        write_json(directory / "metadata.json", {
            "schema_version": 1,
            "kind": "common_loss_surfaces",
            "projection_id": name,
            "config_source_sha256": self.loaded.source_sha256,
            "effective_config_sha256": self.loaded.effective_sha256,
            "projection": {
                "complete_sha256": file_hash(projection_complete),
                "metadata_sha256": projection_manifest["metadata_sha256"],
                "sample_count": len(records),
            },
            "grid": {"shape": [2, 2]},
            "subsets": {
                "train": {"samples": 1000},
                "validation": {"samples": 1000},
            },
            "evaluation": {
                "parameter_dtype": "torch.float32",
                "input_dtype": "torch.float32",
                "amp": False,
                "tf32": False,
            },
            "color_scale": {
                "intervals": 20,
                "shared_by": ["train", "validation"],
            },
            "arrays": {
                key: {"path": f"{key}.npy", "shape": list(value.shape), "dtype": "float64"}
                for key, value in arrays.items()
            },
        })
        self._publish(directory, "completed_loss_surfaces", name)
        return directory

    def _republish_surface(self) -> None:
        (self.surface / "complete.json").unlink()
        self._publish(self.surface, "completed_loss_surfaces", self.projection.name)

    def test_completed_sources_build_common_epoch_run_trajectories(self) -> None:
        inputs = load_animation_inputs(self.output, self.projection)
        self.assertEqual(inputs.epochs, (0, 1, 2))
        self.assertEqual(len(inputs.runs), 9)
        self.assertEqual(
            [run.label for run in inputs.runs[:3]],
            ["B64 seed0", "B256 seed0", "B1024 seed0"],
        )
        self.assertEqual(
            [run.color for run in inputs.runs[:3]],
            ["#D55E00", "#56B4E9", "#CC79A7"],
        )
        records = self._records()
        reversed_styles = _styles(
            [*records[3:6], *records[:3]],
            ["fixture/b256_seed0", "fixture/b64_seed0"],
        )
        self.assertEqual(reversed_styles["fixture/b64_seed0"][0], "#D55E00")
        self.assertEqual(reversed_styles["fixture/b256_seed0"][0], "#56B4E9")
        self.assertEqual([run.coordinates.shape for run in inputs.runs], [(3, 2)] * 9)
        self.assertEqual(inputs.explained_variance_ratio, (0.5, 0.25))
        self.assertEqual(inputs.train_metric_samples, 1000)
        self.assertEqual(inputs.validation_metric_samples, 5000)
        self.assertNotIsInstance(inputs.color_levels, np.memmap)
        self.assertNotIsInstance(inputs.runs[0].coordinates, np.memmap)

    def test_seed_selection_preserves_shared_inputs_and_global_styles(self) -> None:
        inputs = load_animation_inputs(self.output, self.projection)
        self.assertIs(_select_runs(inputs, None), inputs)

        selected = _select_runs(inputs, 1)
        self.assertEqual(
            [run.label for run in selected.runs],
            ["B64 seed1", "B256 seed1", "B1024 seed1"],
        )
        self.assertEqual([run.line_style for run in selected.runs], ["--"] * 3)
        self.assertEqual([run.marker for run in selected.runs], ["s"] * 3)
        self.assertIs(selected.x_values, inputs.x_values)
        self.assertIs(selected.y_values, inputs.y_values)
        self.assertIs(selected.losses, inputs.losses)
        self.assertIs(selected.color_levels, inputs.color_levels)
        self.assertEqual(selected.explained_variance_ratio, inputs.explained_variance_ratio)
        self.assertEqual(selected.epochs, inputs.epochs)

        for invalid in (-1, True):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                ValueError, "non-negative integer",
            ):
                _select_runs(inputs, invalid)
        with self.assertRaisesRegex(ValueError, "no runs for seed 3"):
            _select_runs(inputs, 3)

    def test_every_source_hash_is_verified_before_any_array_is_opened(self) -> None:
        with (self.projection / "mean.npy").open("ab") as handle:
            handle.write(b"corrupt")
        with patch(
            "landscape_exp.animation.np.load",
            side_effect=AssertionError("NumPy arrays must remain unopened until all hashes pass"),
        ):
            with self.assertRaisesRegex(ValueError, "size/hash mismatch: mean.npy"):
                load_animation_inputs(self.output, self.projection)

    def test_saved_actual_metrics_must_equal_projection_records(self) -> None:
        document = read_json(self.surface / "checkpoint_metrics.json")
        records = document["records"]
        self.assertIsInstance(records, list)
        records[1]["metrics"]["val_loss"] = 9.0
        (self.surface / "checkpoint_metrics.json").unlink()
        write_json(self.surface / "checkpoint_metrics.json", document)
        self._republish_surface()
        with self.assertRaisesRegex(ValueError, "differ from projection checkpoint records"):
            load_animation_inputs(self.output, self.projection)

    def test_gif_preserves_epoch_frames_timing_and_fixed_dimensions(self) -> None:
        inputs = load_animation_inputs(self.output, self.projection)
        destination = self.root / "trajectory.gif"
        size, height = _save_gif(inputs, "train", destination, width=480, colors=64)
        self.assertEqual(height, 320)
        self.assertGreater(size, 0)
        with Image.open(destination) as animation:
            self.assertEqual(animation.size, (480, 320))
            self.assertEqual(animation.n_frames, 3)
            palette = animation.getpalette()
            self.assertIsNotNone(palette)
            reserved = {
                tuple(palette[index:index + 3])
                for index in range(0, len(palette), 3)
            }
            self.assertTrue({
                (213, 94, 0),
                (86, 180, 233),
                tuple(int(_HALO_OUTER[index:index + 2], 16) for index in (1, 3, 5)),
                tuple(int(_HALO_INNER[index:index + 2], 16) for index in (1, 3, 5)),
            }.issubset(reserved))
            durations = []
            for index in range(animation.n_frames):
                animation.seek(index)
                durations.append(animation.info["duration"])
        self.assertEqual(durations, [200, 200, 1200])

    def test_paired_renderers_share_trajectory_axes_and_do_not_mutate_sources(self) -> None:
        inputs = load_animation_inputs(self.output, self.projection)
        before = np.asarray(inputs.runs[0].coordinates).copy()
        train = _FrameRenderer(inputs, "train", 480)
        validation = _FrameRenderer(inputs, "validation", 480)
        try:
            train_image = train.render(2)
            validation_image = validation.render(2)
            self.assertEqual(train_image.size, validation_image.size)
            self.assertEqual(train.axis.get_xlim(), validation.axis.get_xlim())
            self.assertEqual(train.axis.get_ylim(), validation.axis.get_ylim())
            self.assertEqual(len(train.lines[0].get_path_effects()), 3)
            self.assertEqual(train.lines[0].get_linewidth(), 0.9)
            self.assertEqual(train.points[0].get_markersize(), 4.0)
            self.assertEqual(len(train.table.get_celld()), 70)
            self.assertEqual(_trajectory_dimensions(1.0), {
                "line_core": 1.5,
                "line_white_halo": 2.5,
                "line_black_halo": 3.5,
                "current_marker_size": 7.0,
                "current_marker_edge": 0.9,
                "current_marker_black_halo": 2.8,
            })
            train_image.close()
            validation_image.close()
        finally:
            train.close()
            validation.close()
        np.testing.assert_array_equal(inputs.runs[0].coordinates, before)

    def test_render_records_seed_and_summary_selection_metadata(self) -> None:
        with patch("landscape_exp.animation._PROFILES", ((480, 64),)):
            selected = render_animation_pair(
                self.loaded,
                self.projection,
                animation_name="phase1_seed1_batch_compare",
                seed=1,
            )
            summary = render_animation_pair(
                self.loaded,
                self.projection,
                animation_name="phase1_all_runs_summary",
            )

        self.assertEqual(selected.selection_kind, "seed")
        self.assertEqual(selected.selected_seed, 1)
        self.assertEqual(selected.source_run_count, 9)
        self.assertEqual(selected.rendered_run_count, 3)
        selected_metadata = read_json(selected.directory / "metadata.json")
        self.assertEqual(selected_metadata["run_selection"], {
            "kind": "seed",
            "seed": 1,
            "source_run_count": 9,
            "rendered_run_count": 3,
        })
        self.assertEqual(len(selected_metadata["runs"]), 3)
        self.assertEqual({run["seed"] for run in selected_metadata["runs"]}, {1})

        self.assertEqual(summary.selection_kind, "all_runs")
        self.assertIsNone(summary.selected_seed)
        self.assertEqual(summary.source_run_count, 9)
        self.assertEqual(summary.rendered_run_count, 9)
        summary_metadata = read_json(summary.directory / "metadata.json")
        self.assertEqual(summary_metadata["run_selection"], {
            "kind": "all_runs",
            "seed": None,
            "source_run_count": 9,
            "rendered_run_count": 9,
        })
        self.assertEqual(len(summary_metadata["runs"]), 9)

    def test_animation_name_is_path_safe(self) -> None:
        self.assertEqual(_validate_animation_name("phase0_seed0_batch_compare"), "phase0_seed0_batch_compare")
        first = _new_animation_id("phase0_seed0_batch_compare")
        second = _new_animation_id("phase0_seed0_batch_compare")
        self.assertNotEqual(first, second)
        self.assertTrue(first.startswith("phase0_seed0_batch_compare_"))
        for value in ("../escape", "Uppercase", "con", "space name"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                _validate_animation_name(value)


if __name__ == "__main__":
    unittest.main()
