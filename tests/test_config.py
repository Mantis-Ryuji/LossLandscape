"""User-run tests of config rejection, provenance and artifact preservation."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from landscape_exp.config import ConfigError, load_config, prepare_run


class ConfigurationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.phase1 = (REPOSITORY_ROOT / "configs/phase1.yaml").read_text(encoding="utf-8")

    def write_config(self, text: str) -> Path:
        path = self.root / "nested" / "experiment.yaml"
        path.parent.mkdir(exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def test_phase0_preserves_the_full_schedule(self) -> None:
        text = (REPOSITORY_ROOT / "configs/phase0.yaml").read_text(encoding="utf-8")
        loaded = load_config(self.write_config(text), project_root=self.root)
        self.assertEqual(loaded.config.training.epochs, 100)
        self.assertEqual(loaded.config.training.warmup_epochs, 0)
        self.assertEqual(loaded.config.training.scheduler, "constant")
        self.assertEqual(loaded.config.training.learning_rate, 1e-3)
        self.assertEqual(loaded.config.end_epoch, 5)
        phase1 = load_config(self.write_config(self.phase1), project_root=self.root).config
        self.assertEqual(phase1.end_epoch, 100)
        for field in ("optimizer", "scheduler", "epochs", "learning_rate", "warmup_epochs", "weight_decay"):
            self.assertEqual(getattr(loaded.config.training, field), getattr(phase1.training, field))
        self.assertFalse((self.root / "artifacts").exists())

    def test_paths_use_project_root_not_yaml_parent(self) -> None:
        loaded = load_config(self.write_config(self.phase1), project_root=self.root)
        self.assertEqual(loaded.config.paths.dataset_root, self.root / "data")
        self.assertEqual(
            loaded.config.run_directory,
            self.root / "artifacts" / "runs" / "phase1" / "b64_seed0",
        )
        self.assertFalse(loaded.config.paths.output_root.exists())

    def test_overrides_are_effective_but_source_bytes_are_preserved(self) -> None:
        source = self.write_config(self.phase1)
        loaded = load_config(source, project_root=self.root, batch_size=256, seed=2, name="phase1_repeat")
        destination = prepare_run(loaded)
        self.assertEqual(loaded.config.run_id, "phase1_repeat/b256_seed2")
        self.assertEqual((destination / "source.yaml").read_bytes(), source.read_bytes())
        effective = json.loads((destination / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(effective["training"]["batch_size"], 256)
        self.assertEqual(effective["training"]["microbatch_size"], 64)
        self.assertEqual(loaded.config.training.accumulation_steps, 4)
        self.assertEqual(effective["experiment"]["seed"], 2)
        manifest = json.loads((destination / "prepared.json").read_text(encoding="utf-8"))
        self.assertEqual(
            manifest["effective_sha256"],
            hashlib.sha256((destination / "config.json").read_bytes()).hexdigest(),
        )
        self.assertEqual(
            manifest["source_sha256"],
            hashlib.sha256((destination / "source.yaml").read_bytes()).hexdigest(),
        )

    def test_existing_run_is_not_overwritten(self) -> None:
        loaded = load_config(self.write_config(self.phase1), project_root=self.root)
        destination = prepare_run(loaded)
        original = (destination / "config.json").read_bytes()
        with self.assertRaises(FileExistsError):
            prepare_run(loaded)
        self.assertEqual((destination / "config.json").read_bytes(), original)

    def test_partial_run_is_not_repaired_or_deleted(self) -> None:
        loaded = load_config(self.write_config(self.phase1), project_root=self.root)
        loaded.config.run_directory.mkdir(parents=True)
        marker = loaded.config.run_directory / "incomplete.txt"
        marker.write_text("keep", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            prepare_run(loaded)
        self.assertEqual(marker.read_text(encoding="utf-8"), "keep")
        self.assertFalse((marker.parent / "prepared.json").exists())

    def test_invalid_inputs_are_rejected_before_creating_artifacts(self) -> None:
        cases = {
            "unknown": self.phase1 + "misspelled_option: true\n",
            "missing": self.phase1.replace("  save_resume: true\n", ""),
            "duplicate": self.phase1.replace("  seed: 0", "  seed: 0\n  seed: 1"),
            "boolean_seed": self.phase1.replace("  seed: 0", "  seed: true"),
            "nonfinite": self.phase1.replace("1.0e-3", ".nan"),
            "legacy_horizon": self.phase1.replace("epochs: 100", "epochs: 50"),
            "legacy_scheduler": self.phase1.replace("scheduler: constant", "scheduler: cosine"),
            "nonzero_warmup": self.phase1.replace("warmup_epochs: 0", "warmup_epochs: 5"),
            "different_lr": self.phase1.replace("1.0e-3", "1.0e-4"),
            "reserved_name": self.phase1.replace("name: phase1", "name: con"),
            "path_name": self.phase1.replace("name: phase1", "name: ../escape"),
            "raw_data_overlap": self.phase1.replace("output_root: ./artifacts", "output_root: ./data/runs"),
            "raw_scratch_overlap": self.phase1.replace("scratch_root: ./artifacts/work", "scratch_root: ./data/work"),
            "integer_interval": self.phase1.replace("checkpoint_interval_epochs: 1", "checkpoint_interval_epochs: 1.0"),
            "gif_limit": self.phase1.replace("max_file_size_mb: 3.0", "max_file_size_mb: 4.0"),
            "lost_validation": self.phase1.replace("show_validation_loss: true", "show_validation_loss: false"),
            "old_batch_set": self.phase1.replace("[64, 256, 1024]", "[16, 64, 256]"),
            "missing_microbatch": self.phase1.replace("  microbatch_size: 64\n", ""),
            "zero_microbatch": self.phase1.replace("microbatch_size: 64", "microbatch_size: 0"),
            "different_microbatch": self.phase1.replace("microbatch_size: 64", "microbatch_size: 32"),
            "boolean_microbatch": self.phase1.replace("microbatch_size: 64", "microbatch_size: true"),
            "legacy_schema": self.phase1.replace("schema_version: 3", "schema_version: 2"),
        }
        for name, source in cases.items():
            with self.subTest(name=name):
                with self.assertRaises(ConfigError):
                    load_config(self.write_config(source), project_root=self.root)
                self.assertFalse((self.root / "artifacts").exists())

    def test_phase0_cannot_become_a_different_batch_comparison(self) -> None:
        text = (REPOSITORY_ROOT / "configs/phase0.yaml").read_text(encoding="utf-8")
        with self.assertRaises(ConfigError):
            load_config(self.write_config(text), project_root=self.root, batch_size=256)

    def test_effective_batches_derive_accumulation_without_changing_lr(self) -> None:
        source = self.write_config(self.phase1)
        for batch, accumulation in ((64, 1), (256, 4), (1024, 16)):
            with self.subTest(batch=batch):
                config = load_config(source, project_root=self.root, batch_size=batch).config
                self.assertEqual(config.training.microbatch_size, 64)
                self.assertEqual(config.training.accumulation_steps, accumulation)
                self.assertEqual(config.training.learning_rate, 1e-3)
                self.assertEqual(config.run_id, f"phase1/b{batch}_seed0")
        for batch in (16, 128, 512, True):
            with self.subTest(batch=batch), self.assertRaises(ConfigError):
                load_config(source, project_root=self.root, batch_size=batch)
        self.assertFalse((self.root / "artifacts").exists())

    def test_boolean_override_is_not_an_integer_seed(self) -> None:
        with self.assertRaises(ConfigError):
            load_config(self.write_config(self.phase1), project_root=self.root, seed=True)

    def test_scratch_contract_rejects_pretrained_and_legacy_model_settings(self) -> None:
        for text in (
            self.phase1.replace("initialization: scratch", "initialization: pretrained"),
            self.phase1.replace("name: convnextv2_tiny", "name: convnextv2_tiny.fcmae"),
            self.phase1.replace("name: convnextv2_tiny", "name: vit_small_patch16_224.dino"),
            self.phase1.replace("schema_version: 3", "schema_version: 1"),
            self.phase1.replace("init_seed: 0", "init_seed: 1"),
        ):
            with self.subTest(text=text):
                with self.assertRaises(ConfigError):
                    load_config(self.write_config(text), project_root=self.root)
                self.assertFalse((self.root / "artifacts").exists())


if __name__ == "__main__":
    unittest.main()
