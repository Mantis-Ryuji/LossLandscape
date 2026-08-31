"""CPU tests with a tiny substituted backbone; never fetch pretrained weights."""

from __future__ import annotations

import hashlib
import io
import json
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
from torch import nn


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from landscape_exp.config import load_config
from landscape_exp.models import (
    create_initial_checkpoint, load_initial_checkpoint, prepare_initial_checkpoint,
)
from landscape_exp.seeds import (
    capture_random_state, restore_random_state, seed_global,
)


class TinyClassifier(nn.Module):
    """A small timm-shaped fixture with both parameters and persistent buffers."""

    def __init__(self, num_classes: int, width: int = 4) -> None:
        super().__init__()
        self.embed_dim = width
        self.backbone = nn.Linear(3, width)
        self.head = nn.Linear(width, num_classes) if num_classes else nn.Identity()
        self.register_buffer("counter", torch.tensor(7, dtype=torch.int64))
        self.register_buffer("scale", torch.tensor(0.5, dtype=torch.float32))
        self.reset_calls = 0
        self.pretrained_cfg = {
            "architecture": "convnextv2_tiny", "tag": "fcmae", "num_classes": 0,
            "input_size": (3, 224, 224), "interpolation": "bicubic",
            "mean": (0.5, 0.5, 0.5), "std": (0.25, 0.25, 0.25),
            "crop_pct": 0.875, "crop_mode": "center",
            "url": "https://example.invalid/tiny-fixture",
        }

    def get_classifier(self) -> nn.Module:
        return self.head

    def reset_classifier(self, num_classes: int) -> None:
        self.reset_calls += 1
        self.head = nn.Linear(self.embed_dim, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(images.mean(dim=(2, 3))) * self.scale)


class InitialModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.previous = capture_random_state()
        self.addCleanup(restore_random_state, self.previous)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        source = self.root / "phase1.yaml"
        source.write_bytes((REPOSITORY_ROOT / "configs/phase1.yaml").read_bytes())
        self.loaded = load_config(source, project_root=self.root)
        self.path = self.loaded.config.paths.init_checkpoint
        self.metadata_path = self.path.with_suffix(".json")
        replacement = patch("landscape_exp.models.timm.create_model", side_effect=self.make_model)
        self.factory = replacement.start()
        self.addCleanup(replacement.stop)

    def make_model(self, name: str, *, pretrained: bool, num_classes: int) -> TinyClassifier:
        self.assertEqual(name, "convnextv2_tiny")
        self.assertFalse(pretrained, "Phase 0/1 must never request pretrained weights")
        model = TinyClassifier(num_classes)
        return model

    def test_creation_records_fp32_parameters_buffers_and_preprocessing(self) -> None:
        with patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA must remain untouched")):
            initial = create_initial_checkpoint(self.loaded)
        self.factory.assert_called_once_with("convnextv2_tiny", pretrained=False, num_classes=10)
        self.assertEqual(initial.model.reset_calls, 0)
        self.assertEqual(initial.metadata["initialization"]["mode"], "scratch")
        self.assertFalse(initial.metadata["initialization"]["pretrained"])
        self.assertIsNone(initial.metadata["pretrained_reference"])
        self.assertFalse(initial.model.training)
        for tensor in initial.model.parameters():
            self.assertEqual(tensor.dtype, torch.float32)
            self.assertEqual(tensor.device.type, "cpu")
            self.assertTrue(tensor.requires_grad)
        payload = torch.load(self.path, map_location="cpu", weights_only=True)
        self.assertEqual(payload["model_state"]["counter"].dtype, torch.int64)
        self.assertEqual(payload["model_state"]["counter"].item(), 7)
        parameter_names = {item["name"] for item in initial.metadata["parameter_spec"]}
        buffer_names = {item["name"] for item in initial.metadata["buffer_spec"]}
        self.assertNotIn("counter", parameter_names)
        self.assertEqual(buffer_names, {"counter", "scale"})
        self.assertEqual(initial.metadata["epoch"], 0)
        self.assertEqual(initial.metadata["global_step"], 0)
        self.assertEqual(initial.metadata["preprocessing"]["resize_size"], 256)
        self.assertEqual(initial.checkpoint_sha256, hashlib.sha256(self.path.read_bytes()).hexdigest())

    def test_all_batch_seed_runs_restore_the_exact_shared_model(self) -> None:
        initial = create_initial_checkpoint(self.loaded)
        self.factory.reset_mock()
        images = initial.preprocessing.evaluation(Image.new("RGB", (32, 32), (128, 64, 32))).unsqueeze(0)
        with torch.no_grad():
            expected = initial.model(images)
        for batch_size in (64, 256, 1024):
            for seed in (0, 1, 2):
                with self.subTest(batch_size=batch_size, seed=seed):
                    run = load_config(self.loaded.source_path, project_root=self.root, batch_size=batch_size, seed=seed)
                    restored = load_initial_checkpoint(run.config)
                    self.assertEqual(restored.checkpoint_sha256, initial.checkpoint_sha256)
                    self.assertFalse(restored.model.training)
                    for key, tensor in initial.model.state_dict().items():
                        torch.testing.assert_close(restored.model.state_dict()[key], tensor, rtol=0, atol=0)
                    with torch.no_grad():
                        torch.testing.assert_close(restored.model(images), expected, rtol=0, atol=0)
        self.assertEqual(self.factory.call_count, 9)
        self.assertTrue(all(call.kwargs == {"pretrained": False, "num_classes": 10}
                            for call in self.factory.call_args_list))

    def test_initial_creation_and_restore_preserve_rng_streams(self) -> None:
        seed_global(123)
        saved = capture_random_state()
        first = create_initial_checkpoint(self.loaded)
        load_initial_checkpoint(self.loaded.config)
        actual = (random.random(), np.random.random(4), torch.rand(4))
        restore_random_state(saved)
        self.assertEqual(actual[0], random.random())
        np.testing.assert_array_equal(actual[1], np.random.random(4))
        torch.testing.assert_close(actual[2], torch.rand(4), rtol=0, atol=0)
        other_config = replace(self.loaded.config, paths=replace(
            self.loaded.config.paths, init_checkpoint=self.root / "another_init" / "theta_0.pt"
        ), experiment=replace(self.loaded.config.experiment, seed=2))
        seed_global(999)
        second = create_initial_checkpoint(replace(self.loaded, config=other_config))
        for key, tensor in first.model.state_dict().items():
            torch.testing.assert_close(second.model.state_dict()[key], tensor, rtol=0, atol=0)

    def test_existing_initial_model_is_verified_not_overwritten(self) -> None:
        create_initial_checkpoint(self.loaded)
        original = (self.path.read_bytes(), self.metadata_path.read_bytes())
        self.factory.reset_mock()
        with self.assertRaises(FileExistsError):
            create_initial_checkpoint(self.loaded)
        self.factory.assert_not_called()
        prepare_initial_checkpoint(self.loaded)
        self.factory.assert_called_once_with("convnextv2_tiny", pretrained=False, num_classes=10)
        self.assertEqual((self.path.read_bytes(), self.metadata_path.read_bytes()), original)

    def test_partial_records_fail_before_any_model_construction(self) -> None:
        for filename in ("theta_0.pt", "theta_0.json"):
            with self.subTest(filename=filename):
                folder = self.root / filename.replace(".", "_")
                folder.mkdir()
                partial = folder / filename
                partial.write_bytes(b"unfinished")
                config = replace(self.loaded.config, paths=replace(
                    self.loaded.config.paths, init_checkpoint=folder / "theta_0.pt"
                ))
                with self.assertRaisesRegex(ValueError, "Incomplete"):
                    prepare_initial_checkpoint(replace(self.loaded, config=config))
                self.assertEqual(partial.read_bytes(), b"unfinished")
                self.assertEqual(len(list(folder.iterdir())), 1)
        self.factory.assert_not_called()

    def test_scratch_construction_failure_does_not_retry_or_create_artifacts(self) -> None:
        saved = capture_random_state()

        def fail_loading(name: str, *, pretrained: bool, num_classes: int) -> nn.Module:
            random.random()
            np.random.random()
            torch.rand(3)
            self.assertFalse(pretrained)
            raise OSError("scratch construction failed")

        self.factory.side_effect = fail_loading
        with self.assertRaisesRegex(OSError, "scratch construction failed"):
            create_initial_checkpoint(self.loaded)
        self.assertFalse(self.path.parent.exists())
        self.factory.assert_called_once_with("convnextv2_tiny", pretrained=False, num_classes=10)
        actual = (random.random(), float(np.random.random()), torch.rand(3))
        restore_random_state(saved)
        self.assertEqual(actual[0], random.random())
        self.assertEqual(actual[1], float(np.random.random()))
        torch.testing.assert_close(actual[2], torch.rand(3), rtol=0, atol=0)

    def test_corrupted_checkpoint_is_rejected_before_loading(self) -> None:
        create_initial_checkpoint(self.loaded)
        damaged = bytearray(self.path.read_bytes())
        damaged[len(damaged) // 2] ^= 1
        self.path.write_bytes(damaged)
        self.factory.reset_mock()
        with patch("landscape_exp.models.torch.load", side_effect=AssertionError("Hash must be checked first")):
            with self.assertRaisesRegex(ValueError, "hash"):
                prepare_initial_checkpoint(self.loaded)
        self.factory.assert_not_called()
        self.assertEqual(self.path.read_bytes(), damaged)

    def test_incompatible_model_runtime_or_metadata_is_rejected(self) -> None:
        create_initial_checkpoint(self.loaded)
        original = self.metadata_path.read_text(encoding="utf-8")
        self.factory.reset_mock()
        changes = (
            ("schema_version", True),
            ("model", {"name": "different_model"}),
            ("runtime", {"torch": "different_version"}),
            ("initialization", {"mode": "pretrained", "seed": 0}),
            ("pretrained_reference", {"tag": "fcmae"}),
            ("created_at_utc", "mismatched embedded metadata"),
        )
        for key, value in changes:
            with self.subTest(key=key):
                metadata = json.loads(original)
                metadata[key] = value
                self.metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_initial_checkpoint(self.loaded.config)
        self.factory.assert_not_called()

    def test_backbone_and_head_use_the_same_shared_default_initialization(self) -> None:
        initial = create_initial_checkpoint(self.loaded)
        seed_global(self.loaded.config.model.init_seed)
        expected = TinyClassifier(10)
        for key, tensor in expected.state_dict().items():
            torch.testing.assert_close(initial.model.state_dict()[key], tensor, rtol=0, atol=0)
        # Even though the fixture carries a pretrained_cfg URL, creation has not
        # loaded that source or replaced the model's default head initialization.
        self.assertEqual(initial.model.reset_calls, 0)
        self.assertIsNone(initial.metadata["pretrained_reference"])

    def test_invalid_state_keys_shapes_dtypes_and_values_are_rejected(self) -> None:
        create_initial_checkpoint(self.loaded)
        original_checkpoint = self.path.read_bytes()
        original_metadata = self.metadata_path.read_text(encoding="utf-8")
        for fault in ("missing", "extra", "shape", "fp16", "nan", "buffer_dtype"):
            with self.subTest(fault=fault):
                payload = torch.load(io.BytesIO(original_checkpoint), map_location="cpu", weights_only=True)
                state = payload["model_state"]
                if fault == "missing":
                    del state["head.bias"]
                elif fault == "extra":
                    state["unexpected"] = torch.zeros(1)
                elif fault == "shape":
                    state["head.weight"] = state["head.weight"][:1]
                elif fault == "fp16":
                    state["head.weight"] = state["head.weight"].half()
                elif fault == "nan":
                    state["head.weight"][0, 0] = float("nan")
                else:
                    state["counter"] = state["counter"].float()
                torch.save(payload, self.path)
                metadata = json.loads(original_metadata)
                metadata["checkpoint_sha256"] = hashlib.sha256(self.path.read_bytes()).hexdigest()
                metadata["checkpoint_size_bytes"] = self.path.stat().st_size
                self.metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Saved model state|saved state"):
                    load_initial_checkpoint(self.loaded.config)

    def test_reconstructed_layout_or_preprocessing_must_match(self) -> None:
        create_initial_checkpoint(self.loaded)
        for mismatch in ("layout", "preprocessing"):
            with self.subTest(mismatch=mismatch):
                def changed_model(name: str, *, pretrained: bool, num_classes: int) -> nn.Module:
                    self.assertFalse(pretrained)
                    model = TinyClassifier(num_classes, width=5 if mismatch == "layout" else 4)
                    if mismatch == "preprocessing":
                        model.pretrained_cfg["crop_pct"] = 0.9
                    return model

                self.factory.side_effect = changed_model
                with self.assertRaisesRegex(ValueError, "layout|preprocessing"):
                    load_initial_checkpoint(self.loaded.config)


if __name__ == "__main__":
    unittest.main()
