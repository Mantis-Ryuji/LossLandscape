"""Small CPU training/restart tests, with no real dataset, download or CUDA work."""

from __future__ import annotations

import csv
import math
import random
import sys
import tempfile
import unittest
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from landscape_exp import checkpoints
from landscape_exp.checkpoints import (
    CompletedEpoch, Segment, completed_lineage, create_segment, load_completed_epoch, read_json, save_epoch,
)
from landscape_exp.config import load_config
from landscape_exp.data import make_loader
from landscape_exp.evaluate import evaluate
from landscape_exp.logging_utils import EpochMetrics
from landscape_exp.seeds import LoaderGenerators, capture_random_state, restore_random_state, seed_global
from landscape_exp.train import (
    EpochSchedule, TrainingLoaders, _prepared_run, gradient_l2, make_optimizer,
    model_snapshot, restore_training_state, run_segment, train_one_epoch,
)


class AugmentedPoints(Dataset):
    """Consume all three CPU RNG families during training only."""

    def __init__(self, points: torch.Tensor, targets: torch.Tensor) -> None:
        self.points, self.targets = points, targets

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        noise = (random.random() + float(np.random.random()) + torch.rand(2)) * 0.01
        return self.points[index] + noise, self.targets[index]


class TinyNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(2, 4)
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Linear(4, 2)
        self.register_buffer("scale", torch.tensor(0.5))

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return self.head(self.dropout(torch.tanh(self.hidden(points)))) * self.scale


@dataclass
class Fixture:
    model: TinyNetwork
    loaders: TrainingLoaders
    optimizer: torch.optim.AdamW
    schedule: EpochSchedule
    generators: LoaderGenerators


class TrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        previous = capture_random_state()
        self.addCleanup(restore_random_state, previous)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.device = torch.device("cpu")
        seed_global(5)
        self.reference = model_snapshot(TinyNetwork())
        self.points = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [-1., 1.], [0., -1.], [2., 0.], [-1., 0.]])
        self.targets = torch.tensor([0, 1, 0, 1, 1, 0, 1])
        self.contract = {"schema_version": 1, "run_id": "fixture/seed11", "fixture": "cpu-fp32"}

    def fixture(self, accumulation_steps: int = 1) -> Fixture:
        model = TinyNetwork()
        model.load_state_dict(self.reference)
        generators = LoaderGenerators.from_seed(11)
        train = make_loader(AugmentedPoints(self.points, self.targets), role="train", batch_size=3,
                            num_workers=0, pin_memory=False, generators=generators)
        subset = make_loader(TensorDataset(self.points[:3], self.targets[:3]), role="train_subset", batch_size=2,
                             num_workers=0, pin_memory=False, generators=generators)
        validation = make_loader(TensorDataset(self.points, self.targets), role="validation", batch_size=3,
                                 num_workers=0, pin_memory=False, generators=generators)
        schedule = EpochSchedule(3, 0, math.ceil(len(train) / accumulation_steps), 0.01, scheduler="constant")
        return Fixture(model, TrainingLoaders(train, subset, validation), make_optimizer(model, schedule, 0.05),
                       schedule, generators)

    def execute(
        self, name: str, end: int, parent: CompletedEpoch | None = None,
        callback: Callable[[Path, EpochMetrics], None] | None = None,
        accumulation_steps: int = 1,
    ) -> tuple[Fixture, Segment, Path]:
        root = self.root / name
        root.mkdir(exist_ok=True)
        fixture = self.fixture(accumulation_steps)
        segment = create_segment(root, self.contract, parent)
        path = run_segment(
            fixture.model, fixture.loaders, fixture.optimizer, fixture.schedule, fixture.generators,
            self.reference, segment, end_epoch=end, seed=11, device=self.device, use_bf16=False,
            parent=parent, on_complete=callback,
            accumulation_steps=accumulation_steps,
        )
        return fixture, segment, path

    def assert_tree_equal(self, actual: object, expected: object) -> None:
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        elif isinstance(expected, dict):
            self.assertEqual(set(actual), set(expected))
            for key in expected:
                self.assert_tree_equal(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(type(actual), type(expected))
            self.assertEqual(len(actual), len(expected))
            for first, second in zip(actual, expected):
                self.assert_tree_equal(first, second)
        else:
            self.assertEqual(actual, expected)

    def test_schedule_uses_one_based_warmup_and_full_cosine(self) -> None:
        schedule = EpochSchedule(50, 5, 3, 1e-4)
        self.assertEqual(schedule.rate(1), 1e-4 / 15)
        self.assertEqual(schedule.rate(15), 1e-4)
        self.assertAlmostEqual(schedule.rate(16), 1e-4 * (1 + math.cos(math.pi / 135)) / 2)
        self.assertEqual(schedule.rate(150), 0.0)
        schedule.completed_updates = 150
        self.assertIsNone(schedule.next_lr)

    def test_constant_schedule_keeps_the_last_update_nonzero(self) -> None:
        schedule = EpochSchedule(100, 0, 3, 1e-3, scheduler="constant")
        optimizer = make_optimizer(nn.Linear(2, 2), schedule, 0.05)
        for completed in (0, 1, 3, 15, 150, 299):
            schedule.completed_updates = completed
            schedule.apply_next(optimizer)
            self.assertEqual(optimizer.param_groups[0]["lr"], 1e-3)
            self.assertEqual(schedule.next_lr, 1e-3)
        schedule.completed_updates = 300
        self.assertEqual(schedule.last_lr, 1e-3)
        self.assertIsNone(schedule.next_lr)
        with self.assertRaises(ValueError):
            schedule.apply_next(optimizer)

    def test_phase0_stops_without_shortening_the_constant_schedule(self) -> None:
        config = load_config("configs/phase0.yaml").config
        steps = math.ceil(config.split.train_size / config.training.batch_size)
        schedule = EpochSchedule(config.training.epochs, config.training.warmup_epochs, steps,
                                 config.training.learning_rate, scheduler=config.training.scheduler)
        schedule.completed_updates = config.end_epoch * steps
        self.assertEqual(schedule.total_updates, 100 * steps)
        self.assertEqual(schedule.last_lr, 1e-3)
        self.assertEqual(schedule.next_lr, 1e-3)
        self.assertEqual(schedule.rate(1), 1e-3)
        self.assertEqual(schedule.rate(100 * steps), 1e-3)
        self.assertEqual(config.end_epoch + 1, 6)

    def test_invalid_or_changed_schedule_is_rejected(self) -> None:
        for values in ((0, 0, 1, .1), (3, 3, 1, .1), (3, 1, 0, .1), (3, 1, 1, float("nan"))):
            with self.subTest(values=values), self.assertRaises(ValueError):
                EpochSchedule(*values)
        schedule = EpochSchedule(3, 1, 3, .01)
        state = {**schedule.state_dict(), "completed_updates": 3, "epochs": 2}
        with self.assertRaises(ValueError):
            schedule.validate_state(state, 3)
        self.assertEqual(schedule.completed_updates, 0)
        for scheduler, warmup in (("unknown", 0), ("constant", 1)):
            with self.subTest(scheduler=scheduler), self.assertRaises(ValueError):
                EpochSchedule(3, warmup, 3, .01, scheduler=scheduler)
        constant = EpochSchedule(3, 0, 3, .01, scheduler="constant")
        for changed in ({"scheduler": "cosine"}, {"schema_version": 1}):
            state = {**constant.state_dict(), "completed_updates": 3, **changed}
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                constant.validate_state(state, 3)
            self.assertEqual(constant.completed_updates, 0)

    def test_evaluation_weights_short_batches_by_sample_count(self) -> None:
        model = nn.Linear(2, 2)
        with torch.no_grad():
            model.weight.copy_(torch.eye(2))
            model.bias.zero_()
        batches = [(self.points[:6], self.targets[:6]), (self.points[6:], self.targets[6:])]
        observed = evaluate(model, batches, device=self.device)
        expected = F.cross_entropy(model(self.points), self.targets).item()
        self.assertAlmostEqual(observed.loss, expected, places=6)
        self.assertEqual(observed.samples, 7)
        self.assertEqual(observed.accuracy, int((model(self.points).argmax(1) == self.targets).sum()) / len(self.targets))
        with self.assertRaises(ValueError):
            evaluate(model, [], device=self.device)

    def test_evaluation_failure_restores_modes_rng_and_precision(self) -> None:
        fixture = self.fixture()
        fixture.model.train()
        fixture.model.dropout.eval()
        modes = [module.training for module in fixture.model.modules()]
        previous_precision = torch.get_float32_matmul_precision()
        self.addCleanup(torch.set_float32_matmul_precision, previous_precision)
        torch.set_float32_matmul_precision("medium")
        flags = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                 torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled(),
                 torch.get_float32_matmul_precision())
        rng = capture_random_state()
        streams = fixture.generators.state_dict()

        def fail() -> Iterator[object]:
            yield self.points[:1], self.targets[:1]
            random.random()
            np.random.random()
            torch.rand(2)
            torch.rand(2, generator=fixture.generators.validation_workers)
            raise RuntimeError("evaluation failed")

        with self.assertRaisesRegex(RuntimeError, "evaluation failed"):
            evaluate(fixture.model, fail(), device=self.device, generators=fixture.generators)
        self.assertEqual([module.training for module in fixture.model.modules()], modes)
        self.assertEqual((torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                          torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled(),
                          torch.get_float32_matmul_precision()), flags)
        self.assert_tree_equal(capture_random_state(), rng)
        self.assert_tree_equal(fixture.generators.state_dict(), streams)

    def test_evaluation_disables_an_outer_autocast(self) -> None:
        model = nn.Linear(2, 2)
        expected = evaluate(model, [(self.points, self.targets)], device=self.device)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = evaluate(model, [(self.points, self.targets)], device=self.device)
        self.assertEqual(actual, expected)

    def test_global_gradient_norm_is_measured_without_clipping(self) -> None:
        model = nn.Linear(2, 1)
        model.weight.grad = torch.tensor([[3., 4.]])
        model.bias.grad = torch.zeros(1)
        self.assertEqual(gradient_l2(model), 5.0)
        torch.testing.assert_close(model.weight.grad, torch.tensor([[3., 4.]]), rtol=0, atol=0)

    def test_accumulation_matches_physical_batches_including_partial_groups(self) -> None:
        # Production tails are 200 (B256) and 968 (B1024). Also exercise an
        # epoch shorter than one effective batch and a one-sample microbatch.
        for samples, microbatch, accumulation in ((13, 3, 3), (7, 3, 4), (456, 64, 4), (1992, 64, 16)):
            with self.subTest(samples=samples, accumulation=accumulation):
                points = self.points.repeat(math.ceil(samples / 7), 1)[:samples]
                targets = self.targets.repeat(math.ceil(samples / 7))[:samples]
                effective = microbatch * accumulation
                expected = nn.Linear(2, 2)
                actual = nn.Linear(2, 2)
                actual.load_state_dict(expected.state_dict())
                reference_optimizer = torch.optim.AdamW(expected.parameters(), lr=.01, weight_decay=.05)
                expected_gradients, loss_sum, correct = [], 0., 0
                for offset in range(0, samples, effective):
                    x, y = points[offset:offset + effective], targets[offset:offset + effective]
                    reference_optimizer.zero_grad(set_to_none=True)
                    logits = expected(x)
                    loss = F.cross_entropy(logits, y)
                    loss.backward()
                    expected_gradients.append(torch.cat([p.grad.detach().flatten().clone() for p in expected.parameters()]))
                    loss_sum += loss.item() * len(y)
                    correct += int((logits.argmax(1) == y).sum())
                    reference_optimizer.step()
                loader = DataLoader(TensorDataset(points, targets), batch_size=microbatch, shuffle=False)
                schedule = EpochSchedule(1, 0, len(expected_gradients), .01, scheduler="constant")
                optimizer = make_optimizer(actual, schedule, .05)
                observed_gradients = []
                original_step = optimizer.step

                def record_step() -> None:
                    observed_gradients.append(torch.cat([p.grad.detach().flatten().clone() for p in actual.parameters()]))
                    original_step()

                with patch.object(optimizer, "step", side_effect=record_step) as step:
                    with patch.object(schedule, "apply_next", wraps=schedule.apply_next) as apply_lr:
                        observed = train_one_epoch(actual, loader, optimizer, schedule, device=self.device,
                                                   use_bf16=False, accumulation_steps=accumulation)
                self.assertEqual(step.call_count, len(expected_gradients))
                self.assertEqual(apply_lr.call_count, len(expected_gradients))
                self.assertEqual(schedule.completed_updates, len(expected_gradients))
                torch.testing.assert_close(observed_gradients, expected_gradients, rtol=1e-5, atol=1e-7)
                torch.testing.assert_close(actual.state_dict(), expected.state_dict(), rtol=1e-5, atol=1e-7)
                torch.testing.assert_close(optimizer.state_dict(), reference_optimizer.state_dict(), rtol=1e-5, atol=1e-7)
                expected_norm = sum(float(g.double().norm()) for g in expected_gradients) / len(expected_gradients)
                self.assertAlmostEqual(observed.gradient_norm, expected_norm, places=6)
                self.assertAlmostEqual(observed.loss, loss_sum / samples, places=6)
                self.assertEqual(observed.accuracy, correct / samples)
                self.assertEqual(observed.samples, samples)

    def test_invalid_accumulation_or_microstep_schedule_fails_before_training(self) -> None:
        for accumulation in (0, -1, True, 1.5, 2):
            fixture = self.fixture()
            with self.subTest(accumulation=accumulation), patch.object(fixture.model, "forward") as forward:
                with self.assertRaises(ValueError):
                    train_one_epoch(fixture.model, fixture.loaders.train, fixture.optimizer, fixture.schedule,
                                    device=self.device, use_bf16=False, accumulation_steps=accumulation)
                forward.assert_not_called()
                self.assertEqual(fixture.schedule.completed_updates, 0)
                self.assertFalse(fixture.optimizer.state)

    def test_nonfinite_later_microbatch_aborts_the_whole_update(self) -> None:
        for failure in ("loss", "gradient"):
            fixture = self.fixture(accumulation_steps=4)
            before = model_snapshot(fixture.model)
            if failure == "loss":
                original_forward = fixture.model.forward
                calls = 0

                def forward(points: torch.Tensor) -> torch.Tensor:
                    nonlocal calls
                    calls += 1
                    result = original_forward(points)
                    return result if calls == 1 else result * float("nan")

                failure_context = patch.object(fixture.model, "forward", side_effect=forward)
                message = "Nonfinite classification"
            else:
                calls = 0

                def gradient_hook(gradient: torch.Tensor) -> torch.Tensor:
                    nonlocal calls
                    calls += 1
                    return gradient if calls == 1 else gradient * float("nan")

                hook = fixture.model.head.weight.register_hook(gradient_hook)
                self.addCleanup(hook.remove)
                failure_context = patch.object(fixture.model, "forward", wraps=fixture.model.forward)
                message = "Nonfinite gradient"
            with self.subTest(failure=failure), failure_context:
                with self.assertRaisesRegex(ValueError, message):
                    train_one_epoch(fixture.model, fixture.loaders.train, fixture.optimizer, fixture.schedule,
                                    device=self.device, use_bf16=False, accumulation_steps=4)
            self.assertGreaterEqual(calls, 2)
            self.assertEqual(fixture.schedule.completed_updates, 0)
            self.assertFalse(fixture.optimizer.state)
            self.assert_tree_equal(model_snapshot(fixture.model), before)

    def test_nonfinite_loss_aborts_before_optimizer_update(self) -> None:
        fixture = self.fixture()
        before = model_snapshot(fixture.model)
        with patch.object(fixture.model, "forward", return_value=torch.full((3, 2), float("nan"))):
            with self.assertRaisesRegex(ValueError, "Nonfinite classification"):
                train_one_epoch(fixture.model, fixture.loaders.train, fixture.optimizer, fixture.schedule,
                                device=self.device, use_bf16=False)
        self.assertEqual(fixture.schedule.completed_updates, 0)
        self.assert_tree_equal(model_snapshot(fixture.model), before)
        self.assertFalse(fixture.optimizer.state)

    def test_nonfinite_gradient_aborts_before_optimizer_update(self) -> None:
        fixture = self.fixture()
        before = model_snapshot(fixture.model)
        hook = fixture.model.head.weight.register_hook(lambda gradient: gradient * float("nan"))
        self.addCleanup(hook.remove)
        with self.assertRaisesRegex(ValueError, "Nonfinite gradient"):
            train_one_epoch(fixture.model, fixture.loaders.train, fixture.optimizer, fixture.schedule,
                            device=self.device, use_bf16=False)
        self.assertEqual(fixture.schedule.completed_updates, 0)
        self.assert_tree_equal(model_snapshot(fixture.model), before)

    def test_epoch_zero_and_csv_preserve_undefined_metrics(self) -> None:
        _, segment, _ = self.execute("zero", 1)
        zero = load_completed_epoch(segment.directory / "epochs/epoch_0000", self.contract)
        for key in ("train_loss", "train_accuracy", "gradient_norm", "learning_rate"):
            self.assertIsNone(zero.metrics[key])
        self.assertEqual(zero.metrics["parameter_displacement"], 0.0)
        self.assertEqual(zero.global_step, 0)
        self.assertEqual(zero.metrics["train_samples"], 0)
        self.assertIsNotNone(zero.metrics["val_loss"])
        self.assert_tree_equal(zero.resume["model_state"], self.reference)
        with (segment.directory / "metrics.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([row["epoch"] for row in rows], ["0", "1"])
        self.assertEqual(rows[0]["train_loss"], "")
        self.assertFalse((zero.directory / "complete.json.tmp").exists())

    def test_epoch_resume_matches_uninterrupted_training_exactly(self) -> None:
        full, full_segment, full_path = self.execute("continuous", 3)
        _, first_segment, first_path = self.execute("resumed", 1)
        parent = load_completed_epoch(first_path, self.contract)
        resumed, child_segment, resumed_path = self.execute("resumed", 3, parent)
        self.assert_tree_equal(model_snapshot(resumed.model), model_snapshot(full.model))
        self.assert_tree_equal(resumed.optimizer.state_dict(), full.optimizer.state_dict())
        self.assert_tree_equal(resumed.generators.state_dict(), full.generators.state_dict())
        self.assertEqual(resumed.schedule.state_dict(), full.schedule.state_dict())
        self.assertEqual(resumed.schedule.scheduler, "constant")
        self.assertEqual(resumed.schedule.last_lr, .01)
        self.assertIsNone(resumed.schedule.next_lr)
        final_full = load_completed_epoch(full_path, self.contract)
        final_resumed = load_completed_epoch(resumed_path, self.contract)
        self.assert_tree_equal(final_resumed.resume["rng_state"], final_full.resume["rng_state"])
        for name in ("train_loss", "train_accuracy", "gradient_norm", "val_loss", "train_subset_loss", "parameter_displacement"):
            self.assertEqual(final_resumed.metrics[name], final_full.metrics[name])
        self.assertNotEqual(first_segment.directory, child_segment.directory)
        self.assertEqual(len(completed_lineage(child_segment.directory, self.contract)), 4)
        self.assertEqual(len(completed_lineage(full_segment.directory, self.contract)), 4)

    def test_resume_from_epoch_zero_reproduces_the_first_epoch(self) -> None:
        first, segment, _ = self.execute("from_zero", 1)
        parent = load_completed_epoch(segment.directory / "epochs/epoch_0000", self.contract)
        resumed, child, _ = self.execute("from_zero", 1, parent)
        self.assert_tree_equal(model_snapshot(resumed.model), model_snapshot(first.model))
        paths = completed_lineage(child.directory, self.contract)
        self.assertEqual(paths[0], parent.directory)
        self.assertEqual(paths[1].parents[1], child.directory)

    def test_accumulated_epoch_and_zero_resume_match_uninterrupted_state(self) -> None:
        for accumulation in (2, 4, 16):
            with self.subTest(accumulation=accumulation):
                full, _, full_path = self.execute(f"accum_full_{accumulation}", 3, accumulation_steps=accumulation)
                final_full = load_completed_epoch(full_path, self.contract)
                for epoch in (0, 1):
                    name = f"accum_resume_{accumulation}_{epoch}"
                    _, first, _ = self.execute(name, 1, accumulation_steps=accumulation)
                    parent = load_completed_epoch(first.directory / f"epochs/epoch_{epoch:04d}", self.contract)
                    resumed, _, last = self.execute(name, 3, parent, accumulation_steps=accumulation)
                    final = load_completed_epoch(last, self.contract)
                    self.assert_tree_equal(model_snapshot(resumed.model), model_snapshot(full.model))
                    self.assert_tree_equal(final.resume, final_full.resume)
                    self.assertEqual(final.global_step, 3 * math.ceil(7 / (3 * accumulation)))
                    self.assertEqual(parent.metrics["batch_size"], 3 * accumulation)
                    self.assertEqual(final.metrics["batch_size"], 3 * accumulation)
                    self.assertEqual(final.metrics["train_samples"], 7)
                    for metric in ("train_loss", "gradient_norm", "val_loss", "parameter_displacement"):
                        self.assertEqual(final.metrics[metric], final_full.metrics[metric])

    def test_resume_rejects_changed_accumulation_even_with_same_update_count(self) -> None:
        _, _, path = self.execute("accum_change", 1, accumulation_steps=4)
        parent = load_completed_epoch(path, self.contract)
        with self.assertRaisesRegex(ValueError, "effective batch size changed"):
            self.execute("accum_change", 2, parent, accumulation_steps=16)

    def test_lineage_cuts_ancestors_and_ignores_incomplete_epochs(self) -> None:
        _, original, _ = self.execute("branch", 3)
        parent = load_completed_epoch(original.directory / "epochs/epoch_0001", self.contract)
        _, child, last = self.execute("branch", 2, parent)
        partial = child.directory / "epochs/epoch_0003"
        partial.mkdir()
        (partial / "resume.pt").write_bytes(b"unfinished")
        paths = completed_lineage(child.directory, self.contract)
        self.assertEqual([path.name for path in paths], ["epoch_0000", "epoch_0001", "epoch_0002"])
        self.assertEqual(paths[-1], last)
        self.assertNotIn(original.directory / "epochs/epoch_0002", paths)
        self.assertEqual(len(completed_lineage(original.directory, self.contract, through_epoch=1)), 2)

    def test_completed_epoch_cannot_be_overwritten(self) -> None:
        _, segment, path = self.execute("immutable", 1)
        original = (path / "resume.pt").read_bytes()
        completed = load_completed_epoch(path, self.contract)
        metrics = EpochMetrics(**completed.metrics)
        with self.assertRaises(ValueError):
            save_epoch(segment, metrics, model_state=completed.resume["model_state"],
                       optimizer_state={}, scheduler_state={}, rng_state={}, loader_state={},
                       epoch_started=0., checkpoint_started=0.)
        self.assertEqual((path / "resume.pt").read_bytes(), original)

    def test_failed_checkpoint_is_preserved_and_cannot_be_resumed(self) -> None:
        original_writer = checkpoints._write_torch

        def fail_resume(path: Path, value: object) -> None:
            if path.parent.name == "epoch_0001" and path.name == "resume.pt":
                raise OSError("simulated save failure")
            original_writer(path, value)

        with patch("landscape_exp.checkpoints._write_torch", side_effect=fail_resume):
            with self.assertRaisesRegex(OSError, "simulated save failure"):
                self.execute("save_failure", 1)
        segment = next((self.root / "save_failure/segments").iterdir())
        partial = segment / "epochs/epoch_0001"
        self.assertTrue((partial / "analysis.pt").exists())
        self.assertFalse((partial / "complete.json").exists())
        with self.assertRaises(ValueError):
            load_completed_epoch(partial, self.contract)
        self.assertEqual(len(completed_lineage(segment, self.contract)), 1)
        parent = load_completed_epoch(segment / "epochs/epoch_0000", self.contract)
        self.execute("save_failure", 1, parent)
        self.assertTrue((partial / "analysis.pt").exists())
        self.assertFalse((partial / "resume.pt").exists())

    def test_corrupt_analysis_is_rejected_before_resume_deserialization(self) -> None:
        _, _, path = self.execute("corrupt", 1)
        with (path / "analysis.pt").open("ab") as handle:
            handle.write(b"corruption")
        with patch("landscape_exp.checkpoints.torch.load", side_effect=AssertionError("Do not deserialize corrupt records")):
            with self.assertRaisesRegex(ValueError, "size/hash"):
                load_completed_epoch(path, self.contract)

    def test_contract_and_optimizer_changes_are_rejected(self) -> None:
        _, _, path = self.execute("changed", 1)
        with self.assertRaisesRegex(ValueError, "identity changed"):
            load_completed_epoch(path, {**self.contract, "fixture": "different"})
        completed = load_completed_epoch(path, self.contract)
        completed.resume["optimizer_state"]["param_groups"][0]["lr"] = .5
        fresh = self.fixture()
        with self.assertRaisesRegex(ValueError, "hyperparameters"):
            restore_training_state(completed, fresh.model, fresh.optimizer, fresh.schedule, fresh.generators)
        self.assert_tree_equal(model_snapshot(fresh.model), self.reference)

    def test_changing_buffers_stops_before_publishing_an_epoch(self) -> None:
        fixture = self.fixture()
        root = self.root / "changed_buffer"
        root.mkdir()
        segment = create_segment(root, self.contract)
        original_forward = fixture.model.forward

        def mutate_buffer(points: torch.Tensor) -> torch.Tensor:
            if fixture.model.training:
                fixture.model.scale.add_(.1)
            return original_forward(points)

        with patch.object(fixture.model, "forward", side_effect=mutate_buffer):
            with self.assertRaisesRegex(ValueError, "Buffer changed"):
                run_segment(fixture.model, fixture.loaders, fixture.optimizer, fixture.schedule,
                            fixture.generators, self.reference, segment, end_epoch=1, seed=11,
                            device=self.device, use_bf16=False)
        self.assertEqual(len(completed_lineage(segment.directory, self.contract)), 1)

    def test_progress_callback_does_not_change_training_randomness(self) -> None:
        def consume_rng(path: Path, metrics: EpochMetrics) -> None:
            random.random()
            np.random.random(9)
            torch.rand(9)

        expected, _, _ = self.execute("quiet", 2)
        observed, _, _ = self.execute("reporting", 2, callback=consume_rng)
        self.assert_tree_equal(model_snapshot(observed.model), model_snapshot(expected.model))

    def test_prepared_run_reuse_requires_explicit_resume_after_a_segment(self) -> None:
        config_path = self.root / "config.yaml"
        config_path.write_bytes((REPOSITORY_ROOT / "configs/phase0.yaml").read_bytes())
        loaded = load_config(config_path, project_root=self.root)
        root = _prepared_run(loaded, resuming=False)
        original = (root / "source.yaml").read_bytes()
        self.assertEqual(_prepared_run(loaded, resuming=False), root)
        create_segment(root, {"run_id": loaded.config.run_id})
        with self.assertRaisesRegex(ValueError, "explicit --resume-from"):
            _prepared_run(loaded, resuming=False)
        self.assertEqual(_prepared_run(loaded, resuming=True), root)
        self.assertEqual((root / "source.yaml").read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
