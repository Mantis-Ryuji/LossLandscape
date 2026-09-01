"""CPU-only tests for strict parameter vectorization and restoration."""

from __future__ import annotations

import hashlib
import json
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from landscape_exp.landscape import (
    assign_parameter_vector, build_parameter_layout, flatten_model_state,
    flatten_parameters, parameter_spec_sha256,
)


class TinyLandscapeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Linear(3, 2)
        self.head = nn.Linear(2, 2, bias=False)
        self.register_buffer("scale", torch.tensor(0.5, dtype=torch.float32))
        self.register_buffer("counter", torch.tensor(7, dtype=torch.int64))


def snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


class ParameterVectorTests(unittest.TestCase):
    def setUp(self) -> None:
        previous = torch.get_rng_state()
        self.addCleanup(torch.set_rng_state, previous)
        torch.manual_seed(17)
        self.reference_model = TinyLandscapeModel()
        self.reference = snapshot(self.reference_model)
        self.layout = build_parameter_layout(self.reference_model)

    def test_layout_uses_named_parameter_order_and_metadata_compatible_hash(self) -> None:
        self.assertEqual(
            self.layout.parameter_names,
            ("first.weight", "first.bias", "head.weight"),
        )
        self.assertEqual(self.layout.buffer_names, ("scale", "counter"))
        records = [item.record() for item in self.layout.parameters]
        expected = hashlib.sha256(json.dumps(
            records, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")).hexdigest()
        self.assertEqual(self.layout.parameter_spec_sha256, expected)
        self.assertEqual(parameter_spec_sha256(self.layout.parameters), expected)

    def test_state_vector_model_roundtrip_excludes_and_preserves_buffers(self) -> None:
        state = {name: tensor.clone() for name, tensor in self.reference.items()}
        for index, name in enumerate(self.layout.parameter_names, start=1):
            state[name].add_(index / 10)
        vector = flatten_model_state(state, self.layout, self.reference)
        expected = torch.cat([state[name].reshape(-1) for name in self.layout.parameter_names])
        torch.testing.assert_close(vector, expected, rtol=0, atol=0)
        self.assertEqual(vector.dtype, torch.float32)
        self.assertEqual(vector.device.type, "cpu")
        self.assertTrue(vector.is_contiguous())

        target = TinyLandscapeModel()
        before_buffers = {name: target.state_dict()[name].clone() for name in self.layout.buffer_names}
        assign_parameter_vector(target, vector, self.layout.parameters)
        for name, parameter in target.named_parameters():
            torch.testing.assert_close(parameter, state[name], rtol=0, atol=0)
        for name in self.layout.buffer_names:
            torch.testing.assert_close(target.state_dict()[name], before_buffers[name], rtol=0, atol=0)

    def test_changed_buffer_or_parameter_layout_is_rejected(self) -> None:
        changed_buffer = {name: tensor.clone() for name, tensor in self.reference.items()}
        changed_buffer["counter"].add_(1)
        with self.assertRaisesRegex(ValueError, "Buffer changed"):
            flatten_model_state(changed_buffer, self.layout, self.reference)

        changed_shape = {name: tensor.clone() for name, tensor in self.reference.items()}
        changed_shape["first.weight"] = changed_shape["first.weight"][:, :2]
        with self.assertRaisesRegex(ValueError, "shape/dtype"):
            flatten_model_state(changed_shape, self.layout, self.reference)

        changed_order = dict(reversed(tuple(self.reference.items())))
        with self.assertRaisesRegex(ValueError, "keys/order"):
            flatten_model_state(changed_order, self.layout, self.reference)

    def test_invalid_vectors_fail_before_any_parameter_is_modified(self) -> None:
        valid = flatten_model_state(self.reference, self.layout, self.reference)
        invalid = (
            valid[:-1],
            valid.double(),
            torch.cat((torch.tensor([float("nan")]), valid[1:])),
        )
        for vector in invalid:
            with self.subTest(shape=tuple(vector.shape), dtype=str(vector.dtype)):
                target = TinyLandscapeModel()
                before = snapshot(target)
                with self.assertRaises(ValueError):
                    assign_parameter_vector(target, vector, self.layout.parameters)
                for name, tensor in target.state_dict().items():
                    torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)

    def test_flatten_rejects_duplicate_missing_and_nonfinite_parameters(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            flatten_parameters(self.reference, ("first.weight", "first.weight"))
        with self.assertRaisesRegex(ValueError, "Missing"):
            flatten_parameters(self.reference, ("missing",))
        nonfinite = {name: tensor.clone() for name, tensor in self.reference.items()}
        nonfinite["first.bias"][0] = float("inf")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            flatten_parameters(nonfinite, self.layout.parameter_names)

    def test_corrupt_parameter_spec_hash_is_rejected(self) -> None:
        corrupt = replace(self.layout, parameter_spec_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "hash"):
            flatten_model_state(self.reference, corrupt, self.reference)


if __name__ == "__main__":
    unittest.main()
