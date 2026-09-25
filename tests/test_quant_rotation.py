"""Function preservation for the block Hadamard rotation primitive."""

import json

import pytest
import torch
import torch.nn.functional as functional
from safetensors.torch import save_file

from openternary.adapters.runtime import _install_rotations
from openternary.quant.rotation import (
    apply_block_rotation,
    cayley_orthogonal,
    fixed_signs_for_module,
    hadamard_last_dim,
    hadamard_segments,
    signed_hadamard_last_dim,
)


@pytest.mark.parametrize("block_size", [2, 4, 8, 16])
def test_hadamard_is_orthogonal_for_linear_inputs(block_size: int) -> None:
    generator = torch.Generator().manual_seed(19)
    inputs = torch.randn(5, block_size * 3, generator=generator)
    weights = torch.randn(7, block_size * 3, generator=generator)
    rotated_inputs = hadamard_last_dim(inputs, block_size)
    rotated_weights = hadamard_last_dim(weights, block_size)
    assert torch.allclose(
        functional.linear(rotated_inputs, rotated_weights), functional.linear(inputs, weights), atol=1e-5
    )
    assert torch.allclose(hadamard_last_dim(rotated_inputs, block_size), inputs, atol=1e-6)


@pytest.mark.parametrize("block_size", [0, 1, 3, 16])
def test_hadamard_rejects_invalid_block_size(block_size: int) -> None:
    with pytest.raises(ValueError, match="block_size"):
        hadamard_last_dim(torch.zeros(2, 12), block_size)


def test_cayley_rotation_preserves_linear_function() -> None:
    generator = torch.Generator().manual_seed(23)
    raw = torch.randn(8, 8, generator=generator) * 0.05
    rotation = cayley_orthogonal(raw - raw.T)
    assert torch.allclose(rotation.T @ rotation, torch.eye(8), atol=1e-6)
    inputs = torch.randn(4, 16, generator=generator)
    weights = torch.randn(5, 16, generator=generator)
    transformed = functional.linear(apply_block_rotation(inputs, rotation), apply_block_rotation(weights, rotation))
    assert torch.allclose(transformed, functional.linear(inputs, weights), atol=1e-5)


def test_signed_hadamard_preserves_linear_function_with_1536_input() -> None:
    generator = torch.Generator().manual_seed(17)
    inputs = torch.randn(3, 1536, generator=generator)
    weights = torch.randn(4, 1536, generator=generator)
    signs = torch.randint(0, 2, (1536,), generator=generator).float() * 2 - 1
    assert hadamard_segments(1536, 1024) == (1024, 512)
    actual = functional.linear(
        signed_hadamard_last_dim(inputs, 1024, signs), signed_hadamard_last_dim(weights, 1024, signs)
    )
    assert torch.allclose(actual, functional.linear(inputs, weights), atol=1e-4, rtol=1e-5)


def test_signed_hadamard_rejects_invalid_signs() -> None:
    with pytest.raises(ValueError, match="signs"):
        signed_hadamard_last_dim(torch.ones(2, 128), 128, torch.zeros(128))


def test_fixed_module_signs_are_reproducible_and_module_specific() -> None:
    first = fixed_signs_for_module("layer.0.q_proj", 128, 42, torch.device("cpu"))
    again = fixed_signs_for_module("layer.0.q_proj", 128, 42, torch.device("cpu"))
    other = fixed_signs_for_module("layer.0.k_proj", 128, 42, torch.device("cpu"))
    assert torch.equal(first, again)
    assert not torch.equal(first, other)
    assert set(first.tolist()) == {-1, 1}


def test_cayley_rejects_nonskew_matrix() -> None:
    with pytest.raises(ValueError, match="antisymmetric"):
        cayley_orthogonal(torch.ones(4, 4))


def test_saved_rotation_input_hook_restores_linear_function(tmp_path) -> None:
    source = torch.nn.Linear(128, 7, bias=False)
    rotated = torch.nn.Module()
    rotated.add_module("linear", torch.nn.Linear(128, 7, bias=False))
    with torch.no_grad():
        rotated.linear.weight.copy_(hadamard_last_dim(source.weight, 128))
    folder = tmp_path / "openternary"
    folder.mkdir()
    save_file({"linear": torch.eye(128)}, str(folder / "rotations.safetensors"))
    metadata = {
        "schema_version": 1,
        "method": "hadamard-learned-cayley",
        "matrix_file": "openternary/rotations.safetensors",
        "transform_order": ["normalized-hadamard", "learned-cayley"],
        "rotations": {"linear": {"block_size": 128, "group_size": 128}},
    }
    (folder / "rotations.json").write_text(json.dumps(metadata), encoding="utf-8")
    (tmp_path / "quantization.json").write_text(json.dumps({"rotation": metadata}), encoding="utf-8")
    _install_rotations(rotated, tmp_path, {"format": "safetensors"})
    inputs = torch.randn(3, 128)
    assert torch.allclose(rotated.linear(inputs), source(inputs), atol=1e-5)


def test_saved_rotation_requires_matching_report(tmp_path) -> None:
    folder = tmp_path / "openternary"
    folder.mkdir()
    (folder / "rotations.json").write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="validated safetensors"):
        _install_rotations(torch.nn.Linear(128, 7), tmp_path, None)


def test_rotated_report_requires_input_transform(tmp_path) -> None:
    (tmp_path / "quantization.json").write_text(json.dumps({"rotation": {"schema_version": 1}}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing their required input transform"):
        _install_rotations(torch.nn.Linear(128, 7), tmp_path, {"format": "safetensors"})
