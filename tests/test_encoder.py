from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from safetensors.torch import save_file

import rosetta_copy.model as model_module
from conftest import TinyCheckpoint, TinyRetrievalModel


def _images() -> list[Image.Image]:
    return [
        Image.new("RGB", (8, 8), (240, 30, 10)),
        Image.new("RGB", (4, 8), (15, 180, 70)),
        Image.new("L", (8, 5), 96),
    ]


def test_checkpoint_is_loaded_and_encoder_is_inference_only(
    tiny_encoder: model_module.RosettaEncoder,
) -> None:
    assert not tiny_encoder.model.training
    assert all(not parameter.requires_grad for parameter in tiny_encoder.model.parameters())
    assert {parameter.device.type for parameter in tiny_encoder.model.parameters()} == {"cpu"}
    assert {parameter.dtype for parameter in tiny_encoder.model.parameters()} == {
        torch.float32
    }


def test_embeddings_are_normalised_deterministic_and_batch_invariant(
    tiny_encoder: model_module.RosettaEncoder,
) -> None:
    images = _images()
    by_one = tiny_encoder.encode(images, batch_size=1)
    by_three = tiny_encoder.encode(images, batch_size=3)
    repeated = tiny_encoder.encode(images, batch_size=2)

    assert by_one.shape == (3, 256)
    assert by_one.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(by_one, axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(by_one, by_three, atol=1e-7)
    np.testing.assert_allclose(by_one, repeated, atol=1e-7)


def test_embedding_has_a_frozen_numerical_reference(
    tiny_encoder: model_module.RosettaEncoder,
) -> None:
    vector = tiny_encoder.encode([Image.new("RGB", (8, 8), (32, 96, 224))])[0]

    expected_prefix = np.array(
        [
            -0.10432690,
            -0.10348234,
            -0.10263775,
            -0.10179325,
            -0.10094882,
            -0.10010400,
            -0.09926000,
            -0.09841500,
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(vector[:8], expected_prefix, rtol=0, atol=1e-6)


def test_iterables_and_empty_batches_are_supported(
    tiny_encoder: model_module.RosettaEncoder,
) -> None:
    generated = tiny_encoder.encode((image for image in _images()), batch_size=2)
    empty = tiny_encoder.encode(iter(()))

    assert generated.shape == (3, 256)
    assert empty.shape == (0, 256)
    assert empty.dtype == np.float32


@pytest.mark.parametrize("bad_batch_size", [0, -1])
def test_invalid_batch_size_is_rejected_even_for_empty_input(
    tiny_encoder: model_module.RosettaEncoder, bad_batch_size: int
) -> None:
    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        tiny_encoder.encode([], batch_size=bad_batch_size)


@pytest.mark.parametrize(
    "single_image",
    ["image.jpg", Path("image.jpg"), Image.new("RGB", (2, 2))],
)
def test_single_image_must_be_wrapped_in_an_iterable(
    tiny_encoder: model_module.RosettaEncoder, single_image
) -> None:
    with pytest.raises(TypeError, match="iterable of images"):
        tiny_encoder.encode(single_image)


def test_missing_and_wrong_checkpoint_are_rejected(
    tmp_path: Path, tiny_runtime: TinyCheckpoint
) -> None:
    with pytest.raises(FileNotFoundError):
        model_module.RosettaEncoder("s224", tmp_path / "missing.safetensors")

    wrong = tmp_path / "wrong.safetensors"
    wrong.write_bytes(b"wrong")
    with pytest.raises(RuntimeError, match="Unexpected checkpoint SHA-256"):
        model_module.RosettaEncoder("s224", wrong)


def test_incompatible_state_dict_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "incompatible.safetensors"
    save_file({"global_head.bias": torch.zeros(12)}, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    config = {
        "input_side": 8,
        "descriptor_dimensions": 256,
        "filename": path.name,
        "sha256": digest,
        "url": "https://example.invalid/incompatible.safetensors",
    }
    monkeypatch.setattr(model_module, "_variant_config", lambda _variant: config)
    monkeypatch.setattr(model_module, "_RetrievalModel", TinyRetrievalModel)

    with pytest.raises(RuntimeError, match=r"Error\(s\) in loading state_dict"):
        model_module.RosettaEncoder("s224", path)


def test_from_pretrained_uses_explicit_or_downloaded_weights(
    tiny_runtime: TinyCheckpoint, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected_download(*_args, **_kwargs):
        raise AssertionError("explicit weights must skip download")

    monkeypatch.setattr(model_module, "download_weights", unexpected_download)
    explicit = model_module.RosettaEncoder.from_pretrained(
        "s224", weights=tiny_runtime.path
    )
    assert explicit.variant == "s224"

    calls: list[tuple] = []

    def fake_download(variant, cache_dir):
        calls.append((variant, cache_dir))
        return tiny_runtime.path

    monkeypatch.setattr(model_module, "download_weights", fake_download)
    downloaded = model_module.RosettaEncoder.from_pretrained(
        "s224", cache_dir="cache"
    )
    assert downloaded.variant == "s224"
    assert calls == [("s224", "cache")]


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_inference(tiny_runtime: TinyCheckpoint) -> None:
    encoder = model_module.RosettaEncoder("s224", tiny_runtime.path, device="cuda")
    vectors = encoder.encode(_images()[:2], batch_size=2)

    assert vectors.shape == (2, 256)
    assert vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=2e-4)
