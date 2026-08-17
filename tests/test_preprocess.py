from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image, UnidentifiedImageError

from rosetta_copy.model import _FILL, _MEAN, _STD, _open_rgb, _preprocess


def _normalised(rgb: tuple[int, int, int]) -> torch.Tensor:
    raw = torch.tensor(rgb, dtype=torch.float32).view(3, 1, 1) / 255.0
    return ((raw - _MEAN) / _STD).flatten()


def test_preprocess_preserves_aspect_ratio_and_uses_documented_fill() -> None:
    image = Image.new("RGB", (4, 2), (255, 0, 0))
    tensor = _preprocess(image, 8)

    assert tensor.shape == (3, 8, 8)
    assert tensor.dtype == torch.float32
    torch.testing.assert_close(tensor[:, 0, 0], _normalised(_FILL))
    torch.testing.assert_close(tensor[:, 4, 4], _normalised((255, 0, 0)))


@pytest.mark.parametrize("mode", ["L", "RGBA"])
def test_non_rgb_pil_inputs_are_converted(mode: str) -> None:
    color = 128 if mode == "L" else (10, 20, 30, 40)
    image = Image.new(mode, (7, 5), color)
    rgb = _open_rgb(image)
    tensor = _preprocess(image, 8)

    assert rgb.mode == "RGB"
    assert tensor.shape == (3, 8, 8)
    assert torch.isfinite(tensor).all()


@pytest.mark.parametrize("extension", ["png", "jpg", "webp", "bmp", "tiff"])
def test_common_image_formats_are_accepted(tmp_path: Path, extension: str) -> None:
    path = tmp_path / f"sample.{extension}"
    Image.new("RGB", (9, 6), (20, 80, 160)).save(path)

    tensor = _preprocess(path, 8)

    assert tensor.shape == (3, 8, 8)
    assert torch.isfinite(tensor).all()


def test_path_and_pil_inputs_match_for_lossless_image(tmp_path: Path) -> None:
    array = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    image = Image.fromarray(array, mode="RGB")
    path = tmp_path / "sample.png"
    image.save(path)

    torch.testing.assert_close(_preprocess(path, 8), _preprocess(image, 8))


def test_missing_and_corrupt_images_raise_useful_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _preprocess(tmp_path / "missing.png", 8)

    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"not an image")
    with pytest.raises(UnidentifiedImageError):
        _preprocess(corrupt, 8)
