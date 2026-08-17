from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from rosetta_copy import RosettaEncoder, download_weights
from rosetta_copy.cli import main as cli_main


pytestmark = [
    pytest.mark.release,
    pytest.mark.skipif(
        os.environ.get("ROSETTA_RUN_RELEASE_TESTS") != "1",
        reason="set ROSETTA_RUN_RELEASE_TESTS=1 to download the public checkpoint",
    ),
]


def test_public_s224_checkpoint_download_load_and_inference(tmp_path: Path) -> None:
    checkpoint = download_weights("s224")
    encoder = RosettaEncoder.from_pretrained("s224", weights=checkpoint, device="cpu")
    image = Image.fromarray(
        (np.arange(224 * 224 * 3, dtype=np.uint32) % 256)
        .astype(np.uint8)
        .reshape(224, 224, 3),
        mode="RGB",
    )

    one_by_one = encoder.encode([image, image], batch_size=1)
    together = encoder.encode([image, image], batch_size=2)

    assert one_by_one.shape == (2, 256)
    assert one_by_one.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(one_by_one, axis=1), 1.0, atol=1e-5)
    np.testing.assert_allclose(one_by_one, together, atol=2e-5)

    expected_prefix = np.array(
        [
            0.02086669,
            0.01173385,
            -0.13823646,
            -0.03194860,
            -0.02925740,
            0.08543193,
            -0.00223410,
            0.01424121,
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(one_by_one[0, :8], expected_prefix, rtol=0, atol=2e-4)

    image_path = tmp_path / "query.png"
    output_path = tmp_path / "query.npy"
    image.save(image_path)
    assert (
        cli_main(
            [
                "embed",
                "s224",
                str(image_path),
                "--weights",
                str(checkpoint),
                "--device",
                "cpu",
                "--output",
                str(output_path),
            ]
        )
        == 0
    )
    np.testing.assert_allclose(np.load(output_path), one_by_one[:1], atol=2e-5)
