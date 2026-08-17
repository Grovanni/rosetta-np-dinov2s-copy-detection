from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import rosetta_copy.cli as cli


def test_download_command_forwards_cache_and_force(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    checkpoint = tmp_path / "checkpoint.safetensors"
    calls: list[tuple] = []

    def fake_download(variant, cache_dir, *, force):
        calls.append((variant, cache_dir, force))
        return checkpoint

    monkeypatch.setattr(cli, "download_weights", fake_download)
    result = cli.main(
        ["download", "s224", "--cache-dir", str(tmp_path), "--force"]
    )

    assert result == 0
    assert calls == [("s224", tmp_path, True)]
    assert capsys.readouterr().out.strip() == str(checkpoint)


def test_embed_command_loads_encoder_and_writes_numpy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    output = tmp_path / "vectors.npy"
    weights = tmp_path / "weights.safetensors"
    images = [tmp_path / "one.png", tmp_path / "two.png"]
    calls: dict = {}
    expected = np.arange(512, dtype=np.float32).reshape(2, 256)

    class StubEncoder:
        @classmethod
        def from_pretrained(cls, variant, **kwargs):
            calls["load"] = (variant, kwargs)
            return cls()

        def encode(self, paths, *, batch_size):
            calls["encode"] = (paths, batch_size)
            return expected

    monkeypatch.setattr(cli, "RosettaEncoder", StubEncoder)
    result = cli.main(
        [
            "embed",
            "s336",
            *(str(path) for path in images),
            "--weights",
            str(weights),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--device",
            "cpu",
            "--batch-size",
            "2",
            "--output",
            str(output),
        ]
    )

    assert result == 0
    assert calls["load"] == (
        "s336",
        {
            "weights": weights,
            "cache_dir": tmp_path / "cache",
            "device": "cpu",
        },
    )
    assert calls["encode"] == (images, 2)
    np.testing.assert_array_equal(np.load(output), expected)
    assert "saved (2, 256) float32" in capsys.readouterr().out


def test_cli_rejects_missing_command_and_unknown_variant() -> None:
    with pytest.raises(SystemExit) as missing:
        cli.main([])
    assert missing.value.code == 2

    with pytest.raises(SystemExit) as unknown:
        cli.main(["download", "unknown"])
    assert unknown.value.code == 2
