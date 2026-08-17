from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pytest

import rosetta_copy.model as model_module


def _config(payload: bytes, *, digest: str | None = None) -> dict:
    return {
        "input_side": 8,
        "descriptor_dimensions": 256,
        "filename": "checkpoint.safetensors",
        "sha256": digest or hashlib.sha256(payload).hexdigest(),
        "url": "https://example.invalid/checkpoint.safetensors",
    }


def _serve(
    monkeypatch: pytest.MonkeyPatch, payload: bytes, seen: dict | None = None
) -> None:
    def fake_urlopen(request, *, timeout):
        if seen is not None:
            seen["url"] = request.full_url
            seen["user_agent"] = request.headers["User-agent"]
            seen["timeout"] = timeout
        return io.BytesIO(payload)

    monkeypatch.setattr(model_module.urllib.request, "urlopen", fake_urlopen)


def test_download_is_atomic_and_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"valid checkpoint bytes"
    config = _config(payload)
    seen: dict = {}
    monkeypatch.setattr(model_module, "_variant_config", lambda _variant: config)
    _serve(monkeypatch, payload, seen)

    result = model_module.download_weights("s224", tmp_path)

    assert result == tmp_path / config["filename"]
    assert result.read_bytes() == payload
    assert model_module._sha256(result) == config["sha256"]
    assert not list(tmp_path.glob("*.download"))
    assert seen == {
        "url": config["url"],
        "user_agent": "rosetta-copy/1.0",
        "timeout": 60,
    }


def test_valid_cached_checkpoint_skips_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"cached checkpoint"
    config = _config(payload)
    destination = tmp_path / config["filename"]
    destination.write_bytes(payload)
    monkeypatch.setattr(model_module, "_variant_config", lambda _variant: config)

    def unexpected_network(*_args, **_kwargs):
        raise AssertionError("network should not be used for a valid cache hit")

    monkeypatch.setattr(model_module.urllib.request, "urlopen", unexpected_network)
    assert model_module.download_weights("s224", tmp_path) == destination


def test_corrupt_cached_checkpoint_is_rejected_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(b"expected")
    destination = tmp_path / config["filename"]
    destination.write_bytes(b"corrupt")
    monkeypatch.setattr(model_module, "_variant_config", lambda _variant: config)

    def unexpected_network(*_args, **_kwargs):
        raise AssertionError("corrupt cache must require an explicit force")

    monkeypatch.setattr(model_module.urllib.request, "urlopen", unexpected_network)
    with pytest.raises(RuntimeError, match="Checksum mismatch for existing file"):
        model_module.download_weights("s224", tmp_path)
    assert destination.read_bytes() == b"corrupt"


def test_force_replaces_a_corrupt_cached_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"replacement"
    config = _config(payload)
    destination = tmp_path / config["filename"]
    destination.write_bytes(b"corrupt")
    monkeypatch.setattr(model_module, "_variant_config", lambda _variant: config)
    _serve(monkeypatch, payload)

    assert model_module.download_weights("s224", tmp_path, force=True) == destination
    assert destination.read_bytes() == payload


def test_bad_download_never_becomes_the_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(b"expected")
    monkeypatch.setattr(model_module, "_variant_config", lambda _variant: config)
    _serve(monkeypatch, b"wrong payload")

    with pytest.raises(RuntimeError, match="Downloaded checkpoint checksum mismatch"):
        model_module.download_weights("s224", tmp_path)
    assert not (tmp_path / config["filename"]).exists()
    assert not list(tmp_path.glob("*.download"))


def test_default_cache_dir_honours_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ROSETTA_COPY_CACHE", str(tmp_path / "custom-cache"))
    assert model_module.default_cache_dir() == tmp_path / "custom-cache"
