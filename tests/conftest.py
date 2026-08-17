from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file

import rosetta_copy.model as model_module


class TinyRetrievalModel(nn.Module):
    """Small deterministic stand-in exercising the public loading path."""

    def __init__(self, _backbone_config: dict, descriptor_dimensions: int) -> None:
        super().__init__()
        self.backbone = nn.AdaptiveAvgPool2d((1, 1))
        self.global_head = nn.Linear(3, descriptor_dimensions)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pooled = self.backbone(pixel_values).flatten(1)
        return F.normalize(self.global_head(pooled), dim=-1)


@dataclass(frozen=True)
class TinyCheckpoint:
    path: Path
    config: dict


@pytest.fixture
def tiny_checkpoint(tmp_path: Path) -> TinyCheckpoint:
    model = TinyRetrievalModel({}, 256)
    with torch.no_grad():
        weights = torch.linspace(-0.75, 0.75, 256 * 3).reshape(256, 3)
        bias = torch.linspace(-0.1, 0.1, 256)
        model.global_head.weight.copy_(weights)
        model.global_head.bias.copy_(bias)

    path = tmp_path / "tiny.safetensors"
    save_file(model.state_dict(), path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    config = {
        "input_side": 8,
        "descriptor_dimensions": 256,
        "filename": path.name,
        "sha256": digest,
        "url": "https://example.invalid/tiny.safetensors",
    }
    return TinyCheckpoint(path=path, config=config)


@pytest.fixture
def tiny_runtime(
    monkeypatch: pytest.MonkeyPatch, tiny_checkpoint: TinyCheckpoint
) -> TinyCheckpoint:
    monkeypatch.setattr(
        model_module, "_variant_config", lambda _variant: tiny_checkpoint.config
    )
    monkeypatch.setattr(model_module, "_RetrievalModel", TinyRetrievalModel)
    return tiny_checkpoint


@pytest.fixture
def tiny_encoder(tiny_runtime: TinyCheckpoint) -> model_module.RosettaEncoder:
    return model_module.RosettaEncoder("s224", tiny_runtime.path, device="cpu")
