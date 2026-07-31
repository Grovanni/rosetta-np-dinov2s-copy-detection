from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
from importlib.resources import files
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from safetensors.torch import load_file
from transformers import Dinov2Config, Dinov2Model

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
_FILL = (124, 116, 104)


def _read_json(name: str) -> dict:
    resource = files("rosetta_copy").joinpath("configs", name)
    return json.loads(resource.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def list_variants() -> tuple[str, ...]:
    return tuple(sorted(_read_json("variants.json")))


def _variant_config(variant: str) -> dict:
    variants = _read_json("variants.json")
    key = variant.lower()
    if key not in variants:
        choices = ", ".join(sorted(variants))
        raise ValueError(f"Unknown variant {variant!r}; choose one of: {choices}")
    return variants[key]


def default_cache_dir() -> Path:
    override = os.environ.get("ROSETTA_COPY_CACHE")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "rosetta-copy"


def download_weights(
    variant: str,
    cache_dir: str | Path | None = None,
    *,
    force: bool = False,
) -> Path:
    """Download one Release asset, atomically, and verify its SHA-256."""

    config = _variant_config(variant)
    root = Path(cache_dir).expanduser() if cache_dir else default_cache_dir()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / config["filename"]

    if destination.is_file() and not force:
        if _sha256(destination) == config["sha256"]:
            return destination
        raise RuntimeError(
            f"Checksum mismatch for existing file: {destination}. "
            "Delete it or pass force=True to download it again."
        )

    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".download", dir=root
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        request = urllib.request.Request(
            config["url"], headers={"User-Agent": "rosetta-copy/1.0"}
        )
        with urllib.request.urlopen(request) as response, temporary.open("wb") as out:
            shutil.copyfileobj(response, out, length=8 * 1024 * 1024)
        observed = _sha256(temporary)
        if observed != config["sha256"]:
            raise RuntimeError(
                f"Downloaded checkpoint checksum mismatch: {observed}"
            )
        os.replace(temporary, destination)
        return destination
    finally:
        temporary.unlink(missing_ok=True)


class _RetrievalModel(nn.Module):
    def __init__(self, backbone_config: dict, descriptor_dimensions: int) -> None:
        super().__init__()
        self.backbone = Dinov2Model(Dinov2Config.from_dict(backbone_config))
        hidden = int(backbone_config["hidden_size"])
        self.global_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 512),
            nn.GELU(),
            nn.Linear(512, descriptor_dimensions),
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        output = self.backbone(pixel_values=pixel_values)
        cls_token = output.last_hidden_state[:, 0]
        return F.normalize(self.global_head(cls_token), dim=-1)


def _open_rgb(image: str | Path | Image.Image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    with Image.open(image) as opened:
        opened.load()
        return opened.convert("RGB")


def _preprocess(image: str | Path | Image.Image, side: int) -> torch.Tensor:
    rgb = _open_rgb(image)
    fitted = ImageOps.pad(
        rgb,
        (side, side),
        method=Image.Resampling.LANCZOS,
        color=_FILL,
        centering=(0.5, 0.5),
    )
    array = np.asarray(fitted, dtype=np.float32).copy() / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return (tensor - _MEAN) / _STD


class RosettaEncoder:
    """Inference wrapper for the S224 and S336 Rosetta checkpoints."""

    def __init__(
        self,
        variant: str,
        weights: str | Path,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
    ) -> None:
        self.variant = variant.lower()
        self.config = _variant_config(self.variant)
        self.input_side = int(self.config["input_side"])
        self.device = torch.device(device)
        self.dtype = dtype or (
            torch.float16 if self.device.type == "cuda" else torch.float32
        )

        checkpoint = Path(weights).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        observed = _sha256(checkpoint)
        if observed != self.config["sha256"]:
            raise RuntimeError(
                f"Unexpected checkpoint SHA-256 for {self.variant}: {observed}"
            )

        model = _RetrievalModel(
            _read_json("backbone.json"),
            int(self.config["descriptor_dimensions"]),
        )
        raw = load_file(str(checkpoint), device="cpu")
        selected = {
            key: value
            for key, value in raw.items()
            if key.startswith("backbone.") or key.startswith("global_head.")
        }
        missing, unexpected = model.load_state_dict(selected, strict=True)
        if missing or unexpected:
            raise RuntimeError(
                f"Incompatible checkpoint: missing={missing}, unexpected={unexpected}"
            )
        model.requires_grad_(False)
        model.eval()
        self.model = model.to(device=self.device, dtype=self.dtype)

    @classmethod
    def from_pretrained(
        cls,
        variant: str = "s336",
        *,
        weights: str | Path | None = None,
        cache_dir: str | Path | None = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
    ) -> "RosettaEncoder":
        checkpoint = (
            Path(weights)
            if weights is not None
            else download_weights(variant, cache_dir)
        )
        return cls(variant, checkpoint, device=device, dtype=dtype)

    @torch.inference_mode()
    def encode(
        self,
        images: Sequence[str | Path | Image.Image]
        | Iterable[str | Path | Image.Image],
        *,
        batch_size: int = 32,
    ) -> np.ndarray:
        items = list(images)
        if not items:
            return np.empty((0, 256), dtype=np.float32)
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        outputs: list[torch.Tensor] = []
        for start in range(0, len(items), batch_size):
            batch = torch.stack(
                [_preprocess(item, self.input_side) for item in items[start : start + batch_size]]
            ).to(device=self.device, dtype=self.dtype)
            vectors = self.model(batch).float().cpu()
            outputs.append(F.normalize(vectors, dim=-1))
        return torch.cat(outputs, dim=0).numpy()

