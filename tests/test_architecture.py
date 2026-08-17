from __future__ import annotations

import torch

from rosetta_copy.model import _RetrievalModel


def test_real_dinov2_wrapper_produces_normalised_descriptors() -> None:
    config = {
        "image_size": 28,
        "patch_size": 14,
        "num_channels": 3,
        "hidden_size": 12,
        "num_hidden_layers": 1,
        "num_attention_heads": 3,
        "intermediate_size": 24,
        "hidden_dropout_prob": 0.0,
        "attention_probs_dropout_prob": 0.0,
        "drop_path_rate": 0.0,
        "layer_norm_eps": 1e-6,
        "qkv_bias": True,
        "use_mask_token": True,
    }
    model = _RetrievalModel(config, descriptor_dimensions=16).eval()

    with torch.inference_mode():
        vectors = model(torch.rand(2, 3, 28, 28))

    assert vectors.shape == (2, 16)
    torch.testing.assert_close(torch.linalg.vector_norm(vectors, dim=1), torch.ones(2))
