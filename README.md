# Rosetta

**Compact 256-dimensional descriptors for robust image copy retrieval.**

Rosetta retrieves likely source images after crops, containment, screenshots, rotations, re-encoding and chained structural transformations. It fine-tunes a [DINOv2](https://github.com/facebookresearch/dinov2) ViT-S/14 backbone and represents each image with one L2-normalized 256-dimensional vector, enabling cosine search without a pairwise reranker.

Rosetta is a specialized retrieval model, not a universal replacement for SSCD. It provides stronger recall in some measured regimes and loses less recall under the published transformation pipeline, while SSCD retains higher absolute micro-AP across the reported benchmarks.

The `NP` suffix retained in the repository URL refers to the original internal `no_privacy` experiment profile. The public model family is simply **Rosetta**.

## Install

Install the tagged public release directly from GitHub:

```bash
pip install "git+https://github.com/Grovanni/rosetta-np-dinov2s-copy-detection.git@v1.0.0"
```

The first `from_pretrained` call downloads only the selected checkpoint and verifies its SHA-256 digest. You can also download it explicitly:

```bash
rosetta-copy download s224
```

Release assets and checksums are listed in [`weights/SHA256SUMS.txt`](weights/SHA256SUMS.txt).

## Search a reference folder

This small exact-search example encodes a reference folder, submits one query and prints its five nearest candidates:

```python
from pathlib import Path

import numpy as np

from rosetta_copy import RosettaEncoder

extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
references = sorted(
    path for path in Path("references").iterdir()
    if path.is_file() and path.suffix.lower() in extensions
)
if not references:
    raise RuntimeError("No reference images found")

encoder = RosettaEncoder.from_pretrained("s224", device="cuda")
reference_vectors = encoder.encode(references, batch_size=64)
query_vector = encoder.encode(["query.jpg"])[0]

scores = reference_vectors @ query_vector
top_k = np.argsort(scores)[-min(5, len(scores)):][::-1]
for rank, index in enumerate(top_k, start=1):
    print(f"{rank}: {references[index]}  cosine={scores[index]:.4f}")
```

The returned arrays are `float32`, shape `(N, 256)`, and L2-normalized. Cast reference vectors to `float16` for 512-byte persistent descriptors. NumPy exact search is sufficient for a small gallery; large deployments can place the same vectors in FAISS, HNSW or another cosine/inner-product index.

A nearest-neighbour score is a retrieval signal, not a portable match decision. Calibrate any acceptance threshold on a disjoint set representative of the deployment gallery.

The CLI can also produce reusable embedding arrays:

```bash
rosetta-copy embed s224 reference-001.jpg reference-002.jpg --output references.npy
```

## Choose a variant

Neither checkpoint is the universal default. They share the same descriptor size but target different operating points:

| Variant | Input | Checkpoint | Operating point |
| --- | ---: | ---: | --- |
| **S224 — balanced / fast** | 224 px | 85.4 MiB | fastest extraction; strongest Rosetta result on natural NDEC |
| **S336 — structural robustness** | 336 px | 87.2 MiB | strongest Rosetta result on DISC21 and augmented NDEC; higher extraction cost |

Both produce one 256d descriptor: 1,024 bytes in `float32` or 512 bytes in `float16`. The checkpoints are separate Release assets, so users download only the selected variant.

Preprocessing is part of the model contract: PIL decode, RGB conversion, aspect-preserving padding to 224×224 or 336×336 with `(124, 116, 104)`, Lanczos resampling, then ImageNet mean/std normalization.

## Benchmark position

All models searched the same exact one-million-image DISC21 reference gallery. The evaluation contains 50,000 DISC21-test queries and 49,252 NDEC queries. The augmented condition re-encodes one deterministic transformed copy of every query and still searches the complete frozen gallery; it is not a self-pair similarity test.

The table reports both Recall@1 and micro-AP to expose the recall/ranking trade-off directly. SSCD is the official `disc_mixup` ResNet-50 global descriptor.

| Dataset and condition | Model | Recall@1 | micro-AP |
| --- | --- | ---: | ---: |
| DISC21 test — natural | S224 | 45.74% | 37.26% |
|  | S336 | 48.35% | 39.97% |
|  | SSCD | **65.29%** | **59.46%** |
| DISC21 test — augmented | S224 | 37.64% | 28.62% |
|  | S336 | 41.34% | 31.98% |
|  | SSCD | **45.56%** | **38.95%** |
| NDEC — natural | S224 | **94.35%** | 39.12% |
|  | S336 | 90.02% | 32.13% |
|  | SSCD | 76.18% | **44.24%** |
| NDEC — augmented | S224 | 76.98% | 26.80% |
|  | S336 | **78.32%** | 24.07% |
|  | SSCD | 57.28% | **27.54%** |

Rosetta's main advantage in these results is high recall in the NDEC regimes and relative retention under added structural transformations. On DISC21, S336 loses 7.01 Recall@1 points under augmentation, versus 19.73 points for SSCD; on NDEC the corresponding losses are 11.70 and 18.91 points. SSCD nevertheless keeps the highest micro-AP in every row and remains substantially stronger on natural DISC21 queries.

Full Recall@1/10/64, fixed-threshold false signals, equal-false-positive comparisons, confidence intervals and per-family results are available in [`docs/RESULTS.md`](docs/RESULTS.md) and [`benchmarks/results.json`](benchmarks/results.json).

## Cost

Measurements below use an NVIDIA RTX 3060 12 GB. Gallery and query embeddings are computed once; exact-search timings cover the full query set against one million references.

| Model | fp16 descriptor | Encode 50k queries | Encode 1M references | Exact search | Peak extraction VRAM |
| --- | ---: | ---: | ---: | ---: | ---: |
| S224 | 512 B | 90.7 s | 64.3 min | 36.9 s | 395 MiB |
| S336 | 512 B | 160.2 s | 82.7 min | 37.1 s | 743 MiB |
| SSCD | 1,024 B | 174.4 s | 76.3 min | 42.3 s | 1,676 MiB |

## When Rosetta fits

Use Rosetta when compact permanent descriptors, inexpensive global retrieval and recall under structural transformations matter. S224 is the economical operating point; S336 trades more extraction compute for greater structural robustness.

Prefer SSCD when its stronger natural DISC21 ranking and higher measured micro-AP match the deployment objective. For small copied regions, extreme crops, composites or high-stakes automatic merges, use either global model only as candidate generation and add regional or geometric verification.

## Documentation

- [`MODEL_CARD.md`](MODEL_CARD.md): architecture, intended use and limitations.
- [`docs/TRAINING.md`](docs/TRAINING.md): data pools, sampling, losses and compute.
- [`docs/BENCHMARK_PROTOCOL.md`](docs/BENCHMARK_PROTOCOL.md): frozen gallery, labels, metrics and statistical protocol.
- [`docs/AUGMENTATIONS.md`](docs/AUGMENTATIONS.md): training families and external robustness pipeline.
- [`docs/RESULTS.md`](docs/RESULTS.md): complete public result tables and interpretation.

## Limitations

- Rosetta retrieves likely source/copy relationships; it does not determine copyright ownership, provenance or whether files should be automatically deleted.
- Scores are model- and gallery-dependent. A threshold from the public benchmark is not a deployment default.
- One global descriptor can miss small localized splices, composites and extreme crops.
- Training used DISC21-derived data. NDEC reuses the DISC21 gallery and part of its domain, so it is not a fully independent out-of-domain benchmark.
- The augmented benchmark uses one balanced composite view per query, not a complete factorial sweep of every transformation family.
- Natural benchmark results were known during development; this is not a blind challenge submission.

## License and citation

Code and Rosetta checkpoint files are released under Apache-2.0. DINOv2 code and pretrained weights are also Apache-2.0; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). DISC21 and NDEC images are not redistributed by this repository and remain subject to their own terms.

If this repository is useful, cite it using [`CITATION.cff`](CITATION.cff), and cite DINOv2, DISC21 and SSCD where appropriate.
