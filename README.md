# Rosetta NP — compact image copy detection

Rosetta NP is a pair of compact [DINOv2](https://github.com/facebookresearch/dinov2)-S/14 image descriptors fine-tuned for image copy detection. Each image becomes one L2-normalized 256-dimensional vector; retrieval is a cosine-similarity search with no pairwise reranker.

Two checkpoints are published because they cover different operating points:

| Variant | Input | Descriptor | Checkpoint | Best fit |
| --- | ---: | ---: | ---: | --- |
| **S224** | 224 px | 256d / 512 B in fp16 | 85.4 MiB | lowest latency and memory |
| **S336** | 336 px | 256d / 512 B in fp16 | 87.2 MiB | stronger DISC21 recall and augmentation robustness |

S336 is the recommended default when accuracy matters most. S224 remains useful for large-scale or latency-sensitive indexing. The weights are separate Release assets: users only download the variant they select.

## Install

```bash
git clone https://github.com/Grovanni/rosetta-np-dinov2s-copy-detection.git
cd rosetta-np-dinov2s-copy-detection
pip install -e .
```

Download one checkpoint:

```bash
rosetta-copy download s336
```

Or pass a local checkpoint explicitly. Release assets and SHA-256 checksums are listed in [`weights/SHA256SUMS.txt`](weights/SHA256SUMS.txt).

## Use

```python
from rosetta_copy import RosettaEncoder

encoder = RosettaEncoder.from_pretrained("s336", device="cuda")
vectors = encoder.encode(["query.jpg", "reference.jpg"])
similarity = float(vectors[0] @ vectors[1])
print(similarity)
```

The returned array is `float32`, shape `(N, 256)`, and L2-normalized. For storage, casting to `float16` uses 512 bytes per image. A minimal CLI is included:

```bash
rosetta-copy embed s336 image.jpg --output embedding.npy
```

Preprocessing is part of the model contract: PIL decode, RGB conversion, aspect-preserving padding to 224×224 or 336×336 with `(124, 116, 104)`, Lanczos resampling, then ImageNet mean/std normalization.

## Main benchmark

All models searched the same exact one-million-image DISC21 reference gallery. The test contains 50,000 DISC21-test queries and 49,252 NDEC queries. The augmented condition re-encodes one deterministic transformed copy of every query; it does **not** compare an image only to its own transform. Every query still retrieves against the complete frozen gallery, so recall and false signals remain measurable.

### Natural queries — Recall@1

| Dataset | S224 | S336 | official SSCD `disc_mixup` |
| --- | ---: | ---: | ---: |
| DISC21 test | 45.74% | **48.35%** | 65.29% |
| NDEC | **94.35%** | 90.02% | 76.18% |

### Augmented queries — Recall@1

| Dataset | S224 | S336 | official SSCD `disc_mixup` |
| --- | ---: | ---: | ---: |
| DISC21 test | 37.64% | 41.34% | **45.56%** |
| NDEC | 76.98% | **78.32%** | 57.28% |

S336 loses 7.01 points of Recall@1 under augmentation on DISC21, versus 19.73 points for SSCD. On NDEC, the drops are 11.70 and 18.91 points respectively. This is a robustness result, not a claim that Rosetta universally dominates SSCD: SSCD remains substantially ahead on natural DISC21 queries.

Full Recall@1/10/64, micro-AP, fixed-threshold false signals, equal-false-positive comparisons, uncertainty intervals, per-family results and runtime measurements are in [`docs/RESULTS.md`](docs/RESULTS.md) and [`benchmarks/results.json`](benchmarks/results.json).

## Documentation

- [`MODEL_CARD.md`](MODEL_CARD.md): intended use, variants and limitations.
- [`docs/TRAINING.md`](docs/TRAINING.md): data pools, sampling, losses and compute.
- [`docs/BENCHMARK_PROTOCOL.md`](docs/BENCHMARK_PROTOCOL.md): frozen gallery, labels, metrics and statistical protocol.
- [`docs/AUGMENTATIONS.md`](docs/AUGMENTATIONS.md): training families and external robustness pipeline.
- [`docs/RESULTS.md`](docs/RESULTS.md): complete public result tables and interpretation.

## Scope and limitations

- Rosetta finds likely source/copy relationships; it does not determine copyright ownership, provenance or whether two files should be automatically deleted.
- Scores are model- and gallery-dependent. Calibrate thresholds on a disjoint validation set representative of the deployment gallery.
- The checkpoints output one global descriptor. Small localized splices, composites and extreme crops may require regional retrieval or geometric verification.
- DISC21-derived data was used for training. NDEC reuses the DISC21 gallery and part of its domain, so it is not a fully independent out-of-domain benchmark.
- The published external benchmark was frozen before evaluation, but the natural benchmark results were already known during development. It should not be described as a blind challenge submission.

## License and citation

Code and Rosetta checkpoint files are released under Apache-2.0. DINOv2 code and pretrained weights are also Apache-2.0; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). DISC21 and NDEC images are not redistributed by this repository and remain subject to their own terms.

If this repository is useful, cite the repository using [`CITATION.cff`](CITATION.cff), and cite DINOv2, DISC21 and SSCD where appropriate.

