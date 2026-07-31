# Model card

## Model summary

Rosetta NP S224 and S336 are global image descriptors for copy detection and near-duplicate retrieval. Both use a DINOv2 ViT-S/14 backbone, the CLS token, and a learned projection head:

`LayerNorm(384) → Linear(384, 512) → GELU → Linear(512, 256) → L2 normalization`.

The model produces one 256-dimensional vector per image. Cosine similarity is therefore the dot product of two output vectors. No regional descriptor, metadata signal, pairwise comparator or geometric verifier is included.

## Published variants

| ID | Input | Parameters | Weight file | SHA-256 |
| --- | ---: | ---: | --- | --- |
| `s224` | 224×224 | 22,385,792 | `rosetta-dinov2s-s224-256d-v1.safetensors` | `4342f100daa15ea246cdec90478d85dff37e3ac32a41005c6e1786c5a2acc7f7` |
| `s336` | 336×336 | 22,385,792 active¹ | `rosetta-dinov2s-s336-256d-v1.safetensors` | `96ededa0fd442c9966f9bf7c4e3f7647bb95c000a1503f43dc48f7b45d2022f4` |

¹ The S336 training checkpoint contains inactive experimental pooling tensors. The public loader intentionally selects only `backbone.*` and `global_head.*`; the active inference graph is the same CLS architecture as S224.

## Intended use

- candidate generation for image deduplication;
- source/copy retrieval in a large reference gallery;
- offline dataset cleaning and human-review queues;
- a compact first stage before regional or geometric verification.

The model is not intended to determine copyright ownership, authorship, malicious intent or whether a file may safely be deleted. High-stakes automatic merges should require deployment-specific calibration and, for localized matches, a second verification stage.

## Preprocessing contract

1. decode with PIL and convert to RGB;
2. preserve aspect ratio and pad to the selected square input;
3. Lanczos resampling, centered, padding color `(124, 116, 104)`;
4. convert to float in `[0, 1]`;
5. normalize with ImageNet mean `(0.485, 0.456, 0.406)` and std `(0.229, 0.224, 0.225)`.

Changing crop/pad behavior changes the descriptor and invalidates published benchmark numbers.

## Training overview

Training used DISC21-derived fit data and a custom transformation library. The family coverage includes identity/re-encoding, resampling and appearance, global geometry, containment, screenshot/page layouts, overlays/annotations, realistic compositions, content change and adversarial cases. Natural positive and no-source queries and mined hard negatives were mixed with transformed positive pairs. Four final transformer blocks and the projection head were fine-tuned.

Exact pool sizes, sampling ratios, steps, learning rates and data caveats are documented in [`docs/TRAINING.md`](docs/TRAINING.md).

## Evaluation

The main public comparison uses the exact one-million-image DISC21 gallery and separately reports DISC21-test and NDEC. S224, S336 and official SSCD `disc_mixup` embeddings were evaluated on identical natural bytes and identical frozen augmented bytes. The augmented evaluation preserves the original query label and searches the full gallery; it is not a self-pair similarity test.

See [`docs/BENCHMARK_PROTOCOL.md`](docs/BENCHMARK_PROTOCOL.md) and [`docs/RESULTS.md`](docs/RESULTS.md).

## Limitations and known failure modes

- Global descriptors can miss small copied regions, extreme crops and localized splices.
- Semantic look-alikes can score highly; a fixed cosine threshold is not portable across galleries.
- S224 loses more spatial detail. S336 costs more compute but stores the same 512-byte fp16 descriptor.
- S336 is not uniformly superior: S224 obtained higher natural Recall@1 on NDEC in the published run.
- Both models are trained on DISC21-derived data. NDEC shares the DISC21 gallery/domain and is not fully independent.
- SSCD was trained specifically for copy detection and remains much stronger on natural DISC21 queries.
- The robustness benchmark applies one composite transformation per query. Per-family rows are stratified by dominant anchor, not a complete causal sweep of isolated transformations.
- Dataset content may carry demographic, geographic and cultural biases inherited from the upstream collections.

## Reproducibility and integrity

The two Release assets are immutable for v1.0.0 and verified by SHA-256. The public results are also provided in machine-readable form. Internal absolute paths and source images are intentionally not published; the latter remain governed by their dataset licenses.

## License

Code and Rosetta weights: Apache-2.0. DINOv2 code and upstream weights: Apache-2.0. Dataset licenses are separate. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

