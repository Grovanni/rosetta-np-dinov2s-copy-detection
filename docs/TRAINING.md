# Training disclosure

This document reports the information needed to understand the two published checkpoints. Pool counts describe available records, not necessarily unique files seen exactly once. Both runs sampled with replacement, so “more pool data” is not equivalent to one exhaustive epoch.

## Common architecture and objective

- Backbone: DINOv2 ViT-S/14 without registers, initialized from `facebook/dinov2-small`.
- Representation: CLS token projected from 384 to 512 to 256 dimensions, then L2-normalized.
- Trainable backbone: last four transformer blocks.
- Main loss: symmetric in-batch contrastive loss, temperature 0.07.
- Additional signal: triplet ranking over positives and mined negatives.
- Retrieval target: one embedding and cosine similarity; certificate heads were out of scope.

## Data authority

Training data came from the public DISC21 training corpus and the DISC21 reference/query structure. The source images are not redistributed here.

The main sealed pack contained:

- 500,000 transformed primary pairs from 500,000 selected training sources;
- 50,000 natural query records: 10,000 positives and 40,000 official no-source negatives;
- all 1,000,000 official DISC21 references for retrieval/mining;
- primary transformed pairs encoded as JPEG quality 90, maximum side 448;
- references normalized to a maximum side of 224 in the local retrieval pack.

The S336 extension added 400,000 novel transformed pairs from 400,000 training sources with zero overlap against the earlier source selection. After split filtering, the S336 trainer exposed 854,881 primary `train_fit` records across the two packs.

The two pack integrity roots used during training were:

- V2: `2000903d028349d05f22dec6c4020fc6db98839dcaf9011a712af472f026f1c2`
- V3 extension: `7e3e674261ca162320e250d0ced725f41c42fadf58ebe41b062574c7aec90032`

These hashes identify local manifests, not downloadable datasets.

## Transformation-family coverage

Across the 900,000 primary pairs before split filtering:

| Family | V2 | V3 extension | Total |
| --- | ---: | ---: | ---: |
| identity / encoding | 49,820 | 40,135 | 89,955 |
| resampling / appearance | 79,594 | 64,065 | 143,659 |
| global geometry | 75,165 | 60,220 | 135,385 |
| containment | 70,254 | 55,660 | 125,914 |
| screenshot / page | 54,622 | 44,171 | 98,793 |
| overlay / annotation | 50,132 | 39,803 | 89,935 |
| realistic composition | 55,348 | 43,877 | 99,225 |
| content change | 34,922 | 28,129 | 63,051 |
| adversarial | 30,143 | 23,940 | 54,083 |

The transformation recipe library was deterministic per record and generated transformed positive pairs before training. This table describes training coverage; the independent robustness pipeline in [`AUGMENTATIONS.md`](AUGMENTATIONS.md) used a separately frozen implementation.

## S224 run

| Item | Value |
| --- | ---: |
| Selected checkpoint | final, step 400 |
| Batch size / steps | 96 / 400 |
| Sampled training pairs | 38,400 |
| Backbone LR | 2e-6 |
| Projection-head LR | 8e-5 |
| Warm-up | 40 steps |
| Sampling slots | 10 primary, 8 positive triplet, 5 natural, 1 hard negative |
| Sampling proportions | 41.67%, 33.33%, 20.83%, 4.17% |
| Available pools | 30,000 primary; 72,112 positive triplets; 12,000 natural; 286,808 mined hard negatives |
| GPU / elapsed | RTX 3060 12 GB / 377 s |
| Peak allocated VRAM | 6,235,986,432 B (5.81 GiB) |
| Random seed | 20260802 |

S224 was a short refinement from an earlier balanced Rosetta checkpoint, not a train-from-scratch run. The internal development gate contained 986 positive and 4,149 no-source natural queries and was explicitly not an independent generalization claim.

## S336 run

| Item | Value |
| --- | ---: |
| Parent run | 5,000 steps; checkpoint frozen at step 4,000 |
| Batch size | 64 |
| Pairs processed by parent run | 320,000 |
| Backbone LR | 1e-7 |
| Projection-head LR | 4e-6 |
| Warm-up | 50 steps |
| Sampling slots | 12 primary, 8 positive triplet, 5 natural, 1 hard negative |
| Sampling proportions | 46.15%, 30.77%, 19.23%, 3.85% |
| Available pools | 854,881 primary; 72,112 positive triplets; 44,865 natural; 286,808 mined hard negatives |
| GPU / elapsed for 5,000 steps | RTX 3060 12 GB / 4,036.7 s |
| Peak allocated VRAM | 9,251,842,048 B (8.62 GiB) |
| Random seed | 20261004 |

The run began from S224, changed the input grid from 16×16 patches to 24×24 patches through DINOv2 positional interpolation, and continued low-learning-rate fine-tuning. Step 4,000 was frozen as the Pareto checkpoint after paired natural and augmentation development comparisons. Its inference pooling remains CLS-only; experimental patch-pooling parameters embedded in the training checkpoint are inactive and ignored by the public loader.

## What was not trained

- no certificate or relation-classification head;
- no pairwise reranker;
- no regional index;
- no geometric verification;
- no test-time multi-view ensemble;
- no distillation from SSCD or RDCD.

