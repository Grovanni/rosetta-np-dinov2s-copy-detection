# Benchmark protocol

## Retrieval task

Each query is encoded once and searched by exact inner product against the same frozen one-million-image DISC21 reference gallery. All embeddings are L2-normalized, so inner product equals cosine similarity. Gallery vectors are stored as float16; score accumulation is float32, with TF32 disabled. The top 64 references are retained.

This is not an “original versus its own augmentation” test. Natural and augmented query variants independently search the full gallery, which preserves distractor competition and permits false-signal measurement.

## Query sets

| Dataset | Queries | Positive source | No source | Notes |
| --- | ---: | ---: | ---: | --- |
| DISC21 test | 50,000 | 10,000 | 40,000 | official test split |
| NDEC | 49,252 | 5,009 | 44,243 | 19,991 DISC easy + 24,252 Open Images hard no-source |
| **Total** | **99,252** | **15,009** | **84,243** | datasets always reported separately |

Query identity is namespaced by dataset. An overlapping textual ID never implies that a DISC21-test and NDEC query are the same asset.

NDEC uses the DISC21 gallery and partially overlaps its domain. It broadens query conditions and hard no-source coverage, but is not a fully independent gallery benchmark.

## Candidates

- Rosetta S224, checkpoint SHA-256 `4342f100…acc7f7`;
- Rosetta S336, checkpoint SHA-256 `96ededa0…2022f4`;
- official SSCD ResNet-50 `disc_mixup`, one global 512-dimensional descriptor.

All candidates received identical natural files and identical augmented JPEG bytes. SSCD is reported as a strong specialized comparator; its descriptor occupies twice Rosetta’s fp16 storage. RDCD and the earlier perceptual-hash cascade were not part of this frozen external run and are therefore not mixed into its tables.

## Metrics

- **Recall@1/10/64:** fraction of positive-source queries whose official source occurs within the first K results.
- **micro-AP (µAP):** official global ISC-style average precision, computed across positive and no-source queries.
- **confidence:** `top1_score − mean(top2_score, top3_score, top4_score)`.
- **fixed-threshold true/false signals:** positives and no-source queries above a threshold calibrated earlier on the frozen Natural100k development benchmark.
- **equal-FP true signals:** descriptive interpolation at the natural condition’s false-signal count; no threshold was reselected to claim a deployable operating point.
- **transitions:** no-source queries newly crossing, persistently crossing, or leaving the frozen threshold after augmentation.

Frozen confidence thresholds:

| Model | Threshold |
| --- | ---: |
| S224 | 0.1582845151 |
| S336 | 0.1523305327 |
| SSCD | 0.1397682577 |

A lower number of false signals after augmentation is not automatically safer: augmentation can lower all confidence values and simultaneously destroy true-positive recall. Fixed-threshold rows must be read with their true-signal counts.

## Paired statistics

- paired cluster bootstrap with 10,000 replicates for natural-to-augmented deltas;
- McNemar tests on paired top-1 successes;
- queries, not retrieved pairs, are the unit of analysis;
- DISC21-test and NDEC are never pooled into a single headline score.

## Protocol status

Models and thresholds were frozen before augmented outcomes. The transformation plan was preregistered and its rendered bytes sealed before model scoring. However, natural results had already informed development; this is a controlled external robustness extension, not a blind competition submission.

