# Results

All rows use the protocol in [`BENCHMARK_PROTOCOL.md`](BENCHMARK_PROTOCOL.md). “Augmented” means a frozen transformed copy of every query searched against the complete one-million-reference gallery.

## Retrieval quality

### DISC21 test — 10,000 positive and 40,000 no-source queries

| Model | Condition | R@1 | R@10 | R@64 | µAP |
| --- | --- | ---: | ---: | ---: | ---: |
| S224 | natural | 45.74 | 55.54 | 62.30 | 37.255 |
| S224 | augmented | 37.64 | 48.01 | 55.67 | 28.615 |
| S336 | natural | 48.35 | 57.66 | 64.46 | 39.966 |
| S336 | augmented | 41.34 | 51.59 | 58.58 | 31.981 |
| SSCD `disc_mixup` | natural | **65.29** | **70.32** | **73.73** | **59.463** |
| SSCD `disc_mixup` | augmented | **45.56** | **52.80** | 57.86 | **38.950** |

Values are percentages. S336 is the strongest Rosetta variant on DISC21. It trails SSCD on natural queries, comes much closer under transformation, and slightly exceeds SSCD at augmented Recall@64 (58.58 versus 57.86).

Natural-to-augmented deltas:

| Model | ΔR@1 | ΔR@10 | ΔR@64 | ΔµAP |
| --- | ---: | ---: | ---: | ---: |
| S224 | -8.10 | -7.53 | -6.63 | -8.640 |
| S336 | **-7.01** | **-6.07** | **-5.88** | **-7.985** |
| SSCD | -19.73 | -17.52 | -15.87 | -20.512 |

### NDEC — 5,009 positive and 44,243 no-source queries

| Model | Condition | R@1 | R@10 | R@64 | µAP |
| --- | --- | ---: | ---: | ---: | ---: |
| S224 | natural | **94.350** | **96.926** | **97.744** | 39.123 |
| S224 | augmented | 76.981 | 85.806 | 90.457 | 26.801 |
| S336 | natural | 90.018 | 96.147 | 97.704 | 32.125 |
| S336 | augmented | **78.319** | **88.920** | **92.893** | 24.068 |
| SSCD `disc_mixup` | natural | 76.183 | 79.976 | 82.472 | **44.241** |
| SSCD `disc_mixup` | augmented | 57.277 | 64.025 | 68.796 | **27.537** |

NDEC shows why both variants are published. S224 is strongest on natural top-1 retrieval; S336 loses fewer points and overtakes it after augmentation. SSCD has lower recall but higher µAP because its global ranking of true and false signals differs materially.

| Model | ΔR@1 | ΔR@10 | ΔR@64 | ΔµAP |
| --- | ---: | ---: | ---: | ---: |
| S224 | -17.369 | -11.120 | -7.287 | -12.322 |
| S336 | **-11.699** | **-7.227** | **-4.811** | **-8.057** |
| SSCD | -18.906 | -15.951 | -13.675 | -16.703 |

## False signals at frozen confidence thresholds

Each cell is `true positive-source signals / false no-source signals`.

| Dataset | Model | Natural | Augmented |
| --- | --- | ---: | ---: |
| DISC21 | S224 | 2,359 / 95 | 1,584 / 52 |
| DISC21 | S336 | 2,636 / 136 | 1,898 / 84 |
| DISC21 | SSCD | 4,876 / 396 | 2,768 / 175 |
| NDEC | S224 | 3,052 / 7,071 | 1,841 / 4,429 |
| NDEC | S336 | 2,717 / 8,759 | 1,940 / 5,858 |
| NDEC | SSCD | 3,169 / 14,363 | 2,014 / 6,832 |

The apparent drop in false signals after augmentation does not mean augmentation improved safety: true-signal counts also collapsed. Thresholds calibrated on one gallery/query distribution are not portable to NDEC without recalibration.

At each model’s **natural false-signal budget**, descriptive equal-FP true-signal counts were:

| Dataset | Model | Natural | Augmented at same FP count |
| --- | --- | ---: | ---: |
| DISC21 | S224 | 2,359 @ 95 FP | 1,786 @ 95 FP |
| DISC21 | S336 | 2,636 @ 136 FP | 2,105 @ 136 FP |
| DISC21 | SSCD | 4,876 @ 396 FP | 3,222 @ 396 FP |
| NDEC | S224 | 3,052 @ 7,071 FP | 2,498 @ 7,071 FP |
| NDEC | S336 | 2,717 @ 8,759 FP | 2,500 @ 8,759 FP |
| NDEC | SSCD | 3,169 @ 14,363 FP | 2,590 @ 14,363 FP |

No-source threshold transitions (`new / persistent / resolved`) were:

| Dataset | S224 | S336 | SSCD |
| --- | ---: | ---: | ---: |
| DISC21 | 12 / 40 / 55 | 24 / 60 / 76 | 28 / 147 / 249 |
| NDEC | 379 / 4,050 / 3,021 | 464 / 5,394 / 3,365 | 98 / 6,734 / 7,629 |

## Dominant augmentation family — Recall@1

These are stratified composite cases, not isolated one-factor tests.

### DISC21 test

| Anchor family | S224 | S336 | SSCD |
| --- | ---: | ---: | ---: |
| identity / encoding | — | 47.80 | **63.26** |
| resampling / appearance | — | 41.64 | **55.14** |
| global geometry | — | **30.58** | 24.84 |
| containment | — | **31.98** | 10.43 |
| screenshot / page | — | 46.08 | **52.17** |
| overlay / annotation | — | 45.45 | **57.91** |
| realistic composition | — | 45.87 | **55.18** |

S224 family rows are omitted here because the audited compact summary retained only the S336/SSCD family matrix. The aggregate S224 results above are complete.

### NDEC

| Anchor family | S336 | SSCD |
| --- | ---: | ---: |
| identity / encoding | **89.94** | 74.58 |
| resampling / appearance | **80.59** | 67.46 |
| global geometry | **56.70** | 33.94 |
| containment | **61.87** | 22.35 |
| screenshot / page | **86.43** | 65.87 |
| overlay / annotation | **84.48** | 67.27 |
| realistic composition | **88.25** | 69.51 |

## Statistical uncertainty

Paired 10,000-replicate cluster-bootstrap 95% confidence intervals for the Recall@1 drop:

| Dataset | S224 | S336 | SSCD |
| --- | ---: | ---: | ---: |
| DISC21 | [-8.75, -7.44] | [-7.64, -6.37] | [-20.52, -18.95] |
| NDEC | [-18.46, -16.31] | [-12.74, -10.71] | [-20.00, -17.80] |

All intervals exclude zero. Statistical significance does not remove the domain-overlap and single-composite-view limitations described in the protocol.

## Compute and storage

Measured on the same RTX 3060 evaluation machine. Query time covers approximately 50,000 images; gallery encoding covers one million images. Search is exact top-64 retrieval over the same gallery.

| Model | Descriptor fp16 | Query encode DISC / NDEC | 1M gallery encode | Search DISC / NDEC | Peak extraction VRAM |
| --- | ---: | ---: | ---: | ---: | ---: |
| S224 | **512 B** | **90.69 / 89.24 s** | **3,858.45 s (64.3 min)** | **36.88 / 36.26 s** | **395 MiB** |
| S336 | **512 B** | 160.23 / 158.51 s | 4,963.87 s (82.7 min) | 37.05 / 36.33 s | 743 MiB |
| SSCD | 1,024 B | 174.39 / 171.84 s | 4,580.07 s (76.3 min) | 42.30 / 42.19 s | 1,676 MiB |

S336 retains the same permanent descriptor size as S224. Its cost is extraction compute: about 1.77× S224 on the query sets. Search time remains nearly identical because both Rosetta variants are 256-dimensional.

