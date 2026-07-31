# Augmentation methodology

Two distinct augmentation systems are relevant and must not be conflated.

## 1. Training transformation library

Training positive pairs were generated from DISC21 fit sources using nine broad families: identity/re-encoding, resampling/appearance, global geometry, containment, screenshot/page, overlay/annotation, realistic composition, content change and adversarial cases. Exact record counts are in [`TRAINING.md`](TRAINING.md).

The training packs were sealed before the published checkpoint runs. Natural positives, official no-source queries and mined hard negatives were sampled separately from transformed positives. No certificate objective was trained.

## 2. Frozen external robustness pipeline

The external test was designed after model training, frozen before its outcomes were computed, and applied byte-for-byte identically to S224, S336 and SSCD. It used a separate implementation from the training recipe library.

One deterministic augmented JPEG was generated for each of the 99,252 natural queries. Anchor families were balanced within each official label partition:

1. **identity / encoding** — no geometric change, JPEG quality 45–60;
2. **resampling / appearance** — downsample to 18–26% of the working size, then upscale;
3. **global geometry** — rotation of approximately ±33–36°;
4. **containment** — the query occupies approximately 40.0–41.8% of the output’s linear extent;
5. **screenshot / page** — offline Chromium/Playwright render of a frozen local HTML layout;
6. **overlay / annotation** — graphic overlay covering approximately 6–12% of the image;
7. **realistic composition** — same-parent main image plus a same-parent thumbnail; no foreign photograph.

Every non-identity family also received moderate appearance jitter and a final JPEG encode:

- brightness: 0.88–1.12;
- contrast: 0.86–1.14;
- saturation: 0.82–1.18;
- final JPEG quality: 62–90, 4:2:0 chroma subsampling.

The completed primary test intentionally excluded isolated destructive crops and label-changing/adversarial edits. Those operations can make the inherited source/no-source truth ambiguous and require a separately adjudicated protocol.

## Label-preservation gates

The generator enforced deterministic, outcome-blind gates:

- retained parent content at least 20%;
- visible output area at least 5%;
- source bounding-box short side at least 48 px;
- occlusion no greater than 60%;
- no foreign query or gallery photograph as an auxiliary asset;
- at most 32 deterministic attempts before failure.

Composition used only the same source image in multiple layout roles. Screenshot pages used procedural or frozen non-photographic assets. This prevents a no-source query from accidentally acquiring a genuine gallery source through augmentation.

## Quality control and integrity

- 75 DISC21 and 63 NDEC outputs were included in manual visual QC, including deterministic exceptions.
- No QC failure or label ambiguity was recorded.
- The complete pixel matrix was shared across all candidates.
- Models, natural embeddings and calibration thresholds were frozen before augmented outcomes were measured.

Integrity roots:

| Artifact | SHA-256 |
| --- | --- |
| preregistration | `a603cbcec3ca0f982c64adca3a6029020ee55de4895d1679afa9bcf7e1c90738` |
| pixel matrix | `96cfa6c5ccec95b2f32a9c39046701948b55da4aa5b72b9b1f5ed53b90f326c0` |
| manual QC receipt | `444b92d1854ef80e4eeb4a1624e41f07383307f9546513da1053dae098b8dee8` |
| comparison code | `959056ff7932a7fc4a83ee06c5c1dd7ac845a47d384475c7d9905cdbd556979f` |
| summary artifacts | `3e18b94372f042a489f113fe1fae3c93ebdf5c93682eeaf2223de2d22350b4d1` |

## Interpretation limit

The family tables stratify one composite augmentation per query by its dominant anchor. They are useful diagnostics, but not a fully crossed, seven-augmentation causal experiment. A future pure-family sweep should apply every isolated family to the same query set.

