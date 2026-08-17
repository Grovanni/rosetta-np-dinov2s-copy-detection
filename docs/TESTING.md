# Software testing

The benchmark documents measure retrieval quality. This test suite has a different purpose: it detects software regressions in packaging, checkpoint handling, preprocessing, inference and the command-line interface.

## Test layers

The default suite is deterministic, CPU-compatible and offline after dependency installation. It covers:

- public variant metadata and published SHA-256 checksums;
- atomic checkpoint download, cache reuse, forced replacement and corruption rejection;
- RGB, grayscale, alpha-channel and common file-format preprocessing;
- real `safetensors` serialization and strict state-dict loading through a tiny deterministic model;
- descriptor shape, `float32` output, L2 normalization, reproducibility and batch-size invariance;
- empty, invalid and corrupt inputs;
- the DINOv2 wrapper with a small real `Dinov2Model` configuration;
- `download` and `embed` CLI behavior;
- a frozen numerical output that catches preprocessing or inference drift;
- CUDA inference when a CUDA runtime is available.

The tiny checkpoint is a software fixture, not a retrieval-quality proxy. It keeps pull-request CI fast while exercising the same checksum, `safetensors`, device, batching and preprocessing paths as the public checkpoints.

A separate release smoke test downloads the official S224 asset, verifies it, constructs the full model, checks a frozen public-model output, compares single-item and batched CPU inference, and runs the real `embed` CLI. It runs weekly and can be started manually from GitHub Actions. Keeping this network-dependent check separate prevents a transient Release or network failure from making every code pull request flaky.

## Local commands

Install the package and test tools:

```bash
python -m pip install -e ".[test]"
```

Run the normal offline suite:

```bash
python -m pytest -m "not release"
```

Include coverage:

```bash
python -m pytest -m "not release" --cov=rosetta_copy --cov-report=term-missing
```

Run the public-checkpoint integration test explicitly:

```bash
ROSETTA_RUN_RELEASE_TESTS=1 python -m pytest tests/test_release_integration.py -m release
```

On PowerShell, set the environment variable first:

```powershell
$env:ROSETTA_RUN_RELEASE_TESTS = "1"
python -m pytest tests/test_release_integration.py -m release
```

CUDA coverage is conditional: the corresponding test is collected everywhere and skipped when `torch.cuda.is_available()` is false. The public GitHub workflows intentionally use CPU runners; GPU behavior should also be checked before a release in the target CUDA environment.

## Continuous integration

`.github/workflows/ci.yml` runs the offline suite on Python 3.10 and 3.12, enforces at least 90% package coverage, builds both distribution formats, reinstalls the wheel and checks the installed CLI. `.github/workflows/release-smoke.yml` owns the slower network-dependent checkpoint validation.
