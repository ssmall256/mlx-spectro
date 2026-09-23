# mlx-spectro — Agent Rules

## What This Is

Spectral frontend for MLX. Fused Metal kernels for STFT/iSTFT, mel/MFCC/CQT frontends, spectral descriptors. Pure Python, no C extensions. All code lives in `src/mlx_spectro/spectral_ops.py` (~8.2k lines) and `src/mlx_spectro/__init__.py` (exports).

## Hard Rules

- **All Metal kernels use `mx.fast.metal_kernel()`** with compile-time template constants (HOP, FRAME, UNROLL_K). Do not use runtime parameters for values that affect thread dispatch.
- **Never break the compiled contract.** `get_compiled()`, `compiled_pair()`, `compiled_pair_nd()` must return callables that produce identical results to eager paths. Test both. All entry points default to the `"bfn"` layout; keep it that way — a split default is how the compiled half silently disagreed with the eager half before.
- **Do not add dependencies.** Core deps are `mlx` and `numpy` only. torch is optional (benchmark/compat only).
- **Autotune cache must remain optional.** Disable with `SPEC_MLX_AUTOTUNE=0`. Never require persisted autotune state for correctness.
- **iSTFT backend policy must be respected.** `"auto"`, `"mlx_fft"`, `"metal"`, `"torch_fallback"` — each must work independently. Don't assume Metal is available.

## Testing

```bash
pytest tests/ -v
```

16 test files. Accuracy assertions use `np.testing.assert_allclose(ours, ref, rtol=1e-6, atol=1e-6)` against torch/torchaudio/librosa. Those tests are skipped via `pytest.importorskip()` when the reference library is absent, so **install the `parity` extra** (`pip install -e ".[dev,parity]"`) before trusting a green run -- with only `dev`, 22 parity tests skip and the suite is green whether or not parity holds. CI installs `parity`. Keep torch out of the `dev` extra.

If you change a Metal kernel or transform output, run the full test suite and verify no regressions in accuracy or shape.

## Benchmarking

Use metalq for any GPU benchmark to avoid thermal contamination:

```bash
metalq submit -w -- python scripts/benchmark.py
metalq submit -w -- python scripts/benchmark_frontends.py
```

Checked-in baselines live in `benchmarks/baselines/`. Run `scripts/check_benchmark_regressions.py` after performance-sensitive changes.

## Key Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `SPEC_MLX_AUTOTUNE` | `"1"` | Metal kernel autotuning |
| `SPEC_MLX_TGX` | — | Force threadgroup size |
| `SPEC_MLX_AUTOTUNE_PERSIST` | `"1"` | Persist autotune to disk |
| `MLX_OLA_FUSE_NORM` | `"1"` | Fused OLA+normalization kernel |

## What Not to Touch

- Do not split `spectral_ops.py` into multiple files without explicit direction. The single-file design is intentional for `mx.compile` scope.
- Do not remove or rename public API symbols exported from `__init__.py`.
- Do not modify Metal kernel templates without running the full benchmark suite.

## Feedback

Before finishing your task, you MUST check whether you encountered any bugs, unexpected behavior, workflow friction, missing features, or surprising performance. If so, record each one:

```bash
devfeedback mlx-spectro "description of bug, friction, or idea"
```
