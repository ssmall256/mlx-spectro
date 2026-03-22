# mlx-spectro

High-performance spectral frontend for Apple MLX. Fused Metal kernels for STFT/iSTFT achieve 2-3x faster STFT and 5-8x faster iSTFT vs PyTorch MPS. Pure Python — no C extensions.

## Architecture

Everything lives in two files under `src/mlx_spectro/`:

| Module | Purpose |
|--------|---------|
| `spectral_ops.py` | All transforms, Metal kernels, caches (~7.7k lines) |
| `__init__.py` | Public API exports |

**Core transform classes** (each has eager + compiled paths):

| Class | Purpose |
|-------|---------|
| `SpectralTransform` | STFT/iSTFT engine with fused Metal kernels |
| `MelSpectrogramTransform` | Mel-scale frontend (mlx_native, torchaudio_compat modes) |
| `FilteredSpectrogramTransform` | Custom filterbank projection (mel, chroma, log-freq) |
| `HybridCQTTransform` | Librosa-compatible hybrid constant-Q transform |
| `MFCCTransform` | Mel-frequency cepstral coefficients |
| `SpectralFeatureTransform` | Shared-STFT descriptor bundles (one STFT → multiple features) |
| `RepeatedShapeCompileCache` | Bounded shape promotion to compiled mode |

**Madmom-compat layer** (functional API, numpy + MLX paths):

| Function family | Purpose |
|-----------------|---------|
| `compute_filtered_spectrogram[_mlx]` | Log-frequency filtered spectrograms with caching |
| `compute_mel_spectrogram[_mlx]` | Mel-filtered spectrograms with caching |
| `madmom_multires_log_diff_features[_mlx]` | Multi-resolution log-spectrogram + spectral diff features |
| `madmom_multires_mel_stack[_mlx]` | Multi-resolution mel-spectrogram channel stacks |
| `madmom_single_resolution_log_stack[_mlx]` | Single-resolution log/mel spectrogram with optional stft_compat backend |
| `spectral_odf` | Onset detection functions (superflux, complex_flux, phase, HFC, etc.) |
| `triangular_filterbank`, `mel_filterbank`, `rectangular_filterbank` | Filterbank builders |
| `frame_starts_from_fps`, `stft_features_at_fps` | FPS-driven framing (fractional hop support) |

**Numpy/MLX dual-path convention**: Functions named `foo` return `np.ndarray` (call `mx.eval` internally). Functions named `foo_mlx` return lazy `mx.array` (no eval barrier). Downstream code that feeds results into MLX models should always use the `_mlx` variant to avoid device→host→device round-trips.

## Key Design Decisions

- **7 fused Metal kernels** via `mx.fast.metal_kernel()`: frame extraction (simple + tiled), OLA, OLA+normalization, power spectrum, STFT backward, iSTFT backward. Templates use compile-time constants (HOP, FRAME, UNROLL_K)
- **`mx.compile` caching**: Per-transform compiled callables via `get_compiled()`, `compiled_pair()`, `compiled_pair_nd()` for fixed-shape hot loops (10-20% speedup)
- **Autotune system**: Discovers optimal threadgroup sizes per kernel, persists to `~/.cache/mlx_spectro/`. Override with `SPEC_MLX_TGX` env var
- **iSTFT backend policy**: `"auto"` (Metal if available), `"mlx_fft"`, `"metal"`, `"torch_fallback"`
- **Compatibility controls**: `center_pad_mode` (reflect/constant), `center_tail_pad` (symmetric/minimal for madmom), mel `mode` (mlx_native/torchaudio_compat)
- **Differentiable STFT/iSTFT**: Full gradient support via Metal backward kernels

## Build & Run

```bash
pip install -e ".[dev]"            # dev install
pip install -e ".[benchmark]"      # + torch + mlx-stft for benchmarking
```

No C extensions, no special build steps. Pure Python + MLX.

## Tests

```bash
pytest tests/ -v
```

11 test files. Tests use `np.testing.assert_allclose()` for parity with torch/torchaudio/librosa. Optional imports skipped via `pytest.importorskip()`. No conftest.py. `test_mlx_numpy_parity.py` verifies `_mlx` variants match numpy counterparts.

## Benchmarks

Scripts in `scripts/`:
- `benchmark.py` — STFT/iSTFT timing + accuracy vs torch MPS
- `benchmark_frontends.py` — eager vs compiled mel/log-mel/MFCC
- `check_benchmark_regressions.py` — compare against `benchmarks/baselines/`

Pattern: 5 warmup, 20 iterations, median latency, markdown tables.

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `SPEC_MLX_AUTOTUNE` | Enable Metal kernel autotuning (default: `"1"`) |
| `SPEC_MLX_TGX` | Force threadgroup size (e.g., `256` or `kernel:256`) |
| `SPEC_MLX_AUTOTUNE_PERSIST` | Persist autotune results to disk (default: `"1"`) |
| `MLX_OLA_FUSE_NORM` | Enable fused OLA+normalization kernel (default: `"1"`) |

## Ecosystem

**No dependencies on** other ~/Code/ projects. **Consumed by**: mlx-audio-separator, basic-pitch-mlx (spectral backend), madmom-mlx (onset/beat/note pipelines), and any MLX audio pipeline needing spectral transforms.

## Feedback

Before finishing your task, you MUST check whether you encountered any bugs, unexpected behavior, workflow friction, missing features, or surprising performance. If so, record each one:

```bash
devfeedback mlx-spectro "description of bug, friction, or idea"
```
