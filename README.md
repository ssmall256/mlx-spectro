# mlx-spectro

High-performance STFT/iSTFT for [Apple MLX](https://github.com/ml-explore/mlx) — **2–3x faster STFT** and **5–8x faster iSTFT** than `torch.stft`/`torch.istft` on MPS, via fused Metal kernels.

```python
from mlx_spectro import SpectralTransform

transform = SpectralTransform(n_fft=2048, hop_length=512, window_fn="hann")

spec = transform.stft(audio)                      # [B, T] → complex spectrogram
reconstructed = transform.istft(spec, length=T)    # complex spectrogram → [B, T]
```

[mlx-audio-separator](https://github.com/ssmall256/mlx-audio-separator) uses mlx-spectro for MLX-native stem separation (Roformer, MDX, Demucs) and runs **1.8–3.1x faster end-to-end** than python-audio-separator on torch+MPS. See [benchmarks](#real-world-mlx-audio-separator) below.

## Install

```bash
pip install mlx-spectro
```

With optional torch fallback support:

```bash
pip install mlx-spectro[torch]
```

## Features

- Fused overlap-add with autotuned Metal kernels
- PyTorch-compatible STFT/iSTFT semantics
- Cached transforms for zero-overhead repeated calls
- Differentiable transforms for training with `mx.grad`
- `mx.compile`-friendly for tight inference loops
- Optional torch fallback for strict numerical parity

## Quick Start

```python
import mlx.core as mx
from mlx_spectro import SpectralTransform

transform = SpectralTransform(
    n_fft=2048,
    hop_length=512,
    window_fn="hann",
)

audio = mx.random.normal((1, 44100))
spec = transform.stft(audio, output_layout="bnf")
reconstructed = transform.istft(spec, length=44100, input_layout="bnf")
```

## API

### `SpectralTransform`

Main class for STFT/iSTFT operations.

```python
SpectralTransform(
    n_fft: int,
    hop_length: int,
    win_length: int | None = None,
    window_fn: str = "hann",       # "hann", "hamming", "rect"
    window: mx.array | None = None,  # custom window array
    periodic: bool = True,
    center: bool = True,
    normalized: bool = False,
    istft_backend_policy: str | None = None,  # "auto", "mlx_fft", "metal", "torch_fallback"
)
```

**Methods:**
- `stft(x, output_layout="bfn")` — Forward STFT. Input: `[T]` or `[B, T]`.
- `istft(z, length=None, ...)` — Inverse STFT. Returns `[B, T]`.
- `compiled_pair(length, layout="bnf", warmup_batch=None)` — Return compiled `(stft_fn, istft_fn)` for steady-state loops (10–20% faster).
- `warmup(batch=1, length=4096)` — Force kernel compilation.

### `get_transform_mlx(**kwargs)`

Factory that returns cached `SpectralTransform` instances for repeated use.

### `make_window(window, window_fn, win_length, n_fft, periodic)`

Create or validate a 1D analysis window.

### `resolve_fft_params(n_fft, hop_length, win_length, pad)`

Resolve effective FFT parameters with PyTorch-compatible defaults.

## Benchmarks

Apple M4 Max, macOS 26.3, MLX 0.30.6, PyTorch 2.10.0, 20 iterations (5 warmup).

### STFT Forward

| Config | mlx-spectro | torch MPS | mlx-stft | vs torch | vs mlx-stft |
|---|---|---|---|---|---|
| B=1 T=16k nfft=512 | 0.16 ms | 0.21 ms | 0.31 ms | 1.4x | 1.9x |
| B=4 T=160k nfft=1024 | 0.37 ms | 1.00 ms | 1.09 ms | **2.7x** | **3.0x** |
| B=8 T=160k nfft=1024 | 0.28 ms | 0.71 ms | 1.53 ms | **2.5x** | **5.6x** |
| B=4 T=1.3M nfft=1024 | 0.77 ms | 2.18 ms | 5.03 ms | **2.8x** | **6.5x** |
| B=8 T=480k nfft=1024 | 0.58 ms | 1.30 ms | 3.73 ms | **2.2x** | **6.4x** |

### iSTFT Forward

| Config | mlx-spectro | torch MPS | mlx-stft | vs torch | vs mlx-stft |
|---|---|---|---|---|---|
| B=1 T=16k nfft=512 | 0.17 ms | 0.49 ms | 0.25 ms | 3.0x | 1.5x |
| B=4 T=160k nfft=1024 | 0.21 ms | 1.00 ms | 0.98 ms | **4.7x** | **4.7x** |
| B=8 T=160k nfft=1024 | 0.30 ms | 1.61 ms | 1.62 ms | **5.4x** | **5.4x** |
| B=4 T=1.3M nfft=1024 | 0.81 ms | 5.76 ms | 6.68 ms | **7.1x** | **8.2x** |
| B=8 T=480k nfft=1024 | 0.60 ms | 4.10 ms | 4.55 ms | **6.8x** | **7.6x** |

### Roundtrip (STFT → iSTFT) Forward + Backward

| Config | mlx-spectro | torch MPS | vs torch |
|---|---|---|---|
| B=4 T=160k nfft=1024 | 0.62 ms | 2.25 ms | **3.6x** |
| B=8 T=160k nfft=1024 | 1.04 ms | 4.38 ms | **4.2x** |
| B=4 T=480k nfft=1024 | 1.59 ms | 6.59 ms | **4.1x** |
| B=4 T=1.3M nfft=1024 | 4.33 ms | 17.63 ms | **4.1x** |
| B=1 T=1.3M nfft=1024 | 1.21 ms | 4.20 ms | **3.5x** |

### Roundtrip Accuracy (STFT → iSTFT max abs error)

| Config | mlx-spectro | torch MPS |
|---|---|---|
| B=1 T=16k nfft=512 | 1.67e-06 | 2.38e-06 |
| B=4 T=160k nfft=2048 | 2.86e-06 | 5.25e-06 |
| B=8 T=480k nfft=1024 | 3.81e-06 | 4.77e-06 |

To reproduce: `python scripts/benchmark.py`

### Real-world: mlx-audio-separator

[mlx-audio-separator](https://github.com/ssmall256/mlx-audio-separator) is an MLX-native music stem separation library supporting Roformer, MDX, Demucs, and more. End-to-end separation speedup vs python-audio-separator (torch on MPS), measured on 30s stereo 44.1 kHz tracks. Apple M4 Max, PyTorch 2.10.0, MLX 0.30.6, ABBA ordering, 2 repeats.

| Model | Arch | torch+MPS (s) | MLX (s) | E2E speedup |
|---|---|--:|--:|--:|
| UVR-MDX-NET-Inst_HQ_3 | MDX | 4.25 | 1.36 | **3.1x** |
| htdemucs | Demucs | 3.35 | 1.29 | **2.6x** |
| Mel-Roformer Karaoke | MDXC | 5.60 | 2.66 | **2.1x** |
| BS-Roformer | MDXC | 6.48 | 3.56 | **1.8x** |

STFT/iSTFT kernel speedups within these pipelines are even larger (2–3x STFT, 5–8x iSTFT vs torch).

### Compiled Mode

For tight inference loops with fixed input shapes, `compiled_pair` eliminates
per-call Python dispatch overhead (10–20% faster for small workloads):

```python
t = SpectralTransform(n_fft=1024, hop_length=256, window_fn="hann")
stft, istft = t.compiled_pair(length=44100, warmup_batch=2)

for chunk in audio_stream:
    z = stft(chunk)
    z = process(z)
    y = istft(z)
    mx.eval(y)
```

Use the eager `t.stft()` / `t.istft()` methods when input shapes vary.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `SPEC_MLX_AUTOTUNE` | `1` | Enable Metal kernel autotuning |
| `SPEC_MLX_TGX` | — | Force threadgroup size (e.g. `256` or `kernel:256`) |
| `SPEC_MLX_AUTOTUNE_PERSIST` | `1` | Persist autotune results to disk |
| `SPEC_MLX_AUTOTUNE_CACHE_PATH` | — | Override autotune cache file path |
| `MLX_OLA_FUSE_NORM` | `1` | Enable fused OLA+normalization kernel |
| `SPEC_MLX_CACHE_STATS` | `0` | Enable cache debug counters |

## License

MIT
