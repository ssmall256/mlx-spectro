# mlx-spectro

High-performance STFT/iSTFT for [Apple MLX](https://github.com/ml-explore/mlx) with fused Metal kernels.

- Fused overlap-add with autotuned Metal kernels
- PyTorch-compatible STFT/iSTFT semantics
- Cached transforms for zero-overhead repeated calls
- Optional torch fallback for strict numerical parity

## Install

```bash
pip install mlx-spectro
```

With optional torch fallback support:

```bash
pip install mlx-spectro[torch]
```

## Quick Start

```python
import mlx.core as mx
from mlx_spectro import SpectralTransform

# Create a transform
transform = SpectralTransform(
    n_fft=2048,
    hop_length=512,
    window_fn="hann",
)

# Forward STFT
audio = mx.random.normal((1, 44100))
spec = transform.stft(audio, output_layout="bnf")

# Inverse STFT
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

Benchmarked across 288 STFT/iSTFT configurations (varying `n_fft`, `hop_length`, frame counts, and length modes), 3 replicates each, with 10 warmup iterations and 100 timed iterations per case.

### iSTFT Performance

| Machine | vs torch.istft (MPS) | vs mlx-stft | Mean latency |
|---|---|---|---|
| M4 Max (Mac16,5) | **3.4x faster** | **2.2x faster** | 0.31 ms |
| M1 Pro (MacBookPro18,4) | **4.3x faster** | **2.8x faster** | 0.58 ms |

### STFT Performance

| Machine | vs torch.stft (MPS) | vs mlx-stft | Mean latency |
|---|---|---|---|
| M4 Max (Mac16,5) | **1.7x faster** | **2.5x faster** | 0.24 ms |
| M1 Pro (MacBookPro18,4) | **2.0x faster** | **2.5x faster** | 0.42 ms |

### Compiled Mode (10–20% faster)

For tight inference loops with fixed input shapes, use `compiled_pair` to
eliminate per-call Python dispatch overhead:

```python
t = SpectralTransform(n_fft=1024, hop_length=256, window_fn="hann")
stft, istft = t.compiled_pair(length=44100, warmup_batch=2)

for chunk in audio_stream:
    z = stft(chunk)
    z = process(z)
    y = istft(z)
    mx.eval(y)
```

Use the eager `t.stft()` / `t.istft()` methods when input shapes vary or
during exploration.

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
