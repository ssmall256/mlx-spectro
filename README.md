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
- `warmup(batch=1, length=4096)` — Force kernel compilation.

### `get_transform_mlx(**kwargs)`

Factory that returns cached `SpectralTransform` instances for repeated use.

### `make_window(window, window_fn, win_length, n_fft, periodic)`

Create or validate a 1D analysis window.

### `resolve_fft_params(n_fft, hop_length, win_length, pad)`

Resolve effective FFT parameters with PyTorch-compatible defaults.

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
