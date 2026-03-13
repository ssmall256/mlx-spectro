# mlx-spectro

High-performance spectral frontends for [Apple MLX](https://github.com/ml-explore/mlx): fast STFT/iSTFT, mel/log-mel/MFCC extraction, reusable filtered spectrograms, descriptor bundles, and hybrid CQT. The core STFT/iSTFT path remains **2–3x faster STFT** and **5–8x faster iSTFT** than `torch.stft`/`torch.istft` on MPS via fused Metal kernels.

```python
from mlx_spectro import SpectralTransform

transform = SpectralTransform(n_fft=2048, hop_length=512, window_fn="hann")

spec = transform.stft(audio)                      # [B, T] → complex spectrogram
reconstructed = transform.istft(spec, length=T)    # complex spectrogram → [B, T]
```

```python
from mlx_spectro import MelSpectrogramTransform

mel = MelSpectrogramTransform(
    sample_rate=24000,
    n_fft=2048,
    hop_length=240,
    n_mels=128,
    top_db=80.0,
    mode="torchaudio_compat",
)
mel_db = mel(audio)  # [B, n_mels, frames]
```

```python
from mlx_spectro import LogMelSpectrogramTransform

log_mel = LogMelSpectrogramTransform(
    sample_rate=16000,
    n_fft=2048,
    hop_length=512,
    n_mels=256,
    f_min=30.0,
    f_max=8000.0,
    power=1.0,
    norm="slaney",
    mel_scale="htk",
    center_pad_mode="constant",
    log_amin=1e-5,
    log_mode="clamp",
)
mel_log = log_mel(audio)  # [B, n_mels, frames]
```

```python
from mlx_spectro import MFCCTransform

mfcc_transform = MFCCTransform(
    sample_rate=16000,
    n_mfcc=13,
    n_fft=400,
    hop_length=160,
    n_mels=40,
)
coeffs = mfcc_transform(audio)  # [B, n_mfcc, frames]
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

- Fused overlap-add with autotuned Metal kernels for fast STFT/iSTFT
- Cached reusable frontends for mel, log-mel, MFCC, filtered spectrograms, and hybrid CQT
- Shared-STFT descriptor extraction and cached descriptor bundles for repeated-call workloads
- PyTorch-, torchaudio-, librosa-, and madmom-style compatibility controls where parity matters
- `mx.compile`-friendly helpers for fixed-shape inference loops, including multi-axis STFT/iSTFT pairs
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

The public API is grouped into six main areas:

- Core STFT/iSTFT
- Mel, log-mel, and MFCC frontends
- Custom filtered frontends
- Spectral descriptors and shared feature bundles
- Hybrid CQT
- Advanced helpers, diagnostics, and typing aliases

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
    center_pad_mode: str = "reflect",   # "reflect" or "constant"
    center_tail_pad: str = "symmetric", # "symmetric" or "minimal"
    normalized: bool = False,
    istft_backend_policy: str | None = None,  # "auto", "mlx_fft", "metal", "torch_fallback"
)
```

**Methods:**
- `stft(x, output_layout="bfn")` — Forward STFT. Input: `[T]` or `[B, T]`.
- `istft(z, length=None, validate=False, *, torch_like=False, allow_fused=True, safety="auto", long_mode_strategy="native", backend_policy=None, input_layout="bfn")` — Inverse STFT. Returns `[B, T]`.
- `compiled_pair(length, layout="bnf", warmup_batch=None)` — Return compiled `(stft_fn, istft_fn)` for steady-state loops (10–20% faster).
- `compiled_pair_nd(length, leading_shape, layout="bnf")` — Return compiled reshape-aware `(stft_fn, istft_fn)` for fixed multi-axis inputs such as `[B, C, T]`.
- `warmup(batch=1, length=4096)` — Force kernel compilation.
- `prewarm_kernels(batch=1, length=None)` — Precompile eager STFT plus fused and legacy iSTFT kernels.
- `prewarm_compiled(batch=1, length=None, ...)` — Precompile cached compiled STFT/iSTFT callables.
- `get_compiled_stft(output_layout="bfn")` / `stft_compiled(x, output_layout="bfn")` — Cached `mx.compile` STFT helpers for fixed-shape loops.
- `get_compiled_istft(...)` / `istft_compiled(z, ...)` — Cached `mx.compile` iSTFT helpers for fixed-shape loops.
- `differentiable_stft(x)` — STFT entry point intended for `mx.grad` / `mx.value_and_grad`, returns `[B, N, F]`.
- `differentiable_istft(z, length=None)` — iSTFT entry point intended for gradients, expects `[B, N, F]`.

**Centering and padding semantics:**
- `center=True, center_pad_mode="reflect", center_tail_pad="symmetric"`: default PyTorch-style centered STFT with reflect padding on both sides. This keeps the current fused Metal fast path.
- `center=True, center_pad_mode="constant", center_tail_pad="symmetric"`: centered STFT with zero padding on both sides, matching the common Torch/librosa constant-pad interpretation.
- `center=True, center_pad_mode="constant", center_tail_pad="minimal"`: centered STFT with zero left padding and only the minimal right padding needed to keep frame count at `ceil(len / hop_length)`. This is useful for madmom-style frontends that should not emit an extra tail frame.

`center_pad_mode="reflect"` currently requires `center_tail_pad="symmetric"`. When using `center_tail_pad="minimal"`, `istft(..., length=...)` must be given an explicit `length`.

**Choosing an interface:**
- Use `stft()` / `istft()` for standard inference and variable-shape inputs.
- Use `compiled_pair()` or the `*_compiled()` helpers when shapes are fixed and you want lower dispatch overhead.
- Use `differentiable_stft()` / `differentiable_istft()` when gradients must flow through the transform.

### Padding Examples

Torch-style centered zero padding:

```python
from mlx_spectro import SpectralTransform

transform = SpectralTransform(
    n_fft=2048,
    hop_length=512,
    window_fn="hann",
    center=True,
    center_pad_mode="constant",
    center_tail_pad="symmetric",
)
```

madmom-style centered framing without an extra tail frame:

```python
import mlx.core as mx
import numpy as np
from mlx_spectro import SpectralTransform

window = mx.array(np.hanning(8192).astype(np.float32))
transform = SpectralTransform(
    n_fft=8192,
    hop_length=4410,
    win_length=8192,
    window=window,
    periodic=False,
    center=True,
    center_pad_mode="constant",
    center_tail_pad="minimal",
)
```

### `MelSpectrogramTransform`

Mel frontend powered by `SpectralTransform`.

```python
MelSpectrogramTransform(
    sample_rate: int = 24000,
    n_fft: int = 2048,
    hop_length: int = 240,
    win_length: int | None = None,
    n_mels: int = 128,
    f_min: float = 0.0,
    f_max: float | None = None,
    power: float = 2.0,
    norm: str | None = None,      # None or "slaney"
    mel_scale: str = "htk",       # "htk" or "slaney"
    top_db: float | None = 80.0,
    output_scale: str = "db",     # "linear", "log", or "db"
    log_amin: float = 1e-5,
    log_mode: str = "clamp",      # "clamp", "add", or "log1p"
    log_scale: float = 1.0,       # used when log_mode="log1p"
    mode: str = "mlx_native",     # "mlx_native" or "torchaudio_compat"; "default" alias -> "mlx_native"
    window_fn: str = "hann",
    periodic: bool = True,
    center: bool = True,
    center_pad_mode: str = "reflect",
    center_tail_pad: str = "symmetric",
    normalized: bool = False,
)
```

**Methods:**
- `spectrogram(x)` — Returns power or magnitude spectrogram `[B, F, N]`.
- `mel_spectrogram(x, output_scale=None, to_db=None)` / `__call__(x, output_scale=None, to_db=None)` — Returns `[B, n_mels, N]`.

**Mode semantics:**
- `mode="mlx_native"`: per-example `top_db` clipping (batch-independent behavior).
- `mode="torchaudio_compat"`: torchaudio-compatible packed-batch clipping semantics for parity-sensitive pipelines.

**Output scale semantics:**
- `output_scale="linear"`: return linear mel values.
- `output_scale="log"`: return natural-log mel using `log_mode`, `log_amin`, and `log_scale`.
- `output_scale="db"`: return dB mel; this preserves the current default behavior.
- `to_db=True` and `to_db=False` remain supported as compatibility aliases for `output_scale="db"` and `output_scale="linear"`.

**Log mode semantics:**
- `log_mode="clamp"`: `log(max(x, log_amin))`
- `log_mode="add"`: `log(x + log_amin)`
- `log_mode="log1p"`: `log1p(log_scale * x)` for frontends that already define a fixed multiplicative scale before logging

### `LogMelSpectrogramTransform`

Convenience wrapper for natural-log mel frontends. It is equivalent to `MelSpectrogramTransform(..., output_scale="log", ...)` and is intended for AMT/ASR-style pipelines that want log-mel directly instead of dB output.

### Feature Extraction

### `FilteredSpectrogramTransform`

Cached shared-STFT frontend for arbitrary filterbank projections.

```python
FilteredSpectrogramTransform(
    filterbank: mx.array | np.ndarray,  # [n_freqs, n_bands]
    sample_rate: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: int | None = None,
    power: float = 1.0,
    output_scale: str = "linear",  # "linear", "log", "db", "log10_plus_one"
    top_db: float | None = None,
    log_amin: float = 1e-5,
    log_mode: str = "clamp",
    window_fn: str = "hann",
    periodic: bool = True,
    center: bool = True,
    center_pad_mode: str = "reflect",
    center_tail_pad: str = "symmetric",
    normalized: bool = False,
)
```

**Methods:**
- `filtered_spectrogram(x)` / `__call__(x)` — Returns `[n_bands, frames]` for 1-D input or `[B, n_bands, frames]` for batched input.

This is the reusable path for project-specific frontends that apply non-mel filterbanks after one STFT magnitude pass, such as beat/chord/log-frequency pipelines.

The transform accepts filterbanks with either `n_fft // 2 + 1` rows or `n_fft // 2` rows. The latter is useful for madmom-style frontends that intentionally drop the Nyquist bin before applying a custom filterbank.

### `filtered_spectrogram(x, *, filterbank, sample_rate=22050, n_fft=2048, hop_length=512, ..., output_scale="linear")`

Functional one-off helper with the same parameters as `FilteredSpectrogramTransform`.

### `log_triangular_fbanks(n_freqs, sample_rate, bands_per_octave, *, f_min, f_max, f_ref=440.0, norm_filters=True, unique_bins=True, include_nyquist=False)`

Build logarithmically spaced triangular filterbanks for custom spectrogram frontends. Use `f_ref=440.0` for the madmom-style `f_ref`-anchored family and `f_ref=None` for the `f_min`-anchored family used by some beat frontends. Returns `[n_freqs, n_bands]`.

### `HybridCQTTransform`

Cached hybrid CQT frontend intended for `librosa.hybrid_cqt`-style pipelines and repeated inference workloads.

```python
HybridCQTTransform(
    sr: int = 22050,
    hop_length: int = 512,
    fmin: float = 32.70319566257483,
    n_bins: int = 84,
    bins_per_octave: int = 12,
    filter_scale: float = 1.0,
    norm: float = 1.0,
    sparsity: float = 0.01,
)
```

**Methods:**
- `hybrid_cqt(x)` / `__call__(x)` — Returns `[n_bins, frames]` for 1-D input or `[B, n_bins, frames]` for batched input.

Hybrid CQT basis construction is implemented directly in-package and cached at init time. Repeated calls stay on MLX tensors; no extra package dependencies are required beyond `numpy`.

### `hybrid_cqt(x, *, sr=22050, hop_length=512, fmin=32.70319566257483, n_bins=84, bins_per_octave=12, filter_scale=1.0, norm=1.0, sparsity=0.01)`

Functional one-off helper with the same parameters as `HybridCQTTransform`.

### `MFCCTransform`

MFCC frontend built on top of `MelSpectrogramTransform`.

```python
MFCCTransform(
    sample_rate: int = 22050,
    n_mfcc: int = 20,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: int | None = None,
    n_mels: int = 128,
    f_min: float = 0.0,
    f_max: float | None = None,
    norm: str | None = "slaney",
    mel_scale: str = "slaney",
    top_db: float | None = 80.0,
    window_fn: str = "hann",
    center: bool = True,
    center_pad_mode: str = "reflect",
    center_tail_pad: str = "symmetric",
    lifter: int = 0,
    dct_norm: str | None = "ortho",
)
```

**Methods:**
- `mfcc(x)` / `__call__(x)` — Returns MFCCs `[n_mfcc, frames]` for 1-D input or `[B, n_mfcc, frames]` for batched input.

MFCC uses librosa-style mel defaults (`norm="slaney"`, `mel_scale="slaney"`) while reusing this package's explicit STFT padding controls.

### `mfcc(x, *, sample_rate=22050, n_mfcc=20, n_fft=2048, hop_length=512, ..., lifter=0, dct_norm="ortho")`

Functional MFCC helper with the same parameters as `MFCCTransform`, intended for one-off extraction. Returns `[n_mfcc, frames]` for 1-D input or `[B, n_mfcc, frames]` for batched input.

### `onset_strength(x, *, sample_rate=22050, n_fft=2048, hop_length=512, n_mels=128, ..., center_pad_mode="reflect", center_tail_pad="symmetric")`

Half-wave rectified spectral flux of a dB-scaled mel spectrogram, matching librosa `onset.onset_strength` conventions. Returns `[frames]` for 1-D input or `[B, frames]` for batched input.

### `onset_strength_multi(x, *, sample_rate=22050, n_fft=2048, hop_length=512, n_mels=128, ..., center_pad_mode="reflect", center_tail_pad="symmetric")`

Per-band half-wave rectified spectral flux (before averaging across frequency). Returns `[n_mels, frames]` for 1-D input or `[B, n_mels, frames]` for batched input.

### `positive_spectral_diff(x, *, lag=None, frame_size=None, hop_size=None, diff_ratio=0.5, time_axis=-1)`

Half-wave rectified frame difference over the chosen time axis. Pass `lag` directly, or pass `frame_size` and `hop_size` to derive the madmom-style spectral-difference lag from a Hann window and `diff_ratio`.

### `chroma_stft(x, *, sample_rate=22050, n_fft=2048, hop_length=512, win_length=None, n_chroma=12, norm=2, ..., tuning=0.0)`

STFT chromagram using a librosa-style chroma filterbank. Returns `[n_chroma, frames]` for 1-D input or `[B, n_chroma, frames]` for batched input.

### `spectral_features(x, *, include=None, sample_rate=22050, n_fft=2048, hop_length=512, ..., n_mfcc=20, n_mels=128)`

Shared-STFT bundle API for extracting any combination of `chroma_stft`, `spectral_centroid`, `spectral_bandwidth`, `spectral_rolloff`, `spectral_contrast`, and `mfcc` from one magnitude pass. Returns an ordered mapping keyed by the requested function names, which is useful when you need several STFT-derived descriptors together. `rms` and `zero_crossing_rate` stay separate because they operate directly on framed waveform samples.

### `SpectralFeatureTransform(*, include=None, sample_rate=22050, n_fft=2048, hop_length=512, ..., n_mfcc=20, n_mels=128)`

Cached reusable version of `spectral_features(...)` for repeated-call workloads. It keeps one STFT transform plus any needed chroma filterbanks, mel filterbanks, DCT matrices, and MFCC lifter weights alive across calls, so it is the intended hot-path API when you repeatedly extract the same descriptor set with fixed parameters.

**Methods:**
- `extract(x)` / `__call__(x)` — Returns the same ordered mapping as `spectral_features(...)`, but reuses cached frontend state.

### `spectral_centroid(x, *, sample_rate=22050, n_fft=2048, hop_length=512, win_length=None, ...)`

Per-frame weighted mean frequency. Returns `[1, frames]` for 1-D input or `[B, 1, frames]` for batched input.

### `spectral_bandwidth(x, *, sample_rate=22050, n_fft=2048, hop_length=512, win_length=None, p=2.0, ...)`

Per-frame spectral spread around the centroid. Returns `[1, frames]` for 1-D input or `[B, 1, frames]` for batched input.

### `spectral_rolloff(x, *, sample_rate=22050, n_fft=2048, hop_length=512, win_length=None, roll_percent=0.85, ...)`

Frequency below which `roll_percent` of spectral magnitude is contained. Returns `[1, frames]` for 1-D input or `[B, 1, frames]` for batched input.

### `spectral_contrast(x, *, sample_rate=22050, n_fft=2048, hop_length=512, win_length=None, n_bands=6, fmin=200.0, quantile=0.02, ...)`

Peak-to-valley contrast in octave-spaced subbands. Returns `[n_bands + 1, frames]` for 1-D input or `[B, n_bands + 1, frames]` for batched input.

### `rms(x, *, frame_length=2048, hop_length=512, center=True, pad_mode="reflect")`

Frame-wise root-mean-square energy computed directly from the waveform. Returns `[1, frames]` for 1-D input or `[B, 1, frames]` for batched input.

### `zero_crossing_rate(x, *, frame_length=2048, hop_length=512, center=True, pad_mode="reflect")`

Frame-wise fraction of sign changes computed directly from the waveform. Returns `[1, frames]` for 1-D input or `[B, 1, frames]` for batched input.

### `get_transform_mlx(**kwargs)`

Factory that returns cached `SpectralTransform` instances for repeated use when
the window is specified symbolically.

```python
get_transform_mlx(
    *,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window_fn: str,
    periodic: bool,
    center: bool,
    normalized: bool,
    window: mx.array | None,
    center_pad_mode: str = "reflect",
    center_tail_pad: str = "symmetric",
    istft_backend_policy: str | None = None,
) -> SpectralTransform
```

If `window` is a concrete MLX array, a bespoke transform is returned instead of
using the shared cache.

### `make_window(window, window_fn, win_length, n_fft, periodic)`

Create or validate a 1D analysis window.

### `resolve_fft_params(n_fft, hop_length, win_length, pad)`

Resolve effective FFT parameters with PyTorch-compatible defaults.

### `melscale_fbanks(n_freqs, f_min, f_max, n_mels, sample_rate, *, norm=None, mel_scale="htk")`

Create torchaudio-compatible triangular mel filter banks. Returns
`[n_freqs, n_mels]`, so mel projection is `spec @ fb`.

### `dct_matrix(n_mfcc, n_mels, *, norm="ortho")`

Create a DCT type-II basis matrix with shape `[n_mfcc, n_mels]` for MFCC extraction. Supports `norm=None` and `norm="ortho"`.

### `amplitude_to_db(x, *, stype="power", top_db=80.0, amin=1e-10, ref_value=1.0, mode="torchaudio_compat")`

Convert magnitude or power spectrograms to dB. `mode="torchaudio_compat"`
matches torchaudio's packed-batch clipping behavior, while
`mode="per_example"` clips each example independently.

### Cache and Diagnostics

- `get_cache_debug_stats(reset=False)` — Return cache counters and lightweight kernel/transform cache snapshots.
- `reset_cache_debug_stats()` — Clear cache/debug counters.
- `spec_mlx_device_key()` — Return the device identifier used for autotune-cache keys.

### Typing Aliases

The package also exports string-literal typing aliases for option-bearing APIs:
`ISTFTBackendPolicy`, `STFTOutputLayout`, `CenterPadMode`,
`CenterTailPad`, `FilteredOutputScale`, `MelScale`, `MelNorm`, `MelMode`,
`MelOutputScale`, `LogMelMode`, and `WindowLike`.

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

To reproduce:
- Full suite: `python scripts/benchmark.py`
- Dispatch overhead profile: `python scripts/benchmark.py --dispatch-profile`
- Feature extraction bundle benchmarks: `python scripts/benchmark_features.py`
- Hybrid CQT benchmarks: `python scripts/benchmark_hybrid_cqt.py`

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
