"""Defaults that used to fail silently.

Two behaviors here were invisible by default:

* A NOLA violation (degenerate overlap-add envelope) was detected, cached, and
  then ignored unless ``torch_like=True`` -- which is not the default. Both the
  Metal kernel and the pure-MLX fallback emit exact 0.0 wherever the envelope is
  below 1e-11, so the reconstruction silently contained gaps.
* The pure-MLX overlap-add fallbacks accumulated in the *input* dtype, so fp16
  input accumulated in fp16 while the Metal kernel they stand in for keeps its
  accumulator in float32. Same operation, different precision, chosen by
  whether a kernel happened to compile.
"""

from __future__ import annotations

import warnings

import mlx.core as mx
import numpy as np
import pytest

import mlx_spectro.spectral_ops as so
from mlx_spectro import SpectralTransform


@pytest.fixture(autouse=True)
def _reset_warn_state():
    """The warning is deduped per config; clear it between tests."""
    so._NOLA_WARNED.clear()
    yield
    so._NOLA_WARNED.clear()


def _degenerate_transform():
    """center=False leaves the first and last n_fft samples partly uncovered,
    so the overlap-add envelope reaches exactly zero at the edges."""
    return SpectralTransform(n_fft=512, hop_length=128, window_fn="hann", center=False)


def test_nola_violation_warns_by_default():
    t = _degenerate_transform()
    x = mx.random.normal((1, 8000))
    z = t.stft(x, output_layout="bnf")
    mx.eval(z)

    with pytest.warns(RuntimeWarning, match="overlap-add envelope is degenerate"):
        mx.eval(t.istft(z, input_layout="bnf"))


def test_nola_violation_still_raises_under_torch_like():
    t = _degenerate_transform()
    x = mx.random.normal((1, 8000))
    z = t.stft(x, output_layout="bnf")
    mx.eval(z)

    with pytest.raises(RuntimeError, match="overlap-add envelope is degenerate"):
        mx.eval(t.istft(z, input_layout="bnf", torch_like=True))


def test_nola_warning_is_emitted_once_per_configuration():
    """A degenerate envelope is a property of the config, not of the call."""
    t = _degenerate_transform()
    x = mx.random.normal((1, 8000))
    z = t.stft(x, output_layout="bnf")
    mx.eval(z)

    with pytest.warns(RuntimeWarning):
        mx.eval(t.istft(z, input_layout="bnf"))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(3):
            mx.eval(t.istft(z, input_layout="bnf"))
    nola = [w for w in caught if "overlap-add envelope" in str(w.message)]
    assert nola == [], f"warning repeated {len(nola)} times for one configuration"


def test_well_conditioned_transform_does_not_warn():
    t = SpectralTransform(n_fft=512, hop_length=128, window_fn="hann", center=True)
    x = mx.random.normal((1, 8000))
    z = t.stft(x, output_layout="bnf")
    mx.eval(z)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mx.eval(t.istft(z, length=8000, input_layout="bnf"))
    nola = [w for w in caught if "overlap-add envelope" in str(w.message)]
    assert nola == [], "COLA-compliant configuration should not warn"


@pytest.mark.parametrize("dtype", [mx.float16, mx.float32])
def test_ola_fallback_accumulates_in_float32(dtype):
    """The fallback must match the Metal kernel's float32 accumulator.

    Many frames overlap each output sample, so a half-precision accumulator
    loses precision the kernel would have kept.
    """
    n_fft, hop, n_frames, B = 256, 64, 40, 2
    mx.random.seed(0)
    frames = (mx.random.normal((B, n_frames, n_fft)) * 0.1).astype(dtype)
    window = mx.ones((n_fft,), dtype=dtype)
    out_len = hop * (n_frames - 1) + n_fft

    got = np.asarray(
        so._run_metal_ola(frames, window, hop, out_len, require_metal=False)
    ).astype(np.float64)

    f64 = np.asarray(frames).astype(np.float64)
    ref = np.zeros((B, out_len), dtype=np.float64)
    for b in range(B):
        for k in range(n_frames):
            ref[b, k * hop:k * hop + n_fft] += f64[b, k]

    scale = max(float(np.max(np.abs(ref))), 1e-12)
    # float16 output still rounds on the final cast; what must not happen is
    # the accumulation itself being done in half precision.
    tol = 2e-3 if dtype == mx.float16 else 1e-5
    assert np.max(np.abs(got - ref)) / scale < tol, (
        f"{dtype}: relative error {np.max(np.abs(got - ref)) / scale:.3e}"
    )
