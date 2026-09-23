"""A hop larger than the frame must still reconstruct.

`UNROLL_K` became a computed template constant in 0.8.0 (`min(FRAME/HOP, 8)`),
replacing a literal `4`. Integer division yields 0 whenever hop > n_fft, and
Metal rejects `#pragma unroll 0` outright, so every iSTFT kernel failed to
build and `istft()` raised for a configuration that worked in 0.7.0. Frames
that do not overlap are unusual but legal, and the library never validated
against it.
"""

from __future__ import annotations

import warnings

import mlx.core as mx
import numpy as np
import pytest

from mlx_spectro import SpectralTransform
from mlx_spectro.spectral_ops import _unroll_k


@pytest.mark.parametrize(
    "frame, hop, expected",
    [
        (2048, 512, 4),    # standard 4x overlap
        (1024, 128, 8),    # capped at 8
        (512, 512, 1),     # no overlap
        (512, 1024, 1),    # hop > frame: would be 0
        (256, 300, 1),     # hop > frame, non-multiple
        (512, 0, 1),       # guarded elsewhere, but must not divide by zero
        (512, -1, 1),
    ],
)
def test_unroll_k_is_always_a_legal_pragma_value(frame, hop, expected):
    assert _unroll_k(frame, hop) == expected


@pytest.mark.parametrize("n_fft, hop_length", [(512, 1024), (256, 300), (1024, 2048)])
def test_istft_reconstructs_when_hop_exceeds_n_fft(n_fft, hop_length):
    """Before the clamp this raised `Unable to build metal library from source`."""
    signal = mx.array(
        np.random.default_rng(7).standard_normal(8000).astype(np.float32)
    )[None]
    transform = SpectralTransform(n_fft=n_fft, hop_length=hop_length)

    with warnings.catch_warnings():
        # Non-overlapping frames legitimately give a degenerate OLA envelope.
        warnings.simplefilter("ignore", RuntimeWarning)
        spec = transform.stft(signal)
        out = transform.istft(spec, length=signal.shape[-1])
    mx.eval(out)

    assert out.shape == signal.shape
    assert bool(mx.all(mx.isfinite(out)))
    assert float(mx.max(mx.abs(out))) > 0.0
