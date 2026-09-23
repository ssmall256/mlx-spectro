"""Threadgroup autotuning must not break a compiled STFT/iSTFT.

Autotuning times candidate threadgroup sizes, and every timing run calls
``mx.eval`` -- which MLX refuses inside ``mx.compile`` or ``vmap``. Before this,
a machine with no tuning cache yet saw every candidate fail for that reason and
got ``RuntimeError: ... no usable threadgroup size``, producing no output at
all. The raise is right when the kernel is genuinely broken for a
configuration; it is wrong when the only problem is that timing is impossible.

``compiled_pair``/``compiled_pair_nd`` were never affected -- they call the
transform eagerly once before compiling -- which is also the remedy the warning
names.
"""

from __future__ import annotations

import warnings

import mlx.core as mx
import numpy as np
import pytest

from mlx_spectro import SpectralTransform
from mlx_spectro import spectral_ops
from mlx_spectro.spectral_ops import _KernelCache, _TRACED_AUTOTUNE_WARNED


N_FFT, HOP, LENGTH = 4096, 1024, 44100


@pytest.fixture
def cold_cache():
    """No tuned entries, and no lazy reload from the developer's real cache."""
    saved_entries = dict(_KernelCache._tgx_cache)
    saved_loaded = _KernelCache._tgx_cache_loaded
    saved_warned = set(_TRACED_AUTOTUNE_WARNED)

    _KernelCache._tgx_cache.clear()
    # Mark it loaded so get_threadgroup_x does not read the on-disk cache back
    # in behind us; without this the fixture is silently a no-op.
    _KernelCache._tgx_cache_loaded = True
    _TRACED_AUTOTUNE_WARNED.clear()
    try:
        yield
    finally:
        _KernelCache._tgx_cache.clear()
        _KernelCache._tgx_cache.update(saved_entries)
        _KernelCache._tgx_cache_loaded = saved_loaded
        _TRACED_AUTOTUNE_WARNED.clear()
        _TRACED_AUTOTUNE_WARNED.update(saved_warned)


def _signal(seed: int = 0):
    rng = np.random.default_rng(seed)
    return mx.array(rng.standard_normal((1, LENGTH)).astype("float32"))


def test_compiled_stft_runs_with_a_cold_tuning_cache(cold_cache):
    t = SpectralTransform(n_fft=N_FFT, hop_length=HOP)
    with pytest.warns(RuntimeWarning, match="autotuning .* was skipped"):
        spec = mx.compile(lambda a: t.stft(a))(_signal())
    mx.eval(spec)
    assert spec.shape[:2] == (1, N_FFT // 2 + 1)


def test_compiled_istft_runs_with_a_cold_tuning_cache(cold_cache):
    t = SpectralTransform(n_fft=N_FFT, hop_length=HOP)
    spec = t.stft(_signal())
    mx.eval(spec)
    _KernelCache._tgx_cache.clear()          # only the iSTFT kernel is cold now
    with pytest.warns(RuntimeWarning, match="autotuning .* was skipped"):
        y = mx.compile(lambda s: t.istft(s, length=LENGTH))(spec)
    mx.eval(y)
    assert y.shape == (1, LENGTH)


def test_the_untuned_default_is_numerically_sound(cold_cache):
    """Threadgroup size is a speed choice. Compare eager against eager so the
    comparison isolates it from mx.compile's own float reassociation."""
    t = SpectralTransform(n_fft=N_FFT, hop_length=HOP)
    x = _signal(1)

    untuned = t.stft(x)                      # cold cache, but not traced
    mx.eval(untuned)
    assert _KernelCache._tgx_cache, "the eager call should have tuned"
    tuned = t.stft(x)                        # now served from the tuned entry
    mx.eval(tuned)
    assert float(mx.max(mx.abs(untuned - tuned))) == 0.0


def test_compiled_output_tracks_eager_when_tuning_was_skipped(cold_cache):
    """mx.compile fuses and so reassociates adds; the gap must stay at float32
    rounding rather than the kernel producing something different."""
    t = SpectralTransform(n_fft=N_FFT, hop_length=HOP)
    x = _signal(2)
    with pytest.warns(RuntimeWarning):
        compiled = mx.compile(lambda a: t.stft(a))(x)
    eager = t.stft(x)
    mx.eval(compiled, eager)
    scale = float(mx.max(mx.abs(eager)))
    assert float(mx.max(mx.abs(compiled - eager))) <= 1e-5 * scale


def test_a_skipped_tuning_is_not_cached(cold_cache):
    """Caching the untested default would serve it to the eager path too."""
    t = SpectralTransform(n_fft=N_FFT, hop_length=HOP)
    with pytest.warns(RuntimeWarning):
        mx.eval(mx.compile(lambda a: t.stft(a))(_signal()))
    assert _KernelCache._tgx_cache == {}, "nothing may be recorded under a trace"

    mx.eval(t.stft(_signal()))
    assert _KernelCache._tgx_cache, "an eager call must still tune and record"


def test_the_warning_fires_once_per_configuration(cold_cache):
    t = SpectralTransform(n_fft=N_FFT, hop_length=HOP)
    with pytest.warns(RuntimeWarning, match="autotuning .* was skipped"):
        mx.eval(mx.compile(lambda a: t.stft(a))(_signal()))

    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        mx.eval(mx.compile(lambda a: t.stft(a))(_signal(3)))
    repeats = [w for w in seen if "autotuning" in str(w.message)]
    assert not repeats, f"warned again: {[str(w.message) for w in repeats]}"


def test_a_genuinely_broken_kernel_still_raises(cold_cache, monkeypatch):
    """The raise exists for a kernel that works at no threadgroup size at all
    -- a `#pragma unroll 0` build error once read as a mysterious istft crash.
    Being inside a trace must not become a blanket excuse that hides it."""
    monkeypatch.setattr(
        spectral_ops, "_is_tracer_error", lambda exc: False, raising=True
    )
    t = SpectralTransform(n_fft=N_FFT, hop_length=HOP)
    with pytest.raises(RuntimeError, match="no usable threadgroup size"):
        mx.eval(mx.compile(lambda a: t.stft(a))(_signal()))
