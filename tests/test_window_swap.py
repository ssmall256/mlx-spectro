"""Tests for window swapping, runtime cache invalidation, and custom window handling in mlx-spectro."""
import mlx.core as mx
import numpy as np
import pytest

from mlx_spectro import SpectralTransform, get_transform_mlx, waveform_overlap_add


def test_set_window_updates_runtime_cache_and_signature():
    t = SpectralTransform(512, 128)
    orig_sig = t._window_cache_sig

    # Custom window
    new_w = mx.ones((512,), dtype=mx.float32) * 0.75
    t.set_window(new_w)

    np.testing.assert_allclose(np.array(t.window), np.array(new_w), rtol=1e-6)
    np.testing.assert_allclose(np.array(t._window_sq), np.array(new_w ** 2), rtol=1e-6)
    assert t._window_cache_sig != orig_sig

    pair = t._window_pair_for_dtype(mx.float32)
    np.testing.assert_allclose(np.array(pair[0]), np.array(new_w), rtol=1e-6)
    np.testing.assert_allclose(np.array(pair[1]), np.array(new_w ** 2), rtol=1e-6)


def test_window_property_setter_compatibility():
    """Ensure legacy mutation pattern (t.window = w; t._window_sq = w**2) syncs runtime caches."""
    t = SpectralTransform(512, 128)

    custom_w = mx.ones((512,), dtype=mx.float32) * 0.6
    t.window = custom_w
    t._window_sq = custom_w ** 2

    np.testing.assert_allclose(np.array(t.window), np.array(custom_w), rtol=1e-6)
    np.testing.assert_allclose(np.array(t._window_sq), np.array(custom_w ** 2), rtol=1e-6)

    pair = t._window_pair_for_dtype(mx.float32)
    np.testing.assert_allclose(np.array(pair[0]), np.array(custom_w), rtol=1e-6)
    np.testing.assert_allclose(np.array(pair[1]), np.array(custom_w ** 2), rtol=1e-6)


def test_with_window_creates_independent_transform():
    t1 = SpectralTransform(512, 128)
    custom_w = mx.ones((512,), dtype=mx.float32) * 0.5

    t2 = t1.with_window(custom_w)
    assert t2 is not t1
    np.testing.assert_allclose(np.array(t2.window), np.array(custom_w), rtol=1e-6)
    assert not np.allclose(np.array(t1.window), np.array(custom_w))


def test_istft_uses_swapped_window():
    """Verify that istft immediately reflects a swapped window without using stale cached windows."""
    x = mx.random.normal((1, 2048))
    t = SpectralTransform(512, 128)
    Z = t.stft(x, output_layout="bnf")

    y_hann = t.istft(Z, length=2048, input_layout="bnf")
    mx.eval(y_hann)

    # Swap to rectangular window
    rect_w = mx.ones((512,), dtype=mx.float32)
    t.set_window(rect_w)

    y_rect = t.istft(Z, length=2048, input_layout="bnf")
    mx.eval(y_rect)

    # Reconstructions must differ because window changed
    diff = float(mx.max(mx.abs(y_hann - y_rect)))
    assert diff > 1e-3, f"Expected distinct istft outputs after window swap, got diff={diff}"


def test_get_transform_mlx_bespoke_window():
    """Passing a custom window to get_transform_mlx returns an unshared bespoke transform."""
    w = mx.ones((512,), dtype=mx.float32)
    t_bespoke = get_transform_mlx(
        n_fft=512,
        hop_length=128,
        win_length=512,
        window_fn="hann",
        periodic=True,
        center=True,
        normalized=False,
        window=w,
    )
    t_cached = get_transform_mlx(
        n_fft=512,
        hop_length=128,
        win_length=512,
        window_fn="hann",
        periodic=True,
        center=True,
        normalized=False,
        window=None,
    )

    assert t_bespoke is not t_cached
    np.testing.assert_allclose(np.array(t_bespoke.window), np.array(w), rtol=1e-6)
    assert not np.allclose(np.array(t_cached.window), np.array(w))
