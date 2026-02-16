"""Tests for mlx_spectro."""

import math

import mlx.core as mx
import numpy as np
import pytest

from mlx_spectro import (
    SpectralTransform,
    get_cache_debug_stats,
    get_transform_mlx,
    make_window,
    reset_cache_debug_stats,
    resolve_fft_params,
    spec_mlx_device_key,
)


# ---------------------------------------------------------------------------
# resolve_fft_params
# ---------------------------------------------------------------------------


class TestResolveFftParams:
    def test_defaults(self):
        n, h, w = resolve_fft_params(1024, None, None, 0)
        assert n == 1024
        assert h == 256  # n_fft // 4
        assert w == 1024

    def test_with_pad(self):
        n, h, w = resolve_fft_params(1024, None, None, 64)
        assert n == 1024 + 128  # n_fft + 2*pad

    def test_explicit_values(self):
        n, h, w = resolve_fft_params(2048, 512, 1024, 0)
        assert (n, h, w) == (2048, 512, 1024)

    def test_invalid_n_fft(self):
        with pytest.raises(ValueError, match="n_fft must be positive"):
            resolve_fft_params(0, None, None, 0)

    def test_invalid_hop(self):
        with pytest.raises(ValueError, match="hop_length must be positive"):
            resolve_fft_params(1024, 0, None, 0)

    def test_win_length_too_large(self):
        with pytest.raises(ValueError, match="win_length.*must be <="):
            resolve_fft_params(1024, 256, 2048, 0)


# ---------------------------------------------------------------------------
# make_window
# ---------------------------------------------------------------------------


class TestMakeWindow:
    def test_hann(self):
        w = make_window(None, "hann", 256, 256, True)
        assert w.shape == (256,)
        assert w.dtype == mx.float32
        # Hann window should be 0 at edges (periodic: first sample is 0)
        assert float(w[0].item()) == pytest.approx(0.0, abs=1e-6)

    def test_hamming(self):
        w = make_window(None, "hamming", 256, 256, True)
        assert w.shape == (256,)
        # Hamming doesn't go to zero
        assert float(w[0].item()) > 0.05

    def test_rect(self):
        w = make_window(None, "rect", 128, 128, True)
        assert w.shape == (128,)
        np.testing.assert_allclose(np.array(w), 1.0)

    def test_padding_when_win_lt_nfft(self):
        w = make_window(None, "hann", 128, 256, True)
        assert w.shape == (256,)
        # Edges should be zero-padded
        assert float(w[0].item()) == pytest.approx(0.0, abs=1e-6)
        assert float(w[-1].item()) == pytest.approx(0.0, abs=1e-6)

    def test_custom_array(self):
        custom = mx.ones((512,))
        w = make_window(custom, "hann", 512, 512, True)
        assert w.shape == (512,)
        np.testing.assert_allclose(np.array(w), 1.0)

    def test_custom_array_wrong_shape(self):
        custom = mx.ones((100,))
        with pytest.raises(ValueError, match="window length must be"):
            make_window(custom, "hann", 512, 512, True)

    def test_unknown_window_fn(self):
        with pytest.raises(ValueError, match="Unknown window_fn"):
            make_window(None, "kaiser", 256, 256, True)


# ---------------------------------------------------------------------------
# SpectralTransform: STFT
# ---------------------------------------------------------------------------


class TestSTFT:
    @pytest.fixture
    def transform(self):
        return SpectralTransform(n_fft=512, hop_length=128, window_fn="hann")

    def test_output_shape_bfn(self, transform):
        x = mx.random.normal((2, 8000))
        z = transform.stft(x, output_layout="bfn")
        B, F, N = z.shape
        assert B == 2
        assert F == 512 // 2 + 1  # onesided
        assert N > 0

    def test_output_shape_bnf(self, transform):
        x = mx.random.normal((2, 8000))
        z = transform.stft(x, output_layout="bnf")
        B, N, F = z.shape
        assert B == 2
        assert F == 512 // 2 + 1

    def test_1d_input(self, transform):
        x = mx.random.normal((4000,))
        z = transform.stft(x)
        assert z.ndim == 3
        assert z.shape[0] == 1  # batch dim added

    def test_invalid_ndim(self, transform):
        x = mx.random.normal((2, 3, 4000))
        with pytest.raises(ValueError, match="1D or 2D"):
            transform.stft(x)


# ---------------------------------------------------------------------------
# SpectralTransform: iSTFT roundtrip
# ---------------------------------------------------------------------------


class TestISTFTRoundtrip:
    @pytest.mark.parametrize("n_fft,hop", [(512, 128), (1024, 256), (256, 64)])
    def test_roundtrip(self, n_fft, hop):
        t = SpectralTransform(n_fft=n_fft, hop_length=hop, window_fn="hann")
        length = 8000
        x = mx.random.normal((1, length))
        z = t.stft(x, output_layout="bnf")
        y = t.istft(z, length=length, input_layout="bnf")
        mx.eval(y)
        np.testing.assert_allclose(
            np.array(y), np.array(x), atol=1e-4, rtol=1e-4
        )

    def test_roundtrip_no_center(self):
        t = SpectralTransform(n_fft=512, hop_length=128, window_fn="hann", center=False)
        x = mx.random.normal((1, 8000))
        z = t.stft(x, output_layout="bnf")
        y = t.istft(z, input_layout="bnf")
        mx.eval(y)
        # Without center, edges have incomplete overlap — skip n_fft samples
        skip = t.n_fft
        min_len = min(x.shape[1], y.shape[1])
        np.testing.assert_allclose(
            np.array(y[:, skip:min_len - skip]),
            np.array(x[:, skip:min_len - skip]),
            atol=1e-4, rtol=1e-4,
        )

    def test_layout_bfn(self):
        t = SpectralTransform(n_fft=512, hop_length=128, window_fn="hann")
        length = 4000
        x = mx.random.normal((1, length))
        z = t.stft(x, output_layout="bfn")
        y = t.istft(z, length=length, input_layout="bfn")
        mx.eval(y)
        np.testing.assert_allclose(
            np.array(y), np.array(x), atol=1e-4, rtol=1e-4
        )

    def test_batch(self):
        t = SpectralTransform(n_fft=512, hop_length=128, window_fn="hann")
        length = 4000
        x = mx.random.normal((4, length))
        z = t.stft(x, output_layout="bnf")
        y = t.istft(z, length=length, input_layout="bnf")
        mx.eval(y)
        np.testing.assert_allclose(
            np.array(y), np.array(x), atol=1e-4, rtol=1e-4
        )


# ---------------------------------------------------------------------------
# get_transform_mlx caching
# ---------------------------------------------------------------------------


class TestGetTransformCached:
    def test_same_params_same_instance(self):
        kwargs = dict(
            n_fft=1024, hop_length=256, win_length=1024,
            window_fn="hann", periodic=True, center=True,
            normalized=False, window=None,
        )
        t1 = get_transform_mlx(**kwargs)
        t2 = get_transform_mlx(**kwargs)
        assert t1 is t2

    def test_different_params_different_instance(self):
        base = dict(
            n_fft=1024, hop_length=256, win_length=1024,
            window_fn="hann", periodic=True, center=True,
            normalized=False, window=None,
        )
        t1 = get_transform_mlx(**base)
        t2 = get_transform_mlx(**{**base, "hop_length": 512})
        assert t1 is not t2

    def test_custom_window_not_cached(self):
        """Custom mx.array windows bypass the LRU cache."""
        w = mx.ones((1024,))
        t = get_transform_mlx(
            n_fft=1024, hop_length=256, win_length=1024,
            window_fn="hann", periodic=True, center=True,
            normalized=False, window=w,
        )
        assert isinstance(t, SpectralTransform)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


class TestUtils:
    def test_device_key(self):
        key = spec_mlx_device_key()
        assert isinstance(key, str)
        assert len(key) > 0

    def test_cache_debug_stats(self):
        reset_cache_debug_stats()
        stats = get_cache_debug_stats()
        assert "enabled" in stats
        assert "counts" in stats
        assert "kernel_cache" in stats

    def test_warmup(self):
        t = SpectralTransform(n_fft=512, hop_length=128, window_fn="hann")
        t.warmup(batch=1, length=2048)  # should not raise


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_win_length_lt_nfft(self):
        t = SpectralTransform(n_fft=512, hop_length=128, win_length=256, window_fn="hann")
        assert t.window.shape == (512,)

    def test_win_length_gt_nfft_raises(self):
        with pytest.raises(ValueError, match="win_length.*must be <="):
            SpectralTransform(n_fft=256, hop_length=64, win_length=512, window_fn="hann")

    def test_normalized_roundtrip(self):
        t = SpectralTransform(n_fft=512, hop_length=128, window_fn="hann", normalized=True)
        length = 4000
        x = mx.random.normal((1, length))
        z = t.stft(x, output_layout="bnf")
        y = t.istft(z, length=length, input_layout="bnf")
        mx.eval(y)
        np.testing.assert_allclose(
            np.array(y), np.array(x), atol=1e-4, rtol=1e-4
        )
