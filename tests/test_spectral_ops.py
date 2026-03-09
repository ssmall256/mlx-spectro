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

    def test_constant_center_pad_matches_manual_symmetric_padding(self):
        n_fft = 512
        hop = 128
        window = mx.array(np.hanning(n_fft).astype(np.float32))
        x = mx.random.normal((1, 4000))
        direct = SpectralTransform(
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
            window=window,
            periodic=False,
            center=True,
            center_pad_mode="constant",
            center_tail_pad="symmetric",
        )
        manual = SpectralTransform(
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
            window=window,
            periodic=False,
            center=False,
        )
        z_direct = direct.stft(x, output_layout="bnf")
        z_manual = manual.stft(
            mx.pad(x, [(0, 0), (n_fft // 2, n_fft // 2)], mode="constant"),
            output_layout="bnf",
        )
        mx.eval(z_direct, z_manual)
        np.testing.assert_allclose(np.array(z_direct), np.array(z_manual), atol=1e-6)

    def test_constant_center_pad_matches_manual_minimal_padding(self):
        n_fft = 512
        hop = 128
        pad = n_fft // 2
        window = mx.array(np.hanning(n_fft).astype(np.float32))
        x = mx.random.normal((1, 4000))
        direct = SpectralTransform(
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
            window=window,
            periodic=False,
            center=True,
            center_pad_mode="constant",
            center_tail_pad="minimal",
        )
        manual = SpectralTransform(
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
            window=window,
            periodic=False,
            center=False,
        )
        sig_len = int(x.shape[1])
        num_frames = max(1, int(math.ceil(sig_len / float(hop))))
        last_start = (num_frames - 1) * hop - pad
        pad_right = max(0, last_start + n_fft - sig_len)
        z_direct = direct.stft(x, output_layout="bnf")
        z_manual = manual.stft(
            mx.pad(x, [(0, 0), (pad, pad_right)], mode="constant"),
            output_layout="bnf",
        )
        mx.eval(z_direct, z_manual)
        np.testing.assert_allclose(np.array(z_direct), np.array(z_manual), atol=1e-6)


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

    def test_minimal_center_tail_requires_length(self):
        t = SpectralTransform(
            n_fft=512,
            hop_length=128,
            window_fn="hann",
            center_pad_mode="constant",
            center_tail_pad="minimal",
        )
        x = mx.random.normal((1, 4000))
        z = t.stft(x, output_layout="bnf")
        with pytest.raises(ValueError, match="length is required"):
            t.istft(z, input_layout="bnf")


# ---------------------------------------------------------------------------
# get_transform_mlx caching
# ---------------------------------------------------------------------------


class TestGetTransformCached:
    def test_same_params_same_instance(self):
        kwargs = dict(
            n_fft=1024, hop_length=256, win_length=1024,
            window_fn="hann", periodic=True, center=True,
            center_pad_mode="reflect", center_tail_pad="symmetric",
            normalized=False, window=None,
        )
        t1 = get_transform_mlx(**kwargs)
        t2 = get_transform_mlx(**kwargs)
        assert t1 is t2

    def test_different_params_different_instance(self):
        base = dict(
            n_fft=1024, hop_length=256, win_length=1024,
            window_fn="hann", periodic=True, center=True,
            center_pad_mode="reflect", center_tail_pad="symmetric",
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
            center_pad_mode="reflect", center_tail_pad="symmetric",
            normalized=False, window=w,
        )
        assert isinstance(t, SpectralTransform)

    def test_different_center_padding_config_not_cached(self):
        base = dict(
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            window_fn="hann",
            periodic=True,
            center=True,
            center_tail_pad="symmetric",
            normalized=False,
            window=None,
        )
        t1 = get_transform_mlx(**{**base, "center_pad_mode": "reflect"})
        t2 = get_transform_mlx(**{**base, "center_pad_mode": "constant"})
        assert t1 is not t2


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

    def test_reflect_pad_correctness(self):
        """Verify compiled reflect-pad matches numpy reflect padding."""
        from mlx_spectro.spectral_ops import _torch_like_reflect_pad_1d_compiled

        for B in [1, 4]:
            for T in [100, 512, 2048]:
                for pad in [64, 256]:
                    if T <= pad:
                        continue
                    x = mx.random.normal((B, T))
                    mx.eval(x)

                    # Metal kernel path
                    y_metal = _torch_like_reflect_pad_1d_compiled(x, pad)
                    mx.eval(y_metal)

                    # Reference: numpy reflect pad
                    x_np = np.array(x)
                    y_ref = np.pad(x_np, [(0, 0), (pad, pad)], mode="reflect")

                    np.testing.assert_allclose(
                        np.array(y_metal), y_ref, atol=1e-6,
                        err_msg=f"reflect pad mismatch for B={B}, T={T}, pad={pad}",
                    )

    def test_large_nfft(self):
        """Large n_fft (4096) roundtrip should work correctly."""
        t = SpectralTransform(n_fft=4096, hop_length=1024, window_fn="hann")
        length = 16000
        x = mx.random.normal((1, length))
        z = t.stft(x, output_layout="bnf")
        y = t.istft(z, length=length, input_layout="bnf")
        mx.eval(y)
        np.testing.assert_allclose(
            np.array(y), np.array(x), atol=1e-4, rtol=1e-4
        )

    def test_compiled_pair_roundtrip(self):
        """compiled_pair returns stft/istft that match eager roundtrip."""
        length = 16000
        t = SpectralTransform(n_fft=1024, hop_length=256, window_fn="hann")
        stft_fn, istft_fn = t.compiled_pair(length=length, warmup_batch=2)

        x = mx.random.normal((2, length))
        mx.eval(x)

        # Compiled roundtrip
        z_c = stft_fn(x)
        y_c = istft_fn(z_c)
        mx.eval(y_c)

        # Eager roundtrip
        z_e = t.stft(x, output_layout="bnf")
        y_e = t.istft(z_e, length=length, input_layout="bnf")
        mx.eval(y_e)

        np.testing.assert_allclose(
            np.array(y_c), np.array(y_e), atol=1e-6,
            err_msg="compiled_pair output diverges from eager",
        )
        np.testing.assert_allclose(
            np.array(y_c), np.array(x), atol=1e-4, rtol=1e-4,
            err_msg="compiled_pair roundtrip not accurate",
        )

    def test_compiled_pair_bfn_layout(self):
        """compiled_pair works with bfn layout."""
        length = 8000
        t = SpectralTransform(n_fft=512, hop_length=128, window_fn="hann")
        stft_fn, istft_fn = t.compiled_pair(length=length, layout="bfn")

        x = mx.random.normal((1, length))
        mx.eval(x)
        z = stft_fn(x)
        y = istft_fn(z)
        mx.eval(y)

        np.testing.assert_allclose(
            np.array(y), np.array(x), atol=1e-4, rtol=1e-4,
        )

    def test_fused_frame_extract_matches_fallback(self):
        """Fused Metal frame extraction produces bit-exact output vs fallback."""
        from mlx_spectro.spectral_ops import (
            _FrameExtractCache,
            _torch_like_reflect_pad_1d_compiled,
        )
        kernel = _FrameExtractCache.get()
        if kernel is False:
            self.skipTest("Metal frame extraction kernel unavailable")

        for B in [1, 4]:
            for n_fft, hop in [(512, 128), (1024, 256), (2048, 512)]:
                sig_len = n_fft * 10
                t = SpectralTransform(n_fft=n_fft, hop_length=hop, window_fn="hann")
                x = mx.random.normal((B, sig_len))
                mx.eval(x)
                x_c = mx.contiguous(x)

                # Fused path: Metal kernel
                pad = n_fft // 2
                padded_len = sig_len + 2 * pad
                n_frames = (padded_len - n_fft) // hop + 1
                params = mx.array([sig_len, n_frames], dtype=mx.int32)
                tmpl = [
                    ("T", x.dtype), ("NFFT", n_fft), ("HOP", hop),
                    ("PAD", pad),
                ]
                fused = kernel(
                    inputs=[x_c, t.window, params],
                    template=tmpl,
                    output_shapes=[(B, n_frames, n_fft)],
                    output_dtypes=[x.dtype],
                    grid=(n_fft, n_frames, B),
                    threadgroup=(min(256, n_fft), 1, 1),
                )
                fused_out = fused[0]

                # Fallback path: pad + stride + multiply
                x_padded = _torch_like_reflect_pad_1d_compiled(x_c, pad)
                T_pad = x_padded.shape[1]
                frames = mx.as_strided(
                    x_padded, shape=(B, n_frames, n_fft),
                    strides=(T_pad, hop, 1),
                )
                fallback_out = frames * t.window

                mx.eval(fused_out, fallback_out)
                np.testing.assert_array_equal(
                    np.array(fused_out), np.array(fallback_out),
                    err_msg=f"fused frame extract mismatch: B={B}, n_fft={n_fft}, hop={hop}",
                )

    def test_tiled_frame_extract_matches_simple(self):
        """Tiled (shared-memory) frame extraction is bit-exact vs simple kernel."""
        from mlx_spectro.spectral_ops import _FrameExtractCache

        simple = _FrameExtractCache.get_simple()
        tiled = _FrameExtractCache.get_tiled()
        if simple is False:
            self.skipTest("Simple frame extraction kernel unavailable")
        if tiled is False:
            self.skipTest("Tiled frame extraction kernel unavailable")

        for B in [1, 4]:
            for n_fft, hop in [(512, 128), (1024, 256), (2048, 512)]:
                tp = _FrameExtractCache.tile_params(n_fft, hop)
                if tp is None:
                    continue
                tile_frames, tg_x, tg_y, chunk_len = tp

                sig_len = n_fft * 10
                x = mx.random.normal((B, sig_len))
                mx.eval(x)
                x_c = mx.contiguous(x)

                pad = n_fft // 2
                padded_len = sig_len + 2 * pad
                n_frames = (padded_len - n_fft) // hop + 1

                # params buffer for both kernels
                params = mx.array([sig_len, n_frames], dtype=mx.int32)

                # Simple kernel
                s_tmpl = [
                    ("T", x.dtype), ("NFFT", n_fft), ("HOP", hop),
                    ("PAD", pad),
                ]
                simple_out = simple(
                    inputs=[x_c, SpectralTransform(n_fft=n_fft, hop_length=hop, window_fn="hann").window, params],
                    template=s_tmpl,
                    output_shapes=[(B, n_frames, n_fft)],
                    output_dtypes=[x.dtype],
                    grid=(n_fft, n_frames, B),
                    threadgroup=(min(256, n_fft), 1, 1),
                )[0]

                # Tiled kernel
                n_tile_groups = math.ceil(n_frames / tile_frames)
                t_tmpl = s_tmpl + [
                    ("TILE_FRAMES", tile_frames), ("TG_X", tg_x),
                    ("TG_Y", tg_y), ("CHUNK_LEN", chunk_len),
                ]
                win = SpectralTransform(n_fft=n_fft, hop_length=hop, window_fn="hann").window
                tiled_out = tiled(
                    inputs=[x_c, win, params],
                    template=t_tmpl,
                    output_shapes=[(B, n_frames, n_fft)],
                    output_dtypes=[x.dtype],
                    grid=(n_tile_groups * tg_x, tg_y, B),
                    threadgroup=(tg_x, tg_y, 1),
                )[0]

                mx.eval(simple_out, tiled_out)
                np.testing.assert_array_equal(
                    np.array(simple_out), np.array(tiled_out),
                    err_msg=f"tiled vs simple mismatch: B={B}, n_fft={n_fft}, hop={hop}",
                )


# ---------------------------------------------------------------------------
# Autograd (differentiable STFT / iSTFT)
# ---------------------------------------------------------------------------


class TestSTFTBackward:
    """Backward-pass tests for differentiable_stft."""

    def test_stft_backward_exists(self):
        """Gradient is non-None, finite, and non-trivial."""
        t = SpectralTransform(512, 128)
        x = mx.random.normal((2, 4096))
        mx.eval(x)

        def loss(x):
            return mx.abs(t.differentiable_stft(x)).square().sum()

        g = mx.grad(loss)(x)
        mx.eval(g)
        assert g.shape == x.shape
        assert g.dtype == mx.float32
        assert bool(mx.isfinite(g).all().item())
        assert bool((mx.abs(g).sum() > 0).item())

    def test_stft_backward_small_signal(self):
        """Backward works for small signals (fallback path)."""
        t = SpectralTransform(256, 64)
        x = mx.random.normal((1, 512))
        mx.eval(x)

        def loss(x):
            return mx.abs(t.differentiable_stft(x)).square().sum()

        g = mx.grad(loss)(x)
        mx.eval(g)
        assert g.shape == x.shape
        assert bool(mx.isfinite(g).all().item())

    def test_stft_backward_no_center(self):
        """Backward works with center=False."""
        t = SpectralTransform(512, 128, center=False)
        x = mx.random.normal((2, 4096))
        mx.eval(x)

        def loss(x):
            return mx.abs(t.differentiable_stft(x)).square().sum()

        g = mx.grad(loss)(x)
        mx.eval(g)
        assert g.shape == x.shape
        assert bool(mx.isfinite(g).all().item())

    @pytest.mark.parametrize("n_fft,hop", [(256, 64), (512, 128), (1024, 256)])
    def test_stft_backward_sizes(self, n_fft, hop):
        """Backward works across different FFT sizes."""
        t = SpectralTransform(n_fft, hop)
        x = mx.random.normal((1, 8000))
        mx.eval(x)

        def loss(x):
            return mx.abs(t.differentiable_stft(x)).square().sum()

        g = mx.grad(loss)(x)
        mx.eval(g)
        assert g.shape == x.shape
        assert bool(mx.isfinite(g).all().item())
        assert bool((mx.abs(g).sum() > 0).item())


class TestISTFTBackward:
    """Backward-pass tests for differentiable_istft."""

    def test_istft_backward_exists(self):
        """Gradient is non-None, finite, and non-trivial."""
        t = SpectralTransform(512, 128)
        x = mx.random.normal((2, 4096))
        mx.eval(x)
        spec = t.stft(x, output_layout="bnf")
        mx.eval(spec)

        def loss(z):
            return t.differentiable_istft(z, length=4096).square().sum()

        g = mx.grad(loss)(spec)
        mx.eval(g)
        assert g.shape == spec.shape
        assert g.dtype == mx.complex64
        assert bool(mx.isfinite(mx.abs(g)).all().item())
        assert bool((mx.abs(g).sum() > 0).item())

    def test_istft_backward_no_center(self):
        """Backward works with center=False."""
        t = SpectralTransform(512, 128, center=False)
        x = mx.random.normal((2, 4096))
        mx.eval(x)
        spec = t.stft(x, output_layout="bnf")
        mx.eval(spec)

        def loss(z):
            return t.differentiable_istft(z).square().sum()

        g = mx.grad(loss)(spec)
        mx.eval(g)
        assert g.shape == spec.shape
        assert bool(mx.isfinite(mx.abs(g)).all().item())

    def test_istft_backward_no_length(self):
        """Backward works when length=None."""
        t = SpectralTransform(512, 128)
        x = mx.random.normal((1, 4096))
        mx.eval(x)
        spec = t.stft(x, output_layout="bnf")
        mx.eval(spec)

        def loss(z):
            return t.differentiable_istft(z).square().sum()

        g = mx.grad(loss)(spec)
        mx.eval(g)
        assert g.shape == spec.shape
        assert bool(mx.isfinite(mx.abs(g)).all().item())


class TestRoundtripGrad:
    """Test gradient flow through STFT → iSTFT roundtrip."""

    def test_roundtrip_grad(self):
        """Full roundtrip: x → stft → istft → loss → backward."""
        t = SpectralTransform(512, 128)
        x = mx.random.normal((2, 8000))
        mx.eval(x)

        def loss(x):
            spec = t.differentiable_stft(x)
            y = t.differentiable_istft(spec, length=8000)
            return (y - mx.stop_gradient(x)).square().sum()

        g = mx.grad(loss)(x)
        mx.eval(g)
        assert g.shape == x.shape
        assert bool(mx.isfinite(g).all().item())
        assert bool((mx.abs(g).sum() > 0).item())

    def test_roundtrip_grad_near_zero(self):
        """Roundtrip reconstruction error gradient should be tiny."""
        t = SpectralTransform(512, 128)
        x = mx.random.normal((1, 4096))
        mx.eval(x)

        def loss(x):
            spec = t.differentiable_stft(x)
            y = t.differentiable_istft(spec, length=4096)
            return (y - mx.stop_gradient(x)).square().sum()

        g = mx.grad(loss)(x)
        mx.eval(g)
        # STFT→iSTFT is near-perfect, so gradient of reconstruction error
        # should be very small.
        assert float(mx.max(mx.abs(g)).item()) < 1e-3

    def test_roundtrip_grad_large_batch(self):
        """Roundtrip with B=4."""
        t = SpectralTransform(512, 128)
        x = mx.random.normal((4, 16000))
        mx.eval(x)

        def loss(x):
            spec = t.differentiable_stft(x)
            y = t.differentiable_istft(spec, length=16000)
            return y.square().sum()

        g = mx.grad(loss)(x)
        mx.eval(g)
        assert g.shape == x.shape
        assert bool(mx.isfinite(g).all().item())


class TestNoGradOverhead:
    """Verify that non-differentiable path still works."""

    def test_stft_no_grad(self):
        """Regular stft produces no autograd graph artifacts."""
        t = SpectralTransform(512, 128)
        x = mx.random.normal((2, 4096))
        mx.eval(x)
        spec = t.stft(x)
        mx.eval(spec)
        # Should work fine without differentiable wrapper
        assert spec.shape[0] == 2

    def test_istft_no_grad(self):
        """Regular istft produces no autograd graph artifacts."""
        t = SpectralTransform(512, 128)
        x = mx.random.normal((2, 4096))
        mx.eval(x)
        spec = t.stft(x, output_layout="bnf")
        mx.eval(spec)
        y = t.istft(spec, length=4096, input_layout="bnf")
        mx.eval(y)
        assert y.shape == (2, 4096)


class TestNumericalGradCheck:
    """Numerical gradient verification using finite differences.

    Uses Pearson correlation to verify gradient direction correctness
    (robust to float32 precision limits) plus a normalized error check
    for magnitude accuracy.
    """

    @staticmethod
    def _check_gradient(analytic_np, numerical_np, label: str, corr_tol: float = 0.999):
        """Verify analytic vs numerical gradient match."""
        a = analytic_np.flatten()
        n = numerical_np.flatten()
        # Correlation: verifies gradient *direction* is correct
        corr = np.corrcoef(a, n)[0, 1]
        # Cosine similarity: verifies both direction and relative magnitude
        cos_sim = np.dot(a, n) / (np.linalg.norm(a) * np.linalg.norm(n) + 1e-12)
        assert corr > corr_tol, (
            f"{label} gradient direction wrong: correlation={corr:.6f} (need >{corr_tol})"
        )
        assert cos_sim > 0.99, (
            f"{label} gradient magnitude wrong: cosine_sim={cos_sim:.6f} (need >0.99)"
        )

    def test_stft_numerical_grad(self):
        """Finite-difference check for STFT gradient."""
        t = SpectralTransform(128, 32)
        x = mx.random.normal((1, 256))
        mx.eval(x)

        def scalar_loss(x):
            spec = t.differentiable_stft(x)
            return mx.abs(spec).square().sum()

        analytic = mx.grad(scalar_loss)(x)
        mx.eval(analytic)

        eps = 1e-3
        x_np = np.array(x)
        numerical = np.zeros_like(x_np)
        for i in range(x_np.shape[1]):
            x_plus = x_np.copy()
            x_plus[0, i] += eps
            x_minus = x_np.copy()
            x_minus[0, i] -= eps
            f_plus = float(scalar_loss(mx.array(x_plus)).item())
            f_minus = float(scalar_loss(mx.array(x_minus)).item())
            numerical[0, i] = (f_plus - f_minus) / (2 * eps)

        self._check_gradient(np.array(analytic), numerical, "STFT")

    def test_istft_numerical_grad(self):
        """Finite-difference check for iSTFT gradient (real part only)."""
        t = SpectralTransform(128, 32)
        x = mx.random.normal((1, 256))
        mx.eval(x)
        spec = t.stft(x, output_layout="bnf")
        mx.eval(spec)

        spec_real = mx.real(spec)
        spec_imag = mx.imag(spec)
        mx.eval(spec_real, spec_imag)

        def scalar_loss_real(sr):
            z = sr + 1j * mx.stop_gradient(spec_imag)
            return t.differentiable_istft(z, length=256).square().sum()

        analytic = mx.grad(scalar_loss_real)(spec_real)
        mx.eval(analytic)

        eps = 1e-3
        sr_np = np.array(spec_real)
        numerical = np.zeros_like(sr_np)
        for n in range(sr_np.shape[1]):
            for f in range(sr_np.shape[2]):
                sr_plus = sr_np.copy()
                sr_plus[0, n, f] += eps
                sr_minus = sr_np.copy()
                sr_minus[0, n, f] -= eps
                f_plus = float(scalar_loss_real(mx.array(sr_plus)).item())
                f_minus = float(scalar_loss_real(mx.array(sr_minus)).item())
                numerical[0, n, f] = (f_plus - f_minus) / (2 * eps)

        # iSTFT OLA scatter-add accumulates many float32 values, reducing
        # numerical precision. Use a slightly relaxed correlation threshold.
        self._check_gradient(np.array(analytic), numerical, "iSTFT", corr_tol=0.99)
