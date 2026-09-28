"""Parity tests: _mlx variants must match their numpy counterparts within tolerance."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mlx_spectro import (
    madmom_multires_log_diff_features,
    madmom_multires_log_diff_features_mlx,
    madmom_multires_mel_stack,
    madmom_multires_mel_stack_mlx,
    madmom_single_resolution_log_stack,
    madmom_single_resolution_log_stack_mlx,
)
from mlx_spectro.spectral_ops import _spectral_diff_mlx, positive_spectral_diff_numpy


def _test_waveform(seconds: float = 1.0, sample_rate: int = 44_100) -> np.ndarray:
    t = np.arange(int(seconds * sample_rate), dtype=np.float32) / np.float32(sample_rate)
    return (
        0.6 * np.sin(2.0 * np.pi * 220.0 * t)
        + 0.3 * np.sin(2.0 * np.pi * 440.0 * t)
    ).astype(np.float32)


def test_multires_log_diff_features_mlx_matches_numpy() -> None:
    waveform = _test_waveform()
    kwargs = dict(
        frame_sizes=(1024, 2048),
        fps=100.0,
        num_bands=6,
        sample_rate=44_100,
        fmin=30.0,
        fmax=17_000.0,
        norm_filters=True,
        output_scale="linear",
        log_fn=np.log10,
        mul=5.0,
        add=1.0,
        diff_ratio=0.25,
    )
    np_result = madmom_multires_log_diff_features(waveform, **kwargs)
    mlx_result = madmom_multires_log_diff_features_mlx(waveform, **kwargs)
    mx.eval(mlx_result)
    np.testing.assert_allclose(
        np.asarray(mlx_result, dtype=np.float32),
        np_result,
        atol=1e-5,
        rtol=1e-5,
    )


def test_multires_log_diff_features_mlx_accepts_mx_array() -> None:
    waveform = mx.array(_test_waveform())
    result = madmom_multires_log_diff_features_mlx(
        waveform,
        frame_sizes=(1024, 2048),
        fps=100.0,
        num_bands=6,
        log_fn=np.log10,
        mul=1.0,
        add=1.0,
    )
    mx.eval(result)
    assert result.dtype == mx.float32
    assert result.ndim == 2
    assert np.all(np.isfinite(np.asarray(result)))


def test_multires_log_diff_features_mlx_per_resolution_bands() -> None:
    waveform = _test_waveform()
    kwargs = dict(
        frame_sizes=(1024, 2048, 4096),
        fps=100.0,
        num_bands=(3, 6, 12),
        sample_rate=44_100,
        fmin=30.0,
        fmax=17_000.0,
        norm_filters=True,
        output_scale="linear",
        log_fn=np.log10,
        mul=1.0,
        add=1.0,
        diff_ratio=0.5,
    )
    np_result = madmom_multires_log_diff_features(waveform, **kwargs)
    mlx_result = madmom_multires_log_diff_features_mlx(waveform, **kwargs)
    mx.eval(mlx_result)
    np.testing.assert_allclose(
        np.asarray(mlx_result, dtype=np.float32),
        np_result,
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.parametrize(
    ("diff_max_bins", "positive_diffs"),
    [(3, True), (3, False), (None, False)],
)
def test_multires_log_diff_features_mlx_frequency_max_stays_lazy(
    monkeypatch, diff_max_bins, positive_diffs
) -> None:
    waveform = mx.array(_test_waveform())
    kwargs = dict(
        frame_sizes=(1024, 2048),
        fps=100.0,
        num_bands=6,
        diff_max_bins=diff_max_bins,
        positive_diffs=positive_diffs,
    )
    expected = madmom_multires_log_diff_features(np.asarray(waveform), **kwargs)
    original_asarray = np.asarray

    def reject_mlx_asarray(value, *args, **kwargs):
        if isinstance(value, mx.array):
            raise AssertionError("MLX array crossed into NumPy")
        return original_asarray(value, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(np, "asarray", reject_mlx_asarray)
        actual = madmom_multires_log_diff_features_mlx(waveform, **kwargs)
        mx.eval(actual)

    np.testing.assert_allclose(np.asarray(actual), expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(("num_bands", "window_size"), [(1, 3), (2, 5), (6, 3), (6, 7)])
def test_spectral_diff_mlx_reflection_matches_numpy(num_bands, window_size) -> None:
    spec = np.arange(8 * num_bands, dtype=np.float32).reshape(8, num_bands)
    expected = positive_spectral_diff_numpy(
        spec, diff_frames=2, diff_max_bins=window_size, positive_diffs=False
    )
    actual = _spectral_diff_mlx(
        mx.array(spec), diff_frames=2, diff_max_bins=window_size, positive_diffs=False
    )
    np.testing.assert_array_equal(np.asarray(actual), expected)


def test_multires_mel_stack_mlx_matches_numpy() -> None:
    waveform = _test_waveform()
    kwargs = dict(
        frame_sizes=(2048, 1024, 4096),
        fps=100.0,
        num_bands=80,
        sample_rate=44_100,
        fmin=27.5,
        fmax=16_000.0,
        norm_filters=True,
        output_scale="linear",
        log_fn=np.log,
        add=np.spacing(1),
        pad_frames=7,
    )
    np_result = madmom_multires_mel_stack(waveform, **kwargs)
    mlx_result = madmom_multires_mel_stack_mlx(waveform, **kwargs)
    mx.eval(mlx_result)
    np.testing.assert_allclose(
        np.asarray(mlx_result, dtype=np.float32),
        np_result,
        atol=1e-5,
        rtol=1e-5,
    )


def test_multires_mel_stack_mlx_shape_and_padding() -> None:
    waveform = mx.array(_test_waveform())
    result = madmom_multires_mel_stack_mlx(
        waveform,
        frame_sizes=(1024, 2048),
        fps=100.0,
        num_bands=40,
        fmin=30.0,
        fmax=17_000.0,
        log_fn=np.log,
        add=np.spacing(1),
        pad_frames=3,
    )
    mx.eval(result)
    assert result.ndim == 3
    assert int(result.shape[2]) == 2
    head = np.asarray(result[:3])
    np.testing.assert_allclose(head[0], head[1], atol=1e-7)
    np.testing.assert_allclose(head[0], head[2], atol=1e-7)


def test_single_resolution_log_stack_mlx_matches_numpy_log_filterbank() -> None:
    waveform = _test_waveform()
    kwargs = dict(
        frame_size=2048,
        fps=100.0,
        num_bands=24,
        sample_rate=44_100,
        fmin=30.0,
        fmax=17_000.0,
        norm_filters=True,
        filterbank="log",
        output_scale="linear",
        log_fn=np.log10,
        mul=1.0,
        add=1.0,
        center_tail_pad="minimal",
    )
    np_result = madmom_single_resolution_log_stack(waveform, **kwargs)
    mlx_result = madmom_single_resolution_log_stack_mlx(waveform, **kwargs)
    mx.eval(mlx_result)
    np.testing.assert_allclose(
        np.asarray(mlx_result, dtype=np.float32),
        np_result,
        atol=1e-5,
        rtol=1e-5,
    )


def test_single_resolution_log_stack_mlx_matches_numpy_mel_filterbank() -> None:
    waveform = _test_waveform()
    kwargs = dict(
        frame_size=4096,
        hop_size=441,
        num_bands=40,
        sample_rate=44_100,
        fmin=60.0,
        fmax=8_000.0,
        norm_filters=True,
        filterbank="mel",
        output_scale="linear",
        log_fn=np.log10,
        add=1.0,
    )
    np_result = madmom_single_resolution_log_stack(waveform, **kwargs)
    mlx_result = madmom_single_resolution_log_stack_mlx(waveform, **kwargs)
    mx.eval(mlx_result)
    np.testing.assert_allclose(
        np.asarray(mlx_result, dtype=np.float32),
        np_result,
        atol=1e-5,
        rtol=1e-5,
    )


def test_single_resolution_log_stack_mlx_padding() -> None:
    waveform = mx.array(_test_waveform())
    result = madmom_single_resolution_log_stack_mlx(
        waveform,
        frame_size=2048,
        fps=100.0,
        num_bands=12,
        log_fn=np.log10,
        add=1.0,
        pad_frames=4,
    )
    mx.eval(result)
    assert result.ndim == 2
    head = np.asarray(result[:4])
    np.testing.assert_allclose(head[0], head[1], atol=1e-7)
    np.testing.assert_allclose(head[0], head[3], atol=1e-7)
