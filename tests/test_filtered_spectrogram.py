import mlx.core as mx
import numpy as np
import pytest

from mlx_spectro import (
    FilteredSpectrogramTransform,
    filtered_spectrogram,
    log_triangular_fbanks,
    positive_spectral_diff,
)


def _to_numpy(x: mx.array) -> np.ndarray:
    mx.eval(x)
    return np.asarray(x, dtype=np.float32)


def _audio(length: int = 16_000, *, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (0.2 * rng.standard_normal(length)).astype(np.float32)


def _simple_filterbank(n_freqs: int, n_bands: int = 8) -> np.ndarray:
    fb = np.zeros((n_freqs, n_bands), dtype=np.float32)
    bins = np.array_split(np.arange(n_freqs), n_bands)
    for idx, band in enumerate(bins):
        fb[band, idx] = 1.0
    return fb


def _reference_log_triangular_fbanks(
    n_freqs: int,
    sample_rate: int,
    bands_per_octave: int,
    *,
    f_min: float,
    f_max: float,
    f_ref: float | None = 440.0,
    norm_filters: bool = True,
    include_nyquist: bool = False,
) -> np.ndarray:
    if include_nyquist:
        bin_frequencies = np.fft.rfftfreq((n_freqs - 1) * 2, 1.0 / sample_rate).astype(np.float32)
    else:
        bin_frequencies = np.fft.fftfreq(n_freqs * 2, 1.0 / sample_rate)[:n_freqs].astype(np.float32)
    if f_ref is None:
        num_octaves = np.log2(f_max / f_min)
        num_bands = int(np.round(bands_per_octave * num_octaves))
        centers = f_min * 2.0 ** (np.arange(num_bands + 2) / float(bands_per_octave))
    else:
        left = np.floor(np.log2(f_min / f_ref) * bands_per_octave)
        right = np.ceil(np.log2(f_max / f_ref) * bands_per_octave)
        centers = f_ref * 2.0 ** (np.arange(left, right) / float(bands_per_octave))
        centers = centers[np.searchsorted(centers, f_min):]
        centers = centers[: np.searchsorted(centers, f_max, side="right")]
    bins = bin_frequencies.searchsorted(centers)
    bins = np.clip(bins, 1, len(bin_frequencies) - 1)
    left_bins = bin_frequencies[bins - 1]
    right_bins = bin_frequencies[bins]
    bins -= centers - left_bins < right_bins - centers
    bins = np.unique(bins)
    filters = []
    for start, center, stop in zip(bins[:-2], bins[1:-1], bins[2:]):
        start_i = int(start)
        center_i = int(center)
        stop_i = int(stop)
        if stop_i - start_i < 2:
            center_i = start_i
            stop_i = start_i + 1
        center_rel = center_i - start_i
        stop_rel = stop_i - start_i
        filt = np.zeros(stop_rel, dtype=np.float32)
        if center_rel > 0:
            filt[:center_rel] = np.linspace(0.0, 1.0, center_rel, endpoint=False)
        filt[center_rel:] = np.linspace(1.0, 0.0, stop_rel - center_rel, endpoint=False)
        if norm_filters:
            total = float(np.sum(filt))
            if total > 0.0:
                filt /= total
        band = np.zeros(n_freqs, dtype=np.float32)
        band[start_i:stop_i] = filt
        filters.append(band)
    return np.stack(filters, axis=1)


def test_filtered_spectrogram_shape_1d():
    x = mx.array(_audio(seed=1))
    fb = _simple_filterbank(257, 8)
    out = filtered_spectrogram(x, filterbank=fb, n_fft=512, hop_length=128)
    mx.eval(out)
    assert out.ndim == 2
    assert out.shape[0] == 8


def test_filtered_spectrogram_shape_batched():
    x_np = _audio(seed=2)
    x = mx.array(np.stack([x_np, x_np]))
    fb = _simple_filterbank(257, 8)
    out = filtered_spectrogram(x, filterbank=fb, n_fft=512, hop_length=128)
    mx.eval(out)
    assert out.ndim == 3
    assert out.shape[:2] == (2, 8)


def test_filtered_spectrogram_log10_plus_one_matches_manual():
    x = mx.array(_audio(seed=3))
    fb = _simple_filterbank(257, 8)
    tr = FilteredSpectrogramTransform(
        filterbank=fb,
        n_fft=512,
        hop_length=128,
        power=1.0,
        output_scale="linear",
    )
    linear = _to_numpy(tr(x))
    logged = _to_numpy(
        filtered_spectrogram(
            x,
            filterbank=fb,
            n_fft=512,
            hop_length=128,
            power=1.0,
            output_scale="log10_plus_one",
        )
    )
    np.testing.assert_allclose(logged, np.log10(linear + 1.0), rtol=1e-6, atol=1e-6)


def test_filtered_spectrogram_log_matches_manual_clamp():
    x = mx.array(_audio(seed=4))
    fb = _simple_filterbank(257, 8)
    tr = FilteredSpectrogramTransform(
        filterbank=fb,
        n_fft=512,
        hop_length=128,
        power=1.0,
        output_scale="linear",
    )
    linear = _to_numpy(tr(x))
    logged = _to_numpy(
        filtered_spectrogram(
            x,
            filterbank=fb,
            n_fft=512,
            hop_length=128,
            power=1.0,
            output_scale="log",
            log_mode="clamp",
            log_amin=1e-5,
        )
    )
    np.testing.assert_allclose(logged, np.log(np.maximum(linear, 1e-5)), rtol=1e-6, atol=1e-6)


def test_filtered_spectrogram_accepts_no_nyquist_filterbank():
    x = mx.array(_audio(seed=11))
    fb = _simple_filterbank(256, 8)
    out = filtered_spectrogram(
        x,
        filterbank=fb,
        n_fft=512,
        hop_length=128,
        output_scale="log10_plus_one",
        periodic=False,
        center=True,
        center_pad_mode="constant",
        center_tail_pad="minimal",
    )
    mx.eval(out)
    assert out.shape[0] == 8


def test_log_triangular_fbanks_matches_reference():
    got = _to_numpy(
        log_triangular_fbanks(
            512,
            44_100,
            6,
            f_min=30.0,
            f_max=17_000.0,
            include_nyquist=False,
        )
    )
    ref = _reference_log_triangular_fbanks(
        512,
        44_100,
        6,
        f_min=30.0,
        f_max=17_000.0,
        include_nyquist=False,
    )
    np.testing.assert_allclose(got, ref, atol=1e-7, rtol=1e-7)


def test_log_triangular_fbanks_matches_madmom_style_reference():
    got = _to_numpy(
        log_triangular_fbanks(
            704,
            22_050,
            24,
            f_min=30.0,
            f_max=17_000.0,
            f_ref=None,
            include_nyquist=False,
        )
    )
    ref = _reference_log_triangular_fbanks(
        704,
        22_050,
        24,
        f_min=30.0,
        f_max=17_000.0,
        f_ref=None,
        include_nyquist=False,
    )
    np.testing.assert_allclose(got, ref, atol=1e-7, rtol=1e-7)


def test_positive_spectral_diff_matches_expected():
    x = mx.array(np.array([[1.0, 3.0, 2.0, 5.0]], dtype=np.float32))
    out = _to_numpy(positive_spectral_diff(x, lag=1))
    ref = np.array([[0.0, 2.0, 0.0, 3.0]], dtype=np.float32)
    np.testing.assert_allclose(out, ref, atol=1e-7)


def test_positive_spectral_diff_time_axis_zero_matches_expected():
    x = mx.array(np.array([[1.0, 4.0], [3.0, 2.0], [2.0, 5.0]], dtype=np.float32))
    out = _to_numpy(positive_spectral_diff(x, lag=1, time_axis=0))
    ref = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 3.0]], dtype=np.float32)
    np.testing.assert_allclose(out, ref, atol=1e-7)


def test_positive_spectral_diff_frame_size_matches_expected():
    x = mx.array(np.array([[1.0, 4.0], [3.0, 2.0], [2.0, 5.0]], dtype=np.float32))
    out = _to_numpy(
        positive_spectral_diff(
            x,
            frame_size=1024,
            hop_size=441,
            diff_ratio=0.5,
            time_axis=0,
        )
    )
    ref = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 3.0]], dtype=np.float32)
    np.testing.assert_allclose(out, ref, atol=1e-7)


def test_filtered_spectrogram_rejects_bad_filterbank():
    with pytest.raises(ValueError, match="filterbank axis 0 must have size"):
        FilteredSpectrogramTransform(filterbank=np.ones((16, 4), dtype=np.float32), n_fft=512)
