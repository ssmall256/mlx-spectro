import mlx.core as mx
import numpy as np
import pytest
from mlx_spectro import (
    FilteredSpectrogramResult,
    FilteredSpectrogramTransform,
    compute_filtered_spectrogram,
    compute_filtered_spectrogram_at_fps,
    compute_filtered_spectrogram_at_starts,
    compute_log_filtered_spectrogram,
    compute_mel_spectrogram,
    fft_frequencies,
    filtered_spectrogram,
    filtered_spectrogram_at_fps,
    filtered_spectrogram_at_starts,
    frame_starts_from_fps,
    log_triangular_fbanks,
    logarithmic_spectrogram,
    mel_filterbank,
    positive_spectral_diff,
    rectangular_filterbank,
    triangular_filterbank,
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


def test_filtered_spectrogram_get_compiled_matches_eager():
    x = mx.array(_audio(seed=12))
    fb = _simple_filterbank(257, 8)
    tr = FilteredSpectrogramTransform(
        filterbank=fb,
        n_fft=512,
        hop_length=128,
        power=1.0,
        output_scale="log10_plus_one",
        periodic=False,
        center=True,
        center_pad_mode="constant",
        center_tail_pad="minimal",
    )
    compiled = tr.get_compiled()
    eager = tr(x)
    compiled_out = compiled(x)
    np.testing.assert_allclose(_to_numpy(eager), _to_numpy(compiled_out), rtol=1e-5, atol=1e-5)


def test_filtered_spectrogram_get_compiled_is_cached():
    fb = _simple_filterbank(257, 8)
    tr = FilteredSpectrogramTransform(filterbank=fb, n_fft=512, hop_length=128)
    compiled1 = tr.get_compiled()
    compiled2 = tr.get_compiled()
    assert compiled1 is compiled2


def test_filtered_spectrogram_at_starts_matches_centered_integer_hop_path():
    x = mx.array(_audio(seed=21))
    fb = _simple_filterbank(257, 8)
    n_fft = 512
    hop = 128
    centered = _to_numpy(
        filtered_spectrogram(
            x,
            filterbank=fb,
            n_fft=n_fft,
            hop_length=hop,
            output_scale="log10_plus_one",
            periodic=False,
            center=True,
            center_pad_mode="constant",
            center_tail_pad="minimal",
        )
    )
    frame_starts = np.arange(centered.shape[-1], dtype=np.int32) * hop - (n_fft // 2)
    explicit = _to_numpy(
        filtered_spectrogram_at_starts(
            x,
            frame_starts=frame_starts,
            filterbank=fb,
            n_fft=n_fft,
            hop_length=hop,
            output_scale="log10_plus_one",
            periodic=False,
            center=True,
            center_pad_mode="constant",
            center_tail_pad="minimal",
        )
    )
    np.testing.assert_allclose(explicit, centered, rtol=1e-6, atol=1e-6)


def test_filtered_spectrogram_at_starts_zero_pads_negative_and_overrun_samples():
    x = mx.array(np.arange(8, dtype=np.float32))
    tr = FilteredSpectrogramTransform(
        filterbank=np.eye(5, dtype=np.float32),
        n_fft=8,
        hop_length=4,
        output_scale="linear",
        periodic=False,
        center=False,
    )
    frame_starts = np.array([-2, 3], dtype=np.int32)
    out = _to_numpy(tr.filtered_spectrogram_at_starts(x, frame_starts=frame_starts))
    window = np.hanning(8).astype(np.float32)
    ref_frames = np.array(
        [
            [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0, 7.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    ref = np.abs(np.fft.rfft(ref_frames * window[None, :], axis=1)).astype(np.float32).T
    np.testing.assert_allclose(out, ref, rtol=1e-6, atol=1e-6)


def test_frame_starts_from_fps_matches_madmom_style_fractional_hop_schedule():
    starts = frame_starts_from_fps(
        12_348,
        frame_size=2048,
        fps=200.0,
        sample_rate=44_100,
        origin="offline",
        end="normal",
    )
    expected = (
        np.arange(len(starts), dtype=np.float64) * (44_100.0 / 200.0)
    ).astype(np.int64) - 1024
    np.testing.assert_array_equal(starts, expected.astype(np.int32))


def test_filtered_spectrogram_at_fps_matches_explicit_frame_starts():
    x = mx.array(_audio(length=12_348, seed=22))
    fb = _simple_filterbank(1025, 8)
    starts = frame_starts_from_fps(
        12_348,
        frame_size=2048,
        fps=200.0,
        sample_rate=44_100,
        origin="offline",
        end="normal",
    )
    explicit = _to_numpy(
        filtered_spectrogram_at_starts(
            x,
            frame_starts=starts,
            filterbank=fb,
            sample_rate=44_100,
            n_fft=2048,
            hop_length=1,
            output_scale="log10_plus_one",
            periodic=False,
            center=True,
            center_pad_mode="constant",
            center_tail_pad="minimal",
        )
    )
    by_fps = _to_numpy(
        filtered_spectrogram_at_fps(
            x,
            fps=200.0,
            filterbank=fb,
            sample_rate=44_100,
            n_fft=2048,
            hop_length=1,
            output_scale="log10_plus_one",
            periodic=False,
            center=True,
            center_pad_mode="constant",
            center_tail_pad="minimal",
            origin="offline",
            end="normal",
        )
    )
    np.testing.assert_allclose(by_fps, explicit, rtol=1e-6, atol=1e-6)


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


def test_fft_frequencies_matches_numpy_rfftfreq():
    got = fft_frequencies(257, 44_100)
    ref = np.fft.rfftfreq(512, d=1.0 / 44_100).astype(np.float32)
    np.testing.assert_allclose(got, ref, atol=1e-7, rtol=1e-7)


def test_triangular_filterbank_matches_log_triangular_reference():
    bin_freqs = fft_frequencies(512, 44_100)
    got = triangular_filterbank(
        bin_freqs,
        6,
        fmin=30.0,
        fmax=17_000.0,
        norm_filters=True,
    )
    ref = _reference_log_triangular_fbanks(
        512,
        44_100,
        6,
        f_min=30.0,
        f_max=17_000.0,
        include_nyquist=True,
    )
    np.testing.assert_allclose(got, ref, atol=1e-7, rtol=1e-7)


def test_mel_and_rectangular_filterbanks_have_expected_shapes():
    bin_freqs = fft_frequencies(512, 44_100)
    mel = mel_filterbank(
        bin_freqs,
        24,
        fmin=30.0,
        fmax=10_000.0,
        norm_filters=True,
    )
    rect = rectangular_filterbank(
        bin_freqs,
        (270.0,),
        fmin=30.0,
        fmax=17_000.0,
        norm_filters=False,
    )
    assert mel.shape == (len(bin_freqs), 24)
    assert rect.shape == (len(bin_freqs), 2)
    assert np.all(mel.sum(axis=0) > 0)
    assert np.all(rect.sum(axis=0) > 0)


def test_logarithmic_spectrogram_matches_manual_formula():
    spec = np.array([[1.0, 3.0], [5.0, 7.0]], dtype=np.float32)
    got = logarithmic_spectrogram(spec, mul=5.0, add=1.0, log_fn=np.log10)
    ref = np.log10(spec * 5.0 + 1.0).astype(np.float32)
    np.testing.assert_allclose(got, ref, atol=1e-7, rtol=1e-7)


def test_compute_filtered_spectrogram_wrappers_match_low_level_paths():
    waveform = _audio(seed=31)
    direct = _to_numpy(
        filtered_spectrogram(
            mx.array(waveform),
            filterbank=log_triangular_fbanks(
                256,
                44_100,
                6,
                f_min=30.0,
                f_max=17_000.0,
                include_nyquist=False,
            ),
            sample_rate=44_100,
            n_fft=512,
            hop_length=441,
            output_scale="linear",
            periodic=False,
            center=True,
            center_pad_mode="constant",
            center_tail_pad="minimal",
        )
    ).T
    wrapped = compute_filtered_spectrogram(
        waveform,
        frame_size=512,
        hop_size=441,
        sample_rate=44_100,
        num_bands=6,
        fmin=30.0,
        fmax=17_000.0,
        norm_filters=True,
        output_scale="linear",
    )
    assert isinstance(wrapped, FilteredSpectrogramResult)
    np.testing.assert_allclose(wrapped.spectrogram, direct, atol=1e-6, rtol=1e-6)


def test_compute_filtered_spectrogram_at_fps_and_starts_wrappers_match():
    waveform = _audio(seed=32)
    starts = frame_starts_from_fps(
        len(waveform),
        frame_size=512,
        fps=100.0,
        sample_rate=44_100,
        origin="offline",
        end="normal",
    )
    by_starts = compute_filtered_spectrogram_at_starts(
        waveform,
        frame_size=512,
        frame_starts=starts,
        sample_rate=44_100,
        num_bands=6,
        fmin=30.0,
        fmax=17_000.0,
        norm_filters=True,
        output_scale="linear",
    )
    by_fps = compute_filtered_spectrogram_at_fps(
        waveform,
        frame_size=512,
        fps=100.0,
        sample_rate=44_100,
        num_bands=6,
        fmin=30.0,
        fmax=17_000.0,
        norm_filters=True,
        output_scale="linear",
        origin="offline",
        end="normal",
    )
    np.testing.assert_allclose(by_starts.spectrogram, by_fps.spectrogram, atol=1e-6, rtol=1e-6)


def test_compute_log_and_mel_spectrogram_wrappers_return_expected_shapes():
    waveform = _audio(seed=33)
    log_spec = compute_log_filtered_spectrogram(
        waveform,
        frame_size=512,
        hop_size=441,
        sample_rate=44_100,
        num_bands=6,
        fmin=30.0,
        fmax=17_000.0,
        norm_filters=True,
    )
    mel_spec = compute_mel_spectrogram(
        waveform,
        frame_size=512,
        hop_size=441,
        sample_rate=44_100,
        num_bands=12,
        fmin=30.0,
        fmax=8_000.0,
        norm_filters=True,
    )
    assert log_spec.spectrogram.ndim == 2
    assert mel_spec.spectrogram.ndim == 2
    assert log_spec.filterbank.shape[1] > 0
    assert mel_spec.filterbank.shape[1] == 12
