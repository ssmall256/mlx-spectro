from __future__ import annotations

import numpy as np

from mlx_spectro import (
    complex_domain_odf,
    diff_frames_from_hann,
    frame_origin_from_mode,
    local_group_delay,
    normalized_weighted_phase_deviation,
    num_frames_for_hop,
    phase_deviation,
    spectral_odf,
    stft_features_at_fps,
    weighted_phase_deviation,
)


def _test_waveform(seconds: float = 1.0, sample_rate: int = 44_100) -> np.ndarray:
    t = np.arange(int(seconds * sample_rate), dtype=np.float32) / np.float32(sample_rate)
    return (
        0.6 * np.sin(2.0 * np.pi * 220.0 * t)
        + 0.3 * np.sin(2.0 * np.pi * 440.0 * t)
    ).astype(np.float32)


def test_frame_origin_modes_match_madmom_conventions() -> None:
    assert frame_origin_from_mode(1024, "offline") == 0
    assert frame_origin_from_mode(1024, "center") == 0
    assert frame_origin_from_mode(1024, "online") == 511
    assert frame_origin_from_mode(1024, "past") == 511
    assert frame_origin_from_mode(1024, "future") == -512
    assert frame_origin_from_mode(1024, 7) == 7


def test_num_frames_for_hop_matches_expected_rounding() -> None:
    assert num_frames_for_hop(44_100, 441.0, end="normal") == 100
    assert num_frames_for_hop(44_100, 441.0, end="extend") == 101


def test_stft_features_at_fps_returns_expected_shapes_and_dtypes() -> None:
    waveform = _test_waveform()
    features = stft_features_at_fps(waveform, frame_size=2048, fps=100.0)
    assert features.stft.dtype == np.complex64
    assert features.magnitude.dtype == np.float32
    assert features.phase.dtype == np.float32
    assert features.bin_frequencies.dtype == np.float32
    assert features.stft.shape == features.magnitude.shape == features.phase.shape
    assert features.bin_frequencies.shape[0] == features.stft.shape[1]


def test_phase_odf_helpers_are_finite_for_regular_phase_progression() -> None:
    phase = np.tile(np.linspace(-np.pi, np.pi, 32, dtype=np.float32), (8, 1))
    magnitude = np.ones_like(phase, dtype=np.float32)
    assert np.all(np.isfinite(phase_deviation(phase)))
    assert np.all(np.isfinite(weighted_phase_deviation(magnitude, phase)))
    assert np.all(np.isfinite(normalized_weighted_phase_deviation(magnitude, phase)))
    assert np.all(np.isfinite(local_group_delay(phase)))
    assert np.all(np.isfinite(complex_domain_odf(magnitude, phase)))


def test_diff_frames_from_hann_matches_known_values() -> None:
    assert diff_frames_from_hann(frame_size=2048, hop_size=441, diff_ratio=0.5) == 1
    assert diff_frames_from_hann(frame_size=2048, hop_size=220, diff_ratio=0.5) >= 1


def test_spectral_odf_superflux_preset_matches_explicit_defaults() -> None:
    waveform = _test_waveform()
    preset = spectral_odf(
        waveform,
        onset_method="superflux",
        fps=200.0,
        filterbank="log",
        num_bands=24,
        fmin=30.0,
        fmax=17_000.0,
        norm_filters=False,
        log_fn=np.log10,
        mul=1.0,
        add=1.0,
        diff_ratio=0.5,
        diff_max_bins=3,
        preset="madmom_superflux",
    )
    explicit = spectral_odf(
        waveform,
        onset_method="superflux",
        fps=200.0,
        filterbank="log",
        num_bands=24,
        fmin=30.0,
        fmax=17_000.0,
        norm_filters=False,
        log_fn=np.log10,
        mul=1.0,
        add=1.0,
        diff_ratio=0.5,
        diff_max_bins=3,
        origin="offline",
        end="normal",
    )
    np.testing.assert_allclose(preset, explicit, atol=1e-6)


def test_spectral_odf_phase_and_complex_methods_are_finite() -> None:
    waveform = _test_waveform()
    for method in (
        "phase_deviation",
        "weighted_phase_deviation",
        "normalized_weighted_phase_deviation",
        "complex_domain",
        "rectified_complex_domain",
    ):
        odf = spectral_odf(
            waveform,
            onset_method=method,
            fps=100.0,
            filterbank=None,
            log_fn=None,
            circular_shift=True,
        )
        assert odf.dtype == np.float32
        assert np.all(np.isfinite(odf))
