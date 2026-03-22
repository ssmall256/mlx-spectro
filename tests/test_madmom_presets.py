from __future__ import annotations

import mlx.core as mx
import numpy as np

from mlx_spectro import (
    compute_filtered_spectrogram,
    hop_size_from_fps,
    logarithmic_spectrogram,
    madmom_multires_log_diff_features,
    madmom_multires_mel_stack,
    madmom_single_resolution_log_stack,
    positive_spectral_diff,
    repeat_pad_frames,
    stack_feature_blocks,
    trim_to_shortest,
)


def _test_waveform(seconds: float = 1.0, sample_rate: int = 44_100) -> np.ndarray:
    t = np.arange(int(seconds * sample_rate), dtype=np.float32) / np.float32(sample_rate)
    return (
        0.6 * np.sin(2.0 * np.pi * 220.0 * t)
        + 0.3 * np.sin(2.0 * np.pi * 440.0 * t)
    ).astype(np.float32)


def test_trim_to_shortest_trims_all_blocks_to_min_time() -> None:
    a = np.ones((5, 3), dtype=np.float32)
    b = np.ones((3, 2), dtype=np.float32)
    c = np.ones((4, 1), dtype=np.float32)
    trimmed = trim_to_shortest([a, b, c])
    assert [block.shape[0] for block in trimmed] == [3, 3, 3]


def test_repeat_pad_frames_repeats_edge_frames() -> None:
    frames = np.arange(6, dtype=np.float32).reshape(3, 2)
    padded = repeat_pad_frames(frames, 2, pad_after=1)
    assert padded.shape == (6, 2)
    np.testing.assert_array_equal(padded[0], frames[0])
    np.testing.assert_array_equal(padded[1], frames[0])
    np.testing.assert_array_equal(padded[-1], frames[-1])


def test_stack_feature_blocks_supports_feature_and_channel_layouts() -> None:
    a = np.ones((4, 2), dtype=np.float32)
    b = np.full((5, 3), 2.0, dtype=np.float32)
    feat = stack_feature_blocks([a, b], layout="feature_stack")
    chan = stack_feature_blocks([a, a], layout="channel_stack")
    assert feat.shape == (4, 5)
    assert chan.shape == (4, 2, 2)


def test_madmom_multires_log_diff_features_matches_manual_construction() -> None:
    waveform = _test_waveform()
    frame_sizes = (1024, 2048)
    features = madmom_multires_log_diff_features(
        waveform,
        frame_sizes=frame_sizes,
        fps=100.0,
        num_bands=6,
        log_fn=np.log10,
        mul=5.0,
        add=1.0,
        diff_ratio=0.25,
    )

    hop_size = int(round(hop_size_from_fps(100.0)))
    manual_blocks = []
    for frame_size in frame_sizes:
        filt = compute_filtered_spectrogram(
            waveform,
            frame_size=frame_size,
            hop_size=hop_size,
            num_bands=6,
            output_scale="linear",
        ).spectrogram
        log_spec = logarithmic_spectrogram(filt, mul=5.0, add=1.0, log_fn=np.log10)
        diff = positive_spectral_diff(
            mx.array(log_spec, dtype=mx.float32),
            frame_size=frame_size,
            hop_size=hop_size,
            diff_ratio=0.25,
            time_axis=0,
        )
        mx.eval(diff)
        manual_blocks.extend((log_spec, np.asarray(diff, dtype=np.float32)))
    expected = stack_feature_blocks(manual_blocks, layout="feature_stack")
    np.testing.assert_allclose(features, expected, atol=1e-6)


def test_madmom_multires_mel_stack_matches_manual_channel_stack() -> None:
    waveform = _test_waveform()
    frame_sizes = (2048, 1024, 4096)
    stacked = madmom_multires_mel_stack(
        waveform,
        frame_sizes=frame_sizes,
        fps=100.0,
        num_bands=80,
        fmin=27.5,
        fmax=16_000.0,
        log_fn=np.log,
        add=np.spacing(1),
        pad_frames=7,
    )
    assert stacked.shape[-1] == len(frame_sizes)
    assert stacked.shape[0] > 14
    np.testing.assert_allclose(stacked[:7], np.repeat(stacked[7:8], 7, axis=0), atol=1e-6)
    np.testing.assert_allclose(stacked[-7:], np.repeat(stacked[-8:-7], 7, axis=0), atol=1e-6)


def test_madmom_single_resolution_log_stack_matches_manual_transform_path() -> None:
    waveform = _test_waveform()
    log_spec = madmom_single_resolution_log_stack(
        waveform,
        frame_size=8192,
        hop_size=4410,
        num_bands=24,
        fmin=60.0,
        fmax=2600.0,
        norm_filters=True,
        filterbank="log",
        log_fn=np.log10,
        mul=1.0,
        add=1.0,
        center_tail_pad="minimal",
    )
    manual = compute_filtered_spectrogram(
        waveform,
        frame_size=8192,
        hop_size=4410,
        num_bands=24,
        fmin=60.0,
        fmax=2600.0,
        norm_filters=True,
        output_scale="linear",
        center_tail_pad="minimal",
    ).spectrogram
    manual = logarithmic_spectrogram(manual, mul=1.0, add=1.0, log_fn=np.log10)
    np.testing.assert_allclose(log_spec, manual, atol=1e-6)


def test_madmom_single_resolution_log_stack_stft_compat_pads_edges() -> None:
    waveform = _test_waveform()
    log_spec = madmom_single_resolution_log_stack(
        waveform,
        frame_size=4096,
        fps=50.0,
        num_bands=24,
        fmin=30.0,
        fmax=10_000.0,
        filterbank="log",
        log_fn=np.log10,
        add=1.0,
        pad_frames=5,
        backend="stft_compat",
    )
    assert log_spec.shape[0] > 10
    np.testing.assert_allclose(log_spec[:5], np.repeat(log_spec[5:6], 5, axis=0), atol=1e-6)
    np.testing.assert_allclose(
        log_spec[-5:], np.repeat(log_spec[-6:-5], 5, axis=0), atol=1e-6
    )
