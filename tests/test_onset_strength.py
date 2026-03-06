import mlx.core as mx
import numpy as np
import pytest

from mlx_spectro import onset_strength, onset_strength_multi


def _to_numpy(x: mx.array) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def _make_test_signal(sr: int = 44100, duration: float = 2.0) -> mx.array:
    """Generate a test signal with transients for onset detection."""
    n = int(sr * duration)
    t = np.linspace(0, duration, n, dtype=np.float32)
    # Sine sweep with amplitude envelope to create onsets
    sig = np.sin(2 * np.pi * 440 * t) * 0.3
    # Add clicks at known positions
    for pos in [0.25, 0.75, 1.25, 1.75]:
        idx = int(pos * sr)
        sig[idx : idx + 100] += 0.7
    return mx.array(sig)


# ── Shape tests ──────────────────────────────────────────────────────────


def test_onset_strength_1d_shape():
    x = _make_test_signal(sr=44100, duration=1.0)
    odf = onset_strength(x, sample_rate=44100, hop_length=441)
    assert odf.ndim == 1
    expected_frames = 1 + len(x) // 441
    assert odf.shape[0] == expected_frames


def test_onset_strength_2d_shape():
    x = _make_test_signal(sr=44100, duration=1.0)
    x_batch = mx.stack([x, x])  # [2, T]
    odf = onset_strength(x_batch, sample_rate=44100, hop_length=441)
    assert odf.ndim == 2
    assert odf.shape[0] == 2
    expected_frames = 1 + x.shape[0] // 441
    assert odf.shape[1] == expected_frames


def test_onset_strength_low_band_shape():
    x = _make_test_signal(sr=44100, duration=1.0)
    odf = onset_strength(x, sample_rate=44100, hop_length=441, n_mels=12, f_max=180.0)
    assert odf.ndim == 1
    expected_frames = 1 + len(x) // 441
    assert odf.shape[0] == expected_frames


def test_onset_strength_non_negative():
    x = _make_test_signal()
    odf = _to_numpy(onset_strength(x, sample_rate=44100, hop_length=441))
    assert np.all(odf >= 0.0)


def test_onset_strength_returns_mx_array():
    x = _make_test_signal(sr=22050, duration=0.5)
    odf = onset_strength(x)
    assert isinstance(odf, mx.array)


def test_onset_strength_invalid_lag():
    x = _make_test_signal(sr=22050, duration=0.5)
    with pytest.raises(ValueError, match="lag"):
        onset_strength(x, lag=0)


# ── Librosa parity tests ────────────────────────────────────────────────


# ── onset_strength_multi tests ────────────────────────────────────────


def test_onset_strength_multi_1d_shape():
    x = _make_test_signal(sr=44100, duration=1.0)
    multi = onset_strength_multi(x, sample_rate=44100, hop_length=441, n_mels=40)
    assert multi.ndim == 2
    assert multi.shape[0] == 40
    expected_frames = 1 + len(x) // 441
    assert multi.shape[1] == expected_frames


def test_onset_strength_multi_2d_shape():
    x = _make_test_signal(sr=44100, duration=1.0)
    x_batch = mx.stack([x, x])
    multi = onset_strength_multi(x_batch, sample_rate=44100, hop_length=441, n_mels=40)
    assert multi.ndim == 3
    assert multi.shape[0] == 2
    assert multi.shape[1] == 40


def test_onset_strength_multi_mean_equals_onset_strength():
    """Averaging onset_strength_multi across mel bands should equal onset_strength."""
    x = _make_test_signal(sr=44100, duration=1.0)
    kwargs = dict(sample_rate=44100, hop_length=441, n_mels=64)
    multi = onset_strength_multi(x, **kwargs)
    single = onset_strength(x, **kwargs)
    np.testing.assert_allclose(
        _to_numpy(mx.mean(multi, axis=-2)),
        _to_numpy(single),
        rtol=1e-5, atol=1e-5,
    )


def test_onset_strength_multi_non_negative():
    x = _make_test_signal()
    multi = _to_numpy(onset_strength_multi(x, sample_rate=44100, hop_length=441))
    assert np.all(multi >= 0.0)


def test_onset_strength_multi_mean_matches_librosa():
    """Mean of our per-band multi output should match librosa's aggregated onset_strength."""
    librosa = pytest.importorskip("librosa")

    x_np = np.random.default_rng(77).standard_normal(44100 * 2).astype(np.float32)
    x_mx = mx.array(x_np)

    # Our multi returns [n_mels, frames]; mean across bands matches onset_strength
    ours_mean = _to_numpy(mx.mean(
        onset_strength_multi(x_mx, sample_rate=44100, hop_length=441), axis=0
    ))

    ref = librosa.onset.onset_strength(
        y=x_np, sr=44100, hop_length=441, center=True,
    ).astype(np.float32)

    assert ours_mean.shape == ref.shape, f"shape mismatch: {ours_mean.shape} vs {ref.shape}"
    interior = slice(5, -5)
    corr = np.corrcoef(ours_mean[interior], ref[interior])[0, 1]
    assert corr > 0.98, f"correlation too low: {corr:.4f}"


# ── Librosa parity tests ────────────────────────────────────────────────


def test_onset_strength_librosa_parity_default():
    librosa = pytest.importorskip("librosa")

    x_np = np.random.default_rng(42).standard_normal(44100 * 2).astype(np.float32)
    x_mx = mx.array(x_np)

    ours = _to_numpy(onset_strength(x_mx, sample_rate=44100, hop_length=441))

    ref = librosa.onset.onset_strength(
        y=x_np, sr=44100, hop_length=441, center=True,
    ).astype(np.float32)

    assert ours.shape == ref.shape, f"shape mismatch: {ours.shape} vs {ref.shape}"
    # Exclude first/last few frames where reflect vs zero padding diverges
    interior = slice(5, -5)
    np.testing.assert_allclose(ours[interior], ref[interior], rtol=5e-2, atol=0.5)
    # Overall correlation should be very high
    corr = np.corrcoef(ours[interior], ref[interior])[0, 1]
    assert corr > 0.98, f"correlation too low: {corr:.4f}"


def test_onset_strength_librosa_parity_low_band():
    librosa = pytest.importorskip("librosa")

    x_np = np.random.default_rng(123).standard_normal(44100 * 2).astype(np.float32)
    x_mx = mx.array(x_np)

    ours = _to_numpy(
        onset_strength(x_mx, sample_rate=44100, hop_length=441, n_mels=12, f_max=180.0)
    )

    ref = librosa.onset.onset_strength(
        y=x_np, sr=44100, hop_length=441, center=True, n_mels=12, fmax=180,
    ).astype(np.float32)

    assert ours.shape == ref.shape, f"shape mismatch: {ours.shape} vs {ref.shape}"
    interior = slice(5, -5)
    np.testing.assert_allclose(ours[interior], ref[interior], rtol=5e-2, atol=0.5)
    corr = np.corrcoef(ours[interior], ref[interior])[0, 1]
    assert corr > 0.98, f"correlation too low: {corr:.4f}"
