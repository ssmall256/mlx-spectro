"""Parity of the MLX VQT against librosa.vqt (skipped if librosa is unavailable)."""
import mlx.core as mx
import numpy as np
import pytest

from mlx_spectro import build_vqt_plan, vqt

librosa = pytest.importorskip("librosa")


def _signal(n=16000 * 4, seed=0):
    rng = np.random.default_rng(seed)
    # a couple of tones + noise so all octaves carry energy
    t = np.arange(n) / 16000.0
    y = 0.5 * np.sin(2 * np.pi * 110 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)
    y += 0.05 * rng.standard_normal(n)
    return y.astype(np.float32)


def _filter_construction_matches_librosa():
    from librosa import filters
    from mlx_spectro import librosa_cqt as L
    fmin = librosa.midi_to_hz(21)
    fl = librosa.interval_frequencies(n_bins=352, fmin=fmin, intervals="equal",
                                      bins_per_octave=48, sort=True)
    fm = L._interval_frequencies(352, fmin, 48)
    al = filters._relative_bandwidth(freqs=fl)
    am = L._relative_bandwidth(fm)
    ll, _ = filters.wavelet_lengths(freqs=fl, sr=16000, window="hann",
                                    filter_scale=1, gamma=0, alpha=al)
    lm, _ = L._wavelet_lengths(fm, 16000, 1, 0, am)
    return max(np.abs(fl - fm).max(), np.abs(al - am).max(), np.abs(ll - lm).max())


def test_filter_construction_bit_exact():
    assert _filter_construction_matches_librosa() == 0.0


def test_vqt_matches_librosa():
    soxr = pytest.importorskip("soxr")  # librosa's resampler; ensures soxr_hq path
    y = _signal()
    ref = np.abs(librosa.vqt(y=y, sr=16000, hop_length=320, fmin=librosa.midi_to_hz(21),
                             n_bins=352, bins_per_octave=48, window="hann", gamma=0,
                             tuning=0.0, res_type="soxr_hq"))

    def soxr_rs(sig, o, t):
        out = soxr.resample(np.array(sig).reshape(-1), o, t, quality="HQ")
        return mx.array(out.astype(np.float32))

    plan = build_vqt_plan(sr=16000, hop_length=320, fmin=librosa.midi_to_hz(21),
                          n_bins=352, bins_per_octave=48)
    out = np.abs(np.array(vqt(mx.array(y), plan, resample_fn=soxr_rs)))
    n = min(ref.shape[1], out.shape[1])
    assert np.abs(ref[:, :n] - out[:, :n]).max() < 1e-4
