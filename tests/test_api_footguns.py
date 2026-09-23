"""Guards for three ways the 0.8.0 API surface could silently mislead a caller.

`mel_filterbank_librosa` shipped with no test at all, `fft_frequencies` takes
librosa's arguments in the opposite order and silently returned a wrong-length
array when called the librosa way, and `vqt` raised an unrelated matmul error
for batched input.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mlx_spectro import (
    build_vqt_plan,
    fft_frequencies,
    mel_filterbank,
    mel_filterbank_librosa,
    vqt,
)

librosa = pytest.importorskip("librosa")


class TestMelFilterbankLibrosaParity:
    @pytest.mark.parametrize("htk", [False, True])
    @pytest.mark.parametrize(
        "sr, n_fft, n_mels", [(22050, 1024, 40), (44100, 2048, 128), (16000, 512, 80)]
    )
    def test_matches_librosa(self, sr, n_fft, n_mels, htk):
        ours = np.asarray(mel_filterbank_librosa(sr, n_fft, n_mels, htk=htk)).T
        ref = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, htk=htk)
        np.testing.assert_allclose(ours, ref, rtol=1e-6, atol=1e-9)

    def test_defaults_match_librosa_defaults(self):
        """The whole point of the name: passing nothing on either side agrees.

        It defaulted to htk=True while librosa defaults to htk=False, so the
        two disagreed exactly when a caller was least likely to check.
        """
        ours = np.asarray(mel_filterbank_librosa(22050, 1024, 40)).T
        ref = librosa.filters.mel(sr=22050, n_fft=1024, n_mels=40)
        np.testing.assert_allclose(ours, ref, rtol=1e-6, atol=1e-9)

    def test_it_is_not_interchangeable_with_mel_filterbank(self):
        """Documented, and pinned so the docs cannot quietly become wrong."""
        slaney = np.asarray(mel_filterbank_librosa(22050, 1024, 40))
        madmom = np.asarray(
            mel_filterbank(fft_frequencies(513, 22050), 40, fmin=0.0, fmax=11025.0)
        )
        assert madmom.max() / slaney.max() > 10.0


class TestFftFrequenciesArgumentOrder:
    @pytest.mark.parametrize(
        "swapped", [(22050, 1024), (44100, 2048), (48000, 4096), (16000, 512)]
    )
    def test_librosa_argument_order_is_rejected(self, swapped):
        """Previously returned a wrong-length array with no error at all."""
        with pytest.raises(ValueError, match="look swapped"):
            fft_frequencies(*swapped)

    @pytest.mark.parametrize(
        "num_fft_bins, sample_rate",
        [(513, 22050), (1025, 44100), (2049, 48000), (129, 16000), (512, 44100)],
    )
    def test_correct_order_is_unaffected(self, num_fft_bins, sample_rate):
        got = fft_frequencies(num_fft_bins, sample_rate)
        assert got.shape == (num_fft_bins,)

    def test_values_match_librosa_when_called_correctly(self):
        n_fft = 1024
        ours = fft_frequencies(n_fft // 2 + 1, 22050)
        ref = librosa.fft_frequencies(sr=22050, n_fft=n_fft)
        np.testing.assert_allclose(ours, ref, rtol=1e-6, atol=1e-4)


class TestVqtErrors:
    def test_batched_input_is_refused_with_a_useful_message(self):
        plan = build_vqt_plan(sr=22050, hop_length=256, n_bins=84, bins_per_octave=12)
        signal = mx.array(
            np.random.default_rng(0).standard_normal(22050).astype(np.float32)
        )
        with pytest.raises(ValueError, match="1-D waveform"):
            vqt(signal[None], plan)

    def test_unsupported_parameters_name_the_alternative(self):
        with pytest.raises(NotImplementedError, match="hybrid_cqt"):
            build_vqt_plan(
                sr=44100, hop_length=512, n_bins=84, bins_per_octave=12
            )
