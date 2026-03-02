import mlx.core as mx
import numpy as np
import pytest

from mlx_spectro import MelSpectrogramTransform, amplitude_to_db, melscale_fbanks


def _to_numpy(x: mx.array) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def test_melscale_fbanks_matches_torchaudio_reference():
    torch = pytest.importorskip("torch")
    ta_f = pytest.importorskip("torchaudio.functional")

    ours = _to_numpy(
        melscale_fbanks(
            n_freqs=1025,
            f_min=0.0,
            f_max=12_000.0,
            n_mels=128,
            sample_rate=24_000,
            norm=None,
            mel_scale="htk",
        )
    )
    ref = (
        ta_f.melscale_fbanks(
            n_freqs=1025,
            f_min=0.0,
            f_max=12_000.0,
            n_mels=128,
            sample_rate=24_000,
            norm=None,
            mel_scale="htk",
        )
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    np.testing.assert_allclose(ours, ref, rtol=1e-5, atol=2e-5)


def test_melscale_fbanks_slaney_matches_torchaudio_reference():
    pytest.importorskip("torch")
    ta_f = pytest.importorskip("torchaudio.functional")

    ours = _to_numpy(
        melscale_fbanks(
            n_freqs=1025,
            f_min=0.0,
            f_max=12_000.0,
            n_mels=128,
            sample_rate=24_000,
            norm="slaney",
            mel_scale="slaney",
        )
    )
    ref = (
        ta_f.melscale_fbanks(
            n_freqs=1025,
            f_min=0.0,
            f_max=12_000.0,
            n_mels=128,
            sample_rate=24_000,
            norm="slaney",
            mel_scale="slaney",
        )
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    np.testing.assert_allclose(ours, ref, rtol=1e-5, atol=2e-5)


def test_amplitude_to_db_matches_torchaudio_reference():
    torch = pytest.importorskip("torch")
    ta_f = pytest.importorskip("torchaudio.functional")

    rng = np.random.default_rng(7)
    x_np = np.abs(rng.standard_normal((2, 3, 17, 29), dtype=np.float32)) + 1e-5
    ours = _to_numpy(amplitude_to_db(mx.array(x_np), stype="power", top_db=80.0))
    ref = (
        ta_f.amplitude_to_DB(
            torch.from_numpy(x_np),
            multiplier=10.0,
            amin=1e-10,
            db_multiplier=np.log10(1.0),
            top_db=80.0,
        )
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    np.testing.assert_allclose(ours, ref, rtol=1e-5, atol=1e-5)


def test_amplitude_to_db_topdb_none_matches_torchaudio_reference():
    torch = pytest.importorskip("torch")
    ta_f = pytest.importorskip("torchaudio.functional")

    rng = np.random.default_rng(23)
    x_np = np.abs(rng.standard_normal((2, 3, 17, 29), dtype=np.float32)) + 1e-6
    ours = _to_numpy(amplitude_to_db(mx.array(x_np), stype="power", top_db=None))
    ref = (
        ta_f.amplitude_to_DB(
            torch.from_numpy(x_np),
            multiplier=10.0,
            amin=1e-10,
            db_multiplier=np.log10(1.0),
            top_db=None,
        )
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    np.testing.assert_allclose(ours, ref, rtol=1e-5, atol=1e-5)


def test_amplitude_to_db_per_example_mode_differs_from_torchaudio_compat():
    x_np = np.array(
        [
            [[1.0, 1e-1, 1e-2], [1e-1, 1e-2, 1e-3]],
            [[1e-4, 1e-5, 1e-6], [1e-5, 1e-6, 1e-7]],
        ],
        dtype=np.float32,
    )
    x = mx.array(x_np)
    compat = _to_numpy(
        amplitude_to_db(x, stype="power", top_db=20.0, mode="torchaudio_compat")
    )
    per_example = _to_numpy(
        amplitude_to_db(x, stype="power", top_db=20.0, mode="per_example")
    )

    assert not np.allclose(compat, per_example)
    # Quiet example should keep a lower floor under per-example clipping.
    assert float(per_example[1].min()) < float(compat[1].min()) - 10.0


def test_mel_spectrogram_torchaudio_compat_matches_reference():
    torch = pytest.importorskip("torch")
    ta_t = pytest.importorskip("torchaudio.transforms")

    rng = np.random.default_rng(11)
    audio_np = rng.standard_normal((2, 24_000), dtype=np.float32) * 0.2

    ours_tr = MelSpectrogramTransform(
        sample_rate=24_000,
        n_fft=2_048,
        hop_length=240,
        n_mels=128,
        power=2.0,
        norm=None,
        mel_scale="htk",
        top_db=80.0,
        mode="torchaudio_compat",
        center=True,
        periodic=True,
        normalized=False,
    )
    ours = _to_numpy(ours_tr(mx.array(audio_np), to_db=True))

    mel = ta_t.MelSpectrogram(
        sample_rate=24_000,
        n_fft=2_048,
        hop_length=240,
        win_length=2_048,
        n_mels=128,
        power=2.0,
        normalized=False,
        center=True,
        norm=None,
        mel_scale="htk",
    )
    amp = ta_t.AmplitudeToDB(stype="power", top_db=80.0)
    ref = amp(mel(torch.from_numpy(audio_np))).cpu().numpy().astype(np.float32)

    assert ours.shape == ref.shape
    # STFT kernels are different implementations; keep a tight but practical tolerance.
    np.testing.assert_allclose(ours, ref, rtol=2e-3, atol=2e-2)


def test_mel_mode_default_alias_maps_to_mlx_native():
    tr = MelSpectrogramTransform(mode="default")
    assert tr.mode == "mlx_native"
