import mlx.core as mx
import numpy as np
import pytest

from mlx_spectro import (
    chroma_stft,
    mfcc,
    rms,
    spectral_bandwidth,
    spectral_centroid,
    spectral_contrast,
    SpectralFeatureTransform,
    spectral_features,
    spectral_rolloff,
    zero_crossing_rate,
)


def _to_numpy(x: mx.array) -> np.ndarray:
    mx.eval(x)
    return np.asarray(x, dtype=np.float32)


def _import_librosa():
    try:
        import librosa
    except Exception as err:
        pytest.skip(f"librosa unavailable: {err}")
    return librosa


def _audio(sr: int = 22_050, duration: float = 1.0, *, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(sr * duration)
    t = np.linspace(0.0, duration, n, endpoint=False, dtype=np.float32)
    tone = 0.2 * np.sin(2.0 * np.pi * 220.0 * t)
    chirp = 0.1 * np.sin(2.0 * np.pi * (220.0 + 660.0 * t) * t)
    noise = 0.01 * rng.standard_normal(n).astype(np.float32)
    return (tone + chirp + noise).astype(np.float32)


def test_feature_shapes_1d():
    x = mx.array(_audio(seed=1))
    assert chroma_stft(x).shape[0] == 12
    assert spectral_centroid(x).shape[0] == 1
    assert spectral_bandwidth(x).shape[0] == 1
    assert spectral_rolloff(x).shape[0] == 1
    assert spectral_contrast(x).shape[0] == 7
    assert rms(x).shape[0] == 1
    assert zero_crossing_rate(x).shape[0] == 1


def test_feature_shapes_batched():
    x_np = _audio(seed=2)
    batch = mx.array(np.stack([x_np, x_np]))
    assert chroma_stft(batch).shape[:2] == (2, 12)
    assert spectral_centroid(batch).shape[:2] == (2, 1)
    assert spectral_bandwidth(batch).shape[:2] == (2, 1)
    assert spectral_rolloff(batch).shape[:2] == (2, 1)
    assert spectral_contrast(batch).shape[:2] == (2, 7)
    assert rms(batch).shape[:2] == (2, 1)
    assert zero_crossing_rate(batch).shape[:2] == (2, 1)


def test_default_outputs_are_finite():
    x = mx.array(_audio(seed=3))
    for fn in (
        chroma_stft,
        spectral_centroid,
        spectral_bandwidth,
        spectral_rolloff,
        spectral_contrast,
        rms,
        zero_crossing_rate,
    ):
        out = _to_numpy(fn(x))
        assert np.isfinite(out).all()
        assert out.shape[-1] > 0


def test_spectral_features_bundle_keys_and_shapes():
    x = mx.array(_audio(seed=4))
    out = spectral_features(
        x,
        include=("spectral_centroid", "mfcc", "chroma_stft"),
        n_fft=1024,
        hop_length=256,
    )
    assert list(out.keys()) == ["spectral_centroid", "mfcc", "chroma_stft"]
    assert out["spectral_centroid"].shape[0] == 1
    assert out["mfcc"].shape[0] == 20
    assert out["chroma_stft"].shape[0] == 12


def test_spectral_features_bundle_matches_standalone_functions():
    x = mx.array(_audio(sr=24_000, seed=5))
    kwargs = {
        "sample_rate": 24_000,
        "n_fft": 1024,
        "hop_length": 240,
        "win_length": 1024,
        "window_fn": "hann",
        "center": True,
        "center_pad_mode": "constant",
        "center_tail_pad": "symmetric",
        "n_chroma": 12,
        "chroma_norm": 2,
        "tuning": 0.0,
        "bandwidth_p": 2.0,
        "roll_percent": 0.85,
        "n_bands": 6,
        "contrast_fmin": 200.0,
        "contrast_quantile": 0.02,
        "n_mfcc": 20,
        "n_mels": 128,
        "f_min": 0.0,
        "f_max": None,
        "mel_norm": "slaney",
        "mel_scale": "slaney",
        "top_db": 80.0,
        "lifter": 0,
        "dct_norm": "ortho",
    }
    bundle = spectral_features(x, include=None, **kwargs)
    np.testing.assert_allclose(
        _to_numpy(bundle["spectral_centroid"]),
        _to_numpy(
            spectral_centroid(
                x,
                sample_rate=kwargs["sample_rate"],
                n_fft=kwargs["n_fft"],
                hop_length=kwargs["hop_length"],
                win_length=kwargs["win_length"],
                window_fn=kwargs["window_fn"],
                center=kwargs["center"],
                center_pad_mode=kwargs["center_pad_mode"],
                center_tail_pad=kwargs["center_tail_pad"],
            )
        ),
        atol=1e-6,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        _to_numpy(bundle["spectral_bandwidth"]),
        _to_numpy(
            spectral_bandwidth(
                x,
                sample_rate=kwargs["sample_rate"],
                n_fft=kwargs["n_fft"],
                hop_length=kwargs["hop_length"],
                win_length=kwargs["win_length"],
                p=kwargs["bandwidth_p"],
                window_fn=kwargs["window_fn"],
                center=kwargs["center"],
                center_pad_mode=kwargs["center_pad_mode"],
                center_tail_pad=kwargs["center_tail_pad"],
            )
        ),
        atol=1e-6,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        _to_numpy(bundle["spectral_rolloff"]),
        _to_numpy(
            spectral_rolloff(
                x,
                sample_rate=kwargs["sample_rate"],
                n_fft=kwargs["n_fft"],
                hop_length=kwargs["hop_length"],
                win_length=kwargs["win_length"],
                roll_percent=kwargs["roll_percent"],
                window_fn=kwargs["window_fn"],
                center=kwargs["center"],
                center_pad_mode=kwargs["center_pad_mode"],
                center_tail_pad=kwargs["center_tail_pad"],
            )
        ),
        atol=1e-6,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        _to_numpy(bundle["spectral_contrast"]),
        _to_numpy(
            spectral_contrast(
                x,
                sample_rate=kwargs["sample_rate"],
                n_fft=kwargs["n_fft"],
                hop_length=kwargs["hop_length"],
                win_length=kwargs["win_length"],
                n_bands=kwargs["n_bands"],
                fmin=kwargs["contrast_fmin"],
                quantile=kwargs["contrast_quantile"],
                window_fn=kwargs["window_fn"],
                center=kwargs["center"],
                center_pad_mode=kwargs["center_pad_mode"],
                center_tail_pad=kwargs["center_tail_pad"],
            )
        ),
        atol=1e-6,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        _to_numpy(bundle["chroma_stft"]),
        _to_numpy(
            chroma_stft(
                x,
                sample_rate=kwargs["sample_rate"],
                n_fft=kwargs["n_fft"],
                hop_length=kwargs["hop_length"],
                win_length=kwargs["win_length"],
                n_chroma=kwargs["n_chroma"],
                norm=kwargs["chroma_norm"],
                window_fn=kwargs["window_fn"],
                center=kwargs["center"],
                center_pad_mode=kwargs["center_pad_mode"],
                center_tail_pad=kwargs["center_tail_pad"],
                tuning=kwargs["tuning"],
            )
        ),
        atol=1e-6,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        _to_numpy(bundle["mfcc"]),
        _to_numpy(
            mfcc(
                x,
                sample_rate=kwargs["sample_rate"],
                n_mfcc=kwargs["n_mfcc"],
                n_fft=kwargs["n_fft"],
                hop_length=kwargs["hop_length"],
                win_length=kwargs["win_length"],
                n_mels=kwargs["n_mels"],
                f_min=kwargs["f_min"],
                f_max=kwargs["f_max"],
                norm=kwargs["mel_norm"],
                mel_scale=kwargs["mel_scale"],
                top_db=kwargs["top_db"],
                window_fn=kwargs["window_fn"],
                center=kwargs["center"],
                center_pad_mode=kwargs["center_pad_mode"],
                center_tail_pad=kwargs["center_tail_pad"],
                lifter=kwargs["lifter"],
                dct_norm=kwargs["dct_norm"],
            )
        ),
        atol=1e-5,
        rtol=1e-5,
    )


def test_spectral_features_invalid_name_raises():
    x = mx.array(_audio(seed=6))
    with pytest.raises(ValueError, match="Unknown spectral feature"):
        spectral_features(x, include=("spectral_centroid", "not_real"))


def test_spectral_feature_transform_matches_function_bundle():
    x = mx.array(_audio(sr=24_000, seed=7))
    kwargs = {
        "include": ("spectral_centroid", "spectral_bandwidth", "chroma_stft", "mfcc"),
        "sample_rate": 24_000,
        "n_fft": 1024,
        "hop_length": 240,
        "win_length": 1024,
        "window_fn": "hann",
        "center": True,
        "center_pad_mode": "constant",
        "center_tail_pad": "symmetric",
        "n_chroma": 12,
        "chroma_norm": 2,
        "tuning": 0.0,
        "bandwidth_p": 2.0,
        "n_mfcc": 20,
        "n_mels": 64,
        "f_min": 0.0,
        "f_max": 12_000.0,
        "mel_norm": "slaney",
        "mel_scale": "slaney",
        "top_db": 80.0,
        "lifter": 0,
        "dct_norm": "ortho",
    }
    direct = spectral_features(x, **kwargs)
    cached = SpectralFeatureTransform(**kwargs)(x)
    for key in kwargs["include"]:
        np.testing.assert_allclose(
            _to_numpy(cached[key]),
            _to_numpy(direct[key]),
            atol=1e-5,
            rtol=1e-5,
        )


def test_spectral_feature_transform_reuses_cached_state_across_instances():
    tr1 = SpectralFeatureTransform(include=("chroma_stft", "mfcc"))
    tr2 = SpectralFeatureTransform(include=("chroma_stft", "mfcc"))
    assert tr1.spectral is tr2.spectral
    assert tr1.chroma_fb is tr2.chroma_fb
    assert tr1.mel_fb is tr2.mel_fb
    assert tr1.dct_mat is tr2.dct_mat


def test_spectral_feature_transform_compile_smoke():
    tr = SpectralFeatureTransform(include=("spectral_centroid", "mfcc"))
    x = mx.array(_audio(seed=8))

    @mx.compile
    def compiled(inp: mx.array):
        out = tr(inp)
        return out["spectral_centroid"], out["mfcc"]

    eager = tr(x)
    compiled_centroid, compiled_mfcc = compiled(x)
    mx.eval(
        eager["spectral_centroid"],
        eager["mfcc"],
        compiled_centroid,
        compiled_mfcc,
    )
    np.testing.assert_allclose(
        _to_numpy(eager["spectral_centroid"]),
        _to_numpy(compiled_centroid),
        atol=1e-5,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        _to_numpy(eager["mfcc"]),
        _to_numpy(compiled_mfcc),
        atol=1e-5,
        rtol=1e-5,
    )


def test_rms_zero_signal():
    x = mx.zeros((4096,), dtype=mx.float32)
    out = _to_numpy(rms(x, center=False))
    np.testing.assert_allclose(out, 0.0, atol=1e-7)


def test_zero_crossing_rate_zero_signal():
    x = mx.zeros((4096,), dtype=mx.float32)
    out = _to_numpy(zero_crossing_rate(x, center=False))
    np.testing.assert_allclose(out, 0.0, atol=1e-7)


def test_zero_crossing_rate_alternating_signal():
    x_np = np.tile(np.array([1.0, -1.0], dtype=np.float32), 1024)
    out = _to_numpy(
        zero_crossing_rate(mx.array(x_np), frame_length=256, hop_length=256, center=False)
    )
    assert float(out.max()) > 0.95


def test_short_audio_returns_single_frame():
    x = mx.array(np.array([0.1, -0.2, 0.3], dtype=np.float32))
    assert rms(x, frame_length=16, hop_length=8).shape == (1, 1)
    assert zero_crossing_rate(x, frame_length=16, hop_length=8).shape == (1, 1)
    assert spectral_centroid(x, n_fft=16, hop_length=8).shape[1] == 1


def test_spectral_features_silence_are_finite():
    x = mx.zeros((4096,), dtype=mx.float32)
    for fn in (spectral_centroid, spectral_bandwidth, spectral_rolloff, spectral_contrast):
        out = _to_numpy(fn(x, center=False))
        assert np.isfinite(out).all()


def test_chroma_stft_matches_librosa_reference():
    librosa = _import_librosa()
    audio_np = _audio(sr=24_000, seed=11)
    ours = _to_numpy(
        chroma_stft(
            mx.array(audio_np),
            sample_rate=24_000,
            n_fft=1024,
            hop_length=240,
            win_length=1024,
            n_chroma=12,
            norm=2,
            center=True,
            center_pad_mode="constant",
            center_tail_pad="symmetric",
            tuning=0.0,
        )
    )
    ref = librosa.feature.chroma_stft(
        y=audio_np,
        sr=24_000,
        n_fft=1024,
        hop_length=240,
        win_length=1024,
        n_chroma=12,
        norm=2,
        center=True,
        pad_mode="constant",
        tuning=0.0,
    ).astype(np.float32)
    assert ours.shape == ref.shape
    np.testing.assert_allclose(ours, ref, rtol=5e-3, atol=5e-2)


def test_spectral_centroid_matches_librosa_reference():
    librosa = _import_librosa()
    audio_np = _audio(sr=24_000, seed=21)
    ours = _to_numpy(
        spectral_centroid(
            mx.array(audio_np),
            sample_rate=24_000,
            n_fft=1024,
            hop_length=240,
            win_length=1024,
            center=True,
            center_pad_mode="constant",
            center_tail_pad="symmetric",
        )
    )
    ref = librosa.feature.spectral_centroid(
        y=audio_np,
        sr=24_000,
        n_fft=1024,
        hop_length=240,
        win_length=1024,
        center=True,
        pad_mode="constant",
    ).astype(np.float32)
    np.testing.assert_allclose(ours, ref, rtol=5e-3, atol=5e-2)


def test_spectral_bandwidth_matches_librosa_reference():
    librosa = _import_librosa()
    audio_np = _audio(sr=24_000, seed=31)
    ours = _to_numpy(
        spectral_bandwidth(
            mx.array(audio_np),
            sample_rate=24_000,
            n_fft=1024,
            hop_length=240,
            win_length=1024,
            p=2.0,
            center=True,
            center_pad_mode="constant",
            center_tail_pad="symmetric",
        )
    )
    ref = librosa.feature.spectral_bandwidth(
        y=audio_np,
        sr=24_000,
        n_fft=1024,
        hop_length=240,
        win_length=1024,
        p=2.0,
        center=True,
        pad_mode="constant",
    ).astype(np.float32)
    np.testing.assert_allclose(ours, ref, rtol=5e-3, atol=5e-2)


def test_spectral_rolloff_matches_librosa_reference():
    librosa = _import_librosa()
    audio_np = _audio(sr=24_000, seed=41)
    ours = _to_numpy(
        spectral_rolloff(
            mx.array(audio_np),
            sample_rate=24_000,
            n_fft=1024,
            hop_length=240,
            win_length=1024,
            roll_percent=0.85,
            center=True,
            center_pad_mode="constant",
            center_tail_pad="symmetric",
        )
    )
    ref = librosa.feature.spectral_rolloff(
        y=audio_np,
        sr=24_000,
        n_fft=1024,
        hop_length=240,
        win_length=1024,
        roll_percent=0.85,
        center=True,
        pad_mode="constant",
    ).astype(np.float32)
    np.testing.assert_allclose(ours, ref, rtol=5e-3, atol=5e-2)


def test_spectral_contrast_matches_librosa_reference():
    librosa = _import_librosa()
    audio_np = _audio(sr=24_000, seed=51)
    ours = _to_numpy(
        spectral_contrast(
            mx.array(audio_np),
            sample_rate=24_000,
            n_fft=2048,
            hop_length=240,
            win_length=2048,
            n_bands=6,
            fmin=200.0,
            quantile=0.02,
            center=True,
            center_pad_mode="constant",
            center_tail_pad="symmetric",
        )
    )
    ref = librosa.feature.spectral_contrast(
        y=audio_np,
        sr=24_000,
        n_fft=2048,
        hop_length=240,
        win_length=2048,
        n_bands=6,
        fmin=200.0,
        quantile=0.02,
        center=True,
        pad_mode="constant",
        linear=False,
    ).astype(np.float32)
    assert ours.shape == ref.shape
    np.testing.assert_allclose(ours, ref, rtol=1e-2, atol=1e-1)


def test_rms_matches_librosa_reference():
    librosa = _import_librosa()
    audio_np = _audio(sr=24_000, seed=61)
    ours = _to_numpy(
        rms(
            mx.array(audio_np),
            frame_length=1024,
            hop_length=240,
            center=True,
            pad_mode="constant",
        )
    )
    ref = librosa.feature.rms(
        y=audio_np,
        frame_length=1024,
        hop_length=240,
        center=True,
        pad_mode="constant",
    ).astype(np.float32)
    np.testing.assert_allclose(ours, ref, rtol=1e-5, atol=1e-6)


def test_zero_crossing_rate_matches_librosa_reference():
    librosa = _import_librosa()
    audio_np = _audio(sr=24_000, seed=71)
    ours = _to_numpy(
        zero_crossing_rate(
            mx.array(audio_np),
            frame_length=1024,
            hop_length=240,
            center=True,
            pad_mode="edge",
        )
    )
    ref = librosa.feature.zero_crossing_rate(
        audio_np,
        frame_length=1024,
        hop_length=240,
        center=True,
    ).astype(np.float32)
    np.testing.assert_allclose(ours, ref, rtol=1e-5, atol=1e-6)
