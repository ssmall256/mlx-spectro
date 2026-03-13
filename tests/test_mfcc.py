import mlx.core as mx
import numpy as np
import pytest

from mlx_spectro import MFCCTransform, dct_matrix, mfcc


def _to_numpy(x: mx.array) -> np.ndarray:
    mx.eval(x)
    return np.asarray(x, dtype=np.float32)


def _audio(length: int, *, scale: float = 0.2, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(length).astype(np.float32) * scale)


def test_dct_matrix_matches_torchaudio_create_dct():
    pytest.importorskip("torch")
    ta_f = pytest.importorskip("torchaudio.functional")

    ours = _to_numpy(dct_matrix(20, 128, norm="ortho"))
    ref = ta_f.create_dct(20, 128, "ortho").cpu().numpy().astype(np.float32).T
    np.testing.assert_allclose(ours, ref, rtol=1e-6, atol=1e-6)


def test_dct_matrix_matches_scipy_dct_on_vector():
    scipy_fft = pytest.importorskip("scipy.fft")

    x_np = _audio(128, scale=1.0, seed=7)
    ours = _to_numpy(dct_matrix(20, 128, norm="ortho") @ mx.array(x_np))
    ref = scipy_fft.dct(x_np, type=2, norm="ortho")[:20].astype(np.float32)
    np.testing.assert_allclose(ours, ref, rtol=1e-6, atol=1e-6)


def test_dct_matrix_rejects_invalid_args():
    with pytest.raises(ValueError, match="n_mfcc must be > 0"):
        dct_matrix(0, 128)
    with pytest.raises(ValueError, match="n_mfcc must be <= n_mels"):
        dct_matrix(129, 128)
    with pytest.raises(ValueError, match="norm must be one of"):
        dct_matrix(20, 128, norm="bad")


def test_mfcc_shape_1d_input():
    tr = MFCCTransform(
        sample_rate=24_000,
        n_mfcc=13,
        n_fft=1024,
        hop_length=240,
        n_mels=40,
    )
    y = tr(mx.array(_audio(24_000, seed=11)))
    mx.eval(y)
    assert y.ndim == 2
    assert y.shape[0] == 13


def test_mfcc_shape_batched_input():
    tr = MFCCTransform(
        sample_rate=24_000,
        n_mfcc=16,
        n_fft=1024,
        hop_length=240,
        n_mels=48,
    )
    x = mx.array(np.stack([_audio(24_000, seed=1), _audio(24_000, seed=2)]))
    y = tr(x)
    mx.eval(y)
    assert y.ndim == 3
    assert y.shape[0] == 2
    assert y.shape[1] == 16


def test_mfcc_defaults_produce_expected_coeff_dim():
    y = mfcc(mx.array(_audio(22_050, seed=17)))
    mx.eval(y)
    assert y.ndim == 2
    assert y.shape[0] == 20


def test_mfcc_function_matches_transform():
    kwargs = dict(
        sample_rate=24_000,
        n_mfcc=13,
        n_fft=1024,
        hop_length=240,
        win_length=1024,
        n_mels=40,
        f_max=12_000.0,
        norm="slaney",
        mel_scale="slaney",
        center_pad_mode="reflect",
    )
    x = mx.array(np.stack([_audio(24_000, seed=21), _audio(24_000, seed=22)]))
    direct = mfcc(x, **kwargs)
    cached = MFCCTransform(**kwargs)(x)
    mx.eval(direct, cached)
    np.testing.assert_allclose(_to_numpy(direct), _to_numpy(cached), rtol=1e-6, atol=1e-6)


def test_mfcc_lifter_matches_formula():
    kwargs = dict(
        sample_rate=24_000,
        n_mfcc=13,
        n_fft=1024,
        hop_length=240,
        n_mels=40,
        f_max=12_000.0,
        norm="slaney",
        mel_scale="slaney",
        center_pad_mode="reflect",
    )
    x = mx.array(_audio(24_000, seed=31))
    base = mfcc(x, lifter=0, **kwargs)
    lifted = mfcc(x, lifter=22, **kwargs)
    weights = 1.0 + (22.0 / 2.0) * np.sin(np.pi * np.arange(1, 14, dtype=np.float32) / 22.0)
    np.testing.assert_allclose(
        _to_numpy(lifted),
        _to_numpy(base) * weights[:, None],
        rtol=1e-5,
        atol=1e-5,
    )


def test_mfcc_transform_rejects_invalid_args():
    with pytest.raises(ValueError, match="n_mfcc must be <= n_mels"):
        MFCCTransform(n_mfcc=64, n_mels=40)
    with pytest.raises(ValueError, match="lifter must be >= 0"):
        MFCCTransform(lifter=-1)
    with pytest.raises(ValueError, match="dct_norm must be one of"):
        MFCCTransform(dct_norm="bad")


def test_mfcc_librosa_parity_constant_padding():
    try:
        import librosa
    except Exception as err:
        pytest.skip(f"librosa unavailable: {err}")

    audio_np = _audio(24_000, seed=41)
    kwargs = dict(
        sample_rate=24_000,
        n_mfcc=13,
        n_fft=1024,
        hop_length=240,
        win_length=1024,
        n_mels=40,
        f_min=0.0,
        f_max=12_000.0,
        norm="slaney",
        mel_scale="slaney",
        top_db=80.0,
        center=True,
        center_pad_mode="constant",
        center_tail_pad="symmetric",
        lifter=0,
        dct_norm="ortho",
    )
    ours = _to_numpy(mfcc(mx.array(audio_np), **kwargs))
    ref = librosa.feature.mfcc(
        y=audio_np,
        sr=24_000,
        n_mfcc=13,
        dct_type=2,
        norm="ortho",
        lifter=0,
        n_fft=1024,
        hop_length=240,
        win_length=1024,
        center=True,
        pad_mode="constant",
        power=2.0,
        n_mels=40,
        fmin=0.0,
        fmax=12_000.0,
        htk=False,
        mel_norm="slaney",
    ).astype(np.float32)
    assert ours.shape == ref.shape
    np.testing.assert_allclose(ours, ref, rtol=2e-3, atol=2e-2)


def test_mfcc_torchaudio_parity_single_waveform():
    torch = pytest.importorskip("torch")
    ta_t = pytest.importorskip("torchaudio.transforms")

    audio_np = _audio(24_000, seed=51)
    kwargs = dict(
        sample_rate=24_000,
        n_mfcc=13,
        n_fft=1024,
        hop_length=240,
        win_length=1024,
        n_mels=40,
        f_min=0.0,
        f_max=12_000.0,
        norm="slaney",
        mel_scale="slaney",
        top_db=80.0,
        center=True,
        center_pad_mode="constant",
        center_tail_pad="symmetric",
        lifter=0,
        dct_norm="ortho",
    )
    ours = _to_numpy(mfcc(mx.array(audio_np), **kwargs))
    tr = ta_t.MFCC(
        sample_rate=24_000,
        n_mfcc=13,
        dct_type=2,
        norm="ortho",
        log_mels=False,
        melkwargs={
            "n_fft": 1024,
            "hop_length": 240,
            "win_length": 1024,
            "n_mels": 40,
            "f_min": 0.0,
            "f_max": 12_000.0,
            "center": True,
            "pad_mode": "constant",
            "power": 2.0,
            "norm": "slaney",
            "mel_scale": "slaney",
        },
    )
    ref = tr(torch.from_numpy(audio_np[None, :])).cpu().numpy().astype(np.float32)[0]
    assert ours.shape == ref.shape
    np.testing.assert_allclose(ours, ref, rtol=2e-3, atol=2e-2)


def test_mfcc_transform_compile_smoke():
    tr = MFCCTransform(
        sample_rate=24_000,
        n_mfcc=13,
        n_fft=1024,
        hop_length=240,
        n_mels=40,
    )
    x = mx.array(_audio(24_000, seed=61))

    @mx.compile
    def compiled(inp: mx.array) -> mx.array:
        return tr(inp)

    eager = tr(x)
    compiled_out = compiled(x)
    mx.eval(eager, compiled_out)
    np.testing.assert_allclose(_to_numpy(eager), _to_numpy(compiled_out), rtol=1e-5, atol=1e-5)


def test_mfcc_transform_get_compiled_matches_eager():
    tr = MFCCTransform(
        sample_rate=24_000,
        n_mfcc=13,
        n_fft=1024,
        hop_length=240,
        n_mels=40,
    )
    x = mx.array(_audio(24_000, seed=71))
    compiled = tr.get_compiled()
    eager = tr(x)
    compiled_out = compiled(x)
    mx.eval(eager, compiled_out)
    np.testing.assert_allclose(_to_numpy(eager), _to_numpy(compiled_out), rtol=1e-5, atol=1e-5)


def test_mfcc_transform_get_compiled_is_cached():
    tr = MFCCTransform(
        sample_rate=24_000,
        n_mfcc=13,
        n_fft=1024,
        hop_length=240,
        n_mels=40,
    )
    compiled1 = tr.get_compiled()
    compiled2 = tr.get_compiled()
    assert compiled1 is compiled2
