import mlx.core as mx
import numpy as np

from mlx_spectro import HybridCQTTransform, hybrid_cqt
from tests.hybrid_cqt_snapshots import HYBRID_CQT_SNAPSHOTS


def _to_numpy(x: mx.array) -> np.ndarray:
    mx.eval(x)
    return np.asarray(x, dtype=np.float32)


def _audio(length: int = 24_000, *, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (0.2 * rng.standard_normal(length)).astype(np.float32)


def _assert_snapshot(name: str) -> None:
    snapshot = HYBRID_CQT_SNAPSHOTS[name]
    transform = HybridCQTTransform(**snapshot["kwargs"])
    audio = mx.array(
        _audio(length=snapshot["audio_length"], seed=snapshot["audio_seed"])
    )
    out = _to_numpy(transform(audio))
    assert out.shape == snapshot["shape"]
    np.testing.assert_allclose(out.sum(dtype=np.float64), snapshot["sum"], atol=1e-6)
    np.testing.assert_allclose(out.mean(dtype=np.float64), snapshot["mean"], atol=1e-7)
    np.testing.assert_allclose(out.std(dtype=np.float64), snapshot["std"], atol=1e-7)
    np.testing.assert_allclose(out.max(), snapshot["max"], atol=1e-6)
    np.testing.assert_allclose(
        out[:3, :6], np.asarray(snapshot["first_block"], dtype=np.float32), atol=1e-6
    )
    middle_row, middle_col = snapshot["middle_offset"]
    np.testing.assert_allclose(
        out[middle_row : middle_row + 3, middle_col : middle_col + 6],
        np.asarray(snapshot["middle_block"], dtype=np.float32),
        atol=1e-6,
    )
    last_row, last_col = snapshot["last_offset"]
    np.testing.assert_allclose(
        out[last_row : last_row + 3, last_col : last_col + 6],
        np.asarray(snapshot["last_block"], dtype=np.float32),
        atol=1e-6,
    )


def test_hybrid_cqt_shape_1d():
    x = mx.array(_audio(seed=1))
    transform = HybridCQTTransform(
        sr=22_050,
        hop_length=256,
        fmin=32.70319566257483,
        n_bins=96,
        bins_per_octave=24,
    )
    out = transform(x)
    mx.eval(out)
    assert out.ndim == 2
    assert out.shape[0] == 96


def test_hybrid_cqt_shape_batched():
    x_np = _audio(seed=2)
    x = mx.array(np.stack([x_np, x_np * 0.5], axis=0))
    transform = HybridCQTTransform(
        sr=22_050,
        hop_length=256,
        fmin=32.70319566257483,
        n_bins=96,
        bins_per_octave=24,
    )
    out = transform(x)
    mx.eval(out)
    assert out.ndim == 3
    assert out.shape[:2] == (2, 96)


def test_hybrid_cqt_wrapper_matches_transform():
    x = mx.array(_audio(seed=3))
    kwargs = dict(
        sr=22_050,
        hop_length=256,
        fmin=32.70319566257483,
        n_bins=96,
        bins_per_octave=24,
        filter_scale=1.0,
        norm=1.0,
        sparsity=0.01,
    )
    transform = HybridCQTTransform(**kwargs)
    got = _to_numpy(transform(x))
    ref = _to_numpy(hybrid_cqt(x, **kwargs))
    np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-6)


def test_hybrid_cqt_small_snapshot():
    _assert_snapshot("small")


def test_hybrid_cqt_lvcr_snapshot():
    _assert_snapshot("lvcr")


def test_hybrid_cqt_silence_is_finite():
    x = mx.zeros((16_000,), dtype=mx.float32)
    out = hybrid_cqt(
        x,
        sr=22_050,
        hop_length=256,
        fmin=32.70319566257483,
        n_bins=84,
        bins_per_octave=12,
    )
    out_np = _to_numpy(out)
    assert np.all(np.isfinite(out_np))
    assert np.all(out_np >= 0.0)


def test_hybrid_cqt_compile_smoke():
    x = mx.array(_audio(seed=5))
    transform = HybridCQTTransform(
        sr=22_050,
        hop_length=256,
        fmin=32.70319566257483,
        n_bins=84,
        bins_per_octave=12,
    )
    compiled = transform.get_compiled()
    eager = _to_numpy(transform(x))
    compiled_out = _to_numpy(compiled(x))
    np.testing.assert_allclose(eager, compiled_out, rtol=1e-5, atol=1e-5)
