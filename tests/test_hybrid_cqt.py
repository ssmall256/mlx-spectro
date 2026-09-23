import mlx.core as mx
import numpy as np

from mlx_spectro import HybridCQTTransform, hybrid_cqt, nnaudio_cqt_kernels
from tests.hybrid_cqt_snapshots import HYBRID_CQT_SNAPSHOTS


def _to_numpy(x: mx.array) -> np.ndarray:
    mx.eval(x)
    return np.asarray(x, dtype=np.float32)


def _audio(length: int = 24_000, *, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (0.2 * rng.standard_normal(length)).astype(np.float32)


# The snapshots were recorded on one machine, and the transform is float32:
# a different GPU reassociates the reductions and lands a few ULPs away. An
# absolute-only tolerance is therefore wrong on the aggregates, whose magnitude
# is set by the input length -- `sum` here is ~10^3, so atol=1e-6 asked for
# agreement three orders of magnitude tighter than float32 can express, and CI
# failed on relative differences of 3e-7 (about 3 ULPs). Relative tolerance with
# an absolute floor for values near zero. A real change to the transform moves
# these far more than 1e-5.
_RTOL = 1e-5
_ATOL = 1e-6


def _assert_snapshot(name: str) -> None:
    snapshot = HYBRID_CQT_SNAPSHOTS[name]
    transform = HybridCQTTransform(**snapshot["kwargs"])
    audio = mx.array(
        _audio(length=snapshot["audio_length"], seed=snapshot["audio_seed"])
    )
    out = _to_numpy(transform(audio))
    assert out.shape == snapshot["shape"]
    check = lambda actual, expected: np.testing.assert_allclose(  # noqa: E731
        actual, expected, rtol=_RTOL, atol=_ATOL
    )
    check(out.sum(dtype=np.float64), snapshot["sum"])
    check(out.mean(dtype=np.float64), snapshot["mean"])
    check(out.std(dtype=np.float64), snapshot["std"])
    check(out.max(), snapshot["max"])
    check(out[:3, :6], np.asarray(snapshot["first_block"], dtype=np.float32))
    middle_row, middle_col = snapshot["middle_offset"]
    check(
        out[middle_row : middle_row + 3, middle_col : middle_col + 6],
        np.asarray(snapshot["middle_block"], dtype=np.float32),
    )
    last_row, last_col = snapshot["last_offset"]
    check(
        out[last_row : last_row + 3, last_col : last_col + 6],
        np.asarray(snapshot["last_block"], dtype=np.float32),
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


# ---------------------------------------------------------------------------
# nnaudio CQT kernel generation
# ---------------------------------------------------------------------------


def test_nnaudio_cqt_kernels_shapes():
    kr, ki, sl, lp = nnaudio_cqt_kernels(
        sr=22050, fmin=27.5, n_bins=309, bins_per_octave=36,
    )
    assert kr.shape == (36, 256)
    assert ki.shape == (36, 256)
    assert sl.shape == (309,)
    assert lp.shape == (256,)
    assert kr.dtype == np.float32
    assert ki.dtype == np.float32
    assert sl.dtype == np.float32
    assert lp.dtype == np.float32


def test_nnaudio_cqt_kernels_l1_normalized():
    kr, ki, _, _ = nnaudio_cqt_kernels(
        sr=22050, fmin=27.5, n_bins=309, bins_per_octave=36,
    )
    for b in range(kr.shape[0]):
        mag = np.sqrt(kr[b] ** 2 + ki[b] ** 2)
        np.testing.assert_allclose(mag.sum(), 1.0, atol=1e-6)


def test_nnaudio_cqt_kernels_hann_window():
    """Kernel envelopes should correlate highly with periodic Hann windows."""
    kr, ki, _, _ = nnaudio_cqt_kernels(
        sr=22050, fmin=27.5, n_bins=309, bins_per_octave=36,
    )
    # Check bin 0 (longest support)
    mag = np.sqrt(kr[0] ** 2 + ki[0] ** 2)
    nz = np.where(mag > 1e-6)[0]
    envelope = mag[nz]
    n = np.arange(len(nz), dtype=np.float64)
    hann = 0.5 - 0.5 * np.cos(2 * np.pi * n / len(nz))
    corr = np.corrcoef(envelope, hann)[0, 1]
    assert corr > 0.999, f"Hann correlation too low: {corr}"


def test_nnaudio_cqt_kernels_lowpass_symmetric():
    """Lowpass filter should be symmetric (linear phase)."""
    _, _, _, lp = nnaudio_cqt_kernels()
    np.testing.assert_allclose(lp, lp[::-1], atol=1e-7)


def test_nnaudio_cqt_kernels_custom_params():
    """Different parameters should produce different shaped outputs."""
    kr, ki, sl, lp = nnaudio_cqt_kernels(
        sr=16000, fmin=55.0, n_bins=84, bins_per_octave=12,
    )
    assert kr.shape[0] == 12  # bins_per_octave filters
    assert sl.shape == (84,)
