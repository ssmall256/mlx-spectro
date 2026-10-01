"""Tests for waveform_overlap_add and waveform_chunk_overlap_add in mlx-spectro."""
import numpy as np
import mlx.core as mx
import pytest

from mlx_spectro import waveform_overlap_add, waveform_chunk_overlap_add


def numpy_reference_ola(
    frames: np.ndarray,
    step: int,
    total_samples: int,
    window: np.ndarray | None = None,
    *,
    normalized: bool = True,
) -> np.ndarray:
    orig_shape = frames.shape
    num_chunks = orig_shape[0]
    prefix_shape = orig_shape[1:-1]
    chunk_len = orig_shape[-1]

    num_channels = int(np.prod(prefix_shape)) if prefix_shape else 1
    frames_flat = frames.reshape(num_chunks, num_channels, chunk_len).astype(np.float64)
    if window is None:
        window = np.ones(chunk_len, dtype=np.float64)
    else:
        window = window.astype(np.float64)

    out = np.zeros((num_channels, total_samples), dtype=np.float64)
    sum_weight = np.zeros(total_samples, dtype=np.float64) if normalized else None

    for k in range(num_chunks):
        off = k * step
        this_len = min(chunk_len, total_samples - off)
        if this_len <= 0:
            continue
        end = off + this_len
        w = window[:this_len]
        out[:, off:end] += frames_flat[k, :, :this_len] * w[None, :]
        if normalized:
            sum_weight[off:end] += w

    if normalized:
        mask = sum_weight > 1e-11
        out[:, mask] /= sum_weight[None, mask]
        out[:, ~mask] = 0.0

    return out.reshape(*prefix_shape, total_samples)


@pytest.mark.parametrize("shape", [
    (5, 1024),                # (num_chunks, chunk_len)
    (8, 2, 2048),             # (num_chunks, channels, chunk_len)
    (4, 4, 2, 4096),          # (num_chunks, stems, channels, chunk_len)
])
@pytest.mark.parametrize("overlap_ratio", [0.0, 0.25, 0.5, 0.75])
@pytest.mark.parametrize("normalized", [True, False])
def test_waveform_overlap_add_parity(shape, overlap_ratio, normalized):
    rng = np.random.default_rng(12345)
    frames_np = rng.standard_normal(shape).astype(np.float32)
    chunk_len = shape[-1]
    step = max(1, int(chunk_len * (1.0 - overlap_ratio)))
    num_chunks = shape[0]
    total_samples = (num_chunks - 1) * step + chunk_len

    # Hann window
    window_np = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(chunk_len) / chunk_len)).astype(np.float32)

    frames_mx = mx.array(frames_np)
    window_mx = mx.array(window_np)

    ref = numpy_reference_ola(frames_np, step, total_samples, window_np, normalized=normalized)
    out_mx = waveform_overlap_add(frames_mx, step, total_samples, window_mx, normalized=normalized)
    mx.eval(out_mx)

    out_np = np.asarray(out_mx)
    max_err = float(np.max(np.abs(ref - out_np)))
    ref_norm = float(np.max(np.abs(ref)))
    rel_err = max_err / (ref_norm + 1e-8)

    assert rel_err < 1e-5, f"Relative error {rel_err} exceeded tolerance"
    assert max_err < 1e-4, f"Max error {max_err} exceeded tolerance"


def test_waveform_overlap_add_default_window_and_total_samples():
    rng = np.random.default_rng(42)
    frames_np = rng.standard_normal((6, 2, 1000)).astype(np.float32)
    step = 500
    frames_mx = mx.array(frames_np)

    # Inferred total_samples = (6-1)*500 + 1000 = 3500
    out_mx = waveform_overlap_add(frames_mx, step=step, window=None, normalized=True)
    mx.eval(out_mx)
    assert out_mx.shape == (2, 3500)

    ref = numpy_reference_ola(frames_np, step, 3500, window=None, normalized=True)
    assert np.allclose(ref, np.asarray(out_mx), atol=1e-5)


def test_waveform_overlap_add_alias():
    frames = mx.ones((3, 2, 100))
    a = waveform_overlap_add(frames, step=50)
    b = waveform_chunk_overlap_add(frames, step=50)
    mx.eval(a, b)
    assert mx.allclose(a, b)


def test_waveform_overlap_add_edge_cases():
    # Empty frames
    empty = mx.zeros((0, 2, 100))
    res = waveform_overlap_add(empty, step=50, total_samples=0)
    mx.eval(res)
    assert res.shape == (2, 0)

    # Single chunk
    single = mx.ones((1, 2, 100))
    res_single = waveform_overlap_add(single, step=50, total_samples=100)
    mx.eval(res_single)
    assert res_single.shape == (2, 100)
    assert float(mx.max(mx.abs(res_single - 1.0))) < 1e-6
