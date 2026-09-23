"""Batched-gradient regression tests for the iSTFT VJP.

`_unpad_cotangent` is the adjoint of `_trim_ola_output` and used to place the
cotangent with `grad_ola.at[:, a:b].add(...)` -- a strided slice scatter-add on
a non-leading axis. MLX before 0.32.0 mis-linearizes the 2-D dispatch grid in
its Metal `slice_update_op_impl` kernel, so that aliased rows onto each other in
a non-atomic read-modify-write and silently produced wrong gradients for any
batch size greater than 1.

The existing suite missed it precisely: `test_istft_numerical_grad` does a real
finite-difference check but only at B=1 (the one case that worked), while
`test_roundtrip_grad_large_batch` uses B=4 but asserts only shape and
finiteness, which a garbage gradient satisfies.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mlx_spectro import SpectralTransform
from mlx_spectro.spectral_ops import _place_rows, _unpad_cotangent


def _numpy_unpad(cotangent, center, n_fft, out_len, length_int, B):
    """Straightforward numpy reference for the adjoint placement."""
    c = np.asarray(cotangent)
    out = np.zeros((B, out_len), dtype=c.dtype)
    if center:
        pad = n_fft // 2
        if length_int is not None:
            w = min(length_int, out_len - pad, c.shape[1])
        else:
            w = min(out_len - 2 * pad, c.shape[1])
        if w > 0:
            out[:, pad:pad + w] = c[:, :w]
        return out
    if length_int is not None:
        w = min(length_int, out_len, c.shape[1])
        if w > 0:
            out[:, :w] = c[:, :w]
        return out
    if c.shape[1] < out_len:
        out[:, :c.shape[1]] = c
        return out
    return c


@pytest.mark.parametrize("B", [1, 2, 4, 8])
@pytest.mark.parametrize("center", [True, False])
@pytest.mark.parametrize("with_length", [True, False])
def test_unpad_cotangent_matches_numpy(B, center, with_length):
    """The adjoint must place the cotangent exactly, at every batch size."""
    n_fft, out_len = 512, 5504
    length_int = out_len - n_fft if with_length else None
    mx.random.seed(0)
    cot = mx.random.normal((B, out_len - n_fft))
    mx.eval(cot)

    got = np.asarray(_unpad_cotangent(cot, center, n_fft, out_len, length_int, B))
    ref = _numpy_unpad(cot, center, n_fft, out_len, length_int, B)

    assert got.shape == ref.shape
    assert np.max(np.abs(got - ref)) == 0.0, (
        f"B={B} center={center} length={with_length}: "
        f"max abs error {np.max(np.abs(got - ref)):.3e}"
    )


@pytest.mark.parametrize("B", [1, 2, 4])
def test_place_rows_is_exact(B):
    mx.random.seed(1)
    vals = mx.random.normal((B, 300))
    out = np.asarray(_place_rows(vals, 64, 250, 1000, B))
    ref = np.zeros((B, 1000), dtype=np.float32)
    ref[:, 64:64 + 250] = np.asarray(vals)[:, :250]
    assert np.max(np.abs(out - ref)) == 0.0


def test_place_rows_zero_width_returns_zeros():
    out = np.asarray(_place_rows(mx.ones((2, 8)), 4, 0, 32, 2))
    assert out.shape == (2, 32)
    assert np.count_nonzero(out) == 0


@pytest.mark.parametrize("B", [1, 2, 4])
def test_istft_gradient_matches_finite_differences_batched(B):
    """Full finite-difference check at batch > 1 -- the missing coverage.

    Kept small so the dense FD sweep stays cheap.
    """
    n_fft, hop, length = 64, 16, 256
    t = SpectralTransform(n_fft, hop)
    mx.random.seed(3)
    x = mx.random.normal((B, length))
    mx.eval(x)
    spec = t.stft(x, output_layout="bnf")
    mx.eval(spec)
    spec_real, spec_imag = mx.real(spec), mx.imag(spec)
    mx.eval(spec_real, spec_imag)

    def loss(sr):
        z = sr + 1j * mx.stop_gradient(spec_imag)
        return t.differentiable_istft(z, length=length).square().sum()

    analytic = np.array(mx.grad(loss)(spec_real))

    eps = 1e-3
    sr_np = np.array(spec_real)
    numerical = np.zeros_like(sr_np)
    for b in range(sr_np.shape[0]):
        for n in range(sr_np.shape[1]):
            for f in range(sr_np.shape[2]):
                plus = sr_np.copy()
                plus[b, n, f] += eps
                minus = sr_np.copy()
                minus[b, n, f] -= eps
                numerical[b, n, f] = (
                    float(loss(mx.array(plus)).item())
                    - float(loss(mx.array(minus)).item())
                ) / (2 * eps)

    a, num = analytic.flatten(), numerical.flatten()
    corr = np.corrcoef(a, num)[0, 1]
    cos = np.dot(a, num) / (np.linalg.norm(a) * np.linalg.norm(num) + 1e-12)
    assert corr > 0.99, f"B={B}: gradient direction wrong, correlation={corr:.6f}"
    assert cos > 0.99, f"B={B}: gradient magnitude wrong, cosine={cos:.6f}"


def test_batched_gradient_is_per_sample_independent():
    """A cheap, dense invariant that a corrupted adjoint cannot satisfy.

    Row b of the gradient must depend only on row b of the input, so stacking
    independent examples must give the same gradients as running them alone.
    Row aliasing -- the exact failure mode of the MLX slice-add bug -- breaks
    this immediately.
    """
    n_fft, hop, length, B = 128, 32, 512, 4
    t = SpectralTransform(n_fft, hop)
    mx.random.seed(5)
    x = mx.random.normal((B, length))
    mx.eval(x)
    spec = t.stft(x, output_layout="bnf")
    mx.eval(spec)

    def loss(z):
        return t.differentiable_istft(z, length=length).square().sum()

    batched = np.array(mx.grad(loss)(spec))
    for b in range(B):
        single = np.array(mx.grad(loss)(spec[b:b + 1]))
        assert np.allclose(batched[b], single[0], rtol=1e-4, atol=1e-5), (
            f"row {b} of the batched gradient differs from computing it alone; "
            "the batch dimension is leaking"
        )
