"""Every STFT entry point hands back the same layout.

`bfn` -- `[B, F, N]`, frames last -- is the layout this library standardises
on. It suits MLX's channel-last grain, and it is what every consumer of this
package actually consumes: models here are built frequency-major, so a `bnf`
result would only be transposed on arrival.

`compiled_pair`/`compiled_pair_nd` used to default to `bnf` while the six
other entry points defaulted to `bfn`, so two halves of one API handed back
different axis orders. Every real caller of the compiled half overrode it back
to `bfn`, which is the clearest evidence available that the split was a bug
rather than a choice.

Nothing pinned any of these defaults before this file existed: a change to one,
or a regression, would have shipped green.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from mlx_spectro import SpectralTransform
from mlx_spectro.spectral_ops import _resolve_stft_output_layout

N_FFT, HOP, LENGTH = 512, 128, 8192
BINS = N_FFT // 2 + 1


@pytest.fixture
def transform():
    return SpectralTransform(N_FFT, HOP, window_fn="hann")


@pytest.fixture
def signal():
    x = mx.random.normal((2, LENGTH))
    mx.eval(x)
    return x


def _layout_of(spec):
    """bfn puts frequency on axis 1; bnf puts it last."""
    if spec.shape[1] == BINS:
        return "bfn"
    if spec.shape[2] == BINS:
        return "bnf"
    raise AssertionError(f"neither axis holds {BINS} bins: {spec.shape}")


class TestEveryEntryPointDefaultsToBfn:
    def test_eager_stft(self, transform, signal):
        assert _layout_of(transform.stft(signal)) == "bfn"

    def test_stft_compiled(self, transform, signal):
        assert _layout_of(transform.stft_compiled(signal)) == "bfn"

    def test_get_compiled_stft(self, transform, signal):
        assert _layout_of(transform.get_compiled_stft()(signal)) == "bfn"

    def test_compiled_pair(self, transform, signal):
        stft_fn, _ = transform.compiled_pair(length=LENGTH)
        assert _layout_of(stft_fn(signal)) == "bfn"

    def test_compiled_pair_nd(self, transform, signal):
        stft_fn, _ = transform.compiled_pair_nd(length=LENGTH, leading_shape=(2,))
        assert _layout_of(stft_fn(signal)) == "bfn"

    def test_istft_accepts_what_stft_produces(self, transform, signal):
        out = transform.istft(transform.stft(signal), length=LENGTH)
        mx.eval(out)
        assert out.shape == signal.shape

    def test_the_eager_and_compiled_halves_agree(self, transform, signal):
        """The split this convergence removed. If these ever diverge again,
        callers get different axis orders from one API depending on which
        method they reached for."""
        stft_fn, _ = transform.compiled_pair(length=LENGTH)
        assert _layout_of(transform.stft(signal)) == _layout_of(stft_fn(signal))


class TestCompiledMatchesEager:
    """AGENTS.md: compiled callables must produce identical results to eager."""

    def test_default_compiled_pair_matches_default_eager(self, transform, signal):
        import numpy as np

        stft_fn, istft_fn = transform.compiled_pair(length=LENGTH)
        zc, ze = stft_fn(signal), transform.stft(signal)
        mx.eval(zc, ze)
        assert zc.shape == ze.shape
        assert np.max(np.abs(np.array(zc) - np.array(ze))) < 1e-4

        yc = istft_fn(zc)
        ye = transform.istft(ze, length=LENGTH)
        mx.eval(yc, ye)
        assert np.max(np.abs(np.array(yc) - np.array(ye))) < 1e-4


class TestResolver:
    def test_none_falls_back_to_the_default(self):
        assert _resolve_stft_output_layout(None) == "bfn"

    @pytest.mark.parametrize("layout", ["bfn", "bnf"])
    def test_explicit_layout_is_honoured(self, transform, signal, layout):
        assert _layout_of(transform.stft(signal, output_layout=layout)) == layout

    def test_unknown_layout_raises(self):
        with pytest.raises(ValueError, match="output_layout must be one of"):
            _resolve_stft_output_layout("nonsense")

    def test_bnf_is_still_reachable(self, transform, signal):
        """Converging the default must not remove the option."""
        stft_fn, istft_fn = transform.compiled_pair(length=LENGTH, layout="bnf")
        z = stft_fn(signal)
        assert _layout_of(z) == "bnf"
        out = istft_fn(z)
        mx.eval(out)
        assert out.shape == signal.shape
