"""The STFT layout defaults, and the sentinel that schedules their convergence.

Two halves of this API disagree: `stft`/`istft` and the `get_compiled_*` /
`*_compiled` methods default to "bfn" (batch, freq, frames), while
`compiled_pair`/`compiled_pair_nd` default to "bnf" (batch, frames, freq).
1.0 converges both on "bnf", the native rFFT order.

Until this file existed the suite pinned *neither* default -- a change to
either would have shipped green, and so would a regression. These tests pin
the current resolution of all six entry points, so the 1.0 change has to be
deliberate.
"""

from __future__ import annotations

import warnings

import mlx.core as mx
import pytest

from mlx_spectro import SpectralTransform
from mlx_spectro.spectral_ops import (
    _STFT_LAYOUT_CONVERGENCE_TARGET,
    _resolve_stft_output_layout,
)

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
    raise AssertionError(f"neither axis is {BINS} bins: {spec.shape}")


def _quiet(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return fn()


class TestDefaultsAreUnchanged:
    """The sentinel must not move any entry point's behaviour."""

    def test_eager_stft_default_is_bfn(self, transform, signal):
        assert _layout_of(_quiet(lambda: transform.stft(signal))) == "bfn"

    def test_stft_compiled_default_is_bfn(self, transform, signal):
        assert _layout_of(_quiet(lambda: transform.stft_compiled(signal))) == "bfn"

    def test_get_compiled_stft_default_is_bfn(self, transform, signal):
        fn = _quiet(transform.get_compiled_stft)
        assert _layout_of(fn(signal)) == "bfn"

    def test_compiled_pair_default_is_bnf(self, transform, signal):
        stft_fn, _ = _quiet(lambda: transform.compiled_pair(length=LENGTH))
        assert _layout_of(stft_fn(signal)) == "bnf"

    def test_compiled_pair_nd_default_is_bnf(self, transform, signal):
        stft_fn, _ = _quiet(
            lambda: transform.compiled_pair_nd(length=LENGTH, leading_shape=(2,))
        )
        assert _layout_of(stft_fn(signal)) == "bnf"

    def test_istft_default_accepts_bfn(self, transform, signal):
        """istft's default must match what the default stft produces."""
        spec = _quiet(lambda: transform.stft(signal))
        out = _quiet(lambda: transform.istft(spec, length=LENGTH))
        mx.eval(out)
        assert out.shape == signal.shape

    def test_the_two_halves_still_disagree(self, transform, signal):
        """Documents the split this deprecation exists to resolve."""
        eager = _layout_of(_quiet(lambda: transform.stft(signal)))
        stft_fn, _ = _quiet(lambda: transform.compiled_pair(length=LENGTH))
        compiled = _layout_of(stft_fn(signal))
        assert eager != compiled, (
            "the eager and compiled defaults have converged; if that was "
            "intentional this deprecation is finished and can be removed"
        )


class TestSentinel:
    def test_auto_resolves_to_the_entry_points_default(self):
        assert _resolve_stft_output_layout("auto", default_layout="bfn") == "bfn"
        assert _resolve_stft_output_layout("auto", default_layout="bnf") == "bnf"

    def test_none_behaves_like_auto(self):
        assert _resolve_stft_output_layout(None, default_layout="bnf") == "bnf"

    def test_explicit_layout_overrides_the_default(self):
        assert _resolve_stft_output_layout("bnf", default_layout="bfn") == "bnf"

    def test_unknown_layout_still_raises(self):
        with pytest.raises(ValueError, match="output_layout must be one of"):
            _resolve_stft_output_layout("nonsense")

    def test_convergence_target_is_the_native_layout(self):
        assert _STFT_LAYOUT_CONVERGENCE_TARGET == "bnf"


class TestWarning:
    def test_relying_on_the_default_warns(self, transform, signal):
        with pytest.warns(FutureWarning, match="relied on the default STFT layout"):
            transform.stft(signal)

    def test_it_is_a_futurewarning_not_a_deprecationwarning(self, transform, signal):
        """DeprecationWarning is hidden outside __main__, so it would not reach
        the application authors who have to act on this."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            transform.stft(signal)
        layout = [w for w in caught if "STFT layout" in str(w.message)]
        assert layout, "no layout warning was emitted"
        assert all(w.category is FutureWarning for w in layout)

    @pytest.mark.parametrize("layout", ["bfn", "bnf"])
    def test_passing_a_layout_is_silent(self, transform, signal, layout):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            transform.stft(signal, output_layout=layout)
        assert [w for w in caught if "STFT layout" in str(w.message)] == []

    def test_the_message_names_the_entry_point_and_the_resolution(
        self, transform, signal
    ):
        with pytest.warns(FutureWarning) as rec:
            transform.stft(signal)
        message = str(rec[0].message)
        assert "stft()" in message
        assert "'bfn'" in message, "must say what it resolved to today"
        assert "1.0" in message, "must say when it changes"

    def test_the_warning_blames_the_caller_not_the_library(self, transform, signal):
        """A warning pointing inside mlx_spectro is useless for migration."""
        with pytest.warns(FutureWarning) as rec:
            transform.stft(signal)
        assert not rec[0].filename.endswith("spectral_ops.py"), (
            f"warning attributed to {rec[0].filename}; stacklevel is wrong"
        )
