"""`sr` and `sample_rate` must be interchangeable on the CQT/VQT entry points.

Those functions mirror librosa, which spells the argument `sr`, while every
transform class in this package spells it `sample_rate`. Both spellings are
accepted so callers do not have to remember which side of that line a given
function sits on.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mlx_spectro import HybridCQTTransform, build_vqt_plan, hybrid_cqt


@pytest.fixture
def signal():
    return mx.array(
        np.random.default_rng(0).standard_normal(22050).astype(np.float32)
    )


def _cqt_kwargs(**over):
    kwargs = {"hop_length": 256, "n_bins": 84, "bins_per_octave": 12}
    kwargs.update(over)
    return kwargs


def test_hybrid_cqt_accepts_either_spelling(signal):
    via_sr = np.asarray(hybrid_cqt(signal, **_cqt_kwargs(sr=22050)))
    via_sample_rate = np.asarray(hybrid_cqt(signal, **_cqt_kwargs(sample_rate=22050)))
    assert np.array_equal(via_sr, via_sample_rate)


def test_hybrid_cqt_transform_accepts_either_spelling(signal):
    via_sr = np.asarray(HybridCQTTransform(**_cqt_kwargs(sr=22050))(signal))
    via_sample_rate = np.asarray(
        HybridCQTTransform(**_cqt_kwargs(sample_rate=22050))(signal)
    )
    assert np.array_equal(via_sr, via_sample_rate)


def test_build_vqt_plan_accepts_either_spelling():
    via_sr = build_vqt_plan(sr=22050, hop_length=256, n_bins=84, bins_per_octave=12)
    via_sample_rate = build_vqt_plan(
        sample_rate=22050, hop_length=256, n_bins=84, bins_per_octave=12
    )
    assert via_sr.sr == via_sample_rate.sr == 22050


@pytest.mark.parametrize(
    "call",
    [
        lambda: build_vqt_plan(sr=16000, sample_rate=22050),
        lambda: HybridCQTTransform(sr=16000, sample_rate=22050),
    ],
)
def test_conflicting_sample_rates_are_rejected(call):
    """Silently preferring one would be worse than refusing."""
    with pytest.raises(TypeError, match="conflicting sample rates"):
        call()


def test_defaults_are_unchanged_when_neither_is_passed(signal):
    assert build_vqt_plan().sr == 16000
    assert HybridCQTTransform().sr == 22050
    assert np.asarray(hybrid_cqt(signal)).shape[0] == 84
