"""STFT/iSTFT must not import NumPy; NumPy-based features still work."""
import subprocess
import sys

import pytest

_PROBE = """
import sys
import mlx.core as mx
import mlx_spectro
from mlx_spectro import SpectralTransform
t = SpectralTransform(n_fft=1024, hop_length=256, window_fn="hann", center=True)
x = mx.random.normal((2, 8192))
mx.eval(t.istft(t.stft(x), length=8192))
stft_fn, istft_fn = t.compiled_pair(length=8192)
mx.eval(istft_fn(stft_fn(x)))
print("numpy" in sys.modules)
"""


def test_stft_path_does_not_import_numpy():
    result = subprocess.run([sys.executable, "-c", _PROBE], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "False", result.stdout + result.stderr


def test_numpy_features_still_resolve_lazily():
    np = pytest.importorskip("numpy")
    from mlx_spectro import spectral_ops

    spec = np.full((4, 3), 9.0, dtype=np.float32)
    out = spectral_ops.logarithmic_spectrogram(spec)
    assert np.allclose(out, np.log10(np.float32(10.0)))
    assert spectral_ops.np is np
