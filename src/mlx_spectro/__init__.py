"""mlx-spectro: High-performance STFT/iSTFT for Apple MLX."""

__version__ = "0.3.0rc1"

from .spectral_ops import (
    ISTFTBackendPolicy,
    MelMode,
    MelNorm,
    MelScale,
    MelSpectrogramTransform,
    STFTOutputLayout,
    SpectralTransform,
    WindowLike,
    amplitude_to_db,
    get_cache_debug_stats,
    get_transform_mlx,
    make_window,
    melscale_fbanks,
    onset_strength,
    onset_strength_multi,
    reset_cache_debug_stats,
    resolve_fft_params,
    spec_mlx_device_key,
)

__all__ = [
    "SpectralTransform",
    "MelSpectrogramTransform",
    "ISTFTBackendPolicy",
    "STFTOutputLayout",
    "MelScale",
    "MelNorm",
    "MelMode",
    "WindowLike",
    "make_window",
    "melscale_fbanks",
    "amplitude_to_db",
    "onset_strength",
    "onset_strength_multi",
    "resolve_fft_params",
    "get_transform_mlx",
    "spec_mlx_device_key",
    "get_cache_debug_stats",
    "reset_cache_debug_stats",
]
