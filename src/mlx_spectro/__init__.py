"""mlx-spectro: High-performance STFT/iSTFT for Apple MLX."""

__version__ = "0.2.2"

from .spectral_ops import (
    ISTFTBackendPolicy,
    STFTOutputLayout,
    SpectralTransform,
    WindowLike,
    get_cache_debug_stats,
    get_transform_mlx,
    make_window,
    reset_cache_debug_stats,
    resolve_fft_params,
    spec_mlx_device_key,
)

__all__ = [
    "SpectralTransform",
    "ISTFTBackendPolicy",
    "STFTOutputLayout",
    "WindowLike",
    "make_window",
    "resolve_fft_params",
    "get_transform_mlx",
    "spec_mlx_device_key",
    "get_cache_debug_stats",
    "reset_cache_debug_stats",
]
