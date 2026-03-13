import mlx.core as mx
import numpy as np

from mlx_spectro import MelSpectrogramTransform, RepeatedShapeCompileCache


def test_repeated_shape_compile_cache_promotes_after_threshold():
    calls: list[tuple[int, ...]] = []

    def factory(shape: tuple[int, ...]):
        calls.append(shape)
        return lambda x: x

    cache = RepeatedShapeCompileCache(factory, min_hits=2, max_compiled_shapes=4)

    assert cache.get((1, 16000)) is None
    compiled = cache.get((1, 16000))
    assert compiled is not None
    assert calls == [(1, 16000)]
    assert cache.get((1, 16000)) is compiled


def test_repeated_shape_compile_cache_evicts_oldest_compiled_shape():
    calls: list[tuple[int, ...]] = []

    def factory(shape: tuple[int, ...]):
        calls.append(shape)
        return object()

    cache = RepeatedShapeCompileCache(factory, min_hits=1, max_compiled_shapes=2)
    first = cache.get((1, 1000))
    second = cache.get((1, 2000))
    third = cache.get((1, 3000))

    assert first is not None and second is not None and third is not None
    info = cache.cache_info()
    assert info["compiled_shapes"] == 2
    assert info["compiled_shape_keys"] == [(1, 2000), (1, 3000)]


def test_repeated_shape_compile_cache_bounds_pending_shapes():
    cache = RepeatedShapeCompileCache(lambda shape: shape, min_hits=3, max_pending_shapes=2)

    assert cache.get((1, 1000)) is None
    assert cache.get((1, 2000)) is None
    assert cache.get((1, 3000)) is None

    info = cache.cache_info()
    assert info["pending_shapes"] == 2


def test_repeated_shape_compile_cache_with_transform_get_compiled():
    transform = MelSpectrogramTransform(
        sample_rate=16_000,
        n_fft=512,
        hop_length=128,
        n_mels=32,
        center=True,
        center_pad_mode="constant",
    )
    cache = RepeatedShapeCompileCache(lambda shape: transform.get_compiled(), min_hits=2)
    x = mx.array(np.random.default_rng(7).standard_normal((1, 16_000)).astype(np.float32))

    assert cache.get(x.shape) is None
    compiled = cache.get(x.shape)
    eager = transform(x)
    compiled_out = compiled(x)
    mx.eval(eager, compiled_out)

    np.testing.assert_allclose(
        np.asarray(eager, dtype=np.float32),
        np.asarray(compiled_out, dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )


def test_repeated_shape_compile_cache_clear_resets_state():
    cache = RepeatedShapeCompileCache(lambda shape: object(), min_hits=1)
    assert cache.get((1, 1024)) is not None
    cache.clear()
    info = cache.cache_info()
    assert info["pending_shapes"] == 0
    assert info["compiled_shapes"] == 0
