"""A transform built on one thread must work on another.

MLX binds an unevaluated array to the stream of the thread that built it, so
any lazily created window or cache used from a second thread failed with
"There is no Stream(gpu, N) in current thread".
"""
import threading

import mlx.core as mx



def _run_in_thread(fn):
    outcome = {}

    def target():
        try:
            outcome["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            outcome["error"] = exc

    thread = threading.Thread(target=target)
    thread.start()
    thread.join()
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


def _fresh_transform():
    """A transform built on this thread and not yet used anywhere."""
    from mlx_spectro import SpectralTransform

    return SpectralTransform(n_fft=1024, hop_length=256, window_fn="hann", center=True)


def test_eager_stft_istft_on_another_thread():
    x = mx.random.normal((2, 8192), key=mx.random.key(0))
    mx.eval(x)
    reference = _fresh_transform()
    expected = reference.istft(reference.stft(x), length=8192)
    mx.eval(expected)

    transform = _fresh_transform()

    def work():
        y = transform.istft(transform.stft(x), length=8192)
        mx.eval(y)
        return y

    assert mx.array_equal(_run_in_thread(work), expected).item()


def test_compiled_pair_built_on_another_thread():
    x = mx.random.normal((2, 8192), key=mx.random.key(1))
    mx.eval(x)
    transform = _fresh_transform()

    def work():
        stft_fn, istft_fn = transform.compiled_pair(length=8192)
        y = istft_fn(stft_fn(x))
        mx.eval(y)
        return y

    y = _run_in_thread(work)
    assert mx.max(mx.abs(y - x)).item() < 1e-4


def test_built_in_window_is_materialized_but_caller_window_is_not():
    import io

    from mlx_spectro import SpectralTransform

    built_in = SpectralTransform(n_fft=256, hop_length=64, window_fn="hann")
    graph = io.StringIO()
    mx.export_to_dot(graph, built_in.window)
    assert "Cos" not in graph.getvalue() and "Sin" not in graph.getvalue()

    window = mx.sin(mx.arange(256, dtype=mx.float32))
    SpectralTransform(n_fft=256, hop_length=64, window=window)
    graph = io.StringIO()
    mx.export_to_dot(graph, window)
    assert "Sin" in graph.getvalue()


def test_transform_built_inside_compile_does_not_eval():
    from mlx_spectro import SpectralTransform

    @mx.compile
    def build_and_run(x):
        transform = SpectralTransform(n_fft=256, hop_length=64, window_fn="hann", center=True)
        return transform.istft(transform.stft(x), length=x.shape[-1])

    x = mx.random.normal((1, 2048), key=mx.random.key(2))
    y = build_and_run(x)
    mx.eval(y)
    assert y.shape == x.shape
