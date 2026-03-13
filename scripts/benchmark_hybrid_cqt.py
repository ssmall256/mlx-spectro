"""Benchmark hybrid CQT construction and repeated-call latency.

Usage:
    python scripts/benchmark_hybrid_cqt.py
    python scripts/benchmark_hybrid_cqt.py --quick
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx
import numpy as np

from mlx_spectro import HybridCQTTransform, hybrid_cqt


def _bench(fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    mx.synchronize()

    times: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        mx.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return times[len(times) // 2]


def _audio(length: int, *, batch: int, seed: int = 0) -> mx.array:
    rng = np.random.default_rng(seed)
    data = (0.2 * rng.standard_normal((batch, length))).astype(np.float32)
    if batch == 1:
        return mx.array(data[0])
    return mx.array(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="mlx-spectro hybrid CQT benchmarks")
    parser.add_argument("--quick", action="store_true", help="use fewer warmup/iteration steps")
    args = parser.parse_args()

    warmup = 2 if args.quick else 5
    iters = 5 if args.quick else 20
    kwargs = dict(
        sr=22_050,
        hop_length=512,
        fmin=23.12465141947715,
        n_bins=288,
        bins_per_octave=36,
        filter_scale=1.0,
        norm=1.0,
        sparsity=0.01,
    )

    print("## mlx-spectro hybrid CQT benchmark")
    print(f"Device: {mx.default_device()}")
    print()

    for batch, length in [(1, 22_050 * 5), (1, 22_050 * 15), (4, 22_050 * 5)]:
        x = _audio(length, batch=batch, seed=batch + length)
        transform = HybridCQTTransform(**kwargs)
        compiled = transform.get_compiled()

        def wrapper_call():
            out = hybrid_cqt(x, **kwargs)
            mx.eval(out)

        def cached_call():
            out = transform(x)
            mx.eval(out)

        def compiled_call():
            out = compiled(x)
            mx.eval(out)

        t0 = time.perf_counter()
        cold = transform(x)
        mx.eval(cold)
        cold_ms = (time.perf_counter() - t0) * 1000.0
        wrapper_ms = _bench(wrapper_call, warmup=warmup, iters=iters)
        cached_ms = _bench(cached_call, warmup=warmup, iters=iters)
        compiled_ms = _bench(compiled_call, warmup=warmup, iters=iters)

        print(f"batch={batch} samples={length}")
        print(f"  cold cached transform: {cold_ms:7.3f} ms")
        print(f"  wrapper one-off:       {wrapper_ms:7.3f} ms")
        print(f"  cached transform:      {cached_ms:7.3f} ms")
        print(f"  compiled transform:    {compiled_ms:7.3f} ms")
        print()


if __name__ == "__main__":
    main()
