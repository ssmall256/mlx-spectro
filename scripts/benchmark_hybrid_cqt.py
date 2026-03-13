"""Benchmark hybrid CQT construction and repeated-call latency.

Usage:
    python scripts/benchmark_hybrid_cqt.py
    python scripts/benchmark_hybrid_cqt.py --quick
"""

from __future__ import annotations

import argparse
import json
import platform
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


def benchmark_hybrid_cqt(*, warmup: int, iters: int, emit_markdown: bool = True) -> dict[str, object]:
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

    if emit_markdown:
        print("## mlx-spectro hybrid CQT benchmark")
        print(f"Device: {mx.default_device()}")
        print()

    rows: list[dict[str, object]] = []
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

        rows.append(
            {
                "batch": batch,
                "samples": length,
                "cold_cached_transform": cold_ms,
                "wrapper_one_off": wrapper_ms,
                "cached_transform": cached_ms,
                "compiled_transform": compiled_ms,
            }
        )

        if emit_markdown:
            print(f"batch={batch} samples={length}")
            print(f"  cold cached transform: {cold_ms:7.3f} ms")
            print(f"  wrapper one-off:       {wrapper_ms:7.3f} ms")
            print(f"  cached transform:      {cached_ms:7.3f} ms")
            print(f"  compiled transform:    {compiled_ms:7.3f} ms")
            print()

    return {
        "benchmark": "hybrid_cqt",
        "meta": {
            "warmup": warmup,
            "iters": iters,
            "quick": warmup <= 2 and iters <= 5,
            "platform": platform.platform(),
            "device": str(mx.default_device()),
        },
        "cases": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="mlx-spectro hybrid CQT benchmarks")
    parser.add_argument("--quick", action="store_true", help="use fewer warmup/iteration steps")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--json-out", type=str, help="write JSON results to a file")
    args = parser.parse_args()

    warmup = 2 if args.quick else 5
    iters = 5 if args.quick else 20
    payload = benchmark_hybrid_cqt(warmup=warmup, iters=iters, emit_markdown=not args.json)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")


if __name__ == "__main__":
    main()
