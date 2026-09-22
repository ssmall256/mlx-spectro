#!/usr/bin/env python3
"""Benchmark cached frontend eager vs compiled paths."""

from __future__ import annotations

import argparse
import json
import platform
import time

import mlx.core as mx
import numpy as np

from mlx_spectro import (
    FilteredSpectrogramTransform,
    LogMelSpectrogramTransform,
    MFCCTransform,
    MelSpectrogramTransform,
)


CONFIGS = [
    (1, 16_000, "B=1 T=16k"),
    (4, 160_000, "B=4 T=160k"),
]


def _eval_tree(value) -> None:
    if isinstance(value, dict):
        mx.eval(*value.values())
        return
    if isinstance(value, tuple):
        mx.eval(*value)
        return
    mx.eval(value)


def _bench(fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        _eval_tree(fn())
    times: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = fn()
        _eval_tree(out)
        times.append((time.perf_counter() - t0) * 1e3)
    times.sort()
    return times[len(times) // 2]


def _bench_pair(eager_fn, compiled_fn, *, warmup: int, iters: int) -> tuple[float, float]:
    """Time two arms with the order alternating between rounds.

    Measuring one arm to completion and then the other lets thermal drift,
    allocator state and cache warmth accumulate across the block and land
    entirely on whichever arm ran last. That is not hypothetical: it is what
    produced the 0.56x log-mel figure in the checked-in baseline, a number that
    does not reproduce -- the same configuration measures ~1.1x on two
    different machines once the arms are interleaved.
    """
    for _ in range(warmup):
        _eval_tree(eager_fn())
        _eval_tree(compiled_fn())

    eager_times: list[float] = []
    compiled_times: list[float] = []
    for i in range(iters):
        order = (
            (("eager", eager_fn, eager_times), ("compiled", compiled_fn, compiled_times))
            if i % 2 == 0
            else (("compiled", compiled_fn, compiled_times), ("eager", eager_fn, eager_times))
        )
        for _name, fn, bucket in order:
            t0 = time.perf_counter()
            _eval_tree(fn())
            bucket.append((time.perf_counter() - t0) * 1e3)

    eager_times.sort()
    compiled_times.sort()
    return (
        eager_times[len(eager_times) // 2],
        compiled_times[len(compiled_times) // 2],
    )


def _print_table(title: str, rows: list[dict[str, object]]) -> None:
    print(f"## {title}")
    print("| Config | eager | compiled | speedup |")
    print("|---|---|---|---|")
    for row in rows:
        print(
            f"| {row['label']} | {row['eager']:.2f} ms | "
            f"{row['compiled']:.2f} ms | {row['speedup']:.2f}x |"
        )
    print()


def _simple_filterbank(n_freqs: int, n_bands: int = 32) -> np.ndarray:
    fb = np.zeros((n_freqs, n_bands), dtype=np.float32)
    bins = np.array_split(np.arange(n_freqs), n_bands)
    for idx, band in enumerate(bins):
        fb[band, idx] = 1.0
    return fb


def benchmark_frontends(*, warmup: int, iters: int, emit_markdown: bool = True) -> dict[str, object]:
    if emit_markdown:
        print("## mlx-spectro frontend benchmarks")
        print(f"- warmup={warmup}")
        print(f"- iters={iters}")
        print()

    mel_rows = []
    logmel_rows = []
    mfcc_rows = []
    filtered_rows = []

    fb = _simple_filterbank(1025, 32)

    for batch, length, label in CONFIGS:
        x = mx.random.normal((batch, length), dtype=mx.float32)
        mx.eval(x)

        mel = MelSpectrogramTransform(
            sample_rate=16_000,
            n_fft=2_048,
            hop_length=512,
            win_length=2_048,
            n_mels=128,
            power=2.0,
            norm="slaney",
            mel_scale="slaney",
            top_db=80.0,
            output_scale="db",
            center=True,
            center_pad_mode="constant",
        )
        logmel = LogMelSpectrogramTransform(
            sample_rate=16_000,
            n_fft=2_048,
            hop_length=512,
            win_length=2_048,
            n_mels=128,
            f_min=30.0,
            f_max=8_000.0,
            power=1.0,
            norm="slaney",
            mel_scale="htk",
            center=True,
            center_pad_mode="constant",
            log_amin=1e-5,
            log_mode="clamp",
        )
        mfcc = MFCCTransform(
            sample_rate=16_000,
            n_mfcc=20,
            n_fft=2_048,
            hop_length=512,
            win_length=2_048,
            n_mels=128,
            f_min=0.0,
            f_max=8_000.0,
            norm="slaney",
            mel_scale="slaney",
            top_db=80.0,
            center=True,
            center_pad_mode="constant",
        )
        filtered = FilteredSpectrogramTransform(
            filterbank=fb,
            sample_rate=16_000,
            n_fft=2_048,
            hop_length=512,
            win_length=2_048,
            power=1.0,
            output_scale="log10_plus_one",
            center=True,
            center_pad_mode="constant",
        )

        mel_eager, mel_comp = _bench_pair(
            lambda mel=mel, x=x: mel(x),
            lambda compiled=mel.get_compiled(), x=x: compiled(x),
            warmup=warmup,
            iters=iters,
        )
        mel_rows.append(
            {"label": label, "eager": mel_eager, "compiled": mel_comp, "speedup": mel_eager / mel_comp}
        )

        logmel_eager, logmel_comp = _bench_pair(
            lambda tr=logmel, x=x: tr(x),
            lambda compiled=logmel.get_compiled(), x=x: compiled(x),
            warmup=warmup,
            iters=iters,
        )
        logmel_rows.append(
            {
                "label": label,
                "eager": logmel_eager,
                "compiled": logmel_comp,
                "speedup": logmel_eager / logmel_comp,
            }
        )

        mfcc_eager, mfcc_comp = _bench_pair(
            lambda tr=mfcc, x=x: tr(x),
            lambda compiled=mfcc.get_compiled(), x=x: compiled(x),
            warmup=warmup,
            iters=iters,
        )
        mfcc_rows.append(
            {"label": label, "eager": mfcc_eager, "compiled": mfcc_comp, "speedup": mfcc_eager / mfcc_comp}
        )

        filtered_eager, filtered_comp = _bench_pair(
            lambda tr=filtered, x=x: tr(x),
            lambda compiled=filtered.get_compiled(), x=x: compiled(x),
            warmup=warmup,
            iters=iters,
        )
        filtered_rows.append(
            {
                "label": label,
                "eager": filtered_eager,
                "compiled": filtered_comp,
                "speedup": filtered_eager / filtered_comp,
            }
        )

    if emit_markdown:
        _print_table("MelSpectrogramTransform", mel_rows)
        _print_table("LogMelSpectrogramTransform", logmel_rows)
        _print_table("MFCCTransform", mfcc_rows)
        _print_table("FilteredSpectrogramTransform", filtered_rows)
    return {
        "benchmark": "frontends",
        "meta": {
            "warmup": warmup,
            "iters": iters,
            "quick": warmup <= 2 and iters <= 6,
            "platform": platform.platform(),
            "device": str(mx.default_device()),
        },
        "families": {
            "MelSpectrogramTransform": mel_rows,
            "LogMelSpectrogramTransform": logmel_rows,
            "MFCCTransform": mfcc_rows,
            "FilteredSpectrogramTransform": filtered_rows,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--json-out", type=str, help="write JSON results to a file")
    args = parser.parse_args()
    if args.quick:
        payload = benchmark_frontends(warmup=2, iters=6, emit_markdown=not args.json)
    else:
        payload = benchmark_frontends(warmup=5, iters=20, emit_markdown=not args.json)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")


if __name__ == "__main__":
    main()
