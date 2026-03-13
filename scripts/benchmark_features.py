#!/usr/bin/env python3
"""Benchmark spectral feature extraction paths.

Focuses on STFT-derived descriptors and compares:
- standalone public feature calls
- sequential multi-feature extraction
- shared-STFT bundle extraction

Usage:
    python scripts/benchmark_features.py
    python scripts/benchmark_features.py --quick
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from collections.abc import Iterable

import mlx.core as mx

from mlx_spectro import (
    chroma_stft,
    mfcc,
    SpectralFeatureTransform,
    spectral_bandwidth,
    spectral_centroid,
    spectral_contrast,
    spectral_features,
    spectral_rolloff,
)
from mlx_spectro.spectral_ops import _spectral_feature_values

FEATURE_NAMES = (
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_rolloff",
    "spectral_contrast",
    "chroma_stft",
    "mfcc",
)

FEATURE_FNS = {
    "spectral_centroid": spectral_centroid,
    "spectral_bandwidth": spectral_bandwidth,
    "spectral_rolloff": spectral_rolloff,
    "spectral_contrast": spectral_contrast,
    "chroma_stft": chroma_stft,
    "mfcc": mfcc,
}

CONFIGS = [
    (1, 16_000, 512, "B=1 T=16k nfft=512"),
    (4, 160_000, 1024, "B=4 T=160k nfft=1024"),
    (8, 480_000, 1024, "B=8 T=480k nfft=1024"),
]


def _eval_tree(value) -> None:
    if isinstance(value, dict):
        mx.eval(*value.values())
        return
    if isinstance(value, tuple):
        mx.eval(*value)
        return
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        mx.eval(*tuple(value))
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


def _shared_kwargs(n_fft: int, hop_length: int) -> dict:
    return {
        "sample_rate": 22_050,
        "n_fft": n_fft,
        "hop_length": hop_length,
        "win_length": n_fft,
        "window_fn": "hann",
        "center": True,
        "center_pad_mode": "reflect",
        "center_tail_pad": "symmetric",
        "n_chroma": 12,
        "chroma_norm": 2,
        "tuning": 0.0,
        "bandwidth_p": 2.0,
        "roll_percent": 0.85,
        "n_bands": 6,
        "contrast_fmin": 200.0,
        "contrast_quantile": 0.02,
        "n_mfcc": 20,
        "n_mels": 128,
        "f_min": 0.0,
        "f_max": None,
        "mel_norm": "slaney",
        "mel_scale": "slaney",
        "top_db": 80.0,
        "lifter": 0,
        "dct_norm": "ortho",
    }


def _single_feature_kwargs(name: str, shared: dict) -> dict:
    base = {
        "sample_rate": shared["sample_rate"],
        "n_fft": shared["n_fft"],
        "hop_length": shared["hop_length"],
        "win_length": shared["win_length"],
        "window_fn": shared["window_fn"],
        "center": shared["center"],
        "center_pad_mode": shared["center_pad_mode"],
        "center_tail_pad": shared["center_tail_pad"],
    }
    if name == "chroma_stft":
        base["n_chroma"] = shared["n_chroma"]
        base["norm"] = shared["chroma_norm"]
        base["tuning"] = shared["tuning"]
    elif name == "spectral_bandwidth":
        base["p"] = shared["bandwidth_p"]
    elif name == "spectral_rolloff":
        base["roll_percent"] = shared["roll_percent"]
    elif name == "spectral_contrast":
        base["n_bands"] = shared["n_bands"]
        base["fmin"] = shared["contrast_fmin"]
        base["quantile"] = shared["contrast_quantile"]
    elif name == "mfcc":
        base["n_mfcc"] = shared["n_mfcc"]
        base["n_mels"] = shared["n_mels"]
        base["f_min"] = shared["f_min"]
        base["f_max"] = shared["f_max"]
        base["norm"] = shared["mel_norm"]
        base["mel_scale"] = shared["mel_scale"]
        base["top_db"] = shared["top_db"]
        base["lifter"] = shared["lifter"]
        base["dct_norm"] = shared["dct_norm"]
    return base


def _print_table(title: str, cols: list[str], rows: list[dict]) -> None:
    print(f"\n## {title}")
    print("| Config | " + " | ".join(cols) + " |")
    print("|---" * (len(cols) + 1) + "|")
    for row in rows:
        values = [row["label"]]
        for col in cols:
            value = row.get(col, "")
            if isinstance(value, (int, float)):
                if "speedup" in col or "gain" in col:
                    values.append(f"{value:.2f}x")
                else:
                    values.append(f"{value:.2f} ms")
            else:
                values.append(str(value))
        print("| " + " | ".join(values) + " |")


def benchmark_per_feature(*, warmup: int, iters: int, emit_markdown: bool = True) -> list[dict]:
    cols = ["eager", "compiled", "speedup"]
    rows: list[dict] = []
    for batch, length, n_fft, label in CONFIGS:
        hop = n_fft // 4
        x = mx.random.normal((batch, length))
        mx.eval(x)
        shared = _shared_kwargs(n_fft, hop)
        for name in FEATURE_NAMES:
            fn = FEATURE_FNS[name]
            kwargs = _single_feature_kwargs(name, shared)

            def eager_call(fn=fn, x=x, kwargs=kwargs):
                return fn(x, **kwargs)

            compiled_call = mx.compile(eager_call)
            eager_ms = _bench(eager_call, warmup=warmup, iters=iters)
            compiled_ms = _bench(compiled_call, warmup=warmup, iters=iters)
            rows.append(
                {
                    "label": f"{label} {name}",
                    "eager": eager_ms,
                    "compiled": compiled_ms,
                    "speedup": eager_ms / compiled_ms if compiled_ms > 0 else float("nan"),
                }
            )
    if emit_markdown:
        _print_table("Per-Feature Latency", cols, rows)
    return rows


def benchmark_bundle(*, warmup: int, iters: int, emit_markdown: bool = True) -> list[dict]:
    cols = [
        "sequential eager",
        "shared eager",
        "cached eager",
        "shared compiled",
        "cached compiled",
        "reuse gain",
    ]
    rows: list[dict] = []
    for batch, length, n_fft, label in CONFIGS:
        hop = n_fft // 4
        x = mx.random.normal((batch, length))
        mx.eval(x)
        shared = _shared_kwargs(n_fft, hop)
        cached_transform = SpectralFeatureTransform(include=FEATURE_NAMES, **shared)

        def sequential_call(x=x, shared=shared):
            return tuple(
                FEATURE_FNS[name](x, **_single_feature_kwargs(name, shared))
                for name in FEATURE_NAMES
            )

        def shared_call(x=x, shared=shared):
            return spectral_features(x, include=FEATURE_NAMES, **shared)

        def shared_compiled_call(x=x, shared=shared):
            return _spectral_feature_values(x, include=FEATURE_NAMES, **shared)[0]

        def cached_call(x=x, tr=cached_transform):
            return tr(x)

        compiled = cached_transform.get_compiled()

        def cached_compiled_call(x=x, fn=compiled):
            return fn(x)

        shared_compiled = mx.compile(shared_compiled_call)
        seq_eager_ms = _bench(sequential_call, warmup=warmup, iters=iters)
        shared_eager_ms = _bench(shared_call, warmup=warmup, iters=iters)
        cached_eager_ms = _bench(cached_call, warmup=warmup, iters=iters)
        shared_compiled_ms = _bench(shared_compiled, warmup=warmup, iters=iters)
        cached_compiled_ms = _bench(cached_compiled_call, warmup=warmup, iters=iters)
        rows.append(
            {
                "label": label,
                "sequential eager": seq_eager_ms,
                "shared eager": shared_eager_ms,
                "cached eager": cached_eager_ms,
                "shared compiled": shared_compiled_ms,
                "cached compiled": cached_compiled_ms,
                "reuse gain": seq_eager_ms / cached_eager_ms if cached_eager_ms > 0 else float("nan"),
            }
        )
    if emit_markdown:
        _print_table("Shared-STFT Bundle", cols, rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark spectral feature extraction paths")
    parser.add_argument("--quick", action="store_true", help="run fewer iterations")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--json-out", type=str, help="write JSON results to a file")
    args = parser.parse_args()

    warmup = 2 if args.quick else 5
    iters = 6 if args.quick else 20

    if not args.json:
        print("## mlx-spectro feature benchmarks")
        print(f"- warmup={warmup}")
        print(f"- iters={iters}")
    per_feature = benchmark_per_feature(warmup=warmup, iters=iters, emit_markdown=not args.json)
    bundle = benchmark_bundle(warmup=warmup, iters=iters, emit_markdown=not args.json)
    payload = {
        "benchmark": "features",
        "meta": {
            "warmup": warmup,
            "iters": iters,
            "quick": bool(args.quick),
            "platform": platform.platform(),
            "device": str(mx.default_device()),
        },
        "per_feature": per_feature,
        "bundle": bundle,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")


if __name__ == "__main__":
    main()
