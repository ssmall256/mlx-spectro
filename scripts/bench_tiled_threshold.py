"""Benchmark tiled vs simple frame-extraction kernel across real-world configs.

Derives test cases from actual mlx-spectro consumers under ~/Code to find the
empirical crossover point between the simple (autotuned) and tiled (shared-memory)
frame extraction Metal kernels.

Usage:
    metalq submit -w -n "tiled-threshold" --uv-project ~/Code/mlx-spectro -- \
        python scripts/bench_tiled_threshold.py
"""

import json
import math
import sys
import time

from pathlib import Path

import mlx.core as mx

from mlx_spectro.spectral_ops import (
    SpectralTransform,
    _FrameExtractCache,
    _KernelCache,
)

# ---------------------------------------------------------------------------
# Real-world configs extracted from ~/Code consumers
# (project, n_fft, hop_length, sample_rate, typical_durations_sec)
# ---------------------------------------------------------------------------
CONSUMER_CONFIGS = [
    # Piano transcription (bytedance): very high overlap
    ("bytedance", 2048, 160, 16000, [5, 15, 30, 60, 120]),
    # Beat tracking (BeatNet): multi-resolution
    ("BeatNet-1024", 1024, 512, 22050, [10, 30, 60, 180]),
    ("BeatNet-2048", 2048, 512, 22050, [10, 30, 60, 180]),
    ("BeatNet-4096", 4096, 512, 22050, [10, 30, 60, 180]),
    # Beat tracking (madmom): multi-resolution with 100fps hop
    ("madmom-1024", 1024, 441, 44100, [10, 30, 60, 180]),
    ("madmom-2048", 2048, 441, 44100, [10, 30, 60, 180]),
    ("madmom-4096", 4096, 441, 44100, [10, 30, 60, 180]),
    # Beat tracking (beat_this)
    ("beat_this", 1024, 441, 22050, [10, 30, 60, 180]),
    # Pitch extraction (FCPE): high overlap
    ("mlxfcpe", 1024, 160, 16000, [5, 15, 30, 60]),
    # Music structure (LinkSeg): very large FFT, low overlap
    ("LinkSeg", 8192, 4410, 44100, [30, 60, 180, 300]),
    # Music structure (SongFormer): high overlap
    ("SongFormer", 2048, 240, 24000, [15, 30, 60, 180]),
    # Source separation (demucs)
    ("demucs", 4096, 1024, 44100, [10, 30, 60, 180]),
    # Source separation (MDX): large FFT
    ("MDX", 6144, 1024, 44100, [10, 30, 60, 180]),
    # Source separation (RoFormer)
    ("RoFormer", 2048, 512, 44100, [10, 30, 60, 180]),
    # Piano transcription (mamba_amt)
    ("mamba_amt", 2048, 512, 16000, [5, 15, 30, 60]),
    # Key recognition: nearly non-overlapping
    ("KeyRecog", 8192, 8820, 44100, [30, 60, 180, 300]),
]

WARMUP = 5
ITERS = 15
BATCH_SIZES = [1, 4, 8]


def median(vals):
    s = sorted(vals)
    return s[len(s) // 2]


def bench_kernel(kernel, inputs, template, output_shapes, output_dtypes, grid, tg):
    """Warmup + timed iterations, return median ms."""
    for _ in range(WARMUP):
        out = kernel(
            inputs=inputs, template=template,
            output_shapes=output_shapes, output_dtypes=output_dtypes,
            grid=grid, threadgroup=tg,
        )
        mx.eval(out[0])

    times = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        out = kernel(
            inputs=inputs, template=template,
            output_shapes=output_shapes, output_dtypes=output_dtypes,
            grid=grid, threadgroup=tg,
        )
        mx.eval(out[0])
        times.append(time.perf_counter() - t0)
    return median(times) * 1000


def main():
    simple_kernel = _FrameExtractCache.get_simple()
    tiled_kernel = _FrameExtractCache.get_tiled()

    if not simple_kernel or simple_kernel is False:
        print("ERROR: simple kernel unavailable", file=sys.stderr)
        sys.exit(1)
    if not tiled_kernel or tiled_kernel is False:
        print("ERROR: tiled kernel unavailable", file=sys.stderr)
        sys.exit(1)

    results = []

    print("## Tiled vs Simple Frame Extract — Real-World Consumer Configs")
    print()
    print(f"Warmup: {WARMUP}, Iterations: {ITERS}")
    print()
    print(
        f"| {'Project':<14} | {'B':>2} | {'n_fft':>5} | {'hop':>5} | "
        f"{'ratio':>5} | {'T':>9} | {'out_MB':>7} | "
        f"{'simple_ms':>10} | {'tiled_ms':>10} | {'winner':>7} | {'factor':>7} |"
    )
    print("|" + "|".join(["-" * w for w in [16, 4, 7, 7, 7, 11, 9, 12, 12, 9, 9]]) + "|")

    for project, nfft, hop, sr, durations in CONSUMER_CONFIGS:
        pad = nfft // 2
        tp = _FrameExtractCache.tile_params(nfft, hop)

        for dur in durations:
            T = sr * dur
            padded_len = T + 2 * pad
            n_frames = (padded_len - nfft) // hop + 1
            ratio = nfft / hop

            for B in BATCH_SIZES:
                out_bytes = B * n_frames * nfft * 4
                out_mb = out_bytes / 1e6

                # Only test cases where the threshold decision matters
                # (within a reasonable range around potential thresholds)
                if out_mb < 5:
                    continue

                mx.random.seed(42)
                x = mx.random.normal((B, T))
                mx.eval(x)
                x = mx.contiguous(x)

                transform = SpectralTransform(n_fft=nfft, hop_length=hop)
                window = transform.window
                fe_params = mx.array([T, n_frames], dtype=mx.int32)

                tmpl_simple = [
                    ("T", mx.float32), ("NFFT", nfft),
                    ("HOP", hop), ("PAD", pad),
                ]

                # Autotune simple kernel
                fe_tgx = _KernelCache.autotune_threadgroup_x(
                    kernel=simple_kernel,
                    kernel_name=f"fused_frame_extract_{mx.float32}",
                    n_fft=nfft, hop=hop,
                    grid=(nfft, n_frames, B),
                    inputs=[x, window, fe_params],
                    template=tmpl_simple,
                    output_shape=(B, n_frames, nfft),
                    output_dtype=mx.float32,
                    default_tgx=min(256, nfft),
                )

                # Benchmark simple kernel
                ms_simple = bench_kernel(
                    simple_kernel,
                    inputs=[x, window, fe_params],
                    template=tmpl_simple,
                    output_shapes=[(B, n_frames, nfft)],
                    output_dtypes=[mx.float32],
                    grid=(nfft, n_frames, B),
                    tg=(fe_tgx, 1, 1),
                )

                # Benchmark tiled kernel (if tiling is possible)
                if tp is not None:
                    tile_frames, tg_x, tg_y, chunk_len = tp
                    tmpl_tiled = [
                        ("T", mx.float32), ("NFFT", nfft),
                        ("HOP", hop), ("PAD", pad),
                        ("TILE_FRAMES", tile_frames), ("TG_X", tg_x),
                        ("TG_Y", tg_y), ("CHUNK_LEN", chunk_len),
                    ]
                    n_tile_groups = math.ceil(n_frames / tile_frames)

                    ms_tiled = bench_kernel(
                        tiled_kernel,
                        inputs=[x, window, fe_params],
                        template=tmpl_tiled,
                        output_shapes=[(B, n_frames, nfft)],
                        output_dtypes=[mx.float32],
                        grid=(n_tile_groups * tg_x, tg_y, B),
                        tg=(tg_x, tg_y, 1),
                    )
                else:
                    ms_tiled = float("inf")

                if ms_tiled < ms_simple:
                    winner = "tiled"
                    factor = ms_simple / ms_tiled
                else:
                    winner = "simple"
                    factor = -(ms_tiled / ms_simple)

                row = {
                    "project": project,
                    "B": B,
                    "n_fft": nfft,
                    "hop": hop,
                    "ratio": round(ratio, 2),
                    "T": T,
                    "out_mb": round(out_mb, 1),
                    "simple_ms": round(ms_simple, 3),
                    "tiled_ms": round(ms_tiled, 3) if ms_tiled != float("inf") else None,
                    "winner": winner,
                    "factor": round(abs(factor), 2),
                }
                results.append(row)

                tiled_str = f"{ms_tiled:.3f}ms" if ms_tiled != float("inf") else "N/A"
                print(
                    f"| {project:<14} | {B:>2} | {nfft:>5} | {hop:>5} | "
                    f"{ratio:>5.1f} | {T:>9} | {out_mb:>6.1f}M | "
                    f"{ms_simple:>8.3f}ms | {tiled_str:>10} | {winner:>7} | {factor:>+6.2f}x |"
                )

    # Summary analysis
    print()
    print("## Analysis")
    print()

    # Find crossover by output size
    tiled_wins = [r for r in results if r["winner"] == "tiled" and r["tiled_ms"] is not None]
    simple_wins = [r for r in results if r["winner"] == "simple"]

    if tiled_wins:
        min_tiled_mb = min(r["out_mb"] for r in tiled_wins)
        max_simple_mb = max(r["out_mb"] for r in simple_wins) if simple_wins else 0

        print(f"- Tiled wins in {len(tiled_wins)}/{len(results)} cases")
        print(f"- Simple wins in {len(simple_wins)}/{len(results)} cases")
        print(f"- Smallest tiled win: {min_tiled_mb:.1f} MB")
        print(f"- Largest simple win: {max_simple_mb:.1f} MB")
        print()

        # Bucket analysis: what % does tiled win at each size range?
        buckets = [(0, 25), (25, 50), (50, 100), (100, 200), (200, 500), (500, 1000), (1000, 5000)]
        print("| MB range | total | tiled wins | simple wins | tiled win % | avg tiled factor |")
        print("|----------|-------|------------|-------------|-------------|------------------|")
        for lo, hi in buckets:
            in_range = [r for r in results if lo <= r["out_mb"] < hi]
            if not in_range:
                continue
            tw = [r for r in in_range if r["winner"] == "tiled"]
            sw = [r for r in in_range if r["winner"] == "simple"]
            pct = 100 * len(tw) / len(in_range) if in_range else 0
            avg_factor = sum(r["factor"] for r in tw) / len(tw) if tw else 0
            print(
                f"| {lo:>4}-{hi:<4} | {len(in_range):>5} | {len(tw):>10} | "
                f"{len(sw):>11} | {pct:>10.0f}% | {avg_factor:>15.2f}x |"
            )
    else:
        print("- Simple kernel wins in ALL cases. Tiled kernel provides no benefit.")

    print()

    # Recommendation
    print("## Recommendation")
    print()
    if not tiled_wins:
        print("Remove the tiled kernel path entirely — simple (autotuned) dominates.")
    else:
        # Find threshold where tiled wins >50% of the time
        for threshold in [25, 50, 75, 100, 150, 200, 300, 500]:
            above = [r for r in results if r["out_mb"] >= threshold]
            if not above:
                continue
            tw_above = [r for r in above if r["winner"] == "tiled"]
            pct = 100 * len(tw_above) / len(above)
            if pct >= 60:
                print(f"Raise threshold to **{threshold} MB** (tiled wins {pct:.0f}% of cases above this).")
                # Check if the advantage is meaningful
                avg_adv = sum(r["factor"] for r in tw_above) / len(tw_above) if tw_above else 1.0
                if avg_adv < 1.05:
                    print(f"  However, average tiled advantage is only {avg_adv:.2f}x — may not justify complexity.")
                    print("  Consider removing tiled path entirely.")
                break
        else:
            print("No clear threshold where tiled consistently wins. Consider removing tiled path.")

    # Dump raw JSON for further analysis
    with open(Path(__file__).parent / "bench_tiled_threshold_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print()
    print("Raw results saved to scripts/bench_tiled_threshold_results.json")


if __name__ == "__main__":
    main()
