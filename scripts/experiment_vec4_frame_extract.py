"""Experiment: Vec4-packed frame extraction Metal kernel.

Result: NOT ADOPTED.  Vec4 wins 70% of cases but averages only 1.05x.  The
benefit is concentrated at n_fft=1024-2048 (1.04-1.09x) and neutral-to-harmful
at n_fft>=4096.  The complexity cost (second kernel source, n_fft-based routing,
autotune cache pollution) outweighs the marginal gain.  Kept as a documented
negative result.

Hypothesis: Writing 4 consecutive FFT-bin floats per thread as a single float4
memory transaction improves write bandwidth utilization in the frame extraction
kernel, which is bandwidth-bound.

Design:
  - Prototype a vec4 variant of _METAL_FUSED_FRAME_EXTRACT_TEMPLATE
  - Grid x-dimension becomes NFFT/4 (each thread handles 4 consecutive bins)
  - Validate bit-exact parity with the existing simple kernel
  - Benchmark across real-world consumer configs from ~/Code projects
  - Report speedup/regression per config

Usage:
    metalq submit -w -n "vec4-frame-extract" -C "$(git rev-parse --show-toplevel)" \
        --uv-project "$(git rev-parse --show-toplevel)" -- \
        python scripts/experiment_vec4_frame_extract.py
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
# Vec4 kernel source
# ---------------------------------------------------------------------------
# Each thread processes 4 consecutive FFT bins (f_base, f_base+1, f_base+2, f_base+3).
# Grid: (NFFT/4, n_frames, B).  Threadgroup: (tgx, 1, 1).
#
# Benefits:
#   - Window reads: 4 consecutive floats (compiler can coalesce or vec-load)
#   - Output writes: 4 consecutive floats in memory (compiler can vec-store)
#   - Signal reads: for interior positions, 4 consecutive reads to consecutive
#     addresses; boundary positions still need per-element indexing
#
# Constraints:
#   - Requires NFFT % 4 == 0 (true for all real-world FFT sizes)
#   - Final thread group may process bins beyond NFFT for non-multiple-of-4
#     sizes, but we guard with an explicit bounds check per bin

_VEC4_FRAME_EXTRACT_SOURCE = """
int sig_len = params[0];
int n_frames = params[1];

int f_base = (int)thread_position_in_grid.x * 4;
int n_idx  = (int)thread_position_in_grid.y;
int b_idx  = (int)thread_position_in_grid.z;

if (n_idx >= n_frames) return;

int sig_offset = b_idx * sig_len;
int out_base   = b_idx * n_frames * NFFT + n_idx * NFFT + f_base;
int padded_base = n_idx * HOP + f_base;

// Process 4 consecutive FFT bins
#pragma unroll
for (int d = 0; d < 4; ++d) {
    int f_idx = f_base + d;
    if (f_idx >= NFFT) return;

    int src_pos = padded_base + d;

    // Reflect-pad boundary mapping
    int orig_idx;
    if (src_pos < PAD) {
        orig_idx = PAD - src_pos;
    } else if (src_pos < PAD + sig_len) {
        orig_idx = src_pos - PAD;
    } else {
        orig_idx = sig_len - 2 - (src_pos - PAD - sig_len);
    }

    float val = (float)signal[sig_offset + orig_idx] * (float)win[f_idx];
    out[out_base + d] = (T)val;
}
"""

# More aggressive vec4: use float4 store when NFFT % 4 == 0 and f_base + 3 < NFFT
_VEC4_STORE_FRAME_EXTRACT_SOURCE = """
int sig_len = params[0];
int n_frames = params[1];

int f_base = (int)thread_position_in_grid.x * 4;
int n_idx  = (int)thread_position_in_grid.y;
int b_idx  = (int)thread_position_in_grid.z;

if (f_base >= NFFT || n_idx >= n_frames) return;

int sig_offset = b_idx * sig_len;
int out_base   = b_idx * n_frames * NFFT + n_idx * NFFT + f_base;
int padded_base = n_idx * HOP + f_base;

// Check if all 4 source positions are in the interior (no reflect-pad boundary)
bool all_interior = (padded_base >= PAD) && (padded_base + 3 < PAD + sig_len);

float v0, v1, v2, v3;

if (all_interior) {
    // Fast path: all 4 reads are consecutive in the original signal
    int base_orig = padded_base - PAD;
    v0 = (float)signal[sig_offset + base_orig]     * (float)win[f_base];
    v1 = (float)signal[sig_offset + base_orig + 1] * (float)win[f_base + 1];
    v2 = (float)signal[sig_offset + base_orig + 2] * (float)win[f_base + 2];
    v3 = (float)signal[sig_offset + base_orig + 3] * (float)win[f_base + 3];
} else {
    // Slow path: per-element reflect-pad indexing
    #pragma unroll
    for (int d = 0; d < 4; ++d) {
        int src_pos = padded_base + d;
        int orig_idx;
        if (src_pos < PAD) {
            orig_idx = PAD - src_pos;
        } else if (src_pos < PAD + sig_len) {
            orig_idx = src_pos - PAD;
        } else {
            orig_idx = sig_len - 2 - (src_pos - PAD - sig_len);
        }
        float val = (float)signal[sig_offset + orig_idx] * (float)win[f_base + d];
        if (d == 0) v0 = val;
        else if (d == 1) v1 = val;
        else if (d == 2) v2 = val;
        else v3 = val;
    }
}

// Write 4 consecutive floats (compiler can optimize to vec store)
out[out_base]     = (T)v0;
out[out_base + 1] = (T)v1;
out[out_base + 2] = (T)v2;
out[out_base + 3] = (T)v3;
"""

# ---------------------------------------------------------------------------
# Benchmarking infrastructure
# ---------------------------------------------------------------------------

WARMUP = 5
ITERS = 15

# Real-world consumer configs (subset covering the range)
CONSUMER_CONFIGS = [
    # (label, n_fft, hop, sr, durations_sec, batch_sizes)
    ("bytedance", 2048, 160, 16000, [5, 30, 120], [1, 4, 8]),
    ("BeatNet-1024", 1024, 512, 22050, [10, 60, 180], [1, 4, 8]),
    ("BeatNet-4096", 4096, 512, 22050, [10, 60, 180], [1, 4, 8]),
    ("madmom-2048", 2048, 441, 44100, [10, 60, 180], [1, 4, 8]),
    ("mlxfcpe", 1024, 160, 16000, [5, 30, 60], [1, 4, 8]),
    ("demucs", 4096, 1024, 44100, [10, 60, 180], [1, 4, 8]),
    ("MDX", 6144, 1024, 44100, [10, 60, 180], [1, 4, 8]),
    ("RoFormer", 2048, 512, 44100, [10, 60, 180], [1, 4, 8]),
    ("mamba_amt", 2048, 512, 16000, [5, 30, 60], [1, 4, 8]),
]


def median(vals):
    s = sorted(vals)
    return s[len(s) // 2]


def bench_kernel(kernel, inputs, template, output_shapes, output_dtypes, grid, tg):
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
    # Compile kernels
    simple_kernel = _FrameExtractCache.get_simple()
    if not simple_kernel or simple_kernel is False:
        print("ERROR: simple kernel unavailable", file=sys.stderr)
        sys.exit(1)

    try:
        vec4_kernel = mx.fast.metal_kernel(
            name="vec4_frame_extract",
            input_names=["signal", "win", "params"],
            output_names=["out"],
            source=_VEC4_FRAME_EXTRACT_SOURCE,
        )
    except Exception as e:
        print(f"ERROR: vec4 kernel compilation failed: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        vec4_store_kernel = mx.fast.metal_kernel(
            name="vec4_store_frame_extract",
            input_names=["signal", "win", "params"],
            output_names=["out"],
            source=_VEC4_STORE_FRAME_EXTRACT_SOURCE,
        )
    except Exception as e:
        print(f"ERROR: vec4_store kernel compilation failed: {e}", file=sys.stderr)
        sys.exit(1)

    print("## Vec4 Frame Extraction Kernel Experiment")
    print()
    print(f"Warmup: {WARMUP}, Iterations: {ITERS}")
    print()

    # -----------------------------------------------------------------------
    # Phase 1: Parity validation
    # -----------------------------------------------------------------------
    print("### Phase 1: Parity Validation")
    print()

    parity_configs = [
        (1, 16000, 512, 128),
        (4, 160000, 1024, 256),
        (8, 480000, 2048, 512),
        (1, 320000, 4096, 1024),
        (4, 44100, 6144, 1024),
        # Edge case: very short signal near n_fft boundary
        (1, 600, 512, 128),
        (2, 1200, 1024, 256),
    ]

    parity_pass = True
    for B, T, nfft, hop in parity_configs:
        pad = nfft // 2
        padded_len = T + 2 * pad
        n_frames = (padded_len - nfft) // hop + 1

        if nfft % 4 != 0:
            continue

        mx.random.seed(42)
        x = mx.random.normal((B, T))
        mx.eval(x)
        x = mx.contiguous(x)

        transform = SpectralTransform(n_fft=nfft, hop_length=hop)
        window = transform.window
        fe_params = mx.array([T, n_frames], dtype=mx.int32)

        tmpl = [("T", mx.float32), ("NFFT", nfft), ("HOP", hop), ("PAD", pad)]

        # Autotune simple kernel
        fe_tgx = _KernelCache.autotune_threadgroup_x(
            kernel=simple_kernel,
            kernel_name=f"fused_frame_extract_{mx.float32}",
            n_fft=nfft, hop=hop,
            grid=(nfft, n_frames, B),
            inputs=[x, window, fe_params],
            template=tmpl,
            output_shape=(B, n_frames, nfft),
            output_dtype=mx.float32,
            default_tgx=min(256, nfft),
        )

        # Run simple kernel (reference)
        ref = simple_kernel(
            inputs=[x, window, fe_params], template=tmpl,
            output_shapes=[(B, n_frames, nfft)], output_dtypes=[mx.float32],
            grid=(nfft, n_frames, B), threadgroup=(fe_tgx, 1, 1),
        )[0]
        mx.eval(ref)

        # Run vec4 kernel
        grid_x_vec4 = math.ceil(nfft / 4)
        vec4_tgx = min(256, grid_x_vec4)
        out_vec4 = vec4_kernel(
            inputs=[x, window, fe_params], template=tmpl,
            output_shapes=[(B, n_frames, nfft)], output_dtypes=[mx.float32],
            grid=(grid_x_vec4, n_frames, B), threadgroup=(vec4_tgx, 1, 1),
        )[0]
        mx.eval(out_vec4)

        # Run vec4_store kernel
        out_vec4s = vec4_store_kernel(
            inputs=[x, window, fe_params], template=tmpl,
            output_shapes=[(B, n_frames, nfft)], output_dtypes=[mx.float32],
            grid=(grid_x_vec4, n_frames, B), threadgroup=(vec4_tgx, 1, 1),
        )[0]
        mx.eval(out_vec4s)

        # Compare
        diff_vec4 = float(mx.max(mx.abs(ref - out_vec4)).item())
        diff_vec4s = float(mx.max(mx.abs(ref - out_vec4s)).item())
        exact_vec4 = bool(mx.array_equal(ref, out_vec4))
        exact_vec4s = bool(mx.array_equal(ref, out_vec4s))

        status_v4 = "EXACT" if exact_vec4 else (f"max_diff={diff_vec4:.2e}" if diff_vec4 < 1e-6 else "FAIL")
        status_v4s = "EXACT" if exact_vec4s else (f"max_diff={diff_vec4s:.2e}" if diff_vec4s < 1e-6 else "FAIL")

        label = f"B={B} T={T} nfft={nfft} hop={hop}"
        print(f"  {label:<40} vec4={status_v4:<12} vec4_store={status_v4s}")

        if diff_vec4 > 1e-6 or diff_vec4s > 1e-6:
            parity_pass = False

    print()
    if not parity_pass:
        print("**PARITY FAILURE** — aborting benchmark phase.")
        sys.exit(1)
    print("All parity checks passed.")
    print()

    # -----------------------------------------------------------------------
    # Phase 2: Performance benchmarking
    # -----------------------------------------------------------------------
    print("### Phase 2: Performance Benchmarking")
    print()
    print(
        f"| {'Project':<14} | {'B':>2} | {'nfft':>5} | {'hop':>5} | "
        f"{'T':>9} | {'MB':>7} | "
        f"{'simple':>9} | {'vec4':>9} | {'vec4s':>9} | "
        f"{'best':>6} | {'factor':>7} |"
    )
    print(
        "|" + "|".join(["-" * w for w in [16, 4, 7, 7, 11, 9, 11, 11, 11, 8, 9]]) + "|"
    )

    results = []

    for label, nfft, hop, sr, durations, batches in CONSUMER_CONFIGS:
        if nfft % 4 != 0:
            continue

        pad = nfft // 2
        grid_x_vec4 = math.ceil(nfft / 4)

        for dur in durations:
            T = sr * dur
            padded_len = T + 2 * pad
            n_frames = (padded_len - nfft) // hop + 1

            for B in batches:
                out_bytes = B * n_frames * nfft * 4
                out_mb = out_bytes / 1e6

                # Focus on the range where simple kernel is used (<100MB)
                # plus some above-threshold cases for completeness
                if out_mb < 2 or out_mb > 500:
                    continue

                mx.random.seed(42)
                x = mx.random.normal((B, T))
                mx.eval(x)
                x = mx.contiguous(x)

                transform = SpectralTransform(n_fft=nfft, hop_length=hop)
                window = transform.window
                fe_params = mx.array([T, n_frames], dtype=mx.int32)

                tmpl = [("T", mx.float32), ("NFFT", nfft), ("HOP", hop), ("PAD", pad)]

                # Autotune simple
                fe_tgx = _KernelCache.autotune_threadgroup_x(
                    kernel=simple_kernel,
                    kernel_name=f"fused_frame_extract_{mx.float32}",
                    n_fft=nfft, hop=hop,
                    grid=(nfft, n_frames, B),
                    inputs=[x, window, fe_params],
                    template=tmpl,
                    output_shape=(B, n_frames, nfft),
                    output_dtype=mx.float32,
                    default_tgx=min(256, nfft),
                )

                # Bench simple
                ms_simple = bench_kernel(
                    simple_kernel,
                    inputs=[x, window, fe_params], template=tmpl,
                    output_shapes=[(B, n_frames, nfft)], output_dtypes=[mx.float32],
                    grid=(nfft, n_frames, B), tg=(fe_tgx, 1, 1),
                )

                # Bench vec4 (unrolled loop)
                vec4_tgx = min(256, grid_x_vec4)
                ms_vec4 = bench_kernel(
                    vec4_kernel,
                    inputs=[x, window, fe_params], template=tmpl,
                    output_shapes=[(B, n_frames, nfft)], output_dtypes=[mx.float32],
                    grid=(grid_x_vec4, n_frames, B), tg=(vec4_tgx, 1, 1),
                )

                # Bench vec4_store (interior fast path + explicit 4-element store)
                ms_vec4s = bench_kernel(
                    vec4_store_kernel,
                    inputs=[x, window, fe_params], template=tmpl,
                    output_shapes=[(B, n_frames, nfft)], output_dtypes=[mx.float32],
                    grid=(grid_x_vec4, n_frames, B), tg=(vec4_tgx, 1, 1),
                )

                # Determine winner
                best_ms = min(ms_simple, ms_vec4, ms_vec4s)
                if best_ms == ms_simple:
                    winner = "simple"
                    factor = 1.0
                elif best_ms == ms_vec4:
                    winner = "vec4"
                    factor = ms_simple / ms_vec4
                else:
                    winner = "vec4s"
                    factor = ms_simple / ms_vec4s

                row = {
                    "project": label, "B": B, "nfft": nfft, "hop": hop,
                    "T": T, "out_mb": round(out_mb, 1),
                    "ms_simple": round(ms_simple, 3),
                    "ms_vec4": round(ms_vec4, 3),
                    "ms_vec4s": round(ms_vec4s, 3),
                    "winner": winner,
                    "factor": round(factor, 3),
                }
                results.append(row)

                print(
                    f"| {label:<14} | {B:>2} | {nfft:>5} | {hop:>5} | "
                    f"{T:>9} | {out_mb:>6.1f}M | "
                    f"{ms_simple:>7.3f}ms | {ms_vec4:>7.3f}ms | {ms_vec4s:>7.3f}ms | "
                    f"{winner:>6} | {factor:>+6.2f}x |"
                )

    # -----------------------------------------------------------------------
    # Phase 3: Analysis
    # -----------------------------------------------------------------------
    print()
    print("### Phase 3: Analysis")
    print()

    total = len(results)
    simple_wins = sum(1 for r in results if r["winner"] == "simple")
    vec4_wins = sum(1 for r in results if r["winner"] == "vec4")
    vec4s_wins = sum(1 for r in results if r["winner"] == "vec4s")

    print(f"Total cases: {total}")
    print(f"- simple wins: {simple_wins} ({100*simple_wins/total:.0f}%)")
    print(f"- vec4 wins: {vec4_wins} ({100*vec4_wins/total:.0f}%)")
    print(f"- vec4_store wins: {vec4s_wins} ({100*vec4s_wins/total:.0f}%)")
    print()

    # Analyze by output size bucket
    buckets = [(0, 10), (10, 50), (50, 100), (100, 500)]
    print("| MB range | total | simple | vec4 | vec4s | best vec4 avg speedup |")
    print("|----------|-------|--------|------|-------|-----------------------|")
    for lo, hi in buckets:
        in_range = [r for r in results if lo <= r["out_mb"] < hi]
        if not in_range:
            continue
        sw = sum(1 for r in in_range if r["winner"] == "simple")
        v4 = sum(1 for r in in_range if r["winner"] == "vec4")
        v4s = sum(1 for r in in_range if r["winner"] == "vec4s")
        # Average speedup of best vec4 variant over simple
        avg_speedup = sum(
            max(r["ms_simple"] / r["ms_vec4"], r["ms_simple"] / r["ms_vec4s"])
            for r in in_range
        ) / len(in_range)
        print(f"| {lo:>4}-{hi:<4} | {len(in_range):>5} | {sw:>6} | {v4:>4} | {v4s:>5} | {avg_speedup:>20.2f}x |")

    print()

    # Analyze by n_fft
    nfft_vals = sorted(set(r["nfft"] for r in results))
    print("| n_fft | total | simple | vec4 | vec4s | best vec4 avg speedup |")
    print("|-------|-------|--------|------|-------|-----------------------|")
    for nfft in nfft_vals:
        in_group = [r for r in results if r["nfft"] == nfft]
        sw = sum(1 for r in in_group if r["winner"] == "simple")
        v4 = sum(1 for r in in_group if r["winner"] == "vec4")
        v4s = sum(1 for r in in_group if r["winner"] == "vec4s")
        avg_speedup = sum(
            max(r["ms_simple"] / r["ms_vec4"], r["ms_simple"] / r["ms_vec4s"])
            for r in in_group
        ) / len(in_group)
        print(f"| {nfft:>5} | {len(in_group):>5} | {sw:>6} | {v4:>4} | {v4s:>5} | {avg_speedup:>20.2f}x |")

    print()

    # Recommendation
    print("### Recommendation")
    print()
    any_vec4_win = vec4_wins + vec4s_wins
    if any_vec4_win == 0:
        print("Vec4 packing provides no benefit for frame extraction. The simple")
        print("autotuned kernel is already optimal. Do not adopt.")
    elif any_vec4_win > total * 0.6:
        best_variant = "vec4" if vec4_wins > vec4s_wins else "vec4_store"
        avg_factor = sum(r["factor"] for r in results if r["winner"] != "simple") / max(any_vec4_win, 1)
        print(f"Vec4 packing ({best_variant}) wins {any_vec4_win}/{total} cases")
        print(f"with average {avg_factor:.2f}x speedup. Worth adopting.")
    else:
        print(f"Mixed results: vec4 wins {any_vec4_win}/{total} cases.")
        print("Benefit is inconsistent across configs. Not recommended for adoption")
        print("unless a specific subset shows compelling gains.")

    # Save results
    with open(Path(__file__).parent / "experiment_vec4_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print()
    print("Raw results saved to scripts/experiment_vec4_results.json")


if __name__ == "__main__":
    main()
