#!/usr/bin/env python3
"""Benchmark suite for mlx-spectro.

Compares mlx-spectro against torch.stft/torch.istft on MPS and the
third-party mlx-stft package.  Prints Markdown tables suitable for
copy-pasting into a README.

Torch MPS comparisons include both timing and accuracy (roundtrip error).
mlx-stft comparisons are timing-only (different parameter conventions).

Usage:
    python scripts/benchmark.py              # full suite
    python scripts/benchmark.py --quick      # reduced iterations
    python scripts/benchmark.py --forward    # forward-only
    python scripts/benchmark.py --backward   # backward-only
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
import warnings

import mlx.core as mx
import numpy as np

from mlx_spectro import SpectralTransform

warnings.filterwarnings("ignore", message=".*was resized.*")
warnings.filterwarnings("ignore", message=".*deprecated.*")

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------

try:
    import torch
    _HAS_TORCH = torch.backends.mps.is_available()
except ImportError:
    _HAS_TORCH = False

try:
    import mlx_stft as _mlx_stft
    _HAS_MLX_STFT = True
except ImportError:
    _HAS_MLX_STFT = False

# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def bench_mlx(fn, *, warmup: int = 5, iters: int = 20) -> float:
    """Return median latency in ms for an MLX workload."""
    for _ in range(warmup):
        fn()
    times: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e3)
    times.sort()
    return times[len(times) // 2]


def bench_mps(fn, *, warmup: int = 5, iters: int = 20) -> float:
    """Return median latency in ms for a torch MPS workload."""
    for _ in range(warmup):
        fn()
    torch.mps.synchronize()
    times: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.mps.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    times.sort()
    return times[len(times) // 2]

# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------

# (batch, signal_length, n_fft, label)
CONFIGS = [
    # Small workloads
    (1,  16_000,   512,  "B=1 T=16k nfft=512"),
    (4,  16_000,   512,  "B=4 T=16k nfft=512"),
    # Medium workloads
    (1,  160_000,  1024, "B=1 T=160k nfft=1024"),
    (4,  160_000,  1024, "B=4 T=160k nfft=1024"),
    (4,  160_000,  2048, "B=4 T=160k nfft=2048"),
    # Large workloads
    (8,  160_000,  1024, "B=8 T=160k nfft=1024"),
    (4,  1_320_000, 1024, "B=4 T=1.3M nfft=1024"),
    (1,  1_320_000, 1024, "B=1 T=1.3M nfft=1024"),
    # Typical audio ML workload
    (8,  480_000,  1024, "B=8 T=480k nfft=1024"),
]

# ---------------------------------------------------------------------------
# Forward benchmarks
# ---------------------------------------------------------------------------

def bench_stft_forward(warmup: int, iters: int) -> None:
    cols = ["mlx-spectro"]
    if _HAS_TORCH:
        cols.append("torch MPS")
    if _HAS_MLX_STFT:
        cols.append("mlx-stft")

    rows: list[dict] = []
    for B, T, nfft, label in CONFIGS:
        hop = nfft // 4
        row: dict = {"label": label}

        # mlx-spectro
        t = SpectralTransform(n_fft=nfft, hop_length=hop, center=True)
        x_mlx = mx.random.normal((B, T))
        mx.eval(x_mlx)

        def ours_fn(x=x_mlx, t=t):
            s = t.stft(x, output_layout="bnf")
            mx.eval(s)
        row["mlx-spectro"] = bench_mlx(ours_fn, warmup=warmup, iters=iters)

        # torch MPS
        if _HAS_TORCH:
            win_mps = torch.hann_window(nfft, device="mps")
            x_mps = torch.randn(B, T, device="mps")

            def torch_fn(x=x_mps, nfft=nfft, hop=hop, win=win_mps):
                torch.stft(x, n_fft=nfft, hop_length=hop, window=win,
                           center=True, return_complex=True)
            row["torch MPS"] = bench_mps(torch_fn, warmup=warmup, iters=iters)

        # mlx-stft
        if _HAS_MLX_STFT:
            stft_ext = _mlx_stft.CompiledSTFT(n_fft=nfft, hop_length=hop, onesided=True)
            x_ext = mx.random.normal((B, T))
            mx.eval(x_ext)

            def ext_fn(x=x_ext, s=stft_ext):
                out = s(x)
                mx.eval(out)
            row["mlx-stft"] = bench_mlx(ext_fn, warmup=warmup, iters=iters)

        rows.append(row)
    _print_timing_table("STFT Forward", cols, rows)


def bench_istft_forward(warmup: int, iters: int) -> None:
    cols = ["mlx-spectro"]
    if _HAS_TORCH:
        cols.append("torch MPS")
    if _HAS_MLX_STFT:
        cols.append("mlx-stft")

    rows: list[dict] = []
    for B, T, nfft, label in CONFIGS:
        hop = nfft // 4
        row: dict = {"label": label}

        # mlx-spectro
        t = SpectralTransform(n_fft=nfft, hop_length=hop, center=True)
        x_mlx = mx.random.normal((B, T))
        mx.eval(x_mlx)
        spec_mlx = t.stft(x_mlx, output_layout="bnf")
        mx.eval(spec_mlx)

        def ours_fn(spec=spec_mlx, t=t, T=T):
            y = t.istft(spec, length=T, input_layout="bnf")
            mx.eval(y)
        row["mlx-spectro"] = bench_mlx(ours_fn, warmup=warmup, iters=iters)

        # torch MPS
        if _HAS_TORCH:
            win_mps = torch.hann_window(nfft, device="mps")
            x_mps = torch.randn(B, T, device="mps")
            spec_mps = torch.stft(x_mps, n_fft=nfft, hop_length=hop,
                                  window=win_mps, center=True,
                                  return_complex=True)

            def torch_fn(spec=spec_mps, nfft=nfft, hop=hop, win=win_mps, T=T):
                torch.istft(spec, n_fft=nfft, hop_length=hop, window=win,
                            center=True, length=T)
            row["torch MPS"] = bench_mps(torch_fn, warmup=warmup, iters=iters)

        # mlx-stft
        if _HAS_MLX_STFT:
            stft_ext = _mlx_stft.CompiledSTFT(n_fft=nfft, hop_length=hop, onesided=True)
            istft_ext = _mlx_stft.ISTFT(n_fft=nfft, hop_length=hop, onesided=True, length=T)
            x_ext = mx.random.normal((B, T))
            mx.eval(x_ext)
            spec_ext = stft_ext(x_ext)
            mx.eval(spec_ext)

            def ext_fn(spec=spec_ext, ist=istft_ext, T=T):
                y = ist(spec, length=T)
                mx.eval(y)
            row["mlx-stft"] = bench_mlx(ext_fn, warmup=warmup, iters=iters)

        rows.append(row)
    _print_timing_table("ISTFT Forward", cols, rows)

# ---------------------------------------------------------------------------
# Compiled vs eager forward benchmarks
# ---------------------------------------------------------------------------

def bench_compiled_forward(warmup: int, iters: int) -> None:
    """Compare eager stft/istft vs stft_compiled/istft_compiled and compiled_pair."""
    cols = ["compiled_pair", "compiled", "uncompiled"]

    stft_rows: list[dict] = []
    istft_rows: list[dict] = []
    for B, T, nfft, label in CONFIGS:
        hop = nfft // 4
        t = SpectralTransform(n_fft=nfft, hop_length=hop, center=True)
        x = mx.random.normal((B, T))
        mx.eval(x)

        # STFT eager
        def eager_stft(x=x, t=t):
            s = t.stft(x, output_layout="bnf")
            mx.eval(s)

        # STFT compiled
        def compiled_stft(x=x, t=t):
            s = t.stft_compiled(x, output_layout="bnf")
            mx.eval(s)

        # compiled_pair
        stft_fn, istft_fn = t.compiled_pair(length=T, layout="bnf", warmup_batch=B)

        def pair_stft(x=x, fn=stft_fn):
            s = fn(x)
            mx.eval(s)

        stft_rows.append({
            "label": label,
            "uncompiled": bench_mlx(eager_stft, warmup=warmup, iters=iters),
            "compiled": bench_mlx(compiled_stft, warmup=warmup, iters=iters),
            "compiled_pair": bench_mlx(pair_stft, warmup=warmup, iters=iters),
        })

        # iSTFT benchmarks
        spec = t.stft(x, output_layout="bnf")
        mx.eval(spec)

        def eager_istft(spec=spec, t=t, T=T):
            y = t.istft(spec, length=T, input_layout="bnf")
            mx.eval(y)

        def compiled_istft(spec=spec, t=t, T=T):
            y = t.istft_compiled(spec, length=T, input_layout="bnf")
            mx.eval(y)

        def pair_istft(spec=spec, fn=istft_fn):
            y = fn(spec)
            mx.eval(y)

        istft_rows.append({
            "label": label,
            "uncompiled": bench_mlx(eager_istft, warmup=warmup, iters=iters),
            "compiled": bench_mlx(compiled_istft, warmup=warmup, iters=iters),
            "compiled_pair": bench_mlx(pair_istft, warmup=warmup, iters=iters),
        })

    _print_timing_table("STFT: Eager vs Compiled", cols, stft_rows, compare_col="uncompiled")
    _print_timing_table("ISTFT: Eager vs Compiled", cols, istft_rows, compare_col="uncompiled")


# ---------------------------------------------------------------------------
# Accuracy comparison (vs torch MPS)
# ---------------------------------------------------------------------------

def bench_accuracy(warmup: int, iters: int) -> None:
    if not _HAS_TORCH:
        print("\n### Accuracy (skipped — torch not available)\n")
        return

    print("\n### Roundtrip Accuracy (vs torch MPS)\n")
    print(
        f"| {'Config':<28} "
        f"| {'mlx-spectro err':>17} "
        f"| {'torch MPS err':>15} |"
    )
    print(f"|{'-'*30}|{'-'*19}|{'-'*17}|")

    for B, T, nfft, label in CONFIGS:
        hop = nfft // 4

        # mlx-spectro roundtrip
        t = SpectralTransform(n_fft=nfft, hop_length=hop, center=True)
        x_mlx = mx.random.normal((B, T))
        mx.eval(x_mlx)
        spec_mlx = t.stft(x_mlx, output_layout="bnf")
        y_mlx = t.istft(spec_mlx, length=T, input_layout="bnf")
        mx.eval(y_mlx)
        err_mlx = float(mx.abs(x_mlx - y_mlx).max())

        # torch MPS roundtrip (use same signal, converted)
        x_np = np.array(x_mlx)
        x_mps = torch.from_numpy(x_np).to("mps")
        win_mps = torch.hann_window(nfft, device="mps")
        spec_mps = torch.stft(x_mps, n_fft=nfft, hop_length=hop,
                               window=win_mps, center=True,
                               return_complex=True)
        y_mps = torch.istft(spec_mps, n_fft=nfft, hop_length=hop,
                             window=win_mps, center=True, length=T)
        torch.mps.synchronize()
        err_torch = (x_mps - y_mps).abs().max().item()

        print(
            f"| {label:<28} "
            f"| {err_mlx:>15.2e} "
            f"| {err_torch:>13.2e} |"
        )

# ---------------------------------------------------------------------------
# Backward benchmarks
# ---------------------------------------------------------------------------

def bench_stft_backward(warmup: int, iters: int) -> None:
    cols = ["mx.compile", "uncompiled"]
    if _HAS_TORCH:
        cols.append("torch MPS")

    rows: list[dict] = []
    for B, T, nfft, label in CONFIGS:
        hop = nfft // 4
        row: dict = {"label": label}

        # mlx-spectro
        t = SpectralTransform(n_fft=nfft, hop_length=hop, center=True)
        x_mlx = mx.random.normal((B, T))
        mx.eval(x_mlx)

        def ours_fn(x=x_mlx, t=t):
            def loss(xi):
                s = t.differentiable_stft(xi)
                return mx.abs(s).square().sum()
            g = mx.grad(loss)(x)
            mx.eval(g)
        row["uncompiled"] = bench_mlx(ours_fn, warmup=warmup, iters=iters)

        # mx.compile variant
        def _stft_grad(x, t=t):
            def loss(xi):
                s = t.differentiable_stft(xi)
                return mx.abs(s).square().sum()
            return mx.grad(loss)(x)
        compiled_stft_grad = mx.compile(_stft_grad)

        def compiled_fn(x=x_mlx, fn=compiled_stft_grad):
            g = fn(x)
            mx.eval(g)
        row["mx.compile"] = bench_mlx(compiled_fn, warmup=warmup, iters=iters)

        # torch MPS
        if _HAS_TORCH:
            win_mps = torch.hann_window(nfft, device="mps")

            def torch_fn(B=B, T=T, nfft=nfft, hop=hop, win=win_mps):
                x = torch.randn(B, T, device="mps", requires_grad=True)
                s = torch.stft(x, n_fft=nfft, hop_length=hop, window=win,
                               center=True, return_complex=True)
                s.abs().pow(2).sum().backward()
            row["torch MPS"] = bench_mps(torch_fn, warmup=warmup, iters=iters)

        rows.append(row)
    _print_timing_table("STFT Forward + Backward", cols, rows)


def bench_istft_backward(warmup: int, iters: int) -> None:
    cols = ["mx.compile", "uncompiled"]
    if _HAS_TORCH:
        cols.append("torch MPS")

    rows: list[dict] = []
    for B, T, nfft, label in CONFIGS:
        hop = nfft // 4
        row: dict = {"label": label}

        # mlx-spectro
        t = SpectralTransform(n_fft=nfft, hop_length=hop, center=True)
        x_mlx = mx.random.normal((B, T))
        mx.eval(x_mlx)
        spec_mlx = t.differentiable_stft(x_mlx)
        mx.eval(spec_mlx)

        def ours_fn(spec=spec_mlx, t=t, T=T):
            def loss(z):
                y = t.differentiable_istft(z, length=T)
                return y.square().sum()
            g = mx.grad(loss)(spec)
            mx.eval(g)
        row["uncompiled"] = bench_mlx(ours_fn, warmup=warmup, iters=iters)

        # mx.compile variant
        def _istft_grad(z, t=t, T=T):
            def loss(zi):
                y = t.differentiable_istft(zi, length=T)
                return y.square().sum()
            return mx.grad(loss)(z)
        compiled_istft_grad = mx.compile(_istft_grad)

        def compiled_fn(spec=spec_mlx, fn=compiled_istft_grad):
            g = fn(spec)
            mx.eval(g)
        row["mx.compile"] = bench_mlx(compiled_fn, warmup=warmup, iters=iters)

        # torch MPS
        if _HAS_TORCH:
            win_mps = torch.hann_window(nfft, device="mps")
            x_mps = torch.randn(B, T, device="mps")
            spec0_mps = torch.stft(x_mps, n_fft=nfft, hop_length=hop,
                                    window=win_mps, center=True,
                                    return_complex=True).detach()

            def torch_fn(spec0=spec0_mps, nfft=nfft, hop=hop, win=win_mps, T=T):
                spec = spec0.requires_grad_(True)
                y = torch.istft(spec, n_fft=nfft, hop_length=hop, window=win,
                                center=True, length=T)
                y.pow(2).sum().backward()
            row["torch MPS"] = bench_mps(torch_fn, warmup=warmup, iters=iters)

        rows.append(row)
    _print_timing_table("ISTFT Forward + Backward", cols, rows)


def bench_roundtrip_backward(warmup: int, iters: int) -> None:
    cols = ["mlx-spectro"]
    if _HAS_TORCH:
        cols.append("torch MPS")

    rows: list[dict] = []
    for B, T, nfft, label in CONFIGS:
        hop = nfft // 4
        row: dict = {"label": label}

        # mlx-spectro
        t = SpectralTransform(n_fft=nfft, hop_length=hop, center=True)
        x_mlx = mx.random.normal((B, T))
        mx.eval(x_mlx)

        def ours_fn(x=x_mlx, t=t, T=T):
            def loss(xi):
                s = t.differentiable_stft(xi)
                y = t.differentiable_istft(s, length=T)
                return y.square().sum()
            g = mx.grad(loss)(x)
            mx.eval(g)
        row["mlx-spectro"] = bench_mlx(ours_fn, warmup=warmup, iters=iters)

        # torch MPS
        if _HAS_TORCH:
            win_mps = torch.hann_window(nfft, device="mps")

            def torch_fn(B=B, T=T, nfft=nfft, hop=hop, win=win_mps):
                x = torch.randn(B, T, device="mps", requires_grad=True)
                s = torch.stft(x, n_fft=nfft, hop_length=hop, window=win,
                               center=True, return_complex=True)
                y = torch.istft(s, n_fft=nfft, hop_length=hop, window=win,
                                center=True, length=T)
                y.pow(2).sum().backward()
            row["torch MPS"] = bench_mps(torch_fn, warmup=warmup, iters=iters)

        rows.append(row)
    _print_timing_table("Roundtrip (STFT → iSTFT) Forward + Backward", cols, rows)

# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------

def _print_timing_table(
    title: str,
    cols: list[str],
    rows: list[dict],
    *,
    compare_col: str | None = None,
) -> None:
    """Print a Markdown timing table with speedup ratios.

    Speedup columns show ``compare_col / col`` for every col != compare_col.
    If *compare_col* is ``None`` (default), speedup columns show
    ``other / first_col`` for every column after the first.
    """
    print(f"\n### {title}\n")

    # Determine which columns get a "vs" speedup column and what they compare against.
    if compare_col is not None:
        ratio_pairs = [(c, compare_col) for c in cols if c != compare_col]
    elif len(cols) > 1:
        ratio_pairs = [(c, cols[0]) for c in cols[1:]]
    else:
        ratio_pairs = []

    # Header
    hdr = f"| {'Config':<28} "
    for c in cols:
        hdr += f"| {c + ' (ms)':>18} "
    for c, ref in ratio_pairs:
        label = f"{c} speedup" if compare_col is not None else f"vs {c}"
        hdr += f"| {label:>20} "
    hdr += "|"
    print(hdr)

    # Separator
    sep = f"|{'-'*30}"
    for _ in cols:
        sep += f"|{'-'*20}"
    for _ in ratio_pairs:
        sep += f"|{'-'*22}"
    sep += "|"
    print(sep)

    # Rows
    for row in rows:
        line = f"| {row['label']:<28} "
        for c in cols:
            line += f"| {row[c]:>16.3f}ms "
        for c, ref in ratio_pairs:
            if compare_col is not None:
                # "compiled speedup": ref (uncompiled) / c (compiled) — >1 = faster
                speedup = row[ref] / max(row[c], 1e-6)
            else:
                # "vs mlx-spectro": c (other) / ref (ours) — >1 = we're faster
                speedup = row[c] / max(row[ref], 1e-6)
            line += f"| {speedup:>18.2f}x "
        line += "|"
        print(line)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="mlx-spectro benchmark suite")
    parser.add_argument("--quick", action="store_true",
                        help="Reduced iterations for fast sanity checks")
    parser.add_argument("--forward", action="store_true",
                        help="Run forward benchmarks only")
    parser.add_argument("--backward", action="store_true",
                        help="Run backward benchmarks only")
    args = parser.parse_args()

    warmup = 3 if args.quick else 5
    iters = 5 if args.quick else 20

    chip = platform.processor() or "unknown"
    mac = platform.mac_ver()[0]
    print(f"## mlx-spectro benchmarks")
    print(f"Machine: macOS {mac}, {chip}")
    print(f"MLX: {mx.__version__}")
    if _HAS_TORCH:
        print(f"PyTorch: {torch.__version__}")
    if _HAS_MLX_STFT:
        print(f"mlx-stft: {_mlx_stft.__version__}")
    avail = []
    if not _HAS_TORCH:
        avail.append("torch MPS (not available)")
    if not _HAS_MLX_STFT:
        avail.append("mlx-stft (not available)")
    if avail:
        print(f"Note: {', '.join(avail)}")
    print(f"Iterations: {iters} (warmup: {warmup})")

    run_forward = not args.backward
    run_backward = not args.forward

    if run_forward:
        bench_stft_forward(warmup, iters)
        bench_istft_forward(warmup, iters)
        bench_compiled_forward(warmup, iters)
        bench_accuracy(warmup, iters)

    if run_backward:
        bench_stft_backward(warmup, iters)
        bench_istft_backward(warmup, iters)
        bench_roundtrip_backward(warmup, iters)


if __name__ == "__main__":
    main()
