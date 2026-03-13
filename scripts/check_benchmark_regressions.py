#!/usr/bin/env python3
"""Compare quick benchmark results against checked-in baselines."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = REPO_ROOT / "benchmarks" / "baselines"
SCRIPT_SPECS = {
    "frontends": ("scripts/benchmark_frontends.py", BASELINE_DIR / "frontends_quick.json"),
    "features": ("scripts/benchmark_features.py", BASELINE_DIR / "features_quick.json"),
    "hybrid_cqt": ("scripts/benchmark_hybrid_cqt.py", BASELINE_DIR / "hybrid_cqt_quick.json"),
}


def _run_json(script_rel: str) -> dict:
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / script_rel), "--quick", "--json"],
        check=True,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    return json.loads(proc.stdout)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _flatten(payload: dict) -> dict[str, float]:
    bench = payload["benchmark"]
    flat: dict[str, float] = {}
    if bench == "frontends":
        for family, rows in payload["families"].items():
            for row in rows:
                for key in ("eager", "compiled"):
                    flat[f"{family}:{row['label']}:{key}"] = float(row[key])
    elif bench == "features":
        for row in payload["per_feature"]:
            for key in ("eager", "compiled"):
                flat[f"per_feature:{row['label']}:{key}"] = float(row[key])
        for row in payload["bundle"]:
            for key in (
                "sequential eager",
                "shared eager",
                "cached eager",
                "shared compiled",
                "cached compiled",
            ):
                flat[f"bundle:{row['label']}:{key}"] = float(row[key])
    elif bench == "hybrid_cqt":
        for row in payload["cases"]:
            label = f"batch={row['batch']} samples={row['samples']}"
            for key in (
                "cold_cached_transform",
                "wrapper_one_off",
                "cached_transform",
                "compiled_transform",
            ):
                flat[f"hybrid:{label}:{key}"] = float(row[key])
    else:
        raise ValueError(f"Unsupported benchmark payload: {bench}")
    return flat


def main() -> None:
    parser = argparse.ArgumentParser(description="Check quick benchmark regressions against baselines")
    parser.add_argument(
        "--allowed-regression",
        type=float,
        default=0.20,
        help="maximum allowed latency regression ratio, e.g. 0.20 allows 20%% slower",
    )
    parser.add_argument(
        "--absolute-slack-ms",
        type=float,
        default=1.0,
        help="extra absolute latency slack in milliseconds to absorb normal benchmark noise",
    )
    args = parser.parse_args()

    failures: list[str] = []
    for name, (script_rel, baseline_path) in SCRIPT_SPECS.items():
        baseline = _load_json(baseline_path)
        current = _run_json(script_rel)
        if baseline["meta"].get("device") != current["meta"].get("device"):
            print(
                f"[warn] {name}: baseline device {baseline['meta'].get('device')} "
                f"!= current device {current['meta'].get('device')}"
            )
        base_flat = _flatten(baseline)
        cur_flat = _flatten(current)
        for metric, base_value in sorted(base_flat.items()):
            if metric not in cur_flat:
                failures.append(f"{name}: missing metric {metric}")
                continue
            cur_value = cur_flat[metric]
            allowed = (base_value * (1.0 + args.allowed_regression)) + args.absolute_slack_ms
            if cur_value > allowed:
                failures.append(
                    f"{name}: {metric} regressed from {base_value:.3f} ms to {cur_value:.3f} ms "
                    f"(allowed <= {allowed:.3f} ms)"
                )

    if failures:
        print("Benchmark regression check failed:")
        for item in failures:
            print(f"- {item}")
        raise SystemExit(1)

    print("Benchmark regression check passed.")


if __name__ == "__main__":
    main()
