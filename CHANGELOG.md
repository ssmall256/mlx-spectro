# Changelog

## 0.7.0

- Added `RepeatedShapeCompileCache` for bounded repeated-shape promotion to compiled mode in wrapper code.
- Completed the single-output frontend compiled contract across mel, log-mel, MFCC, filtered spectrograms, feature bundles, and hybrid CQT.
- Added machine-readable benchmark output plus checked-in quick baselines for frontend, feature-bundle, and hybrid-CQT benchmarks.
- Added `scripts/check_benchmark_regressions.py` for quick local regression checks against the baseline benchmark set.
- Clarified README guidance for choosing eager mode, direct `get_compiled()`, and repeated-shape compile caches.
