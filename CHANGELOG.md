# Changelog

## 0.9.0

### Fixed

- **iSTFT gradients were wrong for any batch size greater than 1.** `_unpad_cotangent`,
  the adjoint of `_trim_ola_output`, placed the cotangent with
  `grad_ola.at[:, a:b].add(...)` — a strided slice scatter-add on a non-leading axis.
  MLX before 0.32.0 mis-linearizes the 2-D dispatch grid in its Metal
  `slice_update_op_impl` kernel, so rows aliased onto each other in a non-atomic
  read-modify-write. Measured relative gradient error was 1.28 at B=2 and 1.47 at B=8 —
  not imprecision, wrong values — and it was completely silent. B=1 was unaffected,
  which is why it went unnoticed. Now uses `mx.slice_update`, which is exact on every
  supported MLX version. Inference paths were never affected.
- A failed NOLA check no longer returns silently. The overlap-add envelope minimum was
  computed, cached, and then ignored unless `torch_like=True`, which is not the default —
  so a degenerate `n_fft`/`hop`/`window` combination silently emitted exact zeros
  wherever the envelope fell below 1e-11, putting unannounced gaps in the reconstruction.
  It now warns by default, once per transform configuration, and still raises under
  `torch_like=True`.
- The pure-MLX overlap-add fallbacks accumulated in the input dtype, so fp16/bf16 input
  accumulated in half precision while the Metal kernels they stand in for accumulate in
  float32. Precision no longer depends on whether a kernel happened to compile.
- A Metal kernel that fails to compile now warns once instead of silently latching a
  permanent, slower, differently-rounded pure-MLX path.

### Changed

- Minimum MLX raised to 0.31.2, matching the rest of the MLX audio stack.

### Tests

- Batched finite-difference gradient checks for `differentiable_istft` at B ∈ {1, 2, 4},
  plus a per-sample independence invariant. The previous suite checked gradients densely
  only at B=1 and asserted merely shape and finiteness at B=4, so it passed with garbage.
- Direct `_unpad_cotangent` checks against a numpy reference across the full
  `center` × `length` × batch matrix.
- NOLA warning/raise behavior and fp16 fallback accumulator precision.

## 0.8.0

### Added

- librosa-compatible VQT/CQT (MLX-native), `mel_filterbank_librosa`, madmom-compat
  presets and onset ODF helpers, `nnaudio_cqt_kernels`, and spectral feature frontends.
  (This release was tagged without a changelog entry; recorded here retroactively.)


## 0.7.0

- Added `RepeatedShapeCompileCache` for bounded repeated-shape promotion to compiled mode in wrapper code.
- Completed the single-output frontend compiled contract across mel, log-mel, MFCC, filtered spectrograms, feature bundles, and hybrid CQT.
- Added machine-readable benchmark output plus checked-in quick baselines for frontend, feature-bundle, and hybrid-CQT benchmarks.
- Added `scripts/check_benchmark_regressions.py` for quick local regression checks against the baseline benchmark set.
- Clarified README guidance for choosing eager mode, direct `get_compiled()`, and repeated-shape compile caches.
