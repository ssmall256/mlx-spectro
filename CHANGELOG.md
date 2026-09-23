# Changelog

## 0.9.4

### Fixed

- Hybrid-CQT snapshot tests compared float32 aggregates with an absolute-only
  tolerance. `sum` is ~10^3 here, so `atol=1e-6` demanded agreement three orders
  of magnitude tighter than float32 can express, and CI failed on relative
  differences of about three ULPs when a different GPU reassociated the
  reduction. Now `rtol=1e-5` with an absolute floor, which a real change to the
  transform still trips by two orders of magnitude.
- `uv.lock` refreshed; it still described the project as 0.7.0 and required
  `mlx>=0.30.3`, both superseded. It ships in the sdist.

## 0.9.3

### Changed

- README documents the `threadgroup autotuning ... was skipped` warning added
  in 0.9.2, alongside the NOLA and Metal-compile ones.

## 0.9.2

### Fixed

- **A compiled `stft` or `istft` raised on a machine with no tuning cache yet.**
  Threadgroup autotuning picks a size by timing candidates, and every timing run
  calls `mx.eval`, which MLX refuses inside `mx.compile` or `vmap`. Every
  candidate therefore failed for the same reason and 0.9.0's "no usable
  threadgroup size" error fired -- correct for a kernel that genuinely does not
  run, wrong when the only problem is that timing is impossible. Tuning is now
  skipped under a trace: the call runs at the default threadgroup size, warns
  once per `(kernel, n_fft, hop)`, and records nothing, so a later eager call
  still measures and caches a real value. Affected `mx.compile` around `stft` or
  `istft`, `get_compiled_stft` and `get_compiled_istft`;
  `compiled_pair`/`compiled_pair_nd` were never affected, because they call the
  transform eagerly once before compiling -- which is also the remedy the
  warning names.

## 0.9.1

### Changed

- `mel_filterbank_librosa` defaults to `htk=False`, matching
  `librosa.filters.mel`, so passing defaults on both sides now agrees to 1e-9.
  It is verified against librosa across three configurations and both `htk`
  settings. Note it is not interchangeable with `mel_filterbank`, which follows
  madmom's convention: area-normalized triangles peaking at 1.0 against
  Slaney-normalized ones peaking near 0.01, roughly 19 dB downstream.
- `fft_frequencies` raises on librosa's argument order rather than returning a
  wrong-length array. It takes `(num_fft_bins, sample_rate)` — the opposite
  order from `librosa.fft_frequencies`, and a bin count rather than `n_fft` —
  and the error names the correct call.
- `vqt` refuses batched input with a message naming the constraint, and its
  `NotImplementedError` for unsupported parameters names `hybrid_cqt` as the
  alternative.
- Release workflows set the version from the workflow input before building,
  wait up to 10 minutes for a new file to reach every CDN mirror, and retry the
  install.

## 0.9.0

### Added

- `blackman` window in `make_window`, using the classic 0.42/0.5/0.08 coefficients that
  `numpy.blackman`, `scipy.signal` and `torch.blackman_window` all use. Honours
  `periodic` the same way `hann` and `hamming` do; matches an analytic reference and
  `numpy.blackman` to float32 precision (~3e-7).
- Two-sided spectra via `onesided=False` on `SpectralTransform` and
  `get_transform_mlx`, emitting all `n_fft` bins instead of `n_fft // 2 + 1`. Matches
  `torch.stft(..., onesided=False)` to ~8e-6, preserves Hermitian symmetry for real
  input, and round-trips through `istft` to ~1.5e-6. `onesided` is part of the transform
  cache key, so one-sided and two-sided transforms of the same shape do not collide.
  `istft` accepts either bin count and infers which it was given; the existing bin-count
  guard now names both widths when it rejects an input.

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

- **`compiled_pair` and `compiled_pair_nd` now default to the `"bfn"` layout**, matching
  the other six STFT entry points. Previously they alone defaulted to `"bnf"`, so two
  halves of one API returned different axis orders for a caller who passed nothing.
  Every real caller of the compiled half already overrode it back to `"bfn"` — the
  clearest evidence available that the split was a defect, not a choice. `"bfn"` also
  suits MLX's channel-last grain and is what spectrogram models consume directly.
  `layout="bnf"` remains available and is unchanged.

  Newly pinned by tests: nothing previously asserted any entry point's default layout,
  so a change to one — or a regression — would have shipped green.

- Minimum MLX raised to 0.31.2, matching the rest of the MLX audio stack.
- `hybrid_cqt`, `HybridCQTTransform` and `build_vqt_plan` accept `sample_rate` as
  well as `sr`. They mirror librosa, which spells it `sr`, while every transform
  class here spells it `sample_rate`; callers should not have to remember which
  side of that line a function sits on. Passing both with different values raises.
  Defaults and the `sr` spelling are unchanged.
- Benchmark scripts write their JSON next to themselves instead of to a path
  relative to the caller's working directory, and no longer carry an absolute
  developer path in their docstrings.
- **`istft()` raised for any configuration with `hop_length > n_fft`.** 0.8.0 made
  the Metal unroll factor a computed template constant, `min(FRAME/HOP, 8)`, which
  is 0 when the hop exceeds the frame. Metal rejects `#pragma unroll 0`, so all
  three iSTFT kernels failed to build and the call raised
  `Unable to build metal library from source`. Non-overlapping frames are unusual
  but legal and worked in 0.7.0. Clamped to 1; output is bit-identical to 0.7.0 at
  every frame/hop combination tested.
- `positive_spectral_diff` truncated a float `hop_size` to int before passing it
  on, so the float-hop support added alongside it never reached that path.
- The kernel autotuner cached an untested default and returned successfully when
  every candidate threadgroup size failed, so a broken kernel surfaced later from an
  unrelated line. That is what turned the unroll bug above into a mysterious
  `istft` crash. It now raises, naming the kernel and the configuration, and caches
  nothing.

### Tests

- CI installs a new `parity` extra (librosa, torch, torchaudio, scipy, soxr). It
  previously installed only `dev`, so all 22 cross-framework parity tests skipped
  and the suite was green whether or not parity held. `test_librosa_cqt.py` — the
  only coverage the new VQT has — never executed anywhere, and
  `mel_filterbank_librosa` shipped with no test at all.
- `hop_length > n_fft` reconstruction, and `_unroll_k` clamping across the
  frame/hop matrix.
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
  59 new names at package level; nothing was removed or renamed.
  (Recorded here retroactively. 0.8.0 was never tagged or published, so these
  changes reach PyPI for the first time in 0.9.0.)
- `vqt` and `build_vqt_plan` cover the parameter sets that need no early
  downsampling, and raise `NotImplementedError` otherwise — `sr=44100` with
  `hop_length=512`, `n_bins=84`, `bins_per_octave=12` is one that does. Use
  `hybrid_cqt` for the general case. `vqt` takes a 1-D signal, not a batch.
- Two mel filterbanks now exist and they are not interchangeable:
  `mel_filterbank` (madmom convention, area-normalized) and
  `mel_filterbank_librosa` (librosa convention). Filter magnitudes differ by
  roughly 87x, about 19 dB. Note `mel_filterbank_librosa` defaults to
  `htk=True` while `librosa.filters.mel` defaults to `htk=False`, so passing
  defaults on both sides does not match.

### Changed

- `SpectralTransform.istft` now emits a `DeprecationWarning` for
  `long_mode_strategy` values other than `"native"` and for
  `backend_policy="torch_fallback"`. Both paths still run unchanged; only the
  warning is new.
- iSTFT Metal kernels take the unroll factor from the overlap ratio
  (`UNROLL_K`) instead of a fixed literal, and the STFT frame-extract kernel
  autotunes its threadgroup size on first use per `(kernel, n_fft, hop)`,
  caching the result to disk. First call on a cold cache is slower; steady
  state is not.


## 0.7.0

- Added `RepeatedShapeCompileCache` for bounded repeated-shape promotion to compiled mode in wrapper code.
- Completed the single-output frontend compiled contract across mel, log-mel, MFCC, filtered spectrograms, feature bundles, and hybrid CQT.
- Added machine-readable benchmark output plus checked-in quick baselines for frontend, feature-bundle, and hybrid-CQT benchmarks.
- Added `scripts/check_benchmark_regressions.py` for quick local regression checks against the baseline benchmark set.
- Clarified README guidance for choosing eager mode, direct `get_compiled()`, and repeated-shape compile caches.
