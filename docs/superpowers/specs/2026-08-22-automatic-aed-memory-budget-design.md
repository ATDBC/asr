# Automatic AED Memory Budget Design

## Goal

Size the two resident BF16 AED workers' variable-cardinality batches from the
memory actually free on physical GPU 0 after model warm-up, while preserving a
fixed precomputed plan and measuring whether the duration-based time and memory
predictions match real execution.

## Memory model and configuration

`[asr].gpu_safety_margin_mib` defaults to `1536`. Immediately after
diarization and the global FireRedVAD refinement, the parallel AED owner reads
GPU 0 free memory and its own two worker process allocations. The shared
transient budget is:

```text
post_warmup_free_mib - gpu_safety_margin_mib
```

Only incremental batch memory participates in this budget. A batch's predicted
increment is the predicted worker peak minus that worker's warmed resident
allocation. `segment_memory_limit_mib` remains only as a broad per-worker
emergency ceiling; it is not the primary planner input.

The default margin is deliberately larger than 100 MiB because CUDA caching,
driver sampling granularity, short-lived kernels, and growth by an unrelated
GPU user can all make a 100 MiB margin unsafe. The implementation reads only
GPU totals and this service's registered worker PIDs and never manages another
project's processes.

## Fixed planning and dispatch

The planner groups duration-sorted segments up to `segment_batch_size` and
`segment_frame_budget`, shrinking batches when two predicted concurrent
increments would exceed the shared budget. Complete batches are assigned to
the two workers by deterministic longest-processing-time partitioning. Batch
membership, lane assignment, and lane order are immutable before the first AED
batch starts.

The owner samples free memory again immediately before dispatch. If the planned
concurrent peak still fits, both fixed worker plans run concurrently. If only
one lane fits, the same plans run serially. If one lane cannot fit, the job
fails before AED with a capacity error. This safety check gates execution but
never replans output order or batch membership.

## Prediction diagnostics

The BF16 cost model predicts each batch from padded feature frames and item
count. Each worker returns per-batch elapsed time and CUDA peak allocated and
reserved memory. Diagnostics join observations to the immutable estimates by
worker and batch position and report absolute error and percentage error for:

- batch elapsed seconds;
- incremental reserved memory above the worker's warmed baseline;
- worker total predicted versus actual AED time;
- planned concurrent increment versus the observed service peak.

Measurements are observational and do not update or reorder the active job's
plan. A later calibration may update production coefficients only from a
separate reviewed benchmark.

## BF16 timestamp compatibility

The installed torchaudio forced-alignment kernel rejects BF16 log probabilities.
A project-owned compatibility wrapper converts only BF16 `log_probs` to FP32 at
the `torchaudio.functional.forced_align` boundary. The AED encoder and decoder
remain BF16, targets are unchanged, and upstream model files under ignored
`models/` are not edited.

## Verification

Unit tests cover budget derivation, concurrent-fit planning, serial fallback,
diagnostic joins, configuration validation, and BF16-only forced-align casting.
After the full suite passes, a production-shaped GPU 0 run on
`audio_test/刘洋洋.wav` records post-warm-up stage times, total and owned GPU
peaks, every AED batch observation, and aggregate prediction error under
`outputs/measurements/`.
