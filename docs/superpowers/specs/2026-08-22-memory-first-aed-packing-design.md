# Memory-First AED Packing Design

## Goal

Replace the old fixed-batch AED scheduler with a memory-first planner that
chooses the exact segment membership of every batch from the real post-warm-up
GPU 0 budget.

## Removed controls

The production configuration and worker protocol no longer expose these AED
controls:

- `segment_batch_size`;
- `segment_frame_budget`;
- `segment_memory_limit_mib`;
- `segment_memory_safety_mib`.

They are removed from `AppConfig`, TOML defaults, CLI arguments, worker client
construction, doctor construction, and diagnostics. AED planning has one
resource control: `gpu_safety_margin_mib`, which defaults to `1536 MiB`.
Punctuation uses its own internal batching and never reuses an AED batch
setting.

## Resource snapshot

After both resident AED workers have loaded and warmed, and after diarization
plus the global speaker-separation FireRedVAD refinement, the owner samples
physical GPU 0:

```text
shared_transient_budget = gpu_free_mib - gpu_safety_margin_mib
```

The snapshot also records each owned AED process's warmed resident allocation.
The planner predicts only transient batch growth above that worker's own
resident baseline. No process belonging to another project is enumerated,
stopped, paused, or modified.

## Memory-first batch construction

Every refined segment is represented by its original index, start/end time,
duration, and estimated feature frames. The planner sorts segments by duration
and greedily fills a candidate batch with the next shortest compatible segment.
The candidate is closed before adding the next segment when its predicted
transient allocation would violate the lane budget. The process repeats until
all segments are assigned. Thus batch cardinality is an output, not an input;
short segments naturally form larger batches and long segments form smaller
ones.

The prediction model uses padded feature frames and item count for time and
transient memory. It is parameterized by each worker's warmed resident memory,
not by a hard-coded `5000 MiB` process baseline. Observed per-batch allocated
and reserved peaks remain diagnostics and calibration evidence only; they never
repartition an active plan.

## Two-worker lane assignment

The complete batch list is created before AED starts. A deterministic
longest-processing-time assignment places complete batches into two worker
lanes while minimizing predicted makespan. The assignment must satisfy:

```text
max(batch_increment_lane_0)
+ max(batch_increment_lane_1)
<= shared_transient_budget
```

If a lane split cannot satisfy that inequality, the planner reduces candidate
batch sizes and rebuilds the complete plan before any AED request. It never
splits a batch after dispatch begins.

## Dispatch safety

Immediately before dispatch, the owner samples GPU 0 again. If the fixed pair
of lanes fits, AED workers run concurrently. If only one lane is safe, the
same fixed lanes run serially. If even the largest single planned batch cannot
fit, the request fails before AED. A changed runtime snapshot can gate
execution but cannot change batch membership or output ordering.

## Diagnostics and verification

`diagnostics.json` records the post-warm-up snapshot, safety margin, shared
budget, every batch's exact indices and predicted costs, lane assignment,
dispatch mode, and per-batch actual/predicted error. The formal GPU0 benchmark
uses `audio_test/刘洋洋.wav`, excludes model loading and explicit warm-up from
stage timing, verifies transcript fingerprint stability, and reports whether
the observed peak approaches the configured safe budget.

## Error handling

An individual segment that cannot fit even as a singleton produces a clear
capacity error before any AED request. Invalid or missing GPU snapshots fail
closed. Worker failures remain fatal for the complete normal job; no partial
transcript is published.
