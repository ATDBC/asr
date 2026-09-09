# Weighted Dual-AED with Streaming Punc Design

## Goal

Extend the fixed weighted-duration experiment from one AED lane to two AED
lanes on physical GPU0. Both lanes must use the same immutable, memory-agnostic
batch plan, while one independent Punc owner consumes completed AED batches.

This is an experimental strategy. It does not replace the production
memory-first scheduler or change its cost-model lifecycle.

## Non-goals

- Do not use free GPU memory, a predicted memory limit, or an admission budget
  to form batches or decide whether the two AED lanes run concurrently.
- Do not dynamically steal batches, replan after dispatch, or use completed
  batch observations to change the current request.
- Do not train a candidate, read development validation or blind holdout data,
  publish a production manifest, or write `models/cost_models/`.
- Do not add a new general scheduling framework.

## Runtime topology

The weighted client accepts either one or two AED workers:

- AED worker-0 hosts the single FireRedVAD helper and one AED model.
- AED worker-1, when selected, hosts only a second AED model.
- The independent Punc owner hosts only FireRedPunc.

FireRedVAD refines the speaker-derived intervals once on worker-0 before any AED
request. With two workers, each worker receives one complete fixed lane. A
failure in either lane fails the whole request. Only project-owned workers may
be started, inspected, cleaned, or stopped.

## Global batch construction

Batch construction remains identical to the validated single-AED experiment:

- sort refined segments by descending duration, with original index as the
  deterministic tie-breaker;
- place each segment into the tightest existing batch that remains within 200
  seconds total duration and 32 items, otherwise create a new batch;
- repeatedly apply the best deterministic single-item relocation that reduces
  the sum of predicted batch costs;
- keep each completed batch indivisible after construction.

The predicted cost of batch `B` remains:

```text
max_duration(B) * (longest_weight + padded_weight * item_count(B))
```

The 200-second and 32-item constraints remain shape constraints, not GPU-memory
admission limits.

## Two-lane partition

After constructing the global batches, partition complete batches using the
existing deterministic longest-processing-time helper:

1. Order batches by descending predicted cost.
2. Assign each batch to the lane with the lower accumulated predicted cost;
   ties go to the lower worker index.
3. Apply the existing deterministic single-batch relocation while it reduces
   the predicted makespan.
4. Restore the global plan order inside each lane. Because the global plan is
   ordered by increasing maximum segment duration, both lanes retain the same
   short-to-long execution policy.

The objective is to minimize the larger predicted lane total, not the sum of
lane totals. Dynamic worker-side task stealing is excluded because it would
make the per-lane plan and fingerprint dependent on runtime completion order.

## AED and Punc overlap

Reuse the existing streaming execution path. Both AED lanes run concurrently.
Each completed AED batch is placed on the bounded Punc queue immediately; the
single Punc owner may punctuate it while either AED lane continues. The Punc
consumer terminates only after receiving one completion sentinel from each AED
lane. Final records are assembled by original segment index, never by completion
order.

The reported AED wall time ends when both AED lanes finish, before waiting for
the final Punc tail. Diagnostics retain per-lane batch timings, Punc model time,
queue wall time, tail time, and allocator peaks for both AED and Punc owners.

## Weight policy

The first dual-AED implementation keeps the current train-only coefficients:

```text
longest_weight = 0.12854663332839417
padded_weight  = 0.008409640427755745
```

They were fitted from 726 repeat-median solo GPU0 BF16 train batches with no
development-validation or blind-holdout reads. For batching and lane
partitioning, their ratio matters; a uniform concurrent slowdown that scales
both coefficients does not change the resulting plan.

Do not refit from the one-hour benchmark. First record dual-lane prediction
errors and actual lane completion imbalance. A later refit is justified only if
concurrent execution changes the relative longest-versus-padded sensitivity,
not merely the absolute runtime scale. Any such refit must be a separate,
approved train-only task using paired sequence evidence, repeat aggregation,
and original-parent-component grouping. Peer-shape effects that cannot be
represented by these two coefficients require a separate contextual correction
rather than contaminating the two intrinsic workload weights.

## Configuration and compatibility

Add one explicit scheduler selector so worker count never silently changes the
meaning of an existing production configuration:

```toml
[asr]
aed_scheduler = "weighted_duration"
aed_worker_count = 2
```

The supported combinations are:

- `weighted_duration` with one or two AED workers;
- `memory_first` with two AED workers.

The experimental branch uses `weighted_duration` in its benchmark
configuration. Production continues to select `memory_first`; this experiment
must not silently replace that routing. An unsupported scheduler name or
`memory_first` with one worker fails configuration validation.

## Implementation shape

Generalize the current single-lane weighted client to own one or two AED
clients. It continues to call `plan_aed_batches()`, then calls the existing
`partition_batches()` for two workers, and finally calls the existing streaming
executor. Warmup, status, PID reporting, cache cleanup, and scheduler diagnostics
derive their worker count from that client list.

The service factory selects the weighted client only for
`aed_scheduler = "weighted_duration"`. It constructs one VAD/AED worker and
one Punc owner for the single mode; for the dual mode it additionally constructs
one AED-only worker. The existing parallel memory-first client remains selected
for `aed_scheduler = "memory_first"`. Neither weighted mode calls the GPU
memory provider or performs memory-based serial fallback. Physical GPU0
enforcement remains unchanged.

## Verification

Unit tests must prove:

- deterministic batch boundaries still satisfy the 200-second and 32-item
  limits;
- two-lane partitioning preserves every batch and every original segment
  exactly once;
- both AED lanes stream into one Punc owner and Punc overlaps unfinished AED;
- weighted dual mode never calls GPU-memory planning or selects serial dispatch;
- worker-0 owns VAD, worker-1 is AED-only, and Punc has a separate PID;
- one-lane weighted behavior remains unchanged;
- status, warmup, cache cleanup, failure propagation, and diagnostics cover the
  selected worker count.

Before commit, run the repository-required focused tests, full test suite, and
`git diff --check`.

## Controlled GPU benchmark

After unit verification, run a new one-hour benchmark on physical GPU0 using
the same audio, runtime identity, BF16 precision, timing contract, and fixed
weights as the single-lane experiment. Record one fresh run and at least three
steady runs, per-lane actual timings, lane imbalance, Punc tail, text checksum,
OOM/warnings, and GPU0 admission/peak/end snapshots.

The benchmark must only inspect or manage this project's processes. GPU device
sampling must follow the active task's exact allowed GPU0 query. The service is
stopped after the benchmark. Results are compared with the current weighted
single-AED result and historical dual-AED evidence, without using benchmark
output to refit weights in the same task.
