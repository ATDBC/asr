# Weighted AED Device Calibration Design

Date: 2026-09-17
Status: approved for implementation planning
Branch: `feature/server-weighted-aed-batching`

## Goal

Productize the existing weighted-duration AED experiment so that an operator can
run one representative, pre-segmented dataset on a new target device and obtain
the two device-specific scheduler coefficients:

```text
cost(batch) = max_duration(batch)
    * (longest_weight + padded_weight * item_count(batch))
```

The calibration is offline. Production loads an immutable JSON artifact before
the first AED request and never updates it from live jobs.

## Scope

The first version calibrates only `longest_weight` and `padded_weight`. It does
not estimate dual-worker contention, memory capacity, a safe batch-size limit,
or Punc/VAD cost. The existing 200-second total-duration and 32-item limits stay
fixed shape constraints, not memory-admission limits.

The calibration and production scheduler both use physical GPU0 as `cuda:0`.
They must not remap another GPU with `CUDA_VISIBLE_DEVICES`, inspect or manage
other projects' processes, read validation/blind labels, or modify historical
measurement outputs.

## Existing Reusable Interfaces

The input reuses the existing `metadata.source_inventory` representation from
the V3 train plan. Each source entry already contains:

- `audio_path`;
- `source_recording_group`;
- `segments`, with `index`, `start_ms`, and `end_ms`;
- `source_checksum`; and
- `segments_checksum`.

The current full train inventory contains 33 source groups and 1,112 segments.
Nine source groups contain at least 32 segments, so the current corpus can cover
the required item-count range without crossing source boundaries.

GPU execution reuses `FireRedWorkerClient.run_aed_plan()` and the worker's
existing explicit AED-plan action. That action already executes caller-owned
batches without VAD or Punc and returns AED time and project-worker CUDA peaks.
The calibrator must not introduce a second inference implementation.

## Command Interface

The product entry point is one resumable command:

```bash
.venv/bin/python tools/calibrate_weighted_aed.py \
  --manifest calibration-segments.json \
  --audio-root /data/calibration \
  --model-dir /models \
  --worker-python /opt/funasr-transcriber/.venv-asr/bin/python \
  --output outputs/device-calibration \
  --resume
```

`--manifest`, `--model-dir`, `--worker-python`, and `--output` are required.
`--audio-root` is required only when the manifest contains relative audio
paths. Absolute paths are used as written. The command accepts the established
plan-shaped JSON and reads only `metadata.source_inventory`; it does not read
transcripts, semantic labels, development validation, or blind holdout data.

The default is three measured repeats per case and BF16 through the existing
`use_half=true` FireRed path. Any future repeat or precision override must be an
explicit command-line choice recorded in the frozen plan and runtime identity.

## Preflight and Frozen Plan

All CPU-side validation and plan construction finish before a worker is
started:

1. Validate the manifest shape, unique source groups, segment indexes and
   bounds, referenced audio files, source checksums, and segment checksums.
2. Resolve relative audio paths against `--audio-root`.
3. Reject a segment longer than 200 seconds.
4. Build deterministic calibration cases within each source group.
5. Validate feature coverage and the rank of the two-column design matrix.
6. Atomically create `plan.json` with the input checksum, runtime request,
   exact case list, repeat count, and plan checksum.

The frozen plan cannot be overwritten. Starting without `--resume` fails when
the output directory already contains a plan. Resuming requires the manifest,
plan, requested precision, model identity, and runtime identity to match.

## Deterministic Case Construction

For every source group, sort valid segments by `(duration_ms, index)`. For each
target item count in `1, 2, 4, 8, 16, 32`, form up to three windows of that
length:

- the shortest window, starting at offset zero;
- the centered window, starting at `(n - count) // 2`; and
- the longest window, ending at `n`.

Deduplicate windows by their ordered segment indexes. Skip a window whose total
duration exceeds 200 seconds. A case never combines segments from different
source recordings.

Before GPU work the resulting plan must contain:

- at least ten distinct source groups;
- at least four distinct item-count levels;
- item count 1;
- at least one item-count level of 16 or greater; and
- a rank-two feature matrix for `max_duration` and
  `max_duration * item_count`.

The current inventory satisfies the source and high-item-count prerequisites.
The exact final case count is determined by the frozen selection rules rather
than hard-coded for this corpus.

## GPU Collection

The calibrator starts exactly one project-owned FireRed AED worker with VAD and
Punc disabled. It uses physical `cuda:0`; it neither starts a second AED worker
nor applies a free-memory budget.

The first valid planned case is executed once as a representative warm-up and
excluded from all records. Every planned case is then executed three times.
CUDA synchronization in the existing worker brackets the reported AED time.
Each completed case appends one JSONL record containing:

- case and source-group identity;
- segment indexes and frozen feature values;
- all repeat times and their median;
- output fingerprints;
- worker-reported allocated/reserved peaks; and
- completion or exact failure information.

All repeat fingerprints for a case must agree. OOM, fingerprint disagreement,
worker failure, non-finite timing, or an invalid worker response is recorded
and stops the run. No ready artifact is produced.

The command records device-level GPU0 snapshots at start and end. It may read
only its registered worker PID for per-process GPU memory and must never signal
or inspect another project's process.

## Resume Semantics

`records.jsonl` is append-only. A record is appended only after all repeats for
one case complete, so interruption during a case simply leaves it incomplete.
`--resume` validates the frozen plan and skips completed successful case IDs.
An existing failed case is a preserved terminal finding; resume reports it and
does not silently retry or replace it. A deliberate retry uses a new output
directory so the original evidence remains intact.

## Aggregation and Fitting

Each case contributes one label: the median of its three AED times. The fitted
model has no intercept and exactly two columns:

```text
x_longest = max_duration_seconds
x_padded  = max_duration_seconds * item_count
y         = median_actual_seconds
```

Fit ordinary least squares with the already-installed numerical stack. Reject
non-finite or non-positive coefficients instead of adding another model family
or a hidden fallback fit.

Repeat noise is reported separately from model error. For every repeat, compute
its absolute percentage deviation from the case median. Reuse the established
stability thresholds: median deviation at most 5% and P95 deviation at most
20%. Exceeding either threshold marks the calibration
`measurement_unstable` and prevents a ready artifact.

## Grouped Diagnostics

Cases are assigned deterministically to five folds by
`source_recording_group`. Every group remains wholly in one fold. For each fold,
fit the same two-coefficient model on the other groups and predict the held-out
groups. Record OOF MAE, median APE, P95 APE, and per-item-count summaries.

The first version reports OOF error but does not invent an unvalidated accuracy
release threshold. Readiness is governed by input coverage, full matrix rank,
repeat stability, fingerprint stability, zero execution failures, and finite
positive coefficients. After those checks pass, refit once on all aggregated
cases to create the final device coefficients.

## Output Layout

The output directory contains:

```text
device-calibration/
├── plan.json
├── records.jsonl
├── calibration-report.json
└── weighted-aed-calibration.json
```

`plan.json` is create-only and `records.jsonl` is append-only.
`calibration-report.json` is written atomically after complete evidence is
available. `weighted-aed-calibration.json` is written only when every readiness
check passes.

The final artifact contains:

- schema version, artifact kind, ready status, and exact formula identifier;
- `longest_weight` and `padded_weight`;
- physical device, actual Torch dtype, GPU name, compute capability, driver,
  CUDA and Torch versions;
- FireRed source commit and AED checkpoint SHA-256;
- manifest, plan, and records checksums;
- source-group, case, batch, and repeat counts;
- feature-support bounds, grouped-fold identity, repeat-noise results, and OOF
  diagnostics; and
- GPU0 start/end snapshots and zero failure/OOM/safety counts.

## Production Loading

Configuration gains one optional setting:

```toml
[asr]
aed_weight_artifact = "/etc/funasr/weighted-aed-calibration.json"
```

When unset, the scheduler continues to use the current built-in server weights.
When set, the service process loads and validates the artifact once before
constructing the weighted AED client. It passes one immutable pair of weights
to both global batch construction and the dual-lane `BatchCostModel`; AED worker
subprocesses do not load the artifact. Therefore all lanes and scheduler
diagnostics use the same coefficients.

The planner helpers accept an explicit weight pair while retaining the current
constants as defaults for compatibility with existing callers and tests. The
complete request plan remains frozen before the first AED request. Live batch
results never update the weights or replan the request.

## Artifact Validation and Fallback

The loader strictly validates:

- schema version, artifact kind, ready status, and formula identifier;
- finite positive weights;
- `cuda:0`, actual dtype, GPU name and compute capability;
- FireRed source identity and AED checkpoint SHA-256; and
- required evidence checksums and support metadata.

A strict mismatch emits an explicit warning and falls back to the built-in
weights. Driver, CUDA, and Torch patch-version changes are recorded as warnings
but do not alone reject an otherwise compatible artifact. Diagnostics record
the selected source (`artifact` or `builtin_fallback`), artifact checksum,
effective weights, and warnings.

The loader never edits source constants or production configuration. Operators
deploy the artifact and set the path explicitly.

## Capacity Boundary

The calibration deliberately has no memory ceiling or dynamic admission budget.
If the target device cannot execute a frozen case within the existing 32-item
and 200-second shape contract, an OOM is preserved as evidence and readiness
fails. Time coefficients cannot make an unsupported batch shape memory-safe;
changing shape limits or adding memory-aware scheduling is a separate task.

## Minimal Implementation Shape

Add:

- `tools/calibrate_weighted_aed.py` for preflight, frozen planning, collection,
  resume, grouped diagnostics, fitting, and artifact generation; and
- `src/funasr_transcriber/weighted_aed_calibration.py` for the small shared
  weight value, artifact validation, checksums, and pure two-column fit helpers.

Modify only the existing configuration, weighted planner/client, service
construction, and their focused tests. Reuse the current FireRed worker
protocol. Do not port the full V3 cost-model framework, introduce a general
calibration framework, add dependencies, or modify historical measurement
files.

## Verification

Unit tests cover:

- deterministic case construction and coverage rejection;
- two-coefficient recovery from synthetic timing rows;
- grouped folds never splitting a source recording;
- repeat-noise and invalid-coefficient rejection;
- artifact identity validation and built-in fallback;
- explicit weights reaching batch planning, lane estimates, and diagnostics;
- configuration parsing; and
- resume checksum and completed-case behavior.

A CPU-only command-level test builds a small synthetic manifest and records file
and verifies the final fit/report path without starting a worker. Repository
completion checks are:

```bash
.venv/bin/python -m pytest -q tests/unit/test_config.py tests/unit/test_firered_client.py
.venv/bin/python -m pytest -q
git diff --check
```

GPU acceptance uses the existing train source inventory on physical GPU0 and
records fresh command output, actual dtype, runtime identity, checksums, GPU0
snapshots, failures, OOMs, and generated weights. It then starts the weighted
client with that artifact and verifies that the same segment input produces a
stable frozen plan across service restarts. The generated calibration output is
not committed and production configuration is not changed automatically.
