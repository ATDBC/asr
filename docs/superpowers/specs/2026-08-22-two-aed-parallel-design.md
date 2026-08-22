# Two-AED Parallel Inference Design

## Goal

Allow the resident service to use two BF16 FireRed AED instances for one normal
job while keeping exactly one Punc model and one global FireRedVAD pass.

## Architecture

After diarization, worker 0 refines all long speaker intervals with the global
FireRedVAD. The main process creates one complete duration/frame/memory batch
plan, then partitions complete batches between two AED workers using deterministic
longest-processing-time scheduling. Both workers execute their assigned fixed
plans without replanning or punctuation. Their segment-level AED results are
merged by original segment index and sent to the single Punc owner in worker 0.

Worker 0 loads VAD, AED, and Punc; worker 1 loads AED only. The public service
owner exposes the same `transcribe` contract as the former single client. Partial
responses never determine output order; `uttid` and original indices are the
authoritative join keys. If either AED fails, the complete normal job fails.

## Configuration and safety

`[asr].aed_worker_count` defaults to `2`; it must be positive. The service keeps
the physical `cuda:0` restriction and performs the existing GPU admission check
before loading both workers. The configured per-worker batch/frame/memory limits
remain unchanged. Two workers are not started when the existing capacity gate
fails. Overlap still drains and closes the entire parallel AED owner before
starting pyannote/ClearVoice.

## Scheduling

`AdaptiveBatchScheduler.plan()` runs once over all refined segments. A pure
partition helper assigns complete planned batches in descending predicted runtime
to the worker with the smaller accumulated predicted runtime; a deterministic
pairwise-improvement pass may exchange batches when it reduces the imbalance.
Each worker receives the exact list of planned batches, including original index,
start, and end timestamps. Runtime observations are diagnostic only.

## Protocol

The worker protocol gains `refine_vad`, explicit `batch_plan` execution, and
`punctuate` actions. AED requests can disable punctuation for that request. The
stable response remains unchanged; scheduler and parallel-owner metadata are
diagnostic extensions only.

## Failure handling and compatibility

The single-worker client remains available for doctor, overlap, and tests. Existing
FIFO behavior, output directories, diagnostics opt-in, GPU safety, and output
schemas remain unchanged. A worker shutdown is limited to subprocesses owned by
this service.

## Testing

Unit tests cover deterministic two-way partitioning, fixed-plan preservation,
worker request serialization, partial-result merging, and host lifecycle. The
full suite must pass before enabling the new default.
