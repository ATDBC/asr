# Standalone Transcriber Agent Notes

## Scope

This repository is the standalone FireRedASR2S transcription service. Work directly on
the current branch; do not create worktrees or delegate work to sub-agents unless the
user explicitly requests it.

## Production architecture

- Production is pinned to physical GPU 0 (`cuda:0`).
- Production defaults to two resident FireRed AED workers and one shared FireRedPunc model.
  Worker 0 also hosts the single FireRedVAD helper; worker 1 loads AED only.
- FireRedVAD is functionally part of speaker separation. It is currently hosted by the
  same `.venv-asr` subprocess for dependency and lifecycle isolation, but it must not be
  described as the AED/Punc worker's functional stage.
- 3D-Speaker diarization runs before normal ASR. Long speaker intervals use the hosted
  FireRedVAD helper, intersected with the speaker interval, before AED/Punc decoding.
- The AED scheduler is memory-first: post-warm-up free GPU memory minus the configured
  safety margin determines exact batch membership. Batch cardinality is never a fixed
  production setting. The complete plan is immutable before the first AED request.
- Production ASR defaults to `use_half = true`. On the current RTX 3090 this selects
  `torch.bfloat16` in the pinned upstream FireRed implementation. `false` is the FP32
  fallback; do not silently change the default back to FP32.
- Overlap is opt-in. When enabled, close the resident FireRed/3D-Speaker owners before
  starting pyannote/ClearVoice and reload the resident owners before the next queued job.
- Resident workers are warmed up after loading. Job timing starts after warm-up; resource
  profiling records total GPU peak and subtracts the initial other-project baseline.

## Safety constraints

- Never stop, pause, migrate, inspect in detail, or otherwise manage another project's
  process. `nvidia-smi` may be used read-only, and resource sampling must filter to this
  project's registered PIDs.
- Keep GPU work on physical GPU 0. Do not use `CUDA_VISIBLE_DEVICES` to remap another GPU
  to logical `cuda:0`.
- Keep `artifacts/release_validation_corpus` ignored by Git. Production outputs belong in
  dedicated output directories, not in the corpus artifacts.
- Before destructive operations, resolve exact targets and prefer recoverable actions.

## Verification

Run focused tests while iterating, then the full suite:

```bash
.venv/bin/python -m pytest -q tests/unit/test_config.py tests/unit/test_firered_client.py
.venv/bin/python -m pytest -q
git diff --check
```

For GPU measurements, exclude model loading and explicit warm-up, record the precision
actually used (`torch.float32`, `torch.bfloat16`, or `torch.float16`), and save reports below
`outputs/measurements/`. Do not claim a benchmark result without fresh command output.

## Commits

Use the repository's existing user identity and make a focused commit after tests pass.
Do not rewrite or discard unrelated user changes in a dirty worktree.
