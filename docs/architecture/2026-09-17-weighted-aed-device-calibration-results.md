# Weighted AED device calibration is ready on the RTX 3090

Date: 2026-09-17
Branch: `feature/server-weighted-aed-batching`
Calibration code commit: `618d3d59523860a7512e951f21a3d1d441811b5e`

## Executive summary

The device calibration completed all 220 cases and produced a ready artifact for the
server's physical GPU0. The fitted scheduler coefficients are:

```text
longest_weight = 0.13321063731004226
padded_weight  = 0.0063924778637545285
```

All 660 measured batches completed without failure, OOM, safety violation, invalid
response, or output-fingerprint disagreement. One excluded warm-up was run before the
three measured repeats of each case. Repeat noise passed comfortably: median deviation
was 0.0711% and P95 was 6.2087%, against limits of 5% and 20%.

The artifact loaded successfully across two independent service restarts. Both runs
used the artifact rather than fallback, produced identical frozen scheduler fields,
and produced byte-identical transcripts. This validates artifact identity and restart
determinism; it does not turn the two-coefficient model into a memory-capacity or
dual-worker-contention model.

## Evidence scope and collection protocol

The input was only
`outputs/measurements/20260827T-firered-cost-model-v3/full/train-plan.json`, and the
calibrator read only its `metadata.source_inventory`. The inventory contained 33
source-recording groups and 1,112 pre-segmented items. No transcript labels,
development-validation records, blind-holdout records, or live production outcomes
were used for fitting, selection, calibration, or tuning.

The frozen run5 plan contained 220 cases with this item-count distribution:

| Item count | Cases |
|---:|---:|
| 1 | 67 |
| 2 | 44 |
| 4 | 29 |
| 8 | 27 |
| 16 | 27 |
| 32 | 26 |

Every case stayed within one source recording and the fixed 200-second/32-item shape
contract. Collection used one project-owned AED worker on physical GPU0 (`cuda:0`),
`torch.bfloat16`, with VAD and Punc disabled. Each case ran once as an excluded warm-up
and then three times for measurement, for 220 excluded warm-up batches and 660 measured
batches. The plan recorded `warmups_per_case=1` and `repeats=3`.

Earlier run4 completed all cases but correctly withheld an artifact because repeat-noise
P95 was 21.0688%. The first repeat alone had P95 41.003%, while repeat 2 and repeat 3
had P95 1.597% and 4.700%. This isolated per-shape first-execution initialization as the
cause. Per-case excluded warm-up reduced final run5 P95 to 6.2087%; the original run1
through run4 directories remain unchanged as preserved diagnostic evidence.

## Checksums

| Evidence | SHA-256 |
|---|---|
| Source manifest (`train-plan.json`) | `3b88c2b8d147d6c878289853bb45ebc1efe0bb92bfb4d49927235fb534159d64` |
| Frozen plan content checksum | `13eb4ff2b247c047649da0a537fd59a34b91feb982f98eeebec01c207cd341e2` |
| `plan.json` file | `57a54526889cf5ccd1eb837ab91713f3252269eb52ec401587e0968d0653be2f` |
| `records.jsonl` | `1d8f84c90886bd8203e5bf4c5060bd411fd93371a6ec6aa10668152d9b70e108` |
| `calibration-report.json` | `53386a7ac024f3c51524f8b9ae79a53edea46dcd5d9eeb2a13cc6942bb7054fa` |
| `weighted-aed-calibration.json` | `4a412b4b4280fd1302259136b86d31a97af9aa946320fce6380f19b4e01538b5` |
| Acceptance configuration | `7dbd2cbfdf5581ddbba2303fb76b4de786b979f426fa3c3b2036166140b5a1d9` |

The artifact embeds the source-manifest, frozen-plan-content, and records checksums; all
three agree with the external evidence above.

## Runtime identity and GPU evidence

| Field | Value |
|---|---|
| Device | `cuda:0` / NVIDIA GeForce RTX 3090 |
| Precision | `torch.bfloat16` |
| Compute capability | `8.6` |
| Driver | `580.142` |
| Worker CUDA | `12.8` |
| Worker Torch | `2.10.0+cu128` |
| FireRed source commit | `4e7d9aaf4482a47cec1724807026b9b151926eb5` |
| AED checkpoint SHA-256 | `4677cbd30988d63ed3e777f6a42a1e5260a3865317f6e15e488bef40954f7054` |

GPU0 had 24,576 MiB total and 19,323 MiB free both before model loading and after the
owned worker closed. The final device-level snapshot was
`index=0, used=4802 MiB, free=19323 MiB, utilization=0%`. Across measured calibration
batches, the project worker reported a maximum 3,227 MiB allocated and 4,000 MiB
reserved. No other project's process was inspected, signalled, paused, migrated, or
managed.

## Fit and stability results

The no-intercept fit uses
`max_duration_seconds * (longest_weight + padded_weight * item_count)`.

| Metric | Result |
|---|---:|
| Source groups | 33 |
| Cases | 220 |
| Measured repeats per case | 3 |
| Repeat-noise median deviation | 0.0711% |
| Repeat-noise P95 deviation | 6.2087% |
| Grouped OOF MAE | 0.1338 s |
| Grouped OOF median APE | 17.3651% |
| Grouped OOF P95 APE | 55.1676% |

Grouped OOF diagnostics by item count were:

| Item count | MAE (s) | Median APE | P95 APE |
|---:|---:|---:|---:|
| 1 | 0.1035 | 14.3054% | 67.5298% |
| 2 | 0.1162 | 17.7457% | 63.1228% |
| 4 | 0.1269 | 25.9071% | 53.4939% |
| 8 | 0.1386 | 23.4461% | 45.5962% |
| 16 | 0.1803 | 23.3025% | 44.2697% |
| 32 | 0.1961 | 12.2336% | 19.0166% |

The readiness contract does not impose an OOF accuracy release threshold, so the
artifact is ready under the approved gates. The high OOF tail, especially for small
batches, remains an explicit limitation: these weights are suitable for deterministic
relative scheduling, not a precise per-batch SLA or a substitute for a richer runtime
model.

## Restart acceptance

The artifact was loaded by the weighted dual-AED service twice from independent service
starts using the same ignored acceptance configuration. Both diagnostics reported:

- planner `weighted_duration_dual_worker`;
- `weight_source=artifact`;
- artifact checksum
  `4a412b4b4280fd1302259136b86d31a97af9aa946320fce6380f19b4e01538b5`;
- the exact fitted weights above; and
- no fallback warning.

The worker artifact records Torch `2.10.0+cu128`, while the service process used Torch
`2.11.0+cu128`; this produced the designed version-difference warning and did not cause
fallback because Torch version is a warning-only identity field. The frozen scheduler
subsets compared equal. Both `transcript.json` files were byte-identical with SHA-256
`2d14bf70a89c6b25a6607b89a65e04b833d2eeddbadce030524d9337799e269d`.
After each acceptance run, only the service created from the acceptance configuration
was stopped, and GPU0 returned to the pre-run device-level snapshot.

## Boundaries and disposition

- The calibration produced only the two requested time weights. It did not fit a
  dual-worker contention correction, memory-capacity model, VAD model, or Punc model.
- No memory ceiling or free-memory batch budget was introduced into weighted scheduling;
  the existing fixed 200-second and 32-item shape limits remain unchanged.
- No validation or blind-holdout labels were read, and no candidate training was run.
- No file under `models/cost_models/` and no production configuration or production
  manifest was modified. The ready artifact remains an ignored measurement output and
  is not deployed automatically.
- Historical calibration runs and V3 measurement evidence were not deleted, reset, or
  overwritten.

Result: the artifact is ready for deliberate deployment on the matching RTX 3090
runtime, subject to the documented OOF-accuracy limitation.
