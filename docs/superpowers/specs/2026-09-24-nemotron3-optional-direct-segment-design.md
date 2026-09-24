# Optional Nemotron 3 diarization with direct speaker-segment AED

Date: 2026-09-24
Status: design for review
Repository: standalone_transcriber, feature/server-weighted-aed-batching

## Intent and evidence

Add Nemotron 3 Diarization as an explicitly selected speaker backend. Keep 3D-Speaker as the unchanged default. Nemotron supplies speaker activity intervals; each interval is cropped directly from the original normalized mixture and transcribed separately by FireRed AED. Overlapping intervals intentionally repeat the original mixture audio in both AED inputs. There is no ClearVoice separation, silence-padded speaker track, overlap deduplication, or automatic default switch.

The 2026-09-24 spike on ten real-overlap clips showed Nemotron could identify two speakers in all ten, and a direct-crop AED comparison on five selected windows recovered much of the content also obtained by ClearVoice separation. It did not establish production accuracy or end-to-end throughput. The prior 21-minute Nemotron run had a longest predicted segment of 9.24 seconds, but this is not a model-imposed maximum. The official 30.4-second offline buffer is not an output-segment limit.

## Public configuration and asset contract

- Add diarization.backend with values 3d_speaker and nemotron3; default is 3d_speaker. A selected backend participates in the service config fingerprint. Existing 3D-specific settings and behavior remain unchanged on the default path.
- Add a Nemotron-specific worker Python path under a new nemotron config section. Keep the tested offline mode, 0.5 activity threshold, physical cuda:0, model revision a435e9867d79e789e90053f9b6d6834053af564a, and Transformers revision f324707307757d9c0b8dac1c4462eceff911fa2f pinned in versioned code/installation metadata rather than silently following latest.
- Provision a separate .venv-nemotron explicitly; do not add Transformers development dependencies to the default .venv or .venv-asr. The default installer and default models download do not fetch Nemotron. An explicit Nemotron installation step prepares the isolated environment; models download with a Nemotron-selected config fetches the pinned checkpoint from the NVIDIA Hugging Face release. Runtime only reads the local verified snapshot and never downloads.
- models verify and doctor verify only the selected speaker backend, plus the always-required FireRed assets. Nemotron verification checks checkpoint files, revision/hash, isolated runtime, a small cuda:0 inference, and release of this project's worker resources. Missing/incompatible assets fail clearly; never silently fall back to 3D-Speaker.
- Nemotron supports at most eight speaker channels. An explicit configured speaker count above eight is rejected; the system cannot reliably detect an unobserved ninth real speaker, so the documented limitation and diagnostic identity must remain visible. The 3D-specific speaker-count search parameters do not control Nemotron.

## Runtime ownership and diarization result

- Add a project-owned persistent Nemotron worker implementing the existing speaker-owner lifecycle and returning immutable SpeakerSegment values on the normalized audio's global millisecond timeline. Host it only when selected. Record its own PID for resource sampling; never inspect or manage other-project PIDs.
- The Nemotron worker loads its pinned local model once, uses offline/chunked inference over the recording, extracts activity by speaker, converts model speaker IDs to positive IDs, sorts by start/speaker/end, and validates finite timestamps, audio bounds, ordering, and at most eight predicted speakers. Overlapping intervals remain overlapping. A model error fails the selected-backend job with a diagnostic.
- All GPU execution stays on physical GPU0. Do not remap another physical GPU into cuda:0. Check available capacity before loading the resident combination; if it cannot be admitted, fail without modifying unrelated processes.
- The legacy 3D-Speaker backend and its long-interval behavior are untouched.

## Direct-crop AED path

For each valid Nemotron SpeakerSegment, write a create-only 16 kHz PCM crop of the original normalized waveform into the job's intermediate directory. Crop at sample-aligned millisecond boundaries and retain source speaker ID, global start/end, original segment index, and input checksum. Two segments such as speaker 1 at 0–6000 ms and speaker 2 at 5000–8000 ms produce two independent AED inputs. The 5000–6000 ms mixture is present in both; it is never separated, masked, or assigned to just one speaker.

- Segments shorter than 100 ms are excluded from AED with a count and warning. No adjacent same-speaker segments are silently merged. A crop that yields no speech is recorded and does not discard other successful crops; if all crops produce no speech, the job fails explicitly.
- An interval can be longer than 30 seconds. Keep its Nemotron speaker ownership, but refine speech inside that crop with FireRedVAD when longer than 10 seconds. If refinement is unavailable or yields an unsafe piece, deterministically subdivide within the crop into pieces no longer than 10 seconds. Every AED piece remains mapped to its parent crop and original absolute offset; the 60-second AED hard limit is never exceeded.
- Extend the resident AED/Punc owner with an explicit multi-crop request path. Each crop is a separate AED input, and work is assigned to the configured one or two resident AED workers. Within one crop, safe VAD subpieces may still be batched; crops from different source paths are not represented as one existing single-path AED plan. The selected weighted-duration scheduler may rank/partition crop work, but its old single-source calibration is not asserted to predict this new workload. The legacy memory-first scheduler is rejected with a clear configuration error for Nemotron until a separate multi-crop contract is implemented; default 3D-Speaker scheduling stays unchanged.
- Punctuation remains owned by the existing independent FireRedPunc worker. Convert crop-local token and sentence times to global times using the original crop offset. Directly attribute each output to its source speaker segment, not midpoint matching against overlapping segments. Keep successful outputs sorted deterministically by global start, speaker ID, and segment index.

## Output and overlap semantics

Existing transcript.txt, transcript.json, and diagnostics.json formats remain present. transcript.json may contain overlapping turns for different speakers; both texts are retained. Mark overlap on a turn or token when its original speaker interval intersects another speaker's activity. TXT lists turns by start time and can show two speakers at the same time. Full text concatenates both speakers' recognized words in deterministic turn order; duplication of a word across speakers is possible and is not automatically removed.

Diagnostics record backend and pinned runtime identity, predicted speaker/segment counts, overlap duration, short/no-speech crop counts, AED crop and subpiece counts, per-owner stage time, own-PID GPU peak, and warnings. Deep traces include crop-to-segment mapping and per-worker assignments without placing the entire plan in basic diagnostics. Intermediates follow the existing keep-intermediate policy. No validation labels are used during inference.

Nemotron direct-crop mode and the legacy pyannote/ClearVoice overlap enhancement are mutually exclusive: config validation and per-request CLI overrides reject their simultaneous selection rather than silently ignoring either option. The default 3D-Speaker overlap feature is unchanged.

## Failure behavior and acceptance

- Unexpected Nemotron, crop, AED contract, Punc, GPU admission, or timestamp errors fail the job explicitly. Failed-job intermediates remain only when keep-intermediate is requested; otherwise the existing service cleanup policy applies. No silent backend fallback, speaker reassignment, waveform separation, or online tuning.
- Unit tests use a 0–6 s / 5–8 s two-speaker fixture to prove two independent original-mixture crops, duplicated 5–6 s audio, correct speaker IDs and global offsets, overlapping JSON turns, and deterministic TXT ordering. Cover nested/crossing segments, a >30 s interval, 10/100/190 ms boundaries, no-speech responses, one/two AED workers, scheduler conflict, model/overlap config conflict, and unchanged default backend.
- Focused tests precede the full unit suite and git diff --check. A GPU0-only opt-in integration run uses the existing real-overlap samples and long recording, records model precision, project-owned GPU peak, model load/prewarm, AED/Punc/cache cleanup, and end-to-end stage times. Compare direct-crop outputs to the source speaker text, while acknowledging coarse real-overlap timing labels. Do not call this production-accuracy acceptance merely because the pipeline completes.
- Document optional install, explicit model download/verification, config, limitations, and fallback procedure. Completing this design and implementation does not switch the default backend or publish a production endorsement.

## Non-goals

No ClearVoice wave separation; no pyannote overlap detection in Nemotron mode; no speaker-voice rematching; no automatic ninth-speaker solution; no AED cost-model retraining; no change to the default 3D-Speaker path or to other projects' processes.
