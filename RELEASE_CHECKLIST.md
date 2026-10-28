# Standalone Transcriber Release Checklist

- [ ] Fresh Linux installation completed from README
- [ ] `.venv`, `.venv-asr`, and `.venv-overlap` are isolated Python 3.11 environments
- [ ] FireRedASR2S source commit is `4e7d9aaf4482a47cec1724807026b9b151926eb5`
- [ ] FireRedVAD / FireRedASR2-AED / FireRedPunc assets verified
- [ ] funasr-transcribe models download completed
- [ ] funasr-transcribe models verify passed
- [ ] GPU 0 reports at least 23 GiB total; under current single-AED architecture, FireRed peak VRAM is ~5.5 GiB and strictly under 6 GiB (at least 13,312 MiB free before the service hot-load when co-resident with optional Nemotron)
- [ ] Historical serial-stage evidence records the former 12 GiB free threshold and 9.04 GiB peak separately from the current single-worker service budget (~5.5 GiB, strictly bounded under 6 GiB)
- [ ] Default production service runs with single AED worker (`aed_worker_count = 1`) and independent Punc owner in BF16
- [ ] funasr-transcribe doctor full model smoke passed without `--skip-cuda`
- [ ] GPU release evidence is present for FireRed, Speaker, and overlap doctor phases
- [ ] Unit and integration tests passed
- [ ] Portable validation corpus SHA-256 verification passed
- [ ] Packaged Mandarin doctor-smoke transcription passed
- [ ] 20-recording RAMC gate passed
- [ ] 50-case overlap gate passed
- [ ] Input format matrix passed
- [ ] 30/60/120-minute stress runs completed without OOM
- [ ] Repeated-output determinism passed
- [ ] `segment_batch_size` GPU 0 calibration report records output equivalence, timing, and per-process memory peak
- [ ] No-overlap run loads FireRed once; overlap run loads FireRed at most twice
- [ ] Two sequential default jobs reuse one hot FireRed/3D-Speaker service owner
- [ ] Profile-enabled job has `diagnostics.json.resource_profile.enabled == true`, verifying FireRed resident peak VRAM is ~5.5 GiB (strictly bounded under 6 GiB) and contains only named project owners
- [ ] `funasr-transcribe service status` reports ready before validation shutdown
- [ ] `funasr-transcribe service stop` drains accepted work, removes this project's socket/state, and releases its FireRed PID
- [ ] Main FireRed failure has no legacy ASR fallback; overlap failures degrade per window
- [ ] TXT, JSON and diagnostics manually inspected
- [ ] No model, audio, token or business-data files are staged
- [ ] No other project process was terminated, paused, restarted, or otherwise managed

For every checked item, record the command, date, commit hash and result artifact path below the item. Never record credentials or personal-data paths.
