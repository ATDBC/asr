# Standalone Transcriber Release Checklist

- [ ] Fresh Linux installation completed from README
- [ ] `.venv`, `.venv-asr`, and `.venv-overlap` are isolated Python 3.11 environments
- [ ] FireRedASR2S source commit is `4e7d9aaf4482a47cec1724807026b9b151926eb5`
- [ ] FireRedVAD / FireRedLID / FireRedASR2-AED / FireRedPunc assets verified
- [ ] funasr-transcribe models download completed
- [ ] funasr-transcribe models verify passed
- [ ] GPU 0 reports at least 23 GiB total and 12 GiB free before every model phase
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
- [ ] No-overlap run loads FireRed once; overlap run loads FireRed at most twice
- [ ] Main FireRed failure has no legacy ASR fallback; overlap failures degrade per window
- [ ] TXT, JSON and diagnostics manually inspected
- [ ] No model, audio, token or business-data files are staged
- [ ] No other project process was terminated, paused, restarted, or otherwise managed

For every checked item, record the command, date, commit hash and result artifact path below the item. Never record credentials or personal-data paths.
