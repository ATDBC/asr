# Standalone Transcriber Release Checklist

- [ ] Fresh Linux installation completed from README
- [ ] funasr-transcribe models download completed
- [ ] funasr-transcribe models verify passed
- [ ] funasr-transcribe doctor passed
- [ ] Unit and integration tests passed
- [ ] Portable validation corpus SHA-256 verification passed
- [ ] Packaged Mandarin doctor-smoke transcription passed
- [ ] 20-recording RAMC gate passed
- [ ] 50-case overlap gate passed
- [ ] Input format matrix passed
- [ ] 30/60/120-minute stress runs completed without OOM
- [ ] Repeated-output determinism passed
- [ ] TXT, JSON and diagnostics manually inspected
- [ ] No model, audio, token or business-data files are staged

For every checked item, record the command, date, commit hash and result artifact path below the item. Never record credentials or personal-data paths.
