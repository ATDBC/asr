# Punc Pipeline Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the quadratic Punc result-validation cost and overlap one resident Punc owner with independently completing AED batches without adding a second Punc model.

**Architecture:** Keep two resident AED workers, move the single Punc model behind a dedicated owner protocol, and feed it compact AED records through a bounded queue as batches finish. Punc microbatches are packed by token/character cost, while final sentence/token order is restored by the original segment index. Contract ownership validation uses monotonic two-pointer scans and remains strict.

**Tech Stack:** Python 3, JSON Lines subprocess protocol, `concurrent.futures`, PyTorch/CUDA, pytest.

---

### Task 1: Replace quadratic sentence ownership checks with linear scans

**Files:**
- Modify: `src/funasr_transcriber/firered_contract.py:128-237,325-362`
- Test: `tests/unit/test_firered_contract.py`

- [ ] **Step 1: Write failing correctness/performance-shape tests**

  Add tests covering adjacent monotonic sentences, a token exactly on the final sentence boundary, and a large synthetic response whose ownership result is identical to the current validator. Expose a private helper test through the module import; assert owners are produced in one forward pass and invalid gaps still raise `FireRedContractError`.

- [ ] **Step 2: Run the focused tests and confirm the new helper is absent/fails**

  Run `pytest tests/unit/test_firered_contract.py -q`.

- [ ] **Step 3: Implement a two-pointer ownership helper**

  Replace `_sentence_owners` with a monotonic interval walk: keep one sentence index, advance it only while the token midpoint is outside the current sentence, assign exactly one owner, and reject gaps/overlaps. Reuse the owners list in normalization and validation; group token indexes in one pass instead of scanning all tokens once per sentence.

- [ ] **Step 4: Run contract tests and benchmark the synthetic case**

  Run `pytest tests/unit/test_firered_contract.py -q` and a short `python -m timeit`/one-off benchmark comparing a 13,200-token/1,398-sentence response before and after the change. The new path must preserve all existing validation failures and avoid the previous multi-million nested scans.

- [ ] **Step 5: Commit the isolated contract change**

  ```bash
  git add src/funasr_transcriber/firered_contract.py tests/unit/test_firered_contract.py
  git commit -m "perf: validate FireRed sentence ownership linearly"
  ```

### Task 2: Add compact, token-budgeted Punc microbatch planning

**Files:**
- Modify: `src/funasr_transcriber/firered_worker.py:26,357-406,828-888`
- Create or modify: `src/funasr_transcriber/punc_scheduler.py`
- Test: `tests/unit/test_punc_scheduler.py`, `tests/unit/test_firered_worker.py`

- [ ] **Step 1: Write failing planner tests**

  Test that compact records retain `index`, `uttid`, confidence, and timestamps; test token/character weighted packing, deterministic order, singleton handling, and a bounded microbatch budget. Test that Punc output carries the original index and can be merged in original order.

- [ ] **Step 2: Run the focused planner/worker tests and confirm failure**

  Run `pytest tests/unit/test_punc_scheduler.py tests/unit/test_firered_worker.py -q`.

- [ ] **Step 3: Implement the planner and compact record conversion**

  Add a small pure module that estimates each segment’s Punc cost from timestamp/token count and packs records up to a configurable token budget, with a conservative item cap. Keep Punc’s CUDA batch size independent from AED memory planning. Update `_punctuate_aed_results` to consume compact records and return indexed punctuated records.

- [ ] **Step 4: Run worker tests and verify output equivalence**

  Run the focused tests and existing contract/client tests. Confirm sentence text, token timestamps, confidences, and original ordering remain unchanged.

- [ ] **Step 5: Commit the planner/compact protocol change**

  ```bash
  git add src/funasr_transcriber/punc_scheduler.py src/funasr_transcriber/firered_worker.py tests/unit/test_punc_scheduler.py tests/unit/test_firered_worker.py
  git commit -m "feat: add token-budgeted compact Punc batches"
  ```

### Task 3: Introduce one dedicated resident Punc owner and bounded streaming protocol

**Files:**
- Modify: `src/funasr_transcriber/firered_worker.py`, `src/funasr_transcriber/firered_client.py`
- Modify: `src/funasr_transcriber/firered_parallel.py`, `src/funasr_transcriber/service.py`
- Modify: `docs/architecture/current-transcription-pipeline.md`
- Test: `tests/unit/test_firered_client.py`, `tests/unit/test_firered_parallel.py`, `tests/integration/test_pipeline_smoke.py`

- [ ] **Step 1: Write failing stream/queue tests**

  Add fake-worker tests proving that a completed AED batch is submitted to Punc before the other AED lane finishes, that the queue has a finite capacity, that Punc remains a single owner, and that final results are reordered by original index. Add protocol tests for batch-completed events and final responses.

- [ ] **Step 2: Run focused tests and confirm the current barrier fails them**

  Run `pytest tests/unit/test_firered_client.py tests/unit/test_firered_parallel.py tests/integration/test_pipeline_smoke.py -q`.

- [ ] **Step 3: Implement the streaming protocol**

  Extend the JSONL worker protocol with an explicit AED-batch result event and a compact Punc request/response. Keep each AED subprocess sequential internally, but have the orchestrator submit completed records to one dedicated Punc owner through a bounded queue. Do not allow concurrent requests on one client pipe. Preserve the existing serial fallback when the GPU gate rejects concurrent AED execution.

- [ ] **Step 4: Integrate lifecycle and diagnostics**

  Start/warm/close the single Punc owner with the service, keep its model resident, and record queue depth, Punc microbatch count, Punc pure-model seconds, Punc normalization seconds, and end-to-end overlap in diagnostics. Ensure `service stop` closes only this project’s registered owners.

- [ ] **Step 5: Run all unit/integration tests and commit**

  Run `pytest -q` and `git diff --check`, then commit:

  ```bash
  git add src/funasr_transcriber/firered_worker.py src/funasr_transcriber/firered_client.py src/funasr_transcriber/firered_parallel.py src/funasr_transcriber/service.py docs/architecture/current-transcription-pipeline.md tests
  git commit -m "feat: stream AED results through one Punc owner"
  ```

### Task 4: Validate on the production corpus and tune Punc budget

**Files:**
- Modify: diagnostics/report tooling only if required by measured fields
- Create: `outputs/measurements/<timestamp>-punc-pipeline/result.json` (ignored output)

- [ ] **Step 1: Run the production audio on GPU0 with the warmed service**

  Use `audio_test/刘洋洋.wav`, BF16, two AED workers, the existing external-process safety rules, and the release output directory. Do not manage unrelated GPU processes.

- [ ] **Step 2: Compare stage and resource metrics**

  Record diarization, VAD, AED wall/worker times, Punc pure-model time, Punc validation time, queue overlap, total time, worker peaks, total GPU peak, and output fingerprint/schema checks. Compare against `20260822T-memory-first-aed-liuyangyang/result.json`.

- [ ] **Step 3: Tune only the Punc token budget**

  Test token budgets 8, 16, and 32 (or the closest safe values) with the same warmed models. Select the lowest wall time that preserves output and stays under the observed Punc safety margin; do not change AED memory-first planning in this task.

- [ ] **Step 4: Run the complete verification suite and report results**

  Run `pytest -q`, `git diff --check`, and verify GPU0 returns to the external-project baseline after service shutdown. Summarize before/after timings, memory, fingerprint/schema status, and any remaining bottleneck.

## Self-review

- The plan covers the three requested optimization mechanisms: linear validation, streaming single-owner Punc, and token-budgeted microbatches.
- No second Punc model is introduced.
- AED memory-first planning and the serial safety fallback remain unchanged.
- All production measurements use the existing GPU0 and process-safety constraints.
