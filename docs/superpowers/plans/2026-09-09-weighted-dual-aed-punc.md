# Weighted Dual-AED with Streaming Punc Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Do not use subagents for this repository.

**Goal:** Add an explicitly selected two-lane fixed weighted-duration AED mode with no GPU-memory admission or memory-based fallback, streaming both AED lanes into one independent Punc owner.

**Architecture:** Reuse the existing global weighted batch builder, deterministic batch partitioner, and streaming Punc executor. Generalize the current single-lane weighted owner to one or two AED workers, while preserving the existing production memory-first owner behind an explicit scheduler selection.

**Tech Stack:** Python 3.11, dataclasses, concurrent.futures, existing FireRed JSONL worker protocol, pytest, TOML configuration.

**Spec:** docs/superpowers/specs/2026-09-09-weighted-dual-aed-punc-design.md

## Global Constraints

- Work only through ssh ai_server in /mnt/hdd/jiajun/project/standalone_transcriber on the current codex/server-weighted-aed-batching branch.
- Do not create a worktree and do not use subagents.
- Use only physical GPU0 as cuda:0; never remap another GPU with CUDA_VISIBLE_DEVICES.
- Weighted-duration mode must not read free GPU memory, apply a GPU-memory batch limit, perform memory admission, or fall back to serial AED dispatch.
- The fixed shape limits remain 200 seconds and 32 items per batch.
- Freeze the complete global batch plan and both lane assignments before the first AED request.
- Only start, inspect, clean, or stop this project's own processes and outputs.
- Do not read development-validation records or blind holdout data, train a candidate, publish a production manifest, or write models/cost_models/.
- Leave artifacts/ and artifacts/release_validation_corpus untouched and Git ignored.
- Preserve the train-only weights exactly:
  - AED_LONGEST_WEIGHT = 0.12854663332839417
  - AED_PADDED_WEIGHT = 0.008409640427755745
- Preserve the production memory-first implementation and make scheduler selection explicit.
- Use the repository's existing Git identity.
- Stop and preserve all outputs on the first command failure or OOM; do not retry automatically.

---

### Task 1: Add explicit AED scheduler configuration

**Files:**
- Modify: config/default.toml:11-19
- Modify: src/funasr_transcriber/config.py:20-40, 96-108, 250-266, 329-340
- Test: tests/unit/test_config.py

**Interfaces:**
- Produces: AsrConfig.aed_scheduler: str
- Valid values: weighted_duration and memory_first
- Valid combinations: weighted_duration with one or two workers; memory_first with two workers
- Preserves the experimental branch's current default behavior: weighted_duration with one worker

- [ ] **Step 1: Write failing configuration tests**

Add tests that load all valid combinations and reject the invalid combinations:

~~~python
@pytest.mark.parametrize(
    ("scheduler", "worker_count"),
    (
        ("weighted_duration", 1),
        ("weighted_duration", 2),
        ("memory_first", 2),
    ),
)
def test_aed_scheduler_accepts_supported_worker_counts(
    tmp_path: Path,
    scheduler: str,
    worker_count: int,
):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        (
            "[asr]\n"
            f'aed_scheduler = "{scheduler}"\n'
            f"aed_worker_count = {worker_count}\n"
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.asr.aed_scheduler == scheduler
    assert config.asr.aed_worker_count == worker_count


@pytest.mark.parametrize(
    "body",
    (
        '[asr]\naed_scheduler = "unknown"\n',
        '[asr]\naed_scheduler = "memory_first"\naed_worker_count = 1\n',
    ),
)
def test_aed_scheduler_rejects_unsupported_configuration(
    tmp_path: Path,
    body: str,
):
    config_path = tmp_path / "config.toml"
    config_path.write_text(body, encoding="utf-8")

    with pytest.raises(ValueError, match="aed_scheduler"):
        load_config(config_path)
~~~

Also extend the existing default assertions:

~~~python
assert config.asr.aed_scheduler == "weighted_duration"
assert config.asr.aed_worker_count == 1
~~~

- [ ] **Step 2: Run the new tests and verify RED**

Run:

~~~bash
.venv/bin/python -m pytest -q \
  tests/unit/test_config.py::test_aed_scheduler_accepts_supported_worker_counts \
  tests/unit/test_config.py::test_aed_scheduler_rejects_unsupported_configuration
~~~

Expected: FAIL because aed_scheduler is currently an unknown configuration key or absent from AsrConfig.

- [ ] **Step 3: Implement the minimal configuration field**

Add the default beside aed_worker_count in both DEFAULTS and config/default.toml:

~~~python
"asr": {
    "worker_python": "/opt/funasr-transcriber/.venv-asr/bin/python",
    "timeout_seconds": 1800,
    "use_half": True,
    "aed_scheduler": "weighted_duration",
    "aed_worker_count": 1,
~~~

~~~toml
[asr]
worker_python = "/opt/funasr-transcriber/.venv-asr/bin/python"
timeout_seconds = 1800
use_half = true
aed_scheduler = "weighted_duration"
aed_worker_count = 1
~~~

Add the dataclass field and builder assignment:

~~~python
@dataclass(frozen=True, slots=True)
class AsrConfig:
    worker_python: Path
    timeout_seconds: int
    use_half: bool
    aed_scheduler: str
    aed_worker_count: int
    gpu_safety_margin_mib: int
    punc_token_budget: int
    punc_max_items: int
~~~

~~~python
aed_scheduler=str(values["asr"]["aed_scheduler"]).strip().lower(),
~~~

Validate exact values and combinations:

~~~python
if config.asr.aed_scheduler not in {"weighted_duration", "memory_first"}:
    raise ValueError(
        "ASR aed_scheduler must be weighted_duration or memory_first"
    )
if (
    config.asr.aed_scheduler == "memory_first"
    and config.asr.aed_worker_count != 2
):
    raise ValueError("ASR memory_first aed_scheduler requires 2 AED workers")
~~~

- [ ] **Step 4: Run focused configuration tests and verify GREEN**

Run:

~~~bash
.venv/bin/python -m pytest -q tests/unit/test_config.py
~~~

Expected: all tests pass.

- [ ] **Step 5: Commit the configuration contract**

Run:

~~~bash
git add config/default.toml src/funasr_transcriber/config.py tests/unit/test_config.py
git commit -m "feat: select AED scheduling strategy"
~~~

---

### Task 2: Generalize the weighted owner to one or two AED lanes

**Files:**
- Modify: src/funasr_transcriber/firered_parallel.py:299-360, 833-946
- Modify: tests/unit/test_firered_parallel.py:210-350

**Interfaces:**
- Replaces internal class SingleAedPuncWorkerClient with WeightedAedPuncWorkerClient
- Constructor:
  WeightedAedPuncWorkerClient(worker0, punc_owner, *, worker1=None, punc_token_budget=256, punc_max_items=16)
- Produces: aed_workers: tuple[Any, ...]
- Reuses: plan_aed_batches(), partition_batches(), serialize_batch_plan(), and ParallelFireRedWorkerClient._transcribe_streaming()
- Keeps one-lane plan boundaries and output behavior unchanged

- [ ] **Step 1: Rename the existing one-lane test and preserve its assertions**

Update the current test to construct WeightedAedPuncWorkerClient with no worker1:

~~~python
owner = parallel_module.WeightedAedPuncWorkerClient(
    aed,
    punc,
    punc_max_items=1,
)

assert owner.pids == (101,)
assert owner.status()["worker_count"] == 1
~~~

Keep the existing overlap, output order, Punc batch count, peak-memory metrics, and weighted planner assertions unchanged.

- [ ] **Step 2: Add a failing two-lane streaming test**

Use the existing fake worker and Punc helpers. Give each worker a distinct PID, create 64 one-second segments so the fixed planner creates at least two complete batches, and assert both workers receive nonempty immutable plans:

~~~python
def test_weighted_dual_aed_partitions_fixed_batches_and_streams_one_punc(
    tmp_path,
):
    punc_completed = threading.Event()

    class DualStreamingWorker(_StreamingWorker):
        def run_aed_plan_stream(self, path, plan, vad, on_event):
            response = super().run_aed_plan_stream(
                path, plan, vad, on_event
            )
            assert punc_completed.wait(timeout=1)
            return response

    class DualPuncOwner(_FakePuncOwner):
        def punctuate_chunk(self, path, records):
            result = super().punctuate_chunk(path, records)
            punc_completed.set()
            return result

    first = DualStreamingWorker(pid=101)
    second = DualStreamingWorker(pid=102)
    punc = DualPuncOwner(pid=103)
    owner = parallel_module.WeightedAedPuncWorkerClient(
        first,
        punc,
        worker1=second,
        punc_max_items=1,
    )
    owner.gpu_memory_provider = lambda *_: pytest.fail(
        "weighted mode must not query GPU memory"
    )

    result = owner.transcribe(
        tmp_path / "audio.wav",
        [[index * 1_000, (index + 1) * 1_000] for index in range(64)],
    )

    plans = [
        call[1]
        for worker in (first, second)
        for call in worker.calls
        if call[0] == "aed_stream"
    ]
    planned_indices = sorted(
        item["index"]
        for lane in plans
        for batch in lane
        for item in batch
    )

    assert result.text
    assert owner.pids == (101, 102)
    assert owner.status()["worker_count"] == 2
    assert len(plans) == 2
    assert all(plans)
    assert planned_indices == list(range(64))
    assert owner.last_response["metrics"]["parallel"]["dispatch_mode"] == "concurrent"
    assert "memory" not in owner.last_response["metrics"]["parallel"]
    assert "shared_budget_mib" not in owner.last_response["metrics"]["scheduler"]
    punc_indices = {
        index
        for action, indices in punc.calls
        if action == "punc_chunk"
        for index in indices
    }
    assert punc_indices == set(range(64))
~~~

The event assertion proves the single Punc owner completes a chunk before either AED stream is allowed to return. Also assert the indices recorded in punc.calls cover records emitted by both nonempty lane plans.

- [ ] **Step 3: Add failing lifecycle tests for the selected lane count**

Add one test proving dual warmup starts both AED workers and Punc concurrently, and one proving cleanup calls only both AED workers:

~~~python
def test_weighted_dual_aed_lifecycle_covers_both_lanes(tmp_path):
    class LifecycleWorker(_FakeWorker):
        def warmup(self, path):
            self.calls.append(("warmup", path))
            return {
                "excluded": True,
                "seconds": 0.1,
                "resident_allocated_mib": 100,
            }

        def cleanup_cuda_cache(self):
            self.calls.append(("cleanup", None))
            return {"seconds": 0.1}

    class LifecyclePunc(_FakePuncOwner):
        def warmup(self, path):
            self.calls.append(("warmup", path))
            return {
                "excluded": True,
                "seconds": 0.1,
                "resident_allocated_mib": 50,
            }

    first = LifecycleWorker(pid=101)
    second = LifecycleWorker(pid=102)
    punc = LifecyclePunc(pid=103)
    owner = parallel_module.WeightedAedPuncWorkerClient(
        first,
        punc,
        worker1=second,
    )
    warmup = owner.warmup(tmp_path / "warmup.wav")
    cleanup = owner.cleanup_cuda_cache()

    assert len(warmup["workers"]) == 2
    assert warmup["punc"] is not None
    assert [record["pid"] for record in cleanup["workers"]] == [101, 102]
    assert not any(call[0] == "cleanup" for call in punc.calls)
~~~

- [ ] **Step 4: Run the new owner tests and verify RED**

Run:

~~~bash
.venv/bin/python -m pytest -q \
  tests/unit/test_firered_parallel.py::test_weighted_dual_aed_partitions_fixed_batches_and_streams_one_punc \
  tests/unit/test_firered_parallel.py::test_weighted_dual_aed_lifecycle_covers_both_lanes
~~~

Expected: FAIL because WeightedAedPuncWorkerClient and its two-lane behavior do not yet exist.

- [ ] **Step 5: Implement the smallest generalized owner**

Rename the class and add an optional second worker:

~~~python
class WeightedAedPuncWorkerClient(ParallelFireRedWorkerClient):
    """One or two fixed weighted AED lanes with one streaming Punc owner."""

    def __init__(
        self,
        worker0: Any,
        punc_owner: Any,
        *,
        worker1: Any | None = None,
        punc_token_budget: int = DEFAULT_PUNC_TOKEN_BUDGET,
        punc_max_items: int = DEFAULT_PUNC_MAX_ITEMS,
    ):
        super().__init__(
            worker0,
            worker1 or worker0,
            punc_owner=punc_owner,
            punc_token_budget=punc_token_budget,
            punc_max_items=punc_max_items,
        )
        self.aed_workers = (
            (worker0,) if worker1 is None else (worker0, worker1)
        )
~~~

Preserve the existing BatchCostModel coefficients exactly. Derive PID, status, warmup, and cleanup behavior from self.aed_workers:

~~~python
@property
def pids(self) -> tuple[int, ...]:
    return tuple(
        pid
        for worker in self.aed_workers
        if isinstance((pid := getattr(worker, "pid", None)), int) and pid > 0
    )
~~~

In transcribe(), keep global batch construction unchanged and partition only after every batch boundary is fixed:

~~~python
worker_count = len(self.aed_workers)
assignments = (
    (plan,)
    if worker_count == 1
    else partition_batches(plan, worker_count=worker_count)
)
worker_plans = [
    serialize_batch_plan(lane)
    for lane in assignments
]
result = self._transcribe_streaming(
    audio_path,
    refined,
    plan,
    assignments,
    worker_plans,
    "single" if worker_count == 1 else "concurrent",
    empty_memory,
    empty_memory,
    0,
    0,
    started,
    vad_seconds,
    plan_seconds,
)
~~~

Remove memory and prediction-accuracy fields from weighted diagnostics exactly as the one-lane implementation does. Set worker_count from len(self.aed_workers), retain the existing one-lane planner name, and add worker batch counts, worker predicted seconds, and serialized worker plans for two lanes.

Do not alter plan_aed_batches(), AED_LONGEST_WEIGHT, AED_PADDED_WEIGHT, the 200-second limit, or the 32-item limit.

- [ ] **Step 6: Run the full parallel-owner test file and verify GREEN**

Run:

~~~bash
.venv/bin/python -m pytest -q tests/unit/test_firered_parallel.py
~~~

Expected: all tests pass.

- [ ] **Step 7: Commit the generalized weighted owner**

Run:

~~~bash
git add src/funasr_transcriber/firered_parallel.py tests/unit/test_firered_parallel.py
git commit -m "feat: run weighted batches on two AED lanes"
~~~

---

### Task 3: Route explicit scheduler and worker-count combinations

**Files:**
- Modify: src/funasr_transcriber/service.py:20-50, 1095-1138
- Modify: tests/unit/test_service.py:190-240
- Test: tests/unit/test_service.py

**Interfaces:**
- Consumes: AsrConfig.aed_scheduler
- Consumes: WeightedAedPuncWorkerClient(..., worker1=None | FireRedWorkerClient)
- Preserves: ParallelFireRedWorkerClient for memory_first with two workers
- Produces logs:
  - firered-worker-0.log for VAD/AED
  - firered-worker-1.log only when two AED workers are selected
  - firered-punc.log for the independent Punc owner

- [ ] **Step 1: Add failing factory topology tests**

Parameterize the weighted factory test over one and two workers:

~~~python
@pytest.mark.parametrize(("worker_count", "expected_clients"), ((1, 2), (2, 3)))
def test_weighted_service_builds_selected_aed_lanes_and_one_punc(
    tmp_path: Path,
    monkeypatch,
    worker_count: int,
    expected_clients: int,
):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        (
            "[asr]\n"
            'aed_scheduler = "weighted_duration"\n'
            f"aed_worker_count = {worker_count}\n"
        ),
        encoding="utf-8",
    )

    owner = build_hot_model_host(
        load_config(config_path),
        ServicePaths.from_work_dir(tmp_path),
    )._asr_factory()

    assert isinstance(owner, WeightedAedPuncWorkerClient)
    assert len(owner.aed_workers) == worker_count
    assert len(clients) == expected_clients
    assert clients[0].options["enable_vad"] is True
    assert clients[0].options["enable_punc"] is False
    assert clients[-1].options["punc_only"] is True
    if worker_count == 2:
        assert clients[1].options["enable_vad"] is False
        assert clients[1].options["enable_punc"] is False
~~~

Keep or add a separate test proving memory_first with two workers still constructs ParallelFireRedWorkerClient rather than the weighted subclass.

- [ ] **Step 2: Replace the admission test with scheduler-based expectations**

The weighted strategy must request no minimum free-memory threshold for either worker count, while memory_first keeps the configured threshold:

~~~python
@pytest.mark.parametrize(
    ("scheduler", "worker_count", "expected_free_mib"),
    (
        ("weighted_duration", 1, 0),
        ("weighted_duration", 2, 0),
        ("memory_first", 2, 13 * 1024),
    ),
)
def test_gpu_admission_depends_on_scheduler(
    tmp_path: Path,
    monkeypatch,
    scheduler: str,
    worker_count: int,
    expected_free_mib: int,
):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        (
            "[asr]\n"
            f'aed_scheduler = "{scheduler}"\n'
            f"aed_worker_count = {worker_count}\n"
        ),
        encoding="utf-8",
    )

    host = build_hot_model_host(
        load_config(config_path),
        ServicePaths.from_work_dir(tmp_path),
    )
    host._gpu_check()

    assert calls[0][1]["minimum_free_mib"] == expected_free_mib
~~~

- [ ] **Step 3: Run the service tests and verify RED**

Run:

~~~bash
.venv/bin/python -m pytest -q \
  tests/unit/test_service.py::test_weighted_service_builds_selected_aed_lanes_and_one_punc \
  tests/unit/test_service.py::test_gpu_admission_depends_on_scheduler
~~~

Expected: FAIL because service routing does not inspect aed_scheduler and two workers still select memory-first.

- [ ] **Step 4: Implement explicit service routing**

Construct the shared worker-0 and Punc owner once inside the weighted branch, and construct worker-1 only for weighted dual mode:

~~~python
if config.asr.aed_scheduler == "weighted_duration":
    worker0 = client(
        "firered-worker-0.log",
        enable_vad=True,
        enable_punc=False,
    )
    punc_owner = client(
        "firered-punc.log",
        enable_vad=False,
        enable_punc=True,
        punc_only=True,
    )
    worker1 = (
        client(
            "firered-worker-1.log",
            enable_vad=False,
            enable_punc=False,
        )
        if worker_count == 2
        else None
    )
    return WeightedAedPuncWorkerClient(
        worker0,
        punc_owner,
        worker1=worker1,
        punc_token_budget=config.asr.punc_token_budget,
        punc_max_items=config.asr.punc_max_items,
    )
~~~

Leave the current two-worker ParallelFireRedWorkerClient construction as the memory_first branch. Change only the free-memory admission selector:

~~~python
minimum_free_mib=(
    0
    if config.asr.aed_scheduler == "weighted_duration"
    else config.service.minimum_free_mib
),
~~~

This removes memory admission from weighted dual mode without removing physical GPU0 and minimum-total-memory validation.

- [ ] **Step 5: Run affected unit tests and verify GREEN**

Run:

~~~bash
.venv/bin/python -m pytest -q \
  tests/unit/test_config.py \
  tests/unit/test_firered_parallel.py \
  tests/unit/test_service.py
~~~

Expected: all tests pass.

- [ ] **Step 6: Commit service routing**

Run:

~~~bash
git add src/funasr_transcriber/service.py tests/unit/test_service.py
git commit -m "feat: select weighted dual AED service"
~~~

---

### Task 4: Self-review and repository verification

**Files:**
- Review only: config/default.toml
- Review only: src/funasr_transcriber/config.py
- Review only: src/funasr_transcriber/firered_parallel.py
- Review only: src/funasr_transcriber/service.py
- Review only: tests/unit/test_config.py
- Review only: tests/unit/test_firered_parallel.py
- Review only: tests/unit/test_service.py

**Interfaces:**
- Confirms the implementation matches the committed design and global constraints
- Produces no new abstraction, dependency, production model, or dataset output

- [ ] **Step 1: Review the complete branch diff**

Run:

~~~bash
git diff 0380fd1..HEAD -- \
  config/default.toml \
  src/funasr_transcriber/config.py \
  src/funasr_transcriber/firered_parallel.py \
  src/funasr_transcriber/service.py \
  tests/unit/test_config.py \
  tests/unit/test_firered_parallel.py \
  tests/unit/test_service.py
~~~

Verify from the diff that:

- plan_aed_batches() and both weight constants are unchanged;
- weighted dual mode calls partition_batches() after global packing;
- both weighted lanes always dispatch concurrently;
- no weighted path calls gpu_memory_provider;
- Punc remains one independent owner;
- memory_first still uses ParallelFireRedWorkerClient;
- only project-owned clients are closed or cleaned;
- artifacts/, validation, blind holdout, candidate training, and models/cost_models/ are untouched.

- [ ] **Step 2: Run the required focused verification**

Run:

~~~bash
.venv/bin/python -m pytest -q \
  tests/unit/test_config.py \
  tests/unit/test_firered_client.py
~~~

Expected: all tests pass.

- [ ] **Step 3: Run the complete test suite**

Run:

~~~bash
.venv/bin/python -m pytest -q
~~~

Expected: all tests pass with only the repository's existing skips.

- [ ] **Step 4: Check patch formatting and repository scope**

Run:

~~~bash
git diff --check
git status --short --branch
~~~

Expected: no diff-check errors. The only unrelated status entry remains the pre-existing untracked artifacts/ directory.

- [ ] **Step 5: Commit any verification-only corrections**

Only if review or tests required a correction, repeat its focused RED/GREEN test and commit only the affected source and test files:

~~~bash
git add config/default.toml \
  src/funasr_transcriber/config.py \
  src/funasr_transcriber/firered_parallel.py \
  src/funasr_transcriber/service.py \
  tests/unit/test_config.py \
  tests/unit/test_firered_parallel.py \
  tests/unit/test_service.py
git commit -m "fix: complete weighted dual AED routing"
~~~

If no correction was required, create no empty commit.

---

### Task 5: Run the controlled GPU0 benchmark without refitting

**Files:**
- Create ignored output: outputs/measurements/20260909T-server-weighted-dual-aed-punc-overlap-liuyangyang/config.toml
- Create ignored outputs beneath: outputs/measurements/20260909T-server-weighted-dual-aed-punc-overlap-liuyangyang/
- Do not modify tracked source or test files during the benchmark

**Interfaces:**
- Input audio: audio_test/刘洋洋.wav
- Smoke input: outputs/measurements/20260909T-server-weighted-single-aed-punc-overlap-liuyangyang/smoke.wav
- Runtime: physical cuda:0, torch.bfloat16, use_half=true
- Configuration: aed_scheduler=weighted_duration and aed_worker_count=2
- Produces: fresh-1, steady-1, steady-2, steady-3, GPU0 samples, and summary.json

- [ ] **Step 1: Confirm repository and GPU0 admission without inspecting processes**

Run:

~~~bash
git status --short --branch
nvidia-smi --id=0 \
  --query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader,nounits
~~~

Expected: the branch contains only the planned commits plus the pre-existing untracked artifacts/. Record the exact GPU0 row. Do not run a compute-process query.

- [ ] **Step 2: Create the benchmark configuration**

Create the new file with the patch tool, without modifying any previous measurement:

~~~toml
[paths]
model_dir = "/mnt/hdd/jiajun/project/standalone_transcriber/models"
work_dir = "/mnt/hdd/jiajun/project/standalone_transcriber/outputs/measurements/20260909T-server-weighted-dual-aed-punc-overlap-liuyangyang/service-work"

[asr]
worker_python = "/mnt/hdd/jiajun/project/standalone_transcriber/.venv-asr/bin/python"
aed_scheduler = "weighted_duration"
aed_worker_count = 2

[service]
startup_timeout_seconds = 600

[diagnostics]
level = "basic"
~~~

Do not set a GPU-memory limit, safety budget override, or CUDA_VISIBLE_DEVICES.

- [ ] **Step 3: Run a separate 60-second smoke lifecycle**

Run:

~~~bash
.venv/bin/funasr-transcribe \
  outputs/measurements/20260909T-server-weighted-single-aed-punc-overlap-liuyangyang/smoke.wav \
  --config outputs/measurements/20260909T-server-weighted-dual-aed-punc-overlap-liuyangyang/config.toml \
  --output-dir outputs/measurements/20260909T-server-weighted-dual-aed-punc-overlap-liuyangyang/smoke-output \
  --diagnostics basic
~~~

Then stop only this benchmark service:

~~~bash
.venv/bin/funasr-transcribe service stop \
  --config outputs/measurements/20260909T-server-weighted-dual-aed-punc-overlap-liuyangyang/config.toml
~~~

Inspect only smoke-output/diagnostics.json and this benchmark's service logs. Verify two AED workers, one Punc owner, concurrent dispatch, no memory-planning fields, no warning, no OOM, and a valid final transcript. On any failure, preserve the directory and stop the plan.

- [ ] **Step 4: Start device-level GPU0 sampling for the formal lifecycle**

Run the exact allowed query repeatedly from a shell owned by this task, writing only to the new measurement directory:

~~~bash
while test ! -e outputs/measurements/20260909T-server-weighted-dual-aed-punc-overlap-liuyangyang/gpu0-sampler.stop
do
  nvidia-smi --id=0 \
    --query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader,nounits
  sleep 0.1
done
~~~

Redirect this loop to gpu0-samples.csv and retain its known shell PID. Do not query GPU processes and do not inspect other projects.

- [ ] **Step 5: Run one fresh and three steady one-hour transcriptions**

Use one service lifecycle. Run these commands sequentially and stop immediately on the first nonzero exit:

~~~bash
.venv/bin/funasr-transcribe audio_test/刘洋洋.wav \
  --config outputs/measurements/20260909T-server-weighted-dual-aed-punc-overlap-liuyangyang/config.toml \
  --output-dir outputs/measurements/20260909T-server-weighted-dual-aed-punc-overlap-liuyangyang/fresh-1 \
  --diagnostics basic

.venv/bin/funasr-transcribe audio_test/刘洋洋.wav \
  --config outputs/measurements/20260909T-server-weighted-dual-aed-punc-overlap-liuyangyang/config.toml \
  --output-dir outputs/measurements/20260909T-server-weighted-dual-aed-punc-overlap-liuyangyang/steady-1 \
  --diagnostics basic

.venv/bin/funasr-transcribe audio_test/刘洋洋.wav \
  --config outputs/measurements/20260909T-server-weighted-dual-aed-punc-overlap-liuyangyang/config.toml \
  --output-dir outputs/measurements/20260909T-server-weighted-dual-aed-punc-overlap-liuyangyang/steady-2 \
  --diagnostics basic

.venv/bin/funasr-transcribe audio_test/刘洋洋.wav \
  --config outputs/measurements/20260909T-server-weighted-dual-aed-punc-overlap-liuyangyang/config.toml \
  --output-dir outputs/measurements/20260909T-server-weighted-dual-aed-punc-overlap-liuyangyang/steady-3 \
  --diagnostics basic
~~~

Do not use --overwrite and do not retry a failed run.

- [ ] **Step 6: Stop only this service and finish sampling**

Run:

~~~bash
.venv/bin/funasr-transcribe service stop \
  --config outputs/measurements/20260909T-server-weighted-dual-aed-punc-overlap-liuyangyang/config.toml
~~~

Create gpu0-sampler.stop, wait for the known sampler PID, and take the exact end snapshot:

~~~bash
nvidia-smi --id=0 \
  --query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader,nounits
~~~

Do not signal, inspect, or manage any process not started by this benchmark.

- [ ] **Step 7: Validate and summarize the benchmark**

Read only the four new diagnostics.json files, transcript outputs, the three new project log files, and gpu0-samples.csv. Create summary.json containing:

- source commit and dirty-state statement;
- exact scheduler, worker topology, weights, batch constraints, batch count, and per-lane batch counts;
- runtime identity: cuda:0, torch.bfloat16, Python, Torch, CUDA, config fingerprint, FireRed source commit, and AED checkpoint SHA256;
- fresh and steady end-to-end, ASR, diarization, VAD, per-lane AED, AED makespan, Punc model, Punc queue, Punc tail, cleanup, and output times;
- predicted and actual lane imbalance;
- per-owner allocated and reserved peaks;
- sampled device-level admission, maximum used, minimum free, maximum utilization, end snapshot, sample count, observed sample interval, and sampler SHA256;
- transcript character count and SHA256 for every run;
- comparison against:
  - outputs/measurements/20260909T-server-weighted-single-aed-punc-overlap-liuyangyang/summary.json
  - outputs/measurements/20260824T-job-timing-cache-cleanup/summary-v2.json
- explicit OOM, warning, invalid-output, safety, and ownership results;
- explicit statements that weights were not refitted, validation and blind holdout were not read, candidate training was not started, no production manifest was written, and models/cost_models/ was not written.

A common scale error in both weights is diagnostic only. Do not change either weight unless a later approved train-only paired-data task demonstrates a changed longest-to-padded ratio.

- [ ] **Step 8: Verify the benchmark did not change tracked scope**

Run:

~~~bash
git status --short --branch
git diff --check
~~~

Expected: benchmark outputs remain Git ignored, tracked files are clean, and the pre-existing untracked artifacts/ remains untouched. Do not commit benchmark outputs unless the user separately requests a tracked report.
