from __future__ import annotations

import torch

from funasr_transcriber.resource_profile import (
    ResourceSampler,
    module_inventory,
    parse_process_memory_mib,
)


def test_parse_process_memory_filters_to_registered_project_pids():
    rows = "\n".join(("100, 900", "200, 1234", "300, 456"))

    assert parse_process_memory_mib(rows, {100, 300}) == {
        100: 900,
        300: 456,
    }


def test_module_inventory_counts_parameters_buffers_and_dtypes():
    module = torch.nn.BatchNorm1d(2)

    inventory = module_inventory("batchnorm", module)

    assert inventory["name"] == "batchnorm"
    assert inventory["parameter_count"] == 4
    assert inventory["parameter_bytes"] == 16
    assert inventory["buffer_bytes"] == 24
    assert inventory["parameters"]["cpu:float32"]["tensors"] == 2


def test_sampler_only_emits_registered_project_processes():
    sampler = ResourceSampler(
        "cuda:0",
        interval_ms=200,
        pid_getters={
            "service": lambda: 100,
            "firered": lambda: 300,
        },
        query=lambda: (
            {"total_mib": 24576, "free_mib": 10000},
            {100: 1, 200: 2, 300: 3},
        ),
        rss_query=lambda pid: pid * 10,
    )

    sampler.sample_once()

    snapshot = sampler.snapshot()
    assert snapshot["process_peak_mib"] == {"service": 1, "firered": 3}
    assert snapshot["cpu_rss_peak_kib"] == {"service": 1000, "firered": 3000}
    assert snapshot["pid_observed"] == {"service": True, "firered": True}
    assert snapshot["sample_count"] == 1
    assert snapshot["other_project_baseline_mib"] == 14572
    assert snapshot["peak_total_used_mib"] == 14576
    assert snapshot["net_service_peak_mib"] == 4


def test_sampler_can_capture_stage_memory_trace_on_demand():
    samples = iter(
        (
            ({"total_mib": 24576, "free_mib": 10000}, {100: 100, 300: 300}),
            ({"total_mib": 24576, "free_mib": 9000}, {100: 100, 300: 500}),
        )
    )
    sampler = ResourceSampler(
        "cuda:0",
        interval_ms=200,
        pid_getters={"service": lambda: 100, "aed_worker_0": lambda: 300},
        query=lambda: next(samples),
        rss_query=lambda pid: None,
        trace_enabled=True,
    )

    sampler.sample_once()
    sampler.sample_once()

    trace = sampler.trace_stage(0.0, float("inf"))

    assert trace["sample_count"] == 2
    assert trace["summary"]["peak_used_mib"] == 15576
    assert trace["summary"]["minimum_free_mib"] == 9000
    assert trace["summary"]["near_peak_threshold_pct"] == 95
    assert trace["summary"]["near_peak_used_fraction"] == 0.5
    assert trace["summary"]["near_peak_process_fraction"]["aed_worker_0"] == 0.5
    assert trace["summary"]["process_peak_mib"] == {
        "service": 100,
        "aed_worker_0": 500,
    }
    assert len(trace["samples"]) == 2


def test_sampler_deduplicates_owner_labels_and_reports_budget_utilization():
    sampler = ResourceSampler(
        "cuda:0",
        interval_ms=200,
        pid_getters={
            "service": lambda: 100,
            "speaker": lambda: 100,
            "aed_worker_0": lambda: 300,
        },
        query=lambda: (
            {"total_mib": 24576, "free_mib": 12517},
            {100: 618, 300: 5000},
        ),
        rss_query=lambda pid: None,
        trace_enabled=True,
    )
    sampler.sample_once()
    sampler._query = lambda: (
        {"total_mib": 24576, "free_mib": 3695},
        {100: 618, 300: 7000},
    )
    sampler.sample_once()

    trace = sampler.trace_stage(
        0.0,
        float("inf"),
        effective_budget_mib=10980,
        safety_margin_mib=1536,
    )

    assert trace["summary"]["aed_transient_increment_mib"] == 8822
    assert trace["summary"]["budget_utilization_pct"] == 80.346
    assert trace["summary"]["remaining_over_safety_mib"] == 2159
    assert trace["samples"][-1]["net_service_mib"] == 7618
