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
