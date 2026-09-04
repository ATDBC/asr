from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


_PATH = Path(__file__).parents[2] / "tools" / "benchmark_aed_budget_matrix.py"
_SPEC = importlib.util.spec_from_file_location("aed_budget_matrix", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_matrix_manifest_has_one_record_per_fraction_and_repeat():
    records = _MODULE.build_matrix_manifest([0.6, 0.8], repeats=2)

    assert len(records) == 4
    assert [(item["fraction"], item["repeat"]) for item in records] == [
        (0.6, 1),
        (0.6, 2),
        (0.8, 1),
        (0.8, 2),
    ]


def test_matrix_manifest_rejects_unsafe_fraction_or_repeat():
    with pytest.raises(ValueError, match="fraction"):
        _MODULE.build_matrix_manifest([0.05], repeats=1)
    with pytest.raises(ValueError, match="repeats"):
        _MODULE.build_matrix_manifest([0.8], repeats=0)
