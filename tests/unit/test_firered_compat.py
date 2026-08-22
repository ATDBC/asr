from __future__ import annotations

from funasr_transcriber.firered_compat import bf16_forced_align_compatible


class _Tensor:
    def __init__(self, dtype: str):
        self.dtype = dtype
        self.float_calls = 0

    def float(self):
        self.float_calls += 1
        return _Tensor("float32")


def test_forced_align_wrapper_casts_only_bf16_log_probs():
    received = []

    def forced_align(log_probs, targets, **kwargs):
        received.append((log_probs, targets, kwargs))
        return "alignment"

    wrapped = bf16_forced_align_compatible(forced_align, bf16_dtype="bf16")
    bf16 = _Tensor("bf16")
    fp32 = _Tensor("float32")
    targets = object()

    assert wrapped(bf16, targets, blank=0) == "alignment"
    assert received[-1][0].dtype == "float32"
    assert received[-1][1] is targets
    assert received[-1][2] == {"blank": 0}
    assert bf16.float_calls == 1

    assert wrapped(fp32, targets, blank=0) == "alignment"
    assert received[-1][0] is fp32
    assert fp32.float_calls == 0
