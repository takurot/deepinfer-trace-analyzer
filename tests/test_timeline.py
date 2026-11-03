import random
from decimal import Decimal as D

import pytest

from builders import TraceBuilder
from deepinfer_trace_analyzer.loader import load_trace
from deepinfer_trace_analyzer.rules import Classifier
from deepinfer_trace_analyzer.steps import split_steps
from deepinfer_trace_analyzer.timeline import decompose


def run(tmp_path, builder, *, device=0, skip_steps=0, prefill_steps=0):
    trace = load_trace(builder.write(tmp_path / "t.json"))
    steps = split_steps(trace, prefill_steps=prefill_steps, skip_steps=skip_steps)
    return decompose(trace, steps, Classifier.default(), device)


def two_stream_step() -> TraceBuilder:
    builder = TraceBuilder().schedule(0).schedule(100)
    builder.gpu("deep_gemm_fp8", 10, 30, stream=7)
    builder.gpu("nccl_all_reduce", 30, 30, stream=8)
    builder.memcpy(70, 5)
    builder.gpu("elementwise", 110, 20).gpu("elementwise", 140, 10)
    return builder


def test_overlapping_compute_and_comm_split_into_exclusive_and_inclusive(tmp_path):
    first = run(tmp_path, two_stream_step()).steps[0]
    assert first.step_time == D(100)
    assert first.busy_union == D(55)
    assert first.idle == D(45)
    assert first.exclusive == {"gemm.deepgemm": D(30), "comm.nccl": D(20), "memcpy": D(5)}
    assert first.inclusive == {"gemm.deepgemm": D(30), "comm.nccl": D(30), "memcpy": D(5)}
    assert first.overlap == D(10)


def test_last_window_runs_to_the_final_event(tmp_path):
    last = run(tmp_path, two_stream_step()).steps[1]
    assert (last.step_time, last.busy_union, last.idle) == (D(50), D(30), D(20))
    assert last.exclusive == {"other": D(30)}


def test_memcpy_beats_comm_and_deepgemm_beats_other_gemm(tmp_path):
    builder = TraceBuilder().schedule(0).schedule(100)
    builder.gpu("nccl", 0, 10).memcpy(0, 10)
    builder.gpu("cutlass_gemm", 20, 10).gpu("deep_gemm", 20, 10)
    builder.gpu("tail", 99, 1)
    step = run(tmp_path, builder).steps[0]
    assert step.exclusive["memcpy"] == D(10) and "comm.nccl" not in step.exclusive
    assert step.exclusive["gemm.deepgemm"] == D(10) and "gemm.other" not in step.exclusive


def test_a_kernel_crossing_a_boundary_is_cut_between_the_windows(tmp_path):
    builder = TraceBuilder().schedule(0).schedule(100)
    builder.kernel("deep_gemm", 90, 30, launch_ts=85)
    first, last = run(tmp_path, builder).steps
    assert first.inclusive == {"gemm.deepgemm": D(10)}
    assert last.inclusive == {"gemm.deepgemm": D(20)}


def test_kernels_launched_in_an_earlier_step_count_as_spill(tmp_path):
    builder = TraceBuilder().schedule(0).schedule(100)
    builder.kernel("deep_gemm", 90, 30, launch_ts=85)
    builder.kernel("deep_gemm", 130, 10, launch_ts=105)
    first, last = run(tmp_path, builder).steps
    assert (first.spill_us, first.matched_us) == (D(0), D(10))
    assert (last.spill_us, last.matched_us) == (D(20), D(30))


def test_kernels_without_a_launch_are_unmatched_not_spill(tmp_path):
    builder = TraceBuilder().schedule(0).schedule(100)
    builder.kernel("a", 10, 10)
    builder.gpu("b", 30, 5, corr=99)
    builder.kernel("c", 50, 20, launch_ts=5)
    step = run(tmp_path, builder).steps[0]
    assert step.gpu_us == D(35)
    assert step.unmatched_us == D(15)
    assert (step.spill_us, step.matched_us) == (D(0), D(20))


def test_gpu_time_before_the_first_schedule_is_dropped(tmp_path):
    builder = TraceBuilder().schedule(100).schedule(200)
    builder.gpu("early", 80, 40).gpu("later", 150, 10)
    result = run(tmp_path, builder)
    assert result.dropped_gpu_us == D(20)
    assert result.steps[0].inclusive == {"other": D(30)}


def test_skipped_steps_are_not_reported_as_dropped(tmp_path):
    builder = TraceBuilder().schedule(0).schedule(100).schedule(200)
    builder.gpu("in_skipped_step", 10, 20).gpu("kept", 150, 10).gpu("edge", 90, 30)
    result = run(tmp_path, builder, skip_steps=1)
    assert result.dropped_gpu_us == D(0)
    assert [s.step.index for s in result.steps] == [1, 2]
    assert result.steps[0].inclusive == {"other": D(20 + 10)}


def test_devices_are_analysed_independently(tmp_path):
    builder = TraceBuilder().schedule(0).schedule(100)
    builder.gpu("deep_gemm", 10, 20, device=0).gpu("nccl", 10, 50, device=1).gpu("tail", 99, 1, device=0)
    assert run(tmp_path, builder, device=0).steps[0].exclusive["gemm.deepgemm"] == D(20)
    assert run(tmp_path, builder, device=1).steps[0].exclusive == {"comm.nccl": D(50)}


def test_a_device_with_no_gpu_events_is_all_idle(tmp_path):
    builder = TraceBuilder().schedule(0).schedule(100).gpu("k", 10, 5)
    step = run(tmp_path, builder, device=3).steps[0]
    assert (step.busy_union, step.idle, step.exclusive) == (D(0), D(100), {})


def test_bubble_time_is_summed_per_window_and_clipped(tmp_path):
    builder = TraceBuilder().schedule(0).schedule(100)
    builder.bubble(10, 5).bubble(90, 30).bubble(150, 20).gpu("k", 199, 1)
    first, last = run(tmp_path, builder).steps
    assert first.bubble == D(15)
    assert last.bubble == D(40)


def test_zero_duration_kernels_are_harmless(tmp_path):
    builder = TraceBuilder().schedule(0).schedule(100).gpu("k", 10, 0).gpu("k2", 99, 1)
    step = run(tmp_path, builder).steps[0]
    assert step.busy_union == D(1)


def test_decimal_microsecond_precision_is_exact(tmp_path):
    builder = TraceBuilder().schedule("1000.000001").schedule("1100.000002")
    builder.gpu("deep_gemm", "1010.000003", "20.000004").gpu("tail", "1099.5", "1")
    first = run(tmp_path, builder).steps[0]
    assert first.step_time == D("100.000001")
    assert first.exclusive["gemm.deepgemm"] == D("20.000004")


@pytest.mark.parametrize("seed", range(8))
def test_exclusive_plus_idle_always_equals_step_time(tmp_path, seed):
    rng = random.Random(seed)
    builder = TraceBuilder()
    for start in range(0, 1000, 100):
        builder.schedule(start)
    names = ["deep_gemm", "cutlass_gemm", "nccl", "flash_mla", "tilelang", "elementwise", "topk"]
    for _ in range(300):
        start = D(rng.randrange(0, 105000)) / 100
        dur = D(rng.randrange(0, 4000)) / 100
        stream = rng.randrange(3)
        if rng.random() < 0.1:
            builder.memcpy(start, dur, stream=stream)
        else:
            builder.gpu(rng.choice(names), start, dur, stream=stream)
    for step in run(tmp_path, builder).steps:
        assert sum(step.exclusive.values()) + step.idle == step.step_time
        assert step.busy_union == sum(step.exclusive.values())
        assert step.overlap == sum(step.inclusive.values()) - step.busy_union
        assert step.overlap >= 0 and step.idle >= 0
        assert all(step.exclusive[c] <= step.inclusive[c] for c in step.exclusive)
