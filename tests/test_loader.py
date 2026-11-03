from decimal import Decimal

import pytest

from builders import TraceBuilder
from deepinfer_trace_analyzer.errors import AnalysisError, InputError
from deepinfer_trace_analyzer.loader import load_trace


def sample() -> TraceBuilder:
    builder = TraceBuilder()
    builder.schedule(100).schedule(200)
    builder.launch_forward(110, 5, reqs=4, tokens=64)
    builder.kernel("deep_gemm_fp8", 120, 30, launch_ts=115, corr=7)
    builder.memcpy(150, 4)
    builder.bubble(160, 10)
    return builder


def test_loads_plain_json_and_splits_cpu_and_gpu(tmp_path):
    trace = load_trace(sample().write(tmp_path / "t.json"))
    assert trace.schedule_starts == [Decimal(100), Decimal(200)]
    assert [(e.name, e.cat, e.device, e.ts, e.dur, e.correlation) for e in trace.gpu_events] == [
        ("deep_gemm_fp8", "kernel", 0, Decimal(120), Decimal(30), 7),
        ("Memcpy HtoD (Pinned -> Device)", "gpu_memcpy", 0, Decimal(150), Decimal(4), None),
    ]
    assert {e.name for e in trace.cpu_events} >= {"cudaLaunchKernel", "deepinfer:bubble"}


def test_gzip_matches_plain(tmp_path):
    plain = load_trace(sample().write(tmp_path / "t.json"))
    zipped = load_trace(sample().write(tmp_path / "t.json.gz", gz=True))
    assert plain.gpu_events == zipped.gpu_events
    assert plain.schedule_starts == zipped.schedule_starts


def test_torch_npu_shape_array_top_level_with_string_values(tmp_path):
    path = sample().write(tmp_path / "npu.json", as_list=True, stringify=True)
    trace = load_trace(path)
    assert trace.schedule_starts == [Decimal(100), Decimal(200)]
    assert trace.gpu_events[0].correlation == 7
    assert trace.gpu_events[0].dur == Decimal(30)


def test_decimal_precision_is_kept(tmp_path):
    builder = TraceBuilder().schedule("1000.123456")
    builder.gpu("k", "1001.000001", "0.000002")
    trace = load_trace(builder.write(tmp_path / "t.json", stringify=True))
    assert trace.gpu_events[0].ts == Decimal("1001.000001")
    assert trace.gpu_events[0].dur == Decimal("0.000002")


def test_float_values_keep_decimal_precision(tmp_path):
    path = tmp_path / "t.json"
    path.write_text(
        '{"traceEvents":[{"ph":"X","name":"deepinfer:schedule","pid":1,"ts":1.000000001,"dur":1},'
        '{"ph":"X","cat":"kernel","name":"k","pid":0,"ts":2.5,"dur":0.1}]}'
    )
    trace = load_trace(path)
    assert trace.schedule_starts == [Decimal("1.000000001")]
    assert trace.gpu_events[0].dur == Decimal("0.1")


def test_ignores_non_complete_events_profiler_tracks_and_annotations(tmp_path):
    builder = sample()
    builder.raw({"ph": "M", "name": "process_name", "pid": 0, "args": {"name": "GPU 0"}})
    builder.raw({"ph": "C", "name": "counter", "pid": 0, "ts": 1, "args": {"v": 1}})
    builder.raw({"ph": "s", "cat": "ac2g", "name": "ac2g", "id": 7, "pid": 0, "tid": 1, "ts": 120})
    builder.raw({"ph": "X", "cat": "kernel", "name": "profiler", "pid": "Spans", "tid": 0, "ts": 1, "dur": 1})
    builder.gpu("annotation", 100, 90, cat="gpu_user_annotation")
    trace = load_trace(builder.write(tmp_path / "t.json"))
    assert [e.name for e in trace.gpu_events] == ["deep_gemm_fp8", "Memcpy HtoD (Pinned -> Device)"]


def test_gpu_events_on_several_devices(tmp_path):
    builder = sample()
    builder.kernel("nccl_all_reduce", 300, 10, device=3)
    trace = load_trace(builder.write(tmp_path / "t.json"))
    assert trace.devices == [0, 3]


def test_unusable_gpu_events_are_reported_not_fatal(tmp_path):
    builder = sample()
    builder.raw({"ph": "X", "cat": "kernel", "name": "weird", "pid": "gpu-a", "tid": 1, "ts": 1, "dur": 1})
    builder.raw({"ph": "X", "cat": "kernel", "name": "nodur", "pid": 0, "tid": 1, "ts": 1})
    trace = load_trace(builder.write(tmp_path / "t.json"))
    assert len(trace.gpu_events) == 2
    assert len(trace.warnings) == 1 and "2 GPU events" in trace.warnings[0]


def test_missing_file_is_input_error(tmp_path):
    with pytest.raises(InputError, match="nope.json"):
        load_trace(tmp_path / "nope.json")


@pytest.mark.parametrize("text", ["{not json", '{"traceEvents": [', '"just a string"', '{"other": []}', ""])
def test_unreadable_documents_are_input_errors(tmp_path, text):
    path = tmp_path / "bad.json"
    path.write_text(text)
    with pytest.raises(InputError):
        load_trace(path)


def test_truncated_gzip_is_input_error(tmp_path):
    good = sample().write(tmp_path / "t.json.gz", gz=True)
    bad = tmp_path / "cut.json.gz"
    bad.write_bytes(good.read_bytes()[:-10])
    with pytest.raises(InputError):
        load_trace(bad)


def test_trace_without_schedule_is_analysis_error(tmp_path):
    builder = TraceBuilder().kernel("k", 1, 1)
    with pytest.raises(AnalysisError, match="deepinfer:schedule"):
        load_trace(builder.write(tmp_path / "t.json"))
