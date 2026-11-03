import json
from decimal import Decimal as D

from make_synthetic_trace import generate

from deepinfer_trace_analyzer.loader import load_trace
from deepinfer_trace_analyzer.render import render_summary
from deepinfer_trace_analyzer.summary import Options, summarize


def test_generated_trace_reproduces_the_numbers_it_promises(tmp_path):
    path = tmp_path / "synthetic.json"
    expected = generate(path, steps=3, blocks_per_step=4)

    summary = summarize(path, Options(prefill_steps=0))
    document = json.loads(render_summary(summary, "json"))
    real_steps = document["steps"][:3]
    assert len(document["steps"]) == 4
    for step in real_steps:
        assert D(step["step_time"]) == D(expected["step_time_us"])
        assert {c: D(v) for c, v in step["exclusive"].items()} == {
            c: D(v) for c, v in expected["exclusive_us"].items()
        }
        assert {c: D(v) for c, v in step["inclusive"].items()} == {
            c: D(v) for c, v in expected["inclusive_us"].items()
        }
        assert D(step["idle"]) == D(expected["idle_us"])
        assert D(step["overlap"]) == D(expected["overlap_us"])

    trace = load_trace(path)
    assert len(trace.gpu_events) == expected["gpu_events"]
    assert summary.latency.matched_fraction == 1


def test_the_file_is_valid_json_with_the_requested_event_shapes(tmp_path):
    path = tmp_path / "synthetic.json"
    expected = generate(path, steps=2, blocks_per_step=2)
    events = json.loads(path.read_text())["traceEvents"]
    assert len(events) == expected["events"]
    kinds = {event.get("cat") for event in events}
    assert {"kernel", "gpu_memcpy", "cuda_runtime", "cpu_op", "user_annotation"} <= kinds
    assert all(isinstance(e["ts"], str) for e in events if e.get("ph") == "X")


def test_launches_precede_their_kernels_by_one_microsecond(tmp_path):
    path = tmp_path / "synthetic.json"
    generate(path, steps=2, blocks_per_step=3)
    by_name = {s.name: s for s in summarize(path, Options(prefill_steps=0)).latency.by_name}
    assert set(by_name) == {"cudaLaunchKernel", "cudaMemcpyAsync"}
    assert all(s.p50 == 1 and s.p99 == 1 for s in by_name.values())
    assert by_name["cudaMemcpyAsync"].count == 6 and by_name["cudaLaunchKernel"].count == 30


def test_every_correlation_id_is_used_once(tmp_path):
    path = tmp_path / "synthetic.json"
    generate(path, steps=2, blocks_per_step=3)
    ids = [
        e["args"]["correlation"]
        for e in json.loads(path.read_text())["traceEvents"]
        if e.get("cat") == "kernel"
    ]
    assert len(ids) == len(set(ids)) == 30
