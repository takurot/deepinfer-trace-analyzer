import json
from decimal import Decimal as D

from builders import TraceBuilder

from deepinfer_trace_analyzer.latency import launch_latency
from deepinfer_trace_analyzer.loader import load_trace
from deepinfer_trace_analyzer.render import render_summary
from deepinfer_trace_analyzer.summary import Options, summarize


def report(tmp_path, builder, device=0):
    return launch_latency(load_trace(builder.write(tmp_path / "t.json")), device)


def base() -> TraceBuilder:
    return TraceBuilder().schedule(0).schedule(10_000)


def test_p50_and_p99_use_nearest_rank(tmp_path):
    builder = base()
    for latency in range(1, 101):
        builder.kernel("k", 1000 + latency * 20 + latency, 5, launch_ts=1000 + latency * 20)
    stats = report(tmp_path, builder).by_name[0]
    assert (stats.name, stats.count, stats.p50, stats.p99) == ("cudaLaunchKernel", 100, D(50), D(99))


def test_a_graph_launch_counts_once_with_the_delay_to_its_first_kernel(tmp_path):
    builder = base()
    corr = builder.corr()
    builder.launch(100, corr=corr, name="cudaGraphLaunch")
    for ts in (130, 120, 140):
        builder.gpu("graph_kernel", ts, 5, corr=corr)
    result = report(tmp_path, builder)
    graph = result.by_name[0]
    assert (graph.name, graph.count, graph.p50, graph.p99) == ("cudaGraphLaunch", 1, D(20), D(20))
    assert (result.matched_events, result.total_events) == (3, 3)


def test_launch_names_are_reported_separately(tmp_path):
    builder = base()
    builder.kernel("a", 110, 5, launch_ts=100)
    corr = builder.corr()
    builder.launch(200, corr=corr, name="cudaGraphLaunch").gpu("g", 250, 5, corr=corr)
    stats = {s.name: s for s in report(tmp_path, builder).by_name}
    assert (stats["cudaLaunchKernel"].p50, stats["cudaGraphLaunch"].p50) == (D(10), D(50))


def test_matched_fraction_counts_gpu_events_with_a_known_launch(tmp_path):
    builder = base()
    builder.kernel("a", 110, 5, launch_ts=100).kernel("b", 210, 5, launch_ts=200)
    builder.gpu("c", 310, 5).gpu("d", 410, 5, corr=999)
    result = report(tmp_path, builder)
    assert (result.matched_events, result.total_events) == (2, 4)
    assert result.matched_fraction == D("0.5")


def test_only_the_requested_device_is_counted(tmp_path):
    builder = base()
    builder.kernel("a", 110, 5, launch_ts=100, device=0).kernel("b", 250, 5, launch_ts=200, device=1)
    assert report(tmp_path, builder, device=1).by_name[0].p50 == D(50)
    assert report(tmp_path, builder, device=0).by_name[0].p50 == D(10)


def test_no_gpu_events_gives_an_empty_report(tmp_path):
    result = report(tmp_path, base())
    assert result.by_name == [] and result.matched_fraction is None


def test_the_earliest_cpu_event_for_a_correlation_is_the_launch(tmp_path):
    builder = base()
    builder.launch(160, corr=1, name="cudaLaunchKernel").launch(100, corr=1, name="cudaGraphLaunch")
    builder.gpu("k", 150, 5, corr=1)
    assert [(s.name, s.p50) for s in report(tmp_path, builder).by_name] == [("cudaGraphLaunch", D(50))]


def test_summary_carries_the_latency_and_renders_it(tmp_path):
    builder = base()
    builder.kernel("deep_gemm", 110, 5, launch_ts=100).gpu("orphan", 200, 5)
    path = builder.write(tmp_path / "t.json")
    summary = summarize(path, Options(prefill_steps=0))
    assert summary.latency.matched_fraction == D("0.5")
    text = render_summary(summary, "table")
    assert "launch latency" in text and "cudaLaunchKernel" in text and "50.0%" in text
    document = json.loads(render_summary(summary, "json"))
    entry = document["launch_latency"]
    assert D(entry["matched_fraction"]) == D("0.5")
    assert entry["by_name"][0]["name"] == "cudaLaunchKernel" and D(entry["by_name"][0]["p50"]) == 10
