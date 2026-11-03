from real_trace_report import Status, evaluate, main

from builders import TraceBuilder
from deepinfer_trace_analyzer.summary import Options, summarize


def checks_for(tmp_path, builder, **options):
    path = builder.write(tmp_path / "t.json")
    summary = summarize(path, Options(prefill_steps=1, **options))
    return {c.name: c for c in evaluate(summary)}


def healthy() -> TraceBuilder:
    builder = TraceBuilder().schedule(0).schedule(100).schedule(200)
    builder.kernel("deep_gemm", 10, 40, launch_ts=5).kernel("nccl", 30, 30, launch_ts=6, stream=8)
    builder.kernel("deep_gemm", 110, 50, launch_ts=105).kernel("flash_mla", 160, 20, launch_ts=155)
    builder.kernel("deep_gemm", 210, 30, launch_ts=205)
    return builder


def test_a_healthy_trace_passes_every_hard_check(tmp_path):
    checks = checks_for(tmp_path, healthy())
    assert checks["other share, prefill"].status is Status.PASS
    assert checks["other share, decode"].status is Status.PASS
    assert checks["unmatched share, decode"].status is Status.PASS
    assert all(c.status is not Status.FAIL for c in checks.values())


def test_too_much_unclassified_time_fails_and_names_the_share(tmp_path):
    builder = TraceBuilder().schedule(0).schedule(100).schedule(200)
    builder.kernel("deep_gemm", 10, 30, launch_ts=5).kernel("mystery", 110, 60, launch_ts=105)
    builder.kernel("mystery", 210, 20, launch_ts=205)
    check = checks_for(tmp_path, builder)["other share, decode"]
    assert check.status is Status.FAIL
    assert "limit 10%" in check.detail and "%" in check.detail


def test_other_share_exactly_at_the_limit_passes(tmp_path):
    builder = TraceBuilder().schedule(0)
    builder.kernel("deep_gemm", 0, 90, launch_ts=-1).kernel("mystery", 90, 10, launch_ts=-1)
    path = builder.write(tmp_path / "t.json")
    checks = {c.name: c for c in evaluate(summarize(path, Options(prefill_steps=0)))}
    assert checks["other share, decode"].status is Status.PASS


def test_unmatched_launches_warn_rather_than_fail(tmp_path):
    builder = TraceBuilder().schedule(0).schedule(100).schedule(200)
    builder.kernel("deep_gemm", 10, 30).kernel("deep_gemm", 110, 30).kernel("deep_gemm", 210, 30)
    check = checks_for(tmp_path, builder)["unmatched share, decode"]
    assert check.status is Status.WARN


def test_graph_launches_and_spill_are_reported_as_information(tmp_path):
    builder = healthy()
    corr = builder.corr()
    builder.launch(150, corr=corr, name="cudaGraphLaunch").gpu("graph_kernel", 160, 5, corr=corr)
    checks = checks_for(tmp_path, builder)
    assert checks["cudaGraphLaunch seen"].status is Status.INFO
    assert "1 launches" in checks["cudaGraphLaunch seen"].detail
    assert checks["spill share, decode"].status is Status.INFO


def test_a_missing_graph_launch_is_called_out(tmp_path):
    check = checks_for(tmp_path, healthy())["cudaGraphLaunch seen"]
    assert check.status is Status.WARN and "none" in check.detail


def test_the_last_step_is_compared_with_the_others(tmp_path):
    builder = TraceBuilder().schedule(0).schedule(100).schedule(200).schedule(300)
    builder.kernel("deep_gemm", 10, 30, launch_ts=5).kernel("deep_gemm", 110, 30, launch_ts=105)
    builder.kernel("deep_gemm", 210, 30, launch_ts=205).kernel("deep_gemm", 300, 30, launch_ts=299)
    check = checks_for(tmp_path, builder)["last step length"]
    assert check.status is Status.WARN and "x the median" in check.detail


def test_main_exits_nonzero_only_on_a_failed_check(tmp_path, capsys):
    good = healthy().write(tmp_path / "good.json")
    bad = (
        TraceBuilder()
        .schedule(0)
        .schedule(100)
        .kernel("mystery", 10, 80, launch_ts=5)
        .write(tmp_path / "bad.json")
    )
    assert main([str(good), "--prefill-steps", "1"]) == 0
    assert "PASS" in capsys.readouterr().out
    assert main([str(bad), "--prefill-steps", "0"]) == 1
    assert "FAIL" in capsys.readouterr().out
