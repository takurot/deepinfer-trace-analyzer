from decimal import Decimal as D

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


def stepped(durations, *, kernel="deep_gemm"):
    """One schedule per entry; the last window is cut by a kernel that ends exactly `last` us in."""
    builder = TraceBuilder()
    start = 0
    for index, duration in enumerate(durations):
        builder.schedule(start)
        end = start + duration
        if index == len(durations) - 1:
            builder.kernel(kernel, end - 1, 1, launch_ts=start)
        start = end
    return builder


def last_step_check(tmp_path, durations, **options):
    path = stepped(durations).write(tmp_path / "t.json")
    summary = summarize(path, Options(prefill_steps=0, **options))
    return {c.name: c for c in evaluate(summary)}["last step length"]


def test_the_last_step_is_judged_against_the_steps_of_its_own_phase(tmp_path):
    path = stepped([1000, 1000, 200, 200, 200]).write(tmp_path / "t.json")
    summary = summarize(path, Options(prefill_steps=2))
    check = {c.name: c for c in evaluate(summary)}["last step length"]
    assert check.status is Status.PASS and "1.00x" in check.detail


def test_last_step_band_is_zero_point_eight_to_one_point_two_inclusive(tmp_path):
    for last, status in ((80, Status.PASS), (79, Status.WARN), (120, Status.PASS), (121, Status.WARN)):
        assert last_step_check(tmp_path, [100, 100, 100, last]).status is status, last


def test_the_median_not_the_mean_sets_the_reference(tmp_path):
    check = last_step_check(tmp_path, [100, 100, 1000, 100])
    assert check.status is Status.PASS and "1.00x" in check.detail


def test_two_steps_in_a_phase_are_too_few_and_three_are_enough(tmp_path):
    assert last_step_check(tmp_path, [100, 100]).status is Status.INFO
    assert last_step_check(tmp_path, [100, 100, 100]).status is Status.PASS
    assert last_step_check(tmp_path, [100, 100, 300]).status is Status.WARN


def single_step(tmp_path, *, matched, unmatched):
    builder = TraceBuilder().schedule(100)
    builder.kernel("deep_gemm", 100, D(matched), launch_ts=99)
    builder.gpu("deep_gemm", 100 + D(matched), D(unmatched))
    path = builder.write(tmp_path / "t.json")
    return {c.name: c for c in evaluate(summarize(path, Options(prefill_steps=0)))}["unmatched share, decode"]


def test_unmatched_share_warns_only_above_five_percent(tmp_path):
    assert single_step(tmp_path, matched="95", unmatched="5").status is Status.PASS
    assert single_step(tmp_path, matched="94.9", unmatched="5.1").status is Status.WARN


def test_a_trace_without_gpu_events_fails_instead_of_passing_vacuously(tmp_path):
    path = TraceBuilder().schedule(0).schedule(100).write(tmp_path / "t.json")
    checks = {c.name: c for c in evaluate(summarize(path, Options(prefill_steps=0)))}
    assert checks["GPU events present"].status is Status.FAIL
    assert main([str(path), "--prefill-steps", "0"]) == 1


def test_a_missing_decode_phase_fails_when_the_split_was_given(tmp_path):
    path = healthy().write(tmp_path / "t.json")
    summary = summarize(path, Options(prefill_steps=3))
    check = {c.name: c for c in evaluate(summary)}["decode phase present"]
    assert check.status is Status.FAIL
    assert main([str(path), "--prefill-steps", "3"]) == 1


def test_a_missing_decode_phase_warns_when_the_split_was_not_given(tmp_path):
    path = healthy().write(tmp_path / "t.json")
    check = {c.name: c for c in evaluate(summarize(path, Options()))}["decode phase present"]
    assert check.status is Status.WARN


def test_the_decode_phase_is_reported_present_when_it_is(tmp_path):
    check = checks_for(tmp_path, healthy())["decode phase present"]
    assert check.status is Status.PASS


def two_devices() -> TraceBuilder:
    builder = TraceBuilder().schedule(0).schedule(100).schedule(200)
    builder.kernel("deep_gemm", 10, 80, launch_ts=5).kernel("deep_gemm", 110, 80, launch_ts=105)
    builder.kernel("mystery", 20, 70, launch_ts=6, device=1).kernel(
        "mystery", 120, 70, launch_ts=106, device=1
    )
    builder.kernel("deep_gemm", 210, 20, launch_ts=205).kernel("mystery", 220, 20, launch_ts=206, device=1)
    return builder


def test_main_device_selects_which_gpu_is_judged(tmp_path):
    path = str(two_devices().write(tmp_path / "t.json"))
    assert main([path, "--prefill-steps", "1"]) == 0
    assert main([path, "--prefill-steps", "1", "--device", "1"]) == 1


def test_main_skip_steps_leaves_early_steps_out(tmp_path):
    builder = TraceBuilder().schedule(0).schedule(100).schedule(200).schedule(300)
    builder.kernel("mystery", 10, 80, launch_ts=5).kernel("deep_gemm", 110, 80, launch_ts=105)
    builder.kernel("deep_gemm", 210, 80, launch_ts=205).kernel("deep_gemm", 310, 20, launch_ts=305)
    path = str(builder.write(tmp_path / "t.json"))
    assert main([path, "--prefill-steps", "0"]) == 1
    assert main([path, "--prefill-steps", "0", "--skip-steps", "1"]) == 0


def test_main_rules_file_can_classify_the_unknown_kernels(tmp_path):
    builder = TraceBuilder().schedule(0).schedule(100)
    builder.kernel("mystery", 10, 80, launch_ts=5).kernel("deep_gemm", 110, 5, launch_ts=105)
    path = str(builder.write(tmp_path / "t.json"))
    rules = tmp_path / "rules.toml"
    rules.write_text('[[rule]]\ncategory = "mine"\npattern = "mystery"\n')
    assert main([path, "--prefill-steps", "0"]) == 1
    assert main([path, "--prefill-steps", "0", "--rules", str(rules)]) == 0


def test_main_lists_unclassified_kernels_and_prints_warnings_to_stderr(tmp_path, capsys):
    path = str(
        TraceBuilder().schedule(0).kernel("mystery_kernel", 1, 5, launch_ts=0).write(tmp_path / "t.json")
    )
    main([path, "--prefill-steps", "0"])
    captured = capsys.readouterr()
    assert "top unclassified kernels" in captured.out and "mystery_kernel" in captured.out
    path = str(TraceBuilder().schedule(0).schedule(10).write(tmp_path / "e.json"))
    main([path, "--prefill-steps", "0"])
    assert "warning:" in capsys.readouterr().err


def test_main_reports_unreadable_input_with_exit_two(tmp_path, capsys):
    assert main([str(tmp_path / "missing.json")]) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err.startswith("error:")
