import copy
import json

import pytest
from bench_summary import _check
from make_synthetic_trace import generate

from deepinfer_trace_analyzer.render import render_summary
from deepinfer_trace_analyzer.summary import Options, summarize


@pytest.fixture
def pair(tmp_path):
    path = tmp_path / "synthetic.json"
    expected = generate(path, steps=3, blocks_per_step=4)
    document = json.loads(render_summary(summarize(path, Options(prefill_steps=0)), "json"))
    return document, expected


def test_the_true_output_passes(pair):
    document, expected = pair
    assert _check(document, expected) == []


@pytest.mark.parametrize(
    "mutate",
    [
        lambda step: step.update(step_time="121"),
        lambda step: step.update(idle="1"),
        lambda step: step.update(overlap="1"),
        lambda step: step["exclusive"].update({"comm.nccl": "999"}),
        lambda step: step["inclusive"].update({"comm.nccl": "999"}),
        lambda step: step.update(busy_union="1"),
    ],
)
def test_any_wrong_number_in_a_real_step_is_reported(pair, mutate):
    document, expected = pair
    broken = copy.deepcopy(document)
    mutate(broken["steps"][1])
    assert _check(broken, expected)


def test_a_missing_or_extra_step_is_reported(pair):
    document, expected = pair
    short = {**document, "steps": document["steps"][:-1]}
    assert _check(short, expected)


def test_a_wrong_trailing_step_is_reported(pair):
    document, expected = pair
    broken = copy.deepcopy(document)
    broken["steps"][-1]["exclusive"] = {"other": "5"}
    assert _check(broken, expected)
