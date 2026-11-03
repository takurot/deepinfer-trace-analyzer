import pytest

from deepinfer_trace_analyzer.errors import AnalysisError, InputError
from deepinfer_trace_analyzer.rules import Classifier


@pytest.mark.parametrize(
    ("name", "cat", "expected"),
    [
        ("ncclDevKernel_AllReduce_Sum_f32", "kernel", "comm.nccl"),
        ("deep_ep::internode::dispatch", "kernel", "comm.deepep"),
        ("void flash_mla_decode_kernel", "kernel", "attn.flashmla"),
        ("deep_gemm::fp8_gemm_nt", "kernel", "gemm.deepgemm"),
        ("m_grouped_fp8_gemm", "kernel", "gemm.deepgemm"),
        ("sm90_xmma_gemm_bf16", "kernel", "gemm.other"),
        ("cutlass_80_tensorop_s16816gemm", "kernel", "gemm.other"),
        ("tilelang_fused_moe", "kernel", "tilelang"),
        ("sampling_topk_kernel", "kernel", "sampler"),
        ("vectorized_elementwise_kernel", "kernel", "other"),
        ("Memcpy HtoD", "gpu_memcpy", "memcpy"),
        ("Memset (Device)", "gpu_memset", "memcpy"),
    ],
)
def test_default_rules(name, cat, expected):
    assert Classifier.default().classify(name, cat) == expected


def test_matching_is_case_insensitive_and_first_rule_wins():
    classifier = Classifier.default()
    assert classifier.classify("NCCL_deep_gemm", "kernel") == "comm.nccl"
    assert classifier.classify("DEEP_GEMM", "kernel") == "gemm.deepgemm"


def test_memcpy_category_ignores_name_rules():
    assert Classifier.default().classify("nccl_copy", "gpu_memcpy") == "memcpy"


def test_user_rules_run_before_defaults(tmp_path):
    rules = tmp_path / "rules.toml"
    rules.write_text('[[rule]]\ncategory = "custom.gemm"\npattern = "deep_gemm"\n')
    classifier = Classifier.from_file(rules)
    assert classifier.classify("deep_gemm_x", "kernel") == "custom.gemm"
    assert classifier.classify("nccl_x", "kernel") == "comm.nccl"


def test_no_default_rules(tmp_path):
    rules = tmp_path / "rules.toml"
    rules.write_text('[[rule]]\ncategory = "only"\npattern = "foo"\n')
    classifier = Classifier.from_file(rules, use_defaults=False)
    assert classifier.classify("foo", "kernel") == "only"
    assert classifier.classify("nccl_x", "kernel") == "other"


def test_no_default_rules_without_a_file_has_only_other():
    classifier = Classifier.from_file(None, use_defaults=False)
    assert classifier.classify("nccl_x", "kernel") == "other"
    assert classifier.classify("copy", "gpu_memcpy") == "memcpy"


def test_priority_orders_compute_then_memcpy_then_comm():
    classifier = Classifier.default()
    order = sorted(
        ["comm.nccl", "memcpy", "other", "tilelang", "gemm.deepgemm", "comm.deepep"], key=classifier.priority
    )
    assert order == ["gemm.deepgemm", "tilelang", "other", "memcpy", "comm.nccl", "comm.deepep"]


def test_user_categories_take_part_in_priority(tmp_path):
    rules = tmp_path / "rules.toml"
    rules.write_text(
        '[[rule]]\ncategory = "comm.mine"\npattern = "x"\n[[rule]]\ncategory = "mine"\npattern = "y"\n'
    )
    classifier = Classifier.from_file(rules)
    assert classifier.priority("mine") < classifier.priority("memcpy") < classifier.priority("comm.mine")


@pytest.mark.parametrize(
    "text",
    [
        "not = [valid",
        '[[rule]]\npattern = "x"\n',
        '[[rule]]\ncategory = "x"\n',
        '[[rule]]\ncategory = "x"\npattern = "("\n',
        '[[rule]]\ncategory = 3\npattern = "x"\n',
        '[[rule]]\ncategory = "other"\npattern = "x"\n',
        '[[rule]]\ncategory = "memcpy"\npattern = "x"\n',
    ],
)
def test_invalid_rule_files_are_analysis_errors(tmp_path, text):
    rules = tmp_path / "rules.toml"
    rules.write_text(text)
    with pytest.raises(AnalysisError):
        Classifier.from_file(rules)


def test_missing_rule_file_is_input_error(tmp_path):
    with pytest.raises(InputError):
        Classifier.from_file(tmp_path / "missing.toml")


def test_categories_are_listed_in_rule_order_then_memcpy_and_other():
    assert Classifier.default().categories() == [
        "comm.nccl",
        "comm.deepep",
        "attn.flashmla",
        "gemm.deepgemm",
        "gemm.other",
        "tilelang",
        "sampler",
        "memcpy",
        "other",
    ]


def test_user_categories_lead_the_category_list(tmp_path):
    rules = tmp_path / "rules.toml"
    rules.write_text(
        '[[rule]]\ncategory = "mine"\npattern = "x"\n[[rule]]\ncategory = "gemm.other"\npattern = "y"\n'
    )
    assert Classifier.from_file(rules).categories()[:3] == ["mine", "gemm.other", "comm.nccl"]


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("deepep_dispatch", "comm.deepep"),
        ("flashmla_fwd", "attn.flashmla"),
        ("sparse_attn_kernel", "attn.flashmla"),
        ("fp8_gemm_kernel", "gemm.deepgemm"),
        ("cutlass_universal", "gemm.other"),
        ("cublasLt_matmul", "gemm.other"),
        ("sampler_top_k_kernel", "sampler"),
        ("sampling_kernel", "sampler"),
        ("topk_kernel", "sampler"),
    ],
)
def test_every_alternative_in_the_default_patterns_matches(name, expected):
    assert Classifier.default().classify(name, "kernel") == expected


@pytest.mark.parametrize(
    "name", ["softmax_kernel", "moe_combine_x", "dispatch_tokens", "notify_peers", "tl_foo"]
)
def test_patterns_the_spec_removed_stay_unclassified(name):
    assert Classifier.default().classify(name, "kernel") == "other"


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("deep_ep_gemm", "comm.deepep"),
        ("flash_mla_gemm", "attn.flashmla"),
        ("tilelang_topk", "tilelang"),
        ("nccl_topk", "comm.nccl"),
        ("deep_gemm_topk", "gemm.deepgemm"),
    ],
)
def test_earlier_rules_win_when_two_match(name, expected):
    assert Classifier.default().classify(name, "kernel") == expected


def test_a_user_rule_reusing_a_default_category_keeps_the_earlier_position(tmp_path):
    rules = tmp_path / "rules.toml"
    rules.write_text('[[rule]]\ncategory = "tilelang"\npattern = "x"\n')
    classifier = Classifier.from_file(rules)
    assert classifier.priority("tilelang") < classifier.priority("gemm.deepgemm")


def test_user_categories_rank_before_default_categories_in_the_same_tier(tmp_path):
    rules = tmp_path / "rules.toml"
    rules.write_text('[[rule]]\ncategory = "mine"\npattern = "x"\n')
    classifier = Classifier.from_file(rules)
    assert classifier.priority("mine") < classifier.priority("gemm.deepgemm")


def test_a_kernel_named_like_a_copy_is_still_classified_by_name():
    assert Classifier.default().classify("Memcpy HtoD", "kernel") == "other"


def test_rule_files_do_not_leak_between_classifiers(tmp_path):
    rules = tmp_path / "rules.toml"
    rules.write_text('[[rule]]\ncategory = "mine"\npattern = "zzz"\n')
    custom = Classifier.from_file(rules)
    assert custom.classify("zzz", "kernel") == "mine"
    assert Classifier.default().classify("zzz", "kernel") == "other"


@pytest.mark.parametrize(
    "text",
    ["rule = 3\n", "rule = [1]\n", '[[rule]]\ncategory = "x"\npattern = 3\n'],
)
def test_wrongly_shaped_rule_entries_are_analysis_errors(tmp_path, text):
    rules = tmp_path / "rules.toml"
    rules.write_text(text)
    with pytest.raises(AnalysisError):
        Classifier.from_file(rules)


def test_rule_file_that_is_not_utf8_is_an_analysis_error(tmp_path):
    rules = tmp_path / "rules.toml"
    rules.write_bytes(b"\xff\xfe\x00")
    with pytest.raises(AnalysisError):
        Classifier.from_file(rules)
