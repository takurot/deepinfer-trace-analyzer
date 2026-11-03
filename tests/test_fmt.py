from decimal import Decimal as D

import pytest
from deepinfer_trace_analyzer.fmt import dec, signed, us


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (D("1E+3"), "1000"),
        (D("0E-26"), "0"),
        (D("-0"), "0"),
        (D("20.000004"), "20.000004"),
        (D("20.500"), "20.5"),
        (D("-1.25"), "-1.25"),
    ],
)
def test_dec_is_plain_notation_without_trailing_zeros(value, expected):
    assert dec(value) == expected


def test_tiny_values_never_print_as_negative_zero():
    assert us(D("-0.0000001")) == "0.000"
    assert signed(D("-0.0000001")) == "+0.000"
    assert signed(D("0.0000001")) == "+0.000"


def test_signed_keeps_real_signs():
    assert signed(D("-20")) == "-20.000" and signed(D("2.5")) == "+2.500"
