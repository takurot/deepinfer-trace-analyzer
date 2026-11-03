from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

from .errors import AnalysisError, InputError
from .model import MEMORY_CATEGORIES

MEMCPY = "memcpy"
OTHER = "other"
RESERVED = frozenset({MEMCPY, OTHER})

DEFAULT_RULES: tuple[tuple[str, str], ...] = (
    ("comm.nccl", "nccl"),
    ("comm.deepep", "deep_ep|deepep"),
    ("attn.flashmla", "flash_mla|flashmla|sparse_attn"),
    ("gemm.deepgemm", "deep_gemm|fp8_gemm|m_grouped"),
    ("gemm.other", "gemm|cutlass|cublas"),
    ("tilelang", "tilelang"),
    ("sampler", "sampl|topk|top_k"),
)

_COMPUTE, _MEMORY, _COMM = 0, 1, 2


@dataclass(frozen=True, slots=True)
class Rule:
    category: str
    pattern: re.Pattern[str]


class Classifier:
    """Ordered regex rules. Memcpy and memset events are decided by `cat`, not by name."""

    def __init__(self, rules: tuple[Rule, ...]) -> None:
        self._rules = rules
        self._order: dict[str, int] = {}
        for index, rule in enumerate(rules):
            self._order.setdefault(rule.category, index)
        self._by_name: dict[str, str] = {}

    @classmethod
    def default(cls) -> Classifier:
        return cls(_compile(DEFAULT_RULES))

    @classmethod
    def from_file(cls, path: str | Path | None, *, use_defaults: bool = True) -> Classifier:
        user = _read_rules_file(Path(path)) if path is not None else ()
        defaults = _compile(DEFAULT_RULES) if use_defaults else ()
        return cls(user + defaults)

    def classify(self, name: str, cat: str) -> str:
        if cat in MEMORY_CATEGORIES:
            return MEMCPY
        category = self._by_name.get(name)
        if category is None:
            category = next((rule.category for rule in self._rules if rule.pattern.search(name)), OTHER)
            self._by_name[name] = category
        return category

    def priority(self, category: str) -> tuple[int, int]:
        """Lower sorts first. Compute beats memcpy beats communication, ties by rule order."""
        if category == MEMCPY:
            return (_MEMORY, 0)
        tier = _COMM if category.startswith("comm.") else _COMPUTE
        return (tier, self._order.get(category, len(self._rules)))

    def categories(self) -> list[str]:
        """Every category this classifier can emit, in rule order, then memcpy and other."""
        return [*self._order, MEMCPY, OTHER]


def _compile(pairs: tuple[tuple[str, str], ...]) -> tuple[Rule, ...]:
    return tuple(Rule(category, re.compile(pattern, re.IGNORECASE)) for category, pattern in pairs)


def _read_rules_file(path: Path) -> tuple[Rule, ...]:
    try:
        document = tomllib.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise InputError(f"{path}: {error.strerror or error}") from error
    except UnicodeDecodeError as error:
        raise AnalysisError(f"{path}: not valid UTF-8 ({error})") from error
    except tomllib.TOMLDecodeError as error:
        raise AnalysisError(f"{path}: invalid TOML ({error})") from error
    entries = document.get("rule", [])
    if not isinstance(entries, list):
        raise AnalysisError(f"{path}: 'rule' must be an array of tables")
    rules = []
    for index, entry in enumerate(entries, start=1):
        category, pattern = (
            (entry.get("category"), entry.get("pattern")) if isinstance(entry, dict) else (None, None)
        )
        if not isinstance(category, str) or not isinstance(pattern, str):
            raise AnalysisError(f"{path}: rule #{index} needs string 'category' and 'pattern'")
        if category in RESERVED:
            raise AnalysisError(f"{path}: rule #{index} uses the reserved category '{category}'")
        try:
            rules.append(Rule(category, re.compile(pattern, re.IGNORECASE)))
        except re.error as error:
            raise AnalysisError(f"{path}: rule #{index} has an invalid pattern ({error})") from error
    return tuple(rules)
