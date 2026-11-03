"""Check that this tool finds the same step starts as DeepInfer's examples/_trace.py.

Needs a DeepInfer checkout, so it is not part of CI:

    uv run python scripts/compare_step_starts.py /path/to/DeepInfer TRACE.json
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

from deepinfer_trace_analyzer.loader import load_trace


def reference_starts(deepinfer: Path, trace_path: Path) -> list:
    sys.path.insert(0, str(deepinfer / "examples"))
    import _trace

    opener = gzip.open if trace_path.suffix == ".gz" else open
    with opener(trace_path, "rt", encoding="utf-8") as stream:
        data = json.load(stream, parse_float=str)
    if isinstance(data, list):
        data = {"traceEvents": data}
    _trace._normalize_trace_events(data)
    _, body = _trace._split_events(data)
    return _trace._step_starts(body)


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    deepinfer, trace_path = Path(argv[1]), Path(argv[2])
    expected = reference_starts(deepinfer, trace_path)
    actual = load_trace(trace_path).schedule_starts
    if [str(x) for x in expected] == [str(x) for x in actual]:
        print(f"OK: {len(actual)} step starts match")
        return 0
    print(f"MISMATCH\n  _trace.py: {expected}\n  analyzer:  {actual}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
