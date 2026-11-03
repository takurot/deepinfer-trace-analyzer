from __future__ import annotations

import gzip
import sys
import zlib
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import IO, Any

import ijson

from .errors import AnalysisError, InputError
from .model import GPU_CATEGORIES, SCHEDULE_NAME, CpuEvent, GpuEvent, Trace

PROFILER_PIDS = frozenset({"Spans", "Traces", ""})
RETAINED_CPU_PREFIX = "deepinfer:"
IGNORED_CATEGORY = "gpu_user_annotation"
_FORMAT_ERRORS = (
    ijson.JSONError,
    ijson.IncompleteJSONError,
    EOFError,
    zlib.error,
    gzip.BadGzipFile,
    UnicodeDecodeError,
)


def _describe(error: Exception) -> str:
    return getattr(error, "strerror", None) or str(error)


def load_trace(path: str | Path) -> Trace:
    path = Path(path)
    prefix = _event_prefix(path)
    cpu_events: list[CpuEvent] = []
    gpu_events: list[GpuEvent] = []
    skipped = 0
    try:
        with _open(path) as stream:
            for raw in ijson.items(stream, prefix, use_float=False):
                if not isinstance(raw, dict) or raw.get("ph") != "X":
                    continue
                if str(raw.get("pid")) in PROFILER_PIDS:
                    continue
                cat = raw.get("cat")
                if cat == IGNORED_CATEGORY:
                    continue
                if cat in GPU_CATEGORIES:
                    gpu = _gpu_event(raw)
                    if gpu is None:
                        skipped += 1
                    else:
                        gpu_events.append(gpu)
                elif _is_retained_cpu(raw):
                    cpu = _cpu_event(raw)
                    if cpu is None:
                        skipped += 1
                    else:
                        cpu_events.append(cpu)
    except _FORMAT_ERRORS as error:
        raise InputError(f"{path}: not a readable chrome trace ({error})") from error
    except OSError as error:
        raise InputError(f"{path}: {_describe(error)}") from error

    if prefix == "traceEvents.item" and not (cpu_events or gpu_events):
        _require_trace_events_array(path)
    trace = Trace(
        path=path,
        cpu_events=cpu_events,
        gpu_events=gpu_events,
        schedule_starts=sorted(e.ts for e in cpu_events if e.name == SCHEDULE_NAME),
        devices=sorted({e.device for e in gpu_events}),
    )
    if skipped:
        trace.warnings.append(
            f"skipped {skipped} events with a missing or non-finite ts, a bad dur, or a non-integer GPU pid"
        )
    if "c" not in ijson.backend:
        trace.warnings.append(f"ijson backend '{ijson.backend}' is pure Python; large traces will be slow")
    if not trace.schedule_starts:
        raise AnalysisError(f"{path}: no '{SCHEDULE_NAME}' events, so steps cannot be delimited")
    return trace


def _open(path: Path) -> IO[bytes]:
    try:
        return gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb")
    except OSError as error:
        raise InputError(f"{path}: {_describe(error)}") from error


def _event_prefix(path: Path) -> str:
    try:
        with _open(path) as stream:
            head = stream.read(4096).lstrip()
    except (OSError, EOFError, zlib.error) as error:
        raise InputError(f"{path}: {_describe(error)}") from error
    if head.startswith(b"["):
        return "item"
    if head.startswith(b"{"):
        return "traceEvents.item"
    raise InputError(f"{path}: top level is neither a JSON object nor an array")


def _require_trace_events_array(path: Path) -> None:
    """Only runs when streaming found no events, to tell an empty array from a missing one."""
    try:
        with _open(path) as stream:
            for prefix, event, _ in ijson.parse(stream):
                if prefix == "traceEvents" and event == "start_array":
                    return
    except _FORMAT_ERRORS as error:
        raise InputError(f"{path}: not a readable chrome trace ({error})") from error
    raise InputError(f"{path}: JSON object has no 'traceEvents' array")


def _decimal(value: Any) -> Decimal | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = value if isinstance(value, Decimal) else Decimal(str(value))
    except InvalidOperation:
        return None
    return number if number.is_finite() else None


def _duration(value: Any) -> Decimal | None:
    number = _decimal(value)
    return number if number is not None and number >= 0 else None


def _int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _correlation(raw: dict) -> int | None:
    args = raw.get("args")
    return _int(args.get("correlation")) if isinstance(args, dict) else None


def _is_retained_cpu(raw: dict) -> bool:
    return str(raw.get("name", "")).startswith(RETAINED_CPU_PREFIX) or _correlation(raw) is not None


def _gpu_event(raw: dict) -> GpuEvent | None:
    ts, dur, device = _decimal(raw.get("ts")), _duration(raw.get("dur")), _int(raw.get("pid"))
    if ts is None or dur is None or device is None:
        return None
    return GpuEvent(sys.intern(str(raw.get("name", ""))), raw["cat"], device, ts, dur, _correlation(raw))


def _cpu_event(raw: dict) -> CpuEvent | None:
    ts, dur = _decimal(raw.get("ts")), _duration(raw.get("dur"))
    if ts is None or dur is None:
        return None
    return CpuEvent(sys.intern(str(raw.get("name", ""))), ts, dur, _correlation(raw))
