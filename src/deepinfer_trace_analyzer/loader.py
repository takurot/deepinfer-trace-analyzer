from __future__ import annotations

import gzip
import zlib
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import IO, Any

import ijson

from .errors import AnalysisError, InputError
from .model import GPU_CATEGORIES, SCHEDULE_NAME, CpuEvent, GpuEvent, Trace

PROFILER_PIDS = frozenset({"Spans", "Traces", ""})
RETAINED_CPU_PREFIX = "deepinfer:"


def load_trace(path: str | Path) -> Trace:
    path = Path(path)
    prefix = _event_prefix(path)
    cpu_events: list[CpuEvent] = []
    gpu_events: list[GpuEvent] = []
    skipped_gpu = 0
    try:
        with _open(path) as stream:
            for raw in ijson.items(stream, prefix, use_float=False):
                if not isinstance(raw, dict) or raw.get("ph") != "X":
                    continue
                if str(raw.get("pid")) in PROFILER_PIDS:
                    continue
                if raw.get("cat") in GPU_CATEGORIES:
                    event = _gpu_event(raw)
                    if event is None:
                        skipped_gpu += 1
                    else:
                        gpu_events.append(event)
                else:
                    event = _cpu_event(raw)
                    if event is not None:
                        cpu_events.append(event)
    except (ijson.JSONError, ijson.IncompleteJSONError, EOFError, zlib.error, gzip.BadGzipFile) as error:
        raise InputError(f"{path}: not a readable chrome trace ({error})") from error
    except OSError as error:
        raise InputError(f"{path}: {error.strerror or error}") from error

    if prefix == "traceEvents.item" and not (cpu_events or gpu_events):
        _require_trace_events_key(path)
    trace = Trace(path=path, cpu_events=cpu_events, gpu_events=gpu_events)
    if skipped_gpu:
        trace.warnings.append(f"skipped {skipped_gpu} GPU events with a missing ts/dur or a non-integer pid")
    if not trace.schedule_starts:
        raise AnalysisError(f"{path}: no '{SCHEDULE_NAME}' events, so steps cannot be delimited")
    return trace


def _open(path: Path) -> IO[bytes]:
    try:
        return gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb")
    except OSError as error:
        raise InputError(f"{path}: {error.strerror or error}") from error


def _event_prefix(path: Path) -> str:
    try:
        with _open(path) as stream:
            head = stream.read(4096).lstrip()
    except (OSError, EOFError, zlib.error) as error:
        raise InputError(f"{path}: {getattr(error, 'strerror', None) or error}") from error
    if head.startswith(b"["):
        return "item"
    if head.startswith(b"{"):
        return "traceEvents.item"
    raise InputError(f"{path}: top level is neither a JSON object nor an array")


def _require_trace_events_key(path: Path) -> None:
    try:
        with _open(path) as stream:
            if any(
                event == "map_key" and value == "traceEvents"
                for prefix, event, value in ijson.parse(stream)
                if prefix == ""
            ):
                return
    except (ijson.JSONError, ijson.IncompleteJSONError) as error:
        raise InputError(f"{path}: not a readable chrome trace ({error})") from error
    raise InputError(f"{path}: JSON object has no 'traceEvents' array")


def _decimal(value: Any) -> Decimal | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return value if isinstance(value, Decimal) else Decimal(str(value))
    except InvalidOperation:
        return None


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


def _gpu_event(raw: dict) -> GpuEvent | None:
    ts, dur, device = _decimal(raw.get("ts")), _decimal(raw.get("dur")), _int(raw.get("pid"))
    if ts is None or dur is None or device is None:
        return None
    return GpuEvent(str(raw.get("name", "")), raw["cat"], device, ts, dur, _correlation(raw))


def _cpu_event(raw: dict) -> CpuEvent | None:
    name = str(raw.get("name", ""))
    correlation = _correlation(raw)
    if not name.startswith(RETAINED_CPU_PREFIX) and correlation is None:
        return None
    ts, dur = _decimal(raw.get("ts")), _decimal(raw.get("dur"))
    if ts is None or dur is None:
        return None
    return CpuEvent(name, ts, dur, correlation)
