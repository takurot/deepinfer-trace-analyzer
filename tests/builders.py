from __future__ import annotations

import gzip
import json
from decimal import Decimal
from pathlib import Path

CPU_PID = 4242


class TraceBuilder:
    """Builds torch-shaped chrome trace events by hand. Times are in microseconds."""

    def __init__(self) -> None:
        self.events: list[dict] = []
        self._next_corr = 1

    def corr(self) -> int:
        value = self._next_corr
        self._next_corr += 1
        return value

    def schedule(self, ts, dur=1) -> TraceBuilder:
        return self.cpu("deepinfer:schedule", ts, dur, cat="user_annotation")

    def bubble(self, ts, dur) -> TraceBuilder:
        return self.cpu("deepinfer:bubble", ts, dur, cat="user_annotation")

    def launch_forward(self, ts, dur, reqs, tokens) -> TraceBuilder:
        return self.cpu(f"deepinfer:launch_forward_N{reqs}_B{tokens}", ts, dur, cat="user_annotation")

    def cpu(self, name, ts, dur, *, cat="cpu_op", corr=None) -> TraceBuilder:
        event = {"ph": "X", "cat": cat, "name": name, "pid": CPU_PID, "tid": 1, "ts": ts, "dur": dur}
        if corr is not None:
            event["args"] = {"correlation": corr}
        self.events.append(event)
        return self

    def launch(self, ts, dur=1, *, corr, name="cudaLaunchKernel") -> TraceBuilder:
        return self.cpu(name, ts, dur, cat="cuda_runtime", corr=corr)

    def gpu(self, name, ts, dur, *, device=0, stream=7, corr=None, cat="kernel") -> TraceBuilder:
        event = {"ph": "X", "cat": cat, "name": name, "pid": device, "tid": stream, "ts": ts, "dur": dur}
        if corr is not None:
            event["args"] = {"correlation": corr}
        self.events.append(event)
        return self

    def kernel(self, name, ts, dur, *, launch_ts=None, device=0, stream=7, corr="auto") -> TraceBuilder:
        """A kernel plus the cudaLaunchKernel that launched it. corr=None leaves it unmatched."""
        if corr == "auto":
            corr = self.corr() if launch_ts is not None else None
        if launch_ts is not None:
            self.launch(launch_ts, corr=corr)
        return self.gpu(name, ts, dur, device=device, stream=stream, corr=corr)

    def memcpy(self, ts, dur, *, device=0, stream=7, cat="gpu_memcpy") -> TraceBuilder:
        return self.gpu("Memcpy HtoD (Pinned -> Device)", ts, dur, device=device, stream=stream, cat=cat)

    def raw(self, event: dict) -> TraceBuilder:
        self.events.append(event)
        return self

    def document(self) -> dict:
        return {"traceEvents": list(self.events)}

    def write(self, path: Path, *, gz=False, as_list=False, stringify=False) -> Path:
        events = [_stringified(e) for e in self.events] if stringify else self.events
        payload = events if as_list else {"traceEvents": events}
        text = json.dumps(payload, default=_decimal_default)
        if gz:
            with gzip.open(path, "wt", encoding="utf-8") as stream:
                stream.write(text)
        else:
            path.write_text(text, encoding="utf-8")
        return path


def _decimal_default(value):
    if isinstance(value, Decimal):
        return float(value)
    raise TypeError(type(value))


def _stringified(event: dict) -> dict:
    out = dict(event)
    for key in ("ts", "dur"):
        if key in out:
            out[key] = str(out[key])
    if "args" in out:
        out["args"] = {k: str(v) for k, v in out["args"].items()}
    return out
