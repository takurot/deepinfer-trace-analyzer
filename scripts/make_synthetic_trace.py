"""Write a torch-shaped chrome trace of a known shape, for tests and for timing the analyzer.

Every step repeats one 30 us block on one device, so the right answer is known in closed form:

    stream 7  deep_gemm 0-10, flash_mla 10-15, elementwise 15-20, memcpy 20-22
    stream 8  nccl all-reduce 5-17 (hidden under compute), nccl send/recv 24-28 (exposed)

    per block, exclusive us: gemm.deepgemm 10, attn.flashmla 5, other 5, memcpy 2, comm.nccl 4, idle 4
    per block, inclusive us: comm.nccl 16 (all-reduce and send/recv), the rest as above, overlap 12

    uv run python scripts/make_synthetic_trace.py out.json --steps 50 --blocks-per-step 600

The file is written as a stream, so sizes far above memory are fine.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BLOCK_US = 30
CPU_PID = 4242
BASE_US = 1_700_000_000_000

# (name, category, stream, offset us, duration us, cat)
KERNELS = (
    ("void deep_gemm::sm100_fp8_gemm_nt_contiguous_kernel<128, 256, 128>", 7, 0, 10, "kernel"),
    ("void flash_mla::sm100_decode_attention_kernel<bf16, 64, 576>", 7, 10, 5, "kernel"),
    (
        "void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<bf16>>",
        7,
        15,
        5,
        "kernel",
    ),
    ("ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)", 8, 5, 12, "kernel"),
    ("Memcpy DtoD (Device -> Device)", 7, 20, 2, "gpu_memcpy"),
    ("ncclDevKernel_SendRecv(ncclDevKernelArgsStorage<4096ul>)", 8, 24, 4, "kernel"),
)
PER_BLOCK = {
    "exclusive_us": {
        "gemm.deepgemm": 10,
        "attn.flashmla": 5,
        "other": 5,
        "memcpy": 2,
        "comm.nccl": 4,
    },
    "inclusive_us": {
        "gemm.deepgemm": 10,
        "attn.flashmla": 5,
        "other": 5,
        "memcpy": 2,
        "comm.nccl": 16,
    },
    "idle_us": 4,
    "overlap_us": 12,
}


def _ts(value: float) -> str:
    return f"{value:.3f}"


def _event(name: str, cat: str, pid: int, tid: int, ts: float, dur: float, args: dict | None = None) -> dict:
    event = {"ph": "X", "cat": cat, "name": name, "pid": pid, "tid": tid, "ts": _ts(ts), "dur": _ts(dur)}
    if args is not None:
        event["args"] = args
    return event


def _block_events(step_start: float, block: int, correlation: int) -> tuple[list[dict], int]:
    start = step_start + block * BLOCK_US
    events = []
    for name, stream, offset, dur, cat in KERNELS:
        correlation += 1
        gpu_ts = start + offset
        launch = "cudaMemcpyAsync" if cat == "gpu_memcpy" else "cudaLaunchKernel"
        events.append(
            _event(
                "aten::mm",
                "cpu_op",
                CPU_PID,
                1,
                gpu_ts - 1.5,
                1.5,
                {
                    "External id": correlation,
                    "Record function id": 0,
                    "Concrete Inputs": ["", "", ""],
                    "Input Dims": [[2048, 7168], [7168, 4096], []],
                    "Input type": ["c10::BFloat16", "c10::BFloat16", "Scalar"],
                    "Input Strides": [[7168, 1], [1, 7168], []],
                },
            )
        )
        events.append(
            _event(launch, "cuda_runtime", CPU_PID, 1, gpu_ts - 1, 0.5, {"correlation": correlation})
        )
        events.append(
            _event(
                name,
                cat,
                0,
                stream,
                gpu_ts,
                dur,
                {
                    "queued": 0,
                    "device": 0,
                    "context": 1,
                    "stream": stream,
                    "correlation": correlation,
                    "registers per thread": 128,
                    "shared memory": 0,
                    "blocks per SM": 1.5,
                    "warps per SM": 12.0,
                    "grid": [132, 1, 1],
                    "block": [256, 1, 1],
                    "est. achieved occupancy %": 25,
                    "External id": correlation,
                },
            )
        )
    return events, correlation


def generate(path: str | Path, *, steps: int, blocks_per_step: int) -> dict:
    """Write the trace and return what the analyzer must report for each of the first `steps` steps."""
    window = blocks_per_step * BLOCK_US
    correlation = 0
    total = gpu = 0
    with Path(path).open("w", encoding="utf-8") as stream:
        stream.write('{"traceEvents":[')
        first = True

        def emit(events: list[dict]) -> None:
            nonlocal first, total
            for event in events:
                stream.write(("" if first else ",") + json.dumps(event, separators=(",", ":")))
                first = False
            total += len(events)

        for step in range(steps):
            step_start = BASE_US + step * window
            emit(
                [
                    _event("deepinfer:schedule", "user_annotation", CPU_PID, 1, step_start, 1),
                    _event(
                        "deepinfer:launch_forward_N256_B256", "user_annotation", CPU_PID, 1, step_start + 2, 1
                    ),
                    _event("deepinfer:bubble", "user_annotation", CPU_PID, 1, step_start + 4, 1),
                ]
            )
            for block in range(blocks_per_step):
                events, correlation = _block_events(step_start, block, correlation)
                emit(events)
                gpu += len(KERNELS)
        emit([_event("deepinfer:schedule", "user_annotation", CPU_PID, 1, BASE_US + steps * window, 1)])
        stream.write("]}")
    return {
        "steps": steps,
        "blocks_per_step": blocks_per_step,
        "step_time_us": str(window),
        "exclusive_us": {c: str(v * blocks_per_step) for c, v in PER_BLOCK["exclusive_us"].items()},
        "inclusive_us": {c: str(v * blocks_per_step) for c, v in PER_BLOCK["inclusive_us"].items()},
        "idle_us": str(PER_BLOCK["idle_us"] * blocks_per_step),
        "overlap_us": str(PER_BLOCK["overlap_us"] * blocks_per_step),
        "gpu_events": gpu,
        "events": total,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--blocks-per-step", type=int, default=600)
    args = parser.parse_args(argv)
    expected = generate(args.output, steps=args.steps, blocks_per_step=args.blocks_per_step)
    sidecar = Path(args.output).with_suffix(".expected.json")
    sidecar.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    size_mb = Path(args.output).stat().st_size / 1e6
    print(
        f"wrote {args.output} ({size_mb:.1f} MB, {expected['events']} events) and {sidecar}", file=sys.stderr
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
