#!/usr/bin/env python3
"""Collect a runtime backend fingerprint for H20/Triton-CUDA experiments.

The script intentionally detects target attributes at runtime instead of hardcoding H20 values.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def run_text(cmd: list[str]) -> str | None:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=10).strip()
    except Exception:
        return None


def git_commit() -> str | None:
    return run_text(["git", "rev-parse", "HEAD"])


def collect_torch() -> dict[str, Any]:
    info: dict[str, Any] = {
        "torch_available": False,
        "cuda_available": False,
    }
    try:
        import torch  # type: ignore

        info["torch_available"] = True
        info["torch_version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["torch_cuda_version"] = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)
            info.update(
                {
                    "device_index": device,
                    "device_name": torch.cuda.get_device_name(device),
                    "compute_capability": f"{props.major}.{props.minor}",
                    "total_memory_bytes": int(props.total_memory),
                    "multi_processor_count": int(props.multi_processor_count),
                }
            )
    except Exception as exc:  # pragma: no cover - environment dependent
        info["torch_error"] = repr(exc)
    return info


def collect_triton_version() -> str | None:
    try:
        import triton  # type: ignore

        return getattr(triton, "__version__", None)
    except Exception:
        return None


def collect(args: argparse.Namespace) -> dict[str, Any]:
    torch_info = collect_torch()
    nvidia_smi = run_text([
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,uuid",
        "--format=csv,noheader",
    ])
    driver_version = None
    if nvidia_smi:
        first = nvidia_smi.splitlines()[0].split(",")
        if len(first) >= 2:
            driver_version = first[1].strip()

    return {
        "schema_version": "0.1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "vendor": "nvidia",
        "backend": args.backend,
        "device_name": torch_info.get("device_name"),
        "device_class": args.device_class,
        "torch": torch_info,
        "toolchain": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch_info.get("torch_version"),
            "triton": collect_triton_version(),
            "cuda": torch_info.get("torch_cuda_version"),
            "driver": driver_version,
        },
        "runtime": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "TRITON_CACHE_DIR": os.environ.get("TRITON_CACHE_DIR"),
            "TORCHINDUCTOR_CACHE_DIR": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
            "warmup": args.warmup,
            "repeats": args.repeats,
            "seed": args.seed,
        },
        "repo": {
            "git_commit": git_commit(),
            "kb_version": args.kb_version,
            "harness_version": args.harness_version,
        },
        "nvidia_smi_query": nvidia_smi,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="Output JSON path.")
    parser.add_argument("--backend", default="triton_cuda")
    parser.add_argument("--device-class", default="h20_runtime_detected")
    parser.add_argument("--kb-version", default="v0")
    parser.add_argument("--harness-version", default="h20-harness-v0")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(collect(args), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
