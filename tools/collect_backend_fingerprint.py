#!/usr/bin/env python3
"""Collect a runtime backend fingerprint for H20 experiments.

This script records the environment that makes benchmark and tuning records
valid. It does not hard-code H20 hardware facts; it detects what the runtime
reports and stores the result in JSON.
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
from pathlib import Path
from typing import Any

from ecc_utils import git_commit, run_cmd, utc_now, write_json


def detect_torch() -> dict[str, Any]:
    info: dict[str, Any] = {
        "torch": "not_importable",
        "cuda_available": False,
    }
    try:
        import torch  # type: ignore
    except Exception as exc:
        info["torch_import_error"] = str(exc)
        return info

    info["torch"] = getattr(torch, "__version__", "unknown")
    info["cuda_available"] = bool(torch.cuda.is_available())
    info["torch_cuda"] = str(getattr(torch.version, "cuda", "unknown"))

    if not torch.cuda.is_available():
        return info

    device_idx = int(os.environ.get("ECC_KB_DEVICE_INDEX", "0"))
    props = torch.cuda.get_device_properties(device_idx)
    info.update(
        {
            "device_index": device_idx,
            "device_name": torch.cuda.get_device_name(device_idx),
            "compute_capability": f"{props.major}.{props.minor}",
            "total_memory_bytes": int(props.total_memory),
            "multi_processor_count": int(props.multi_processor_count),
            "device_uuid": getattr(props, "uuid", "unknown"),
        }
    )
    return info


def detect_triton() -> dict[str, str]:
    try:
        import triton  # type: ignore

        return {"triton": getattr(triton, "__version__", "unknown")}
    except Exception as exc:
        return {"triton": "not_importable", "triton_import_error": str(exc)}


def detect_driver() -> dict[str, str]:
    queries = [
        ["nvidia-smi", "--query-gpu=driver_version,name,uuid,memory.total", "--format=csv,noheader"],
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
    ]
    last_error = ""
    for cmd in queries:
        code, out, err = run_cmd(cmd, timeout=10)
        last_error = err
        if code == 0 and out:
            return {"nvidia_smi_query": " ".join(cmd), "nvidia_smi": out}
    return {"nvidia_smi": "unavailable", "nvidia_smi_error": last_error}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="nvidia-triton-cuda")
    parser.add_argument("--vendor", default="nvidia")
    parser.add_argument("--kb-version", default="v0")
    parser.add_argument("--harness-version", default="h20-harness-v0")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=500)
    parser.add_argument("--output", "--out", required=True)
    args = parser.parse_args()

    torch_info = detect_torch()
    triton_info = detect_triton()
    driver_info = detect_driver()
    repo_root = Path.cwd()

    toolchain = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": str(torch_info.get("torch", "unknown")),
        "torch_cuda": str(torch_info.get("torch_cuda", "unknown")),
        "triton": str(triton_info.get("triton", "unknown")),
        "driver": str(driver_info.get("nvidia_smi", "unknown")).splitlines()[0] if driver_info else "unknown",
    }

    fingerprint = {
        "schema_version": "0.1",
        "created_at": utc_now(),
        "vendor": args.vendor,
        "backend": args.backend,
        "device_name": str(torch_info.get("device_name", "unknown")),
        "device_class": "h20_runtime_detected" if "H20" in str(torch_info.get("device_name", "")) else "runtime_detected",
        "compute_capability": str(torch_info.get("compute_capability", "unknown")),
        "total_memory_bytes": torch_info.get("total_memory_bytes", None),
        "multi_processor_count": torch_info.get("multi_processor_count", None),
        "device_uuid": str(torch_info.get("device_uuid", "unknown")),
        "hardware": torch_info,
        "software": {
            "python_executable": sys.executable,
            "toolchain": toolchain,
            "triton": triton_info,
            "driver": driver_info,
        },
        "toolchain": toolchain,
        "runtime": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "ECC_KB_DEVICE_INDEX": os.environ.get("ECC_KB_DEVICE_INDEX", "0"),
            "warmup": args.warmup,
            "repeats": args.repeats,
            "random_seeds": [],
        },
        "repo": {
            "git_commit": git_commit(repo_root),
            "kb_version": args.kb_version,
            "harness_version": args.harness_version,
        },
    }

    write_json(args.output, fingerprint)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
