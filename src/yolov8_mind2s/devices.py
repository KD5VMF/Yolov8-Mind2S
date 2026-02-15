from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class DeviceInfo:
    name: str
    kind: str  # "cuda" | "intel" | "cpu"
    detail: str


def _torch_cuda_available() -> bool:
    try:
        import torch  # type: ignore
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _torch_cuda_count() -> int:
    try:
        import torch  # type: ignore
        return int(torch.cuda.device_count())
    except Exception:
        return 0


def _openvino_devices() -> List[str]:
    try:
        import openvino as ov  # type: ignore
        core = ov.Core()
        return list(core.available_devices)
    except Exception:
        return []


def available_devices() -> List[DeviceInfo]:
    devices: List[DeviceInfo] = []

    # CPU always.
    devices.append(DeviceInfo(name="cpu", kind="cpu", detail="CPU (default)"))

    # NVIDIA CUDA (eGPU etc.)
    if _torch_cuda_available():
        n = _torch_cuda_count()
        for i in range(max(n, 1)):
            devices.append(DeviceInfo(name=f"cuda:{i}", kind="cuda", detail="NVIDIA CUDA"))

    # Intel OpenVINO devices (CPU/GPU/NPU) exposed by OpenVINO Runtime.
    ov_devs = _openvino_devices()
    # OpenVINO uses names like "CPU", "GPU", "NPU" etc.
    # We present them in Ultralytics' device string format: intel:cpu/intel:gpu/intel:npu.
    map_ov = {"CPU": "intel:cpu", "GPU": "intel:gpu", "NPU": "intel:npu"}
    for d in ov_devs:
        if d in map_ov:
            devices.append(DeviceInfo(name=map_ov[d], kind="intel", detail=f"OpenVINO {d}"))
        else:
            # Keep unknown OpenVINO devices discoverable.
            devices.append(DeviceInfo(name=f"openvino:{d}", kind="intel", detail=f"OpenVINO {d}"))

    return devices


def pick_default_device(prefer: Optional[str] = None) -> str:
    """Choose a sensible default.
    Priority (unless prefer is set):
      1) CUDA (if present)
      2) Intel NPU (if present)
      3) Intel GPU (if present)
      4) CPU
    """
    if prefer:
        return prefer

    devs = [d.name for d in available_devices()]
    for cand in ("cuda:0", "intel:npu", "intel:gpu", "intel:cpu", "cpu"):
        if cand in devs:
            return cand
    return "cpu"


def is_windows() -> bool:
    return platform.system().lower().startswith("win")


def is_linux() -> bool:
    return platform.system().lower() == "linux"
