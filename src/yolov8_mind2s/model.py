from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

from .utils import ensure_dir


def _is_intel_device(device: str) -> bool:
    d = device.lower().strip()
    return d.startswith("intel:") or d.startswith("openvino:")


def _openvino_export_dir(model_id: str) -> Path:
    # Normalize "yolov8x.pt" -> "yolov8x"
    stem = Path(model_id).name
    for suf in (".pt", ".onnx", ".engine"):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
    return ensure_dir(Path("models") / "openvino" / stem)


def maybe_export_openvino(model_id: str, int8: bool = False, imgsz: int = 640) -> Path:
    """Ensure an OpenVINO export exists for the model and return export directory."""
    export_dir = _openvino_export_dir(model_id)
    xml = export_dir / "model.xml"
    if xml.exists():
        return export_dir

    # Export
    from ultralytics import YOLO  # type: ignore

    model = YOLO(model_id)  # downloads if needed
    export_dir = model.export(
        format="openvino",
        imgsz=imgsz,
        int8=bool(int8),
        half=False,
        dynamic=False,
    )
    # Ultralytics returns a path-like; normalize.
    return Path(str(export_dir))


def load_model(model_id: str, device: str, prefer_int8_on_npu: bool, imgsz: int) -> Tuple["YOLO", str]:
    """Load YOLO model, exporting to OpenVINO if device is Intel/OpenVINO."""
    from ultralytics import YOLO  # type: ignore

    dev = device.strip()

    if _is_intel_device(dev):
        # Ultralytics expects intel:* device strings when using OpenVINO exports.
        # We auto-export and then load from the export directory.
        int8 = prefer_int8_on_npu and dev.lower().endswith("npu")
        export_dir = maybe_export_openvino(model_id, int8=int8, imgsz=imgsz)
        model = YOLO(str(export_dir))
        # For OpenVINO, device string should be passed to predict(). Use intel:cpu/intel:gpu/intel:npu.
        # If we got openvino:DEVICE, pass through OpenVINO name.
        if dev.lower().startswith("openvino:"):
            dev = dev.split(":", 1)[1]  # e.g. "GPU" / "CPU" / "NPU"
        return model, dev

    # CUDA or CPU
    model = YOLO(model_id)
    return model, dev
