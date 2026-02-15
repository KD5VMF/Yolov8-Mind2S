from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

from .utils import COCO80


def _is_intel_device(device: str) -> bool:
    d = device.lower().strip()
    return d.startswith("intel:") or d.startswith("openvino:")


def _model_stem(model_id: str) -> str:
    p = Path(model_id)
    stem = p.stem
    for suf in (".pt", ".onnx", ".engine", ".xml"):
        if p.name.endswith(suf):
            stem = p.name[: -len(suf)]
            break
    return stem


def _openvino_export_dir(model_id: str) -> Path:
    stem = _model_stem(model_id)
    return Path(f"{stem}_openvino_model")


def maybe_export_openvino(model_id: str, *, int8: bool = False, imgsz: int = 640) -> Path:
    """Ensure an OpenVINO export exists for the model and return export directory."""
    export_dir = _openvino_export_dir(model_id)
    xml = export_dir / "model.xml"
    if xml.exists():
        return export_dir

    from ultralytics import YOLO  # type: ignore

    model = YOLO(model_id)  # downloads if needed
    exported = model.export(
        format="openvino",
        imgsz=imgsz,
        int8=bool(int8),
        half=False,
        dynamic=False,
    )
    return Path(str(exported))


def get_class_names(yolo) -> Dict[int, str]:
    """Get class names WITHOUT triggering Ultralytics' `YOLO.names` property.

    Ultralytics' `YOLO.names` is a property that may call predictor setup and can
    accidentally try CUDA device selection on some OpenVINO setups. We avoid that.
    """
    try:
        m = getattr(yolo, "model", None)
        names = getattr(m, "names", None)
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        if isinstance(names, (list, tuple)) and names:
            return {i: str(n) for i, n in enumerate(names)}
    except Exception:
        pass

    try:
        pred = getattr(yolo, "predictor", None)
        m = getattr(pred, "model", None)
        names = getattr(m, "names", None)
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        if isinstance(names, (list, tuple)) and names:
            return {i: str(n) for i, n in enumerate(names)}
    except Exception:
        pass

    return {i: n for i, n in enumerate(COCO80)}


def load_model(model_id: str, *, device: str, prefer_int8_on_npu: bool, imgsz: int) -> Tuple["YOLO", str]:
    """Load YOLO model, exporting to OpenVINO if device is Intel/OpenVINO."""
    from ultralytics import YOLO  # type: ignore

    dev = (device or "").strip() or "cpu"

    if _is_intel_device(dev):
        # Helpful error if OpenVINO isn't installed
        try:
            import openvino  # type: ignore  # noqa: F401
        except Exception as e:
            raise RuntimeError(
                "OpenVINO is not installed. For Intel GPU/NPU acceleration run:\n"
                "  pip install -r requirements-openvino.txt"
            ) from e

        int8 = bool(prefer_int8_on_npu and dev.lower().endswith("npu"))
        export_dir = maybe_export_openvino(model_id, int8=int8, imgsz=imgsz)

        model = YOLO(str(export_dir))

        pred_dev = dev
        if dev.lower().startswith("openvino:"):
            pred_dev = dev.split(":", 1)[1].strip()

        return model, pred_dev

    model = YOLO(model_id)
    return model, dev
