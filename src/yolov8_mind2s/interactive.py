from __future__ import annotations

import sys
from typing import Dict, List, Optional

from .devices import available_devices, pick_default_device
from .detect import run_detect
from .model import load_model
from .utils import columns, parse_class_selection


BANNER = r"""     __   __   ___  _      ___        __  __ _           _  ___  ___
 \ \ / /  / _ \| |    / _ \ ___  |  \/  (_)_ __   __| |/ _ \/ __|
  \ V /  | (_) | |__ | (_) / _ \ | |\/| | | '_ \ / _` | (_) \__ \
   \_/    \___/|____| \___/\___/ |_|  |_|_| .__/ \__,_|\___/|___/
                                          |_|  Mind 2S Edition
"""


def prompt(msg: str, default: str = "") -> str:
    if default:
        msg = f"{msg} [{default}]: "
    else:
        msg = f"{msg}: "
    s = input(msg).strip()
    return s if s else default


def interactive_main() -> int:
    print(BANNER)
    print()

    # Device selection
    devs = available_devices()
    print("Detected devices:")
    for i, d in enumerate(devs, start=1):
        print(f"  {i:2d}. {d.name:12}  ({d.detail})")
    print()

    default_dev = pick_default_device()
    dev_in = prompt("Pick device number (or type a device string)", str([d.name for d in devs].index(default_dev) + 1 if default_dev in [d.name for d in devs] else default_dev))
    device = default_dev
    if dev_in.isdigit():
        idx = int(dev_in) - 1
        if 0 <= idx < len(devs):
            device = devs[idx].name
    else:
        device = dev_in.strip() or default_dev

    # Model selection
    model_id = prompt("Model (yolov8n.pt / yolov8s.pt / yolov8m.pt / yolov8l.pt / yolov8x.pt / custom.pt)", "yolov8x.pt")

    imgsz = int(prompt("Image size (imgsz)", "640"))
    conf = float(prompt("Confidence threshold", "0.25"))
    iou = float(prompt("IoU threshold", "0.45"))
    max_det = int(prompt("Max detections per frame", "300"))
    half = prompt("Use FP16 (CUDA only)? (y/n)", "n").lower().startswith("y")
    fullscreen = prompt("Fullscreen window? (y/n)", "y").lower().startswith("y")
    save_video = prompt("Save annotated MP4? (y/n)", "n").lower().startswith("y")
    save_json = prompt("Save detections JSONL? (y/n)", "n").lower().startswith("y")
    out_dir = prompt("Output folder", "runs")
    source = prompt("Camera index / video path", "0")

    prefer_int8_on_npu = prompt("If using intel:npu, export INT8 OpenVINO model (recommended)? (y/n)", "y").lower().startswith("y")

    # Load model once to get accurate class list for selection (also triggers auto-download).
    print()
    print("[INFO] Loading model (and exporting if needed) ...")
    model, pred_dev = load_model(model_id, device=device, prefer_int8_on_npu=prefer_int8_on_npu, imgsz=imgsz)
    names: Dict[int, str] = getattr(model, "names", {}) or {}
    n = len(names) if names else 80

    # Class selection
    print()
    print("Classes (choose what to detect):")
    items = []
    for k in range(n):
        nm = names.get(k, str(k))
        # 1-based display
        items.append(f"{k+1:2d}:{nm}")
    print(columns(items, col_width=22, cols=4))
    print()
    cls_in = prompt("Enter class numbers (comma-separated), ranges (1-5), or ALL", "ALL")
    classes = None
    try:
        classes = parse_class_selection(cls_in, n_classes=n)
    except Exception as e:
        print(f"[WARN] {e} -> using ALL")
        classes = None

    print()
    print("[INFO] Starting detection. Press 'q' (or ESC) to quit.")
    return run_detect(
        model_id=model_id,
        source=source,
        device=device,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        classes=classes,
        max_det=max_det,
        half=half,
        fullscreen=fullscreen,
        show=True,
        save_video=save_video,
        save_json=save_json,
        out_dir=out_dir,
        width=0,
        height=0,
        cam_fps=0,
        overlay_fps=True,
        prefer_int8_on_npu=prefer_int8_on_npu,
    )


if __name__ == "__main__":
    raise SystemExit(interactive_main())
