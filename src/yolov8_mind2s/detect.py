from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np  # type: ignore

from .devices import available_devices, pick_default_device
from .model import get_class_names, load_model
from .utils import FpsCounter, ensure_dir, letterbox_to_screen, stable_color_for_class


@dataclass
class Detection:
    cls_id: int
    name: str
    conf: float
    xyxy: Tuple[int, int, int, int]


class FrameGrabber(threading.Thread):
    def __init__(self, cap):
        super().__init__(daemon=True)
        self.cap = cap
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._ts: float = 0.0
        self._running = True

    def run(self) -> None:
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._lock:
                self._frame = frame
                self._ts = time.time()

    def latest(self) -> Tuple[Optional[np.ndarray], float]:
        with self._lock:
            if self._frame is None:
                return None, 0.0
            return self._frame.copy(), self._ts

    def stop(self) -> None:
        self._running = False


def _open_capture(source: str, width: int, height: int, fps: int):
    import cv2  # type: ignore

    try:
        src = int(source)
    except ValueError:
        src = source

    backend = cv2.CAP_V4L2 if sys.platform.startswith("linux") else 0
    cap = cv2.VideoCapture(src, backend)

    if width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    if height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    if fps > 0:
        cap.set(cv2.CAP_PROP_FPS, float(fps))

    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1.0)
    except Exception:
        pass

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source!r}")
    return cap


def _screen_size() -> Tuple[int, int]:
    try:
        from screeninfo import get_monitors  # type: ignore
        m = get_monitors()[0]
        return int(m.width), int(m.height)
    except Exception:
        return 1280, 720


def _draw_detections(img: np.ndarray, dets: Sequence[Detection], *, show_labels: bool = True) -> np.ndarray:
    import cv2  # type: ignore

    out = img
    for d in dets:
        x1, y1, x2, y2 = d.xyxy
        color = stable_color_for_class(d.cls_id)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        if show_labels:
            label = f"{d.name} {d.conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            y = max(0, y1 - th - 6)
            cv2.rectangle(out, (x1, y), (x1 + tw + 6, y1), color, -1)
            cv2.putText(out, label, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    return out


def _results_to_detections(result, names: Dict[int, str]) -> List[Detection]:
    dets: List[Detection] = []
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return dets

    try:
        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
    except Exception:
        xyxy = np.array(boxes.xyxy)
        cls = np.array(boxes.cls)
        conf = np.array(boxes.conf)

    for i in range(len(xyxy)):
        x1, y1, x2, y2 = [int(v) for v in xyxy[i].tolist()]
        cid = int(cls[i])
        dets.append(Detection(
            cls_id=cid,
            name=str(names.get(cid, cid)),
            conf=float(conf[i]),
            xyxy=(x1, y1, x2, y2),
        ))
    return dets


def run_detect(
    *,
    model_id: str,
    source: str,
    device: str,
    imgsz: int,
    conf: float,
    iou: float,
    classes: Optional[List[int]],
    max_det: int,
    half: bool,
    fullscreen: bool,
    show: bool,
    save_video: bool,
    save_json: bool,
    out_dir: str,
    width: int,
    height: int,
    cam_fps: int,
    overlay_fps: bool,
    prefer_int8_on_npu: bool,
) -> int:
    import cv2  # type: ignore

    device = (device or "").strip() or pick_default_device()
    model, pred_device = load_model(model_id, device=device, prefer_int8_on_npu=prefer_int8_on_npu, imgsz=imgsz)
    names = get_class_names(model)

    cap = _open_capture(source, width, height, cam_fps)
    grabber = FrameGrabber(cap)
    grabber.start()

    out_dir_p = ensure_dir(Path(out_dir))
    ts = time.strftime("%Y%m%d_%H%M%S")

    writer = None
    out_path = None
    if save_video:
        out_path = out_dir_p / f"detect_{ts}.mp4"
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) or (width if width > 0 else 1280)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) or (height if height > 0 else 720)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, 30.0, (w, h))

    jf = None
    json_path = None
    if save_json:
        json_path = out_dir_p / f"detections_{ts}.jsonl"
        jf = json_path.open("w", encoding="utf-8")

    screen_w, screen_h = _screen_size()
    if show:
        cv2.namedWindow("YOLOv8", cv2.WINDOW_NORMAL)
        if fullscreen:
            cv2.setWindowProperty("YOLOv8", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    fps_cap = FpsCounter()
    fps_inf = FpsCounter()
    last_ts = 0.0

    try:
        while True:
            frame, fts = grabber.latest()
            if frame is None:
                time.sleep(0.01)
                continue
            if fts == last_ts:
                time.sleep(0.001)
                continue
            last_ts = fts

            fps_cap.tick()

            results = model.predict(
                source=frame,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                device=pred_device,
                classes=classes,
                max_det=max_det,
                half=bool(half),
                verbose=False,
            )

            fps_inf.tick()
            result = results[0] if isinstance(results, list) and results else results
            dets = _results_to_detections(result, names)
            annotated = _draw_detections(frame, dets)

            if overlay_fps:
                txt = f"cap {fps_cap.fps:5.1f} fps | inf {fps_inf.fps:5.1f} fps | device {device}"
                cv2.putText(annotated, txt, (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            if writer is not None:
                writer.write(annotated)

            if jf is not None:
                payload = {
                    "ts": fts,
                    "device": device,
                    "detections": [{"cls": d.cls_id, "name": d.name, "conf": d.conf, "xyxy": list(d.xyxy)} for d in dets],
                }
                jf.write(json.dumps(payload) + "\n")
                jf.flush()

            if show:
                view = letterbox_to_screen(annotated, screen_w, screen_h) if fullscreen else annotated
                cv2.imshow("YOLOv8", view)
                k = cv2.waitKey(1) & 0xFF
                if k in (ord("q"), 27):
                    break
    finally:
        grabber.stop()
        try:
            grabber.join(timeout=1.0)
        except Exception:
            pass
        cap.release()
        if writer is not None:
            writer.release()
        if jf is not None:
            jf.close()
        if show:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    if out_path:
        print(f"[OK] Saved video: {out_path}")
    if json_path:
        print(f"[OK] Saved detections: {json_path}")

    return 0


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="yolo-mind2s-detect",
        description="Real-time YOLOv8 detection tuned for Khadas Mind 2S (Intel CPU/GPU/NPU) + optional NVIDIA CUDA.",
    )
    p.add_argument("--model", default="yolov8x.pt", help="Model id/path (e.g. yolov8n.pt, yolov8x.pt, custom.pt).")
    p.add_argument("--source", default="0", help="Video source: webcam index (e.g. 0) or path/URL.")
    p.add_argument("--device", default="", help="Device: auto/cpu/cuda:0/intel:npu/intel:gpu/intel:cpu (see --list-devices).")
    p.add_argument("--list-devices", action="store_true", help="List detected devices and exit.")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    p.add_argument("--iou", type=float, default=0.45, help="IoU threshold.")
    p.add_argument("--classes", default="", help="Class filter: 'ALL' or '1,3,7' or '0,2,6' or '1-5,8'.")
    p.add_argument("--max-det", type=int, default=300, help="Max detections per frame.")
    p.add_argument("--half", action="store_true", help="Use FP16 on CUDA (ignored elsewhere).")
    p.add_argument("--fullscreen", action="store_true", help="Fullscreen display (letterboxed).")
    p.add_argument("--no-display", action="store_true", help="Headless mode (no window).")
    p.add_argument("--save-video", action="store_true", help="Save annotated MP4.")
    p.add_argument("--save-json", action="store_true", help="Save detections as JSONL.")
    p.add_argument("--out-dir", default="runs", help="Output directory for saved files.")
    p.add_argument("--width", type=int, default=0, help="Camera capture width (0=default).")
    p.add_argument("--height", type=int, default=0, help="Camera capture height (0=default).")
    p.add_argument("--cam-fps", type=int, default=0, help="Camera requested FPS (0=default).")
    p.add_argument("--no-fps", action="store_true", help="Disable FPS overlay.")
    p.add_argument("--prefer-int8-on-npu", action="store_true", help="When using intel:npu, export INT8 OpenVINO model if needed.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    from .utils import parse_class_selection

    args = build_argparser().parse_args(argv)

    if args.list_devices:
        print("Detected devices:")
        for d in available_devices():
            print(f" - {d.name:12} ({d.detail})")
        return 0

    device = args.device.strip() or pick_default_device()
    if device.lower() == "auto":
        device = pick_default_device()

    classes = None
    if args.classes.strip():
        try:
            classes = parse_class_selection(args.classes, n_classes=80)
        except Exception as e:
            print(f"[WARN] Class selection parse failed: {e}. Falling back to ALL.")
            classes = None

    return run_detect(
        model_id=args.model,
        source=args.source,
        device=device,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        classes=classes,
        max_det=args.max_det,
        half=args.half,
        fullscreen=args.fullscreen,
        show=not args.no_display,
        save_video=args.save_video,
        save_json=args.save_json,
        out_dir=args.out_dir,
        width=args.width,
        height=args.height,
        cam_fps=args.cam_fps,
        overlay_fps=not args.no_fps,
        prefer_int8_on_npu=bool(args.prefer_int8_on_npu),
    )


if __name__ == "__main__":
    raise SystemExit(main())
