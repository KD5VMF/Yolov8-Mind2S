from __future__ import annotations

import argparse
import asyncio
import io
import os
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np  # type: ignore
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from ..detect import _open_capture, _draw_detections, _results_to_detections
from ..devices import pick_default_device
from ..model import load_model

import cv2  # type: ignore


app = FastAPI(title="YOLOv8 Mind2S Web Detect")


@dataclass
class WebConfig:
    model: str = "yolov8x.pt"
    source: str = "0"
    device: str = "auto"
    imgsz: int = 640
    conf: float = 0.25
    iou: float = 0.45
    max_det: int = 300
    prefer_int8_on_npu: bool = True


class Detector:
    def __init__(self, cfg: WebConfig):
        self.cfg = cfg
        self._lock = threading.Lock()
        self._running = False
        self._frame_jpeg: Optional[bytes] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        with self._lock:
            self._running = False

    def latest_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._frame_jpeg

    def _loop(self) -> None:
        cfg = self.cfg
        device = cfg.device
        if device.lower() == "auto" or not device.strip():
            device = pick_default_device()

        model, pred_device = load_model(cfg.model, device=device, prefer_int8_on_npu=cfg.prefer_int8_on_npu, imgsz=cfg.imgsz)
        names = getattr(model, "names", {}) or {}

        cap = _open_capture(cfg.source, 0, 0, 0)

        try:
            while True:
                with self._lock:
                    if not self._running:
                        break
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.01)
                    continue

                results = model.predict(
                    source=frame,
                    imgsz=cfg.imgsz,
                    conf=cfg.conf,
                    iou=cfg.iou,
                    device=pred_device,
                    max_det=cfg.max_det,
                    verbose=False,
                )
                result = results[0] if isinstance(results, list) and results else results
                dets = _results_to_detections(result, names)
                annotated = _draw_detections(frame, dets, show_labels=True)

                ok2, jpg = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                if not ok2:
                    continue
                with self._lock:
                    self._frame_jpeg = jpg.tobytes()
        finally:
            cap.release()


_cfg = WebConfig()
_det = Detector(_cfg)
_det.start()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    # Tiny HTML UI. Use <img> MJPEG.
    return HTMLResponse(
        f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>YOLOv8 Mind2S Web Detect</title>
  <style>
    body {{ font-family: system-ui, Arial, sans-serif; background: #0b0f14; color: #e7eef7; margin: 0; }}
    .wrap {{ max-width: 1200px; margin: 0 auto; padding: 16px; }}
    .card {{ background: #111826; border: 1px solid #1b2a40; border-radius: 12px; padding: 12px; }}
    .row {{ display: flex; gap: 12px; flex-wrap: wrap; align-items: center; }}
    code {{ background: #0a1220; padding: 2px 6px; border-radius: 6px; }}
    img {{ width: 100%; height: auto; border-radius: 12px; border: 1px solid #1b2a40; }}
    .small {{ opacity: 0.8; font-size: 14px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h2>YOLOv8 Mind 2S Web Detect</h2>
    <div class="row small">
      <div class="card">Model: <code>{_cfg.model}</code></div>
      <div class="card">Device: <code>{_cfg.device}</code></div>
      <div class="card">Source: <code>{_cfg.source}</code></div>
      <div class="card">imgsz: <code>{_cfg.imgsz}</code> conf: <code>{_cfg.conf}</code> iou: <code>{_cfg.iou}</code></div>
    </div>
    <p class="small">Stream URL: <code>/stream</code></p>
    <img src="/stream" />
  </div>
</body>
</html>"""
    )


def _mjpeg_generator():
    boundary = b"frame"
    while True:
        frame = _det.latest_jpeg()
        if frame is None:
            time.sleep(0.02)
            continue
        yield (b"--" + boundary + b"\r\n"
               b"Content-Type: image/jpeg\r\n"
               b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n" +
               frame + b"\r\n")
        time.sleep(0.02)


@app.get("/stream")
async def stream():
    return StreamingResponse(_mjpeg_generator(), media_type="multipart/x-mixed-replace; boundary=frame")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="YOLOv8 Mind2S MJPEG web stream")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--model", default=_cfg.model)
    p.add_argument("--source", default=_cfg.source)
    p.add_argument("--device", default=_cfg.device)
    p.add_argument("--imgsz", type=int, default=_cfg.imgsz)
    p.add_argument("--conf", type=float, default=_cfg.conf)
    p.add_argument("--iou", type=float, default=_cfg.iou)
    p.add_argument("--max-det", type=int, default=_cfg.max_det)
    p.add_argument("--prefer-int8-on-npu", action="store_true")
    args = p.parse_args(argv)

    _cfg.model = args.model
    _cfg.source = args.source
    _cfg.device = args.device
    _cfg.imgsz = args.imgsz
    _cfg.conf = args.conf
    _cfg.iou = args.iou
    _cfg.max_det = args.max_det
    if args.prefer_int8_on_npu:
        _cfg.prefer_int8_on_npu = True

    # Restart detector with new cfg
    global _det
    _det.stop()
    time.sleep(0.1)
    _det = Detector(_cfg)
    _det.start()

    import uvicorn  # type: ignore
    uvicorn.run("yolov8_mind2s.web.app:app", host=args.host, port=args.port, reload=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
