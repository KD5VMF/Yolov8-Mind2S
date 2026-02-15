from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2  # type: ignore
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from ..detect import _draw_detections, _open_capture, _results_to_detections
from ..devices import pick_default_device
from ..model import get_class_names, load_model


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
        device = cfg.device.strip() or "auto"
        if device.lower() == "auto":
            device = pick_default_device()

        model, pred_device = load_model(cfg.model, device=device, prefer_int8_on_npu=cfg.prefer_int8_on_npu, imgsz=cfg.imgsz)
        names = get_class_names(model)

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
                if ok2:
                    with self._lock:
                        self._frame_jpeg = jpg.tobytes()
        finally:
            cap.release()


def create_app(cfg: WebConfig) -> FastAPI:
    app = FastAPI(title="YOLOv8 Mind2S Web Detect")
    det = Detector(cfg)
    det.start()

    @app.on_event("shutdown")
    def _shutdown():
        det.stop()

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        html = f'''<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>YOLOv8 Mind2S Web Detect</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; }}
    .box {{ max-width: 980px; }}
    img {{ width: 100%; border-radius: 10px; }}
    code {{ background: #f4f4f4; padding: 2px 6px; border-radius: 6px; }}
  </style>
</head>
<body>
  <div class="box">
    <h2>YOLOv8 Mind 2S Web Detect</h2>
    <p><b>Model:</b> <code>{cfg.model}</code></p>
    <p><b>Device:</b> <code>{cfg.device}</code></p>
    <p><b>Source:</b> <code>{cfg.source}</code></p>
    <p><b>imgsz:</b> <code>{cfg.imgsz}</code> &nbsp; <b>conf:</b> <code>{cfg.conf}</code> &nbsp; <b>iou:</b> <code>{cfg.iou}</code></p>
    <p><b>Stream:</b> <code>/stream</code></p>
    <img src="/stream" />
  </div>
</body>
</html>'''
        return HTMLResponse(html)

    def gen():
        boundary = b"--frame"
        while True:
            jpg = det.latest_jpeg()
            if jpg is None:
                time.sleep(0.02)
                continue
            yield boundary + b"\r\n"
            yield b"Content-Type: image/jpeg\r\n"
            yield f"Content-Length: {len(jpg)}\r\n\r\n".encode("utf-8")
            yield jpg + b"\r\n"
            time.sleep(0.01)

    @app.get("/stream")
    async def stream():
        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

    return app


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Web MJPEG stream for YOLOv8 Mind2S.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--model", default="yolov8x.pt")
    p.add_argument("--source", default="0")
    p.add_argument("--device", default="auto")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--max-det", type=int, default=300)
    p.add_argument("--prefer-int8-on-npu", action="store_true")
    return p


_cfg = WebConfig()
app = create_app(_cfg)


def main(argv=None) -> int:
    args = build_argparser().parse_args(argv)
    _cfg.model = args.model
    _cfg.source = args.source
    _cfg.device = args.device
    _cfg.imgsz = args.imgsz
    _cfg.conf = args.conf
    _cfg.iou = args.iou
    _cfg.max_det = args.max_det
    _cfg.prefer_int8_on_npu = bool(args.prefer_int8_on_npu)

    import uvicorn  # type: ignore
    uvicorn.run("yolov8_mind2s.web.app:app", host=args.host, port=args.port, reload=False, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
