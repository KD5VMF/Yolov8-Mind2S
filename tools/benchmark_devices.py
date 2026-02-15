#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
import time
from typing import List

import cv2  # type: ignore

from yolov8_mind2s.devices import available_devices
from yolov8_mind2s.model import load_model


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Quick device benchmark (webcam/video).")
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--source", default="0")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--frames", type=int, default=120)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--devices", default="", help="Comma list or 'auto' (default: auto-detect & test).")
    p.add_argument("--prefer-int8-on-npu", action="store_true")
    args = p.parse_args(argv)

    devs = [d.name for d in available_devices() if d.name != "cpu"] + ["cpu"]
    if args.devices.strip():
        if args.devices.strip().lower() == "auto":
            test = devs
        else:
            test = [x.strip() for x in args.devices.split(",") if x.strip()]
    else:
        test = [cand for cand in ("cuda:0", "intel:npu", "intel:gpu", "intel:cpu", "cpu") if cand in devs]

    try:
        src = int(args.source)
    except ValueError:
        src = args.source

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source {args.source!r}")
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Could not read first frame.")

    print("Benchmarking:")
    print(f" model:  {args.model}")
    print(f" source: {args.source}")
    print(f" imgsz:  {args.imgsz}")
    print(f" frames: {args.frames} (warmup {args.warmup})")
    print()

    best = None
    best_fps = 0.0

    for dev in test:
        print(f"== {dev} ==")
        model, pred_dev = load_model(args.model, device=dev, prefer_int8_on_npu=args.prefer_int8_on_npu, imgsz=args.imgsz)

        for _ in range(args.warmup):
            model.predict(source=frame, imgsz=args.imgsz, device=pred_dev, verbose=False)

        times: List[float] = []
        for _ in range(args.frames):
            ok, fr = cap.read()
            if not ok:
                break
            t0 = time.perf_counter()
            model.predict(source=fr, imgsz=args.imgsz, device=pred_dev, verbose=False)
            times.append(time.perf_counter() - t0)

        if not times:
            print(" no frames measured")
            continue

        avg = statistics.mean(times)
        fps = 1.0 / avg if avg > 0 else 0.0
        p95 = statistics.quantiles(times, n=20)[-1]
        print(f" avg: {avg*1000:.1f} ms (~{fps:.1f} fps)")
        print(f" p95: {p95*1000:.1f} ms")
        print()

        if fps > best_fps:
            best_fps = fps
            best = dev

    cap.release()
    print(f"Best: {best} (~{best_fps:.1f} fps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
