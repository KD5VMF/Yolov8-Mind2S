#!/usr/bin/env python3
"""Convenience launcher: Web MJPEG stream (FastAPI/Uvicorn)."""
from yolov8_mind2s.web.app import main

if __name__ == "__main__":
    raise SystemExit(main())
