from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np  # type: ignore


@dataclass
class FpsCounter:
    window_s: float = 1.0
    _t0: float = 0.0
    _n: int = 0
    fps: float = 0.0

    def __post_init__(self) -> None:
        self._t0 = time.perf_counter()

    def tick(self, n: int = 1) -> float:
        self._n += n
        t = time.perf_counter()
        dt = t - self._t0
        if dt >= self.window_s:
            self.fps = self._n / dt
            self._n = 0
            self._t0 = t
        return self.fps


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def parse_class_selection(user: str, n_classes: int) -> Optional[List[int]]:
    """Return None for ALL, else a sorted unique list of 0-based class indices."""
    s = user.strip().lower()
    if s in ("all", "*", "a", ""):
        return None

    tokens = re.split(r"[\s,]+", s)
    out: List[int] = []
    for tok in tokens:
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-", 1)
            if not a.isdigit() or not b.isdigit():
                raise ValueError(f"Bad range token: {tok!r}")
            lo = int(a)
            hi = int(b)
            if lo > hi:
                lo, hi = hi, lo
            out.extend(list(range(lo, hi + 1)))
        else:
            if not tok.isdigit():
                raise ValueError(f"Bad class token: {tok!r}")
            out.append(int(tok))

    # Accept 1-based input if it matches 1..n
    if out and min(out) >= 1 and max(out) <= n_classes:
        out = [x - 1 for x in out]

    cleaned = sorted({x for x in out if 0 <= x < n_classes})
    if not cleaned:
        raise ValueError("No valid classes selected.")
    return cleaned


def columns(items: Sequence[str], col_width: int = 22, cols: int = 4) -> str:
    lines: List[str] = []
    for i in range(0, len(items), cols):
        row = items[i : i + cols]
        lines.append("".join(s.ljust(col_width) for s in row))
    return "\n".join(lines)


def letterbox_to_screen(img: np.ndarray, screen_w: int, screen_h: int) -> np.ndarray:
    """Resize while preserving aspect ratio, pad with black."""
    h, w = img.shape[:2]
    if w == 0 or h == 0:
        return img

    scale = min(screen_w / w, screen_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = img if (new_w == w and new_h == h) else _cv2_resize(img, new_w, new_h)
    canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)

    x0 = (screen_w - new_w) // 2
    y0 = (screen_h - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def _cv2_resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    import cv2  # type: ignore
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)


def stable_color_for_class(cls_id: int) -> Tuple[int, int, int]:
    """Deterministic BGR color."""
    r = (cls_id * 73 + 41) % 256
    g = (cls_id * 151 + 17) % 256
    b = (cls_id * 199 + 89) % 256
    return int(b), int(g), int(r)


# COCO-80 class names as a safe fallback (Ultralytics default models)
COCO80 = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
    "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
    "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",
    "wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange",
    "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed",
    "dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush",
]
