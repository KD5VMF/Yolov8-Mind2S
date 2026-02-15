# Yolov8-Mind2S (Intel CPU/GPU/NPU + optional NVIDIA CUDA)

This is a refreshed, **Khadas Mind 2S–optimized** version of your YOLOv8 webcam demo repo.

It keeps the original “pick classes + run webcam detection” workflow, but adds:
- **Intel OpenVINO acceleration** for **intel:cpu / intel:gpu / intel:npu**
- Optional **NVIDIA CUDA** acceleration (eGPU / discrete GPU)
- A faster capture loop (threaded) + FPS overlay
- A simple **web MJPEG stream** (FastAPI/Uvicorn)
- A quick **device benchmark** tool

> Intel AI PCs combine CPU+GPU+NPU, and OpenVINO is the standard way to run/optimize models across those engines. citeturn8view0

## What you get

**Interactive mode (closest to your original repo):**
- choose device
- choose model
- choose classes (numbers / ranges / ALL)
- fullscreen display
- optional save MP4 + JSONL detections

**CLI mode:**
- same features, but scripted via flags

**Web mode:**
- open a browser to view the annotated camera stream

## Quick start (Ubuntu on Mind 2S)

```bash
git clone <YOUR_NEW_REPO_URL>
cd Yolov8-Mind2S
./scripts/setup_mind2s_ubuntu.sh
./scripts/run_interactive.sh
```

### Intel GPU / NPU acceleration (OpenVINO)

Install the OpenVINO runtime:

```bash
source ~/envYoloMind2S/bin/activate
pip install -r requirements-openvino.txt
```

**Important (NPU):** OpenVINO’s NPU plugin requires an **NPU driver** to be installed on the system. citeturn8view1  
(The OpenVINO docs link to the current Linux/Windows driver installers. citeturn8view1)

### NVIDIA CUDA (optional)

If you have an NVIDIA GPU attached (like your RTX 3060 eGPU), install a CUDA-enabled PyTorch build and then run with `--device cuda:0`.
(PyTorch’s install command depends on your exact CUDA version.)

## Run modes

### 1) Interactive (menu UI)

```bash
source ~/envYoloMind2S/bin/activate
python ./yolo_interactive.py
```

### 2) CLI (scriptable)

List devices:

```bash
source ~/envYoloMind2S/bin/activate
python ./yolo_detect.py --list-devices
```

Run webcam, Intel NPU, export INT8 if needed:

```bash
python ./yolo_detect.py --source 0 --device intel:npu --model yolov8x.pt --prefer-int8-on-npu --fullscreen
```

Run webcam, Intel GPU:

```bash
python ./yolo_detect.py --source 0 --device intel:gpu --model yolov8x.pt --fullscreen
```

Run webcam, NVIDIA CUDA:

```bash
python ./yolo_detect.py --source 0 --device cuda:0 --model yolov8x.pt --half --fullscreen
```

Save outputs:

```bash
python ./yolo_detect.py --source 0 --device auto --save-video --save-json --out-dir runs
```

### 3) Web MJPEG stream

```bash
source ~/envYoloMind2S/bin/activate
python ./yolo_web.py --host 0.0.0.0 --port 8000 --device auto --source 0 --model yolov8x.pt
```

Then open: `http://<server-ip>:8000/`

### 4) Quick benchmark

```bash
source ~/envYoloMind2S/bin/activate
./tools/benchmark_devices.py --model yolov8n.pt --source 0 --frames 120
```

## Notes & known gotchas

- **OpenVINO device switching:** Some Ultralytics + OpenVINO setups historically had issues where the first inference “locks” the device selection; safest approach is to run one process per device choice (restart to change). citeturn8view2

## Project layout

- `yolo_interactive.py` – menu UI (recommended)
- `yolo_detect.py` – CLI
- `yolo_web.py` – web stream
- `src/yolov8_mind2s/` – core package code
- `scripts/` – setup + run helpers
- `legacy/` – wrappers that match your original filenames

## License

MIT (see `LICENSE`)
