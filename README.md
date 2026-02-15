# Yolov8-Mind2S (Intel CPU/GPU/NPU + optional NVIDIA CUDA)

This repo is a Khadas **Mind 2S–optimized** YOLOv8 webcam demo with:

- **Intel OpenVINO acceleration** for **intel:cpu / intel:gpu / intel:npu**
- Optional **NVIDIA CUDA** acceleration (eGPU / discrete GPU)
- Threaded capture + FPS overlay
- Interactive menu UI + CLI mode
- Optional **web MJPEG stream** (FastAPI/Uvicorn)
- A quick **device benchmark** tool

> If you want **Intel Arc (iGPU)** or **NPU** acceleration, install OpenVINO + the proper Intel drivers.

---

## Windows 11 (Anaconda) quick start (Mind 2S)

### 1) Create and activate the conda environment

Open **Anaconda Prompt**:

```bat
conda create -n envYolo python=3.11 -y
conda activate envYolo
```

### 2) Get the repo

**Option A (Git):**
```bat
cd %USERPROFILE%\AI
git clone https://github.com/KD5VMF/Yolov8-Mind2S.git
cd Yolov8-Mind2S
```

**Option B (ZIP):**
- Download the repo ZIP from GitHub
- Extract it to: `%USERPROFILE%\AI\Yolov8-Mind2S`
- `cd` into that folder

### 3) Install Python dependencies

```bat
python -m pip install -U pip
pip install -r requirements.txt
```

### 4) Intel CPU/GPU/NPU (OpenVINO) acceleration (recommended)

```bat
pip install -r requirements-openvino.txt
python -c "import openvino as ov; print('OpenVINO devices:', ov.Core().available_devices)"
```

If you **don’t** see `GPU` and/or `NPU` in that list, install the latest Mind 2S Intel drivers package and reboot.

### 5) Run (Interactive UI)

```bat
python yolo_interactive.py
```

Pick one of:
- `intel:npu` (fast/efficient, best when available)
- `intel:gpu` (Intel Arc iGPU)
- `cuda:0` (NVIDIA GPU, if present)
- `cpu`

---

## Ubuntu quick start (Mind 2S)

```bash
git clone https://github.com/KD5VMF/Yolov8-Mind2S.git
cd Yolov8-Mind2S
./scripts/setup_mind2s_ubuntu.sh
./scripts/run_interactive.sh
```

Then for Intel OpenVINO:

```bash
source ~/envYoloMind2S/bin/activate
pip install -r requirements-openvino.txt
```

---

## Run modes

### Interactive (menu UI)

```bash
python ./yolo_interactive.py
```

### CLI (scriptable)

List devices:
```bash
python ./yolo_detect.py --list-devices
```

Intel NPU:
```bash
python ./yolo_detect.py --source 0 --device intel:npu --model yolov8x.pt --prefer-int8-on-npu --fullscreen
```

Web MJPEG stream:
```bash
python ./yolo_web.py --host 0.0.0.0 --port 8000 --device auto --source 0 --model yolov8x.pt
```

Quick benchmark:
```bash
python ./tools/benchmark_devices.py --model yolov8n.pt --source 0 --frames 120
```

---

## Notes

- **OpenVINO export + INT8:** Ultralytics uses `nncf` to produce INT8 models; this repo includes it in `requirements-openvino.txt`.
- **One process per device:** safest way to switch devices is to restart the script.

---

## Project layout

- `yolo_interactive.py` – menu UI (recommended)
- `yolo_detect.py` – CLI
- `yolo_web.py` – web stream
- `src/yolov8_mind2s/` – core package code
- `scripts/` – Ubuntu setup + run helpers
- `tools/` – benchmark tool
- `legacy/` + `systemd/` – placeholders for future wrappers/services

---

## License

MIT (see `LICENSE`)
