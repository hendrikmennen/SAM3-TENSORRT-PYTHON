## SAM3 TensorRT Pipeline

This project provides a complete pipeline to run **SAM3** (Segment Anything Model 3) with **TensorRT**:

- **System audit** for CUDA / TensorRT readiness (`Check.py`)
- **ONNX export** of the SAM3 submodules (`SAM3_PyTorch_To_Onnx.py`)
- **TensorRT engine building** from ONNX (`Build_Engines.py`)
- **High‑performance inference** with text prompts (`SAM3_TensorRT_Inference.py`)

The workflow is designed around FP16 TensorRT engines with dynamic shapes and explicit batch.

---

## 1. Environment Setup

**Python packages**

Use the provided commands in `Requirements_Install_Commands.txt` (Linux‑oriented). At minimum you will need:

- **Core**: `torch`, `torchvision`, `numpy`, `opencv-python`, `tokenizers`, `transformers`, `onnx`, `onnxruntime-gpu`
- **Deployment / tooling**: `cuda-python`, `tensorrt` (Python bindings), `onnx_graphsurgeon`, `nvidia-modelopt`, etc.

Example (adapt from the text file as needed for your OS):

```bash
pip install torch torchvision --upgrade
pip uninstall torchaudio -y
pip install litserve uvicorn python-multipart
pip install onnx onnxscript onnxslim onnxruntime-gpu "numpy>=2.2.6" opencv-python \
    "transformers" \
    matplotlib aiohttp locust pandas tabulate cuda-python==12.8.0 --upgrade
pip install tritonclient[http] opencv-python-headless tokenizers
pip install nvidia-modelopt "numpy>=2.2.6" tensorboard scikit-learn "protobuf>=4.25.1"
pip install onnx_graphsurgeon
```

**TensorRT**

- Install TensorRT that matches your CUDA toolkit.
- Ensure `trtexec` is on your `PATH` and TensorRT libraries are visible in your library path.
- For Linux, `Requirements_Install_Commands.txt` includes an example `apt-get` + `pip` install and `PATH` / `LD_LIBRARY_PATH` exports.
- REMEMBER TO MATCH CUDA-PYTHON LESS THAN OR EQUAL TO CUDA-TOOLKIT INSTALLED
---

## 2. Download or Export ONNX Models

You have two options: **download pre‑exported ONNX** or **export from PyTorch yourself**.

### 2.1. Download pre‑exported ONNX (recommended)

From `Export Instrcutions.txt`:

```bash
hf download --local-dir "Onnx-Models" kishanstar2003/SAM3_ONNX_FP16
```

This will create an `Onnx-Models` directory containing:

- `vision-encoder.onnx`
- `text-encoder.onnx`
- `geometry-encoder.onnx`
- `decoder.onnx`
- `tokenizer.json` (copy later to the engines directory)

### 2.2. Export ONNX from the SAM3 PyTorch model

1. Download the original SAM3 weights:

```bash
hf download facebook/sam3 --local-dir sam3
```

2. Run the ONNX export script (`SAM3_PyTorch_To_Onnx.py` is a self‑contained exporter):

```bash
python SAM3_PyTorch_To_Onnx.py --all \
  --model-path ./sam3 \
  --output-dir Onnx-Models \
  --device cuda
```

Key points:

- The script exports four modules via wrappers:
  - `VisionEncoderWrapper` → `vision-encoder.onnx`
  - `TextEncoderWrapper` → `text-encoder.onnx`
  - `GeometryEncoderWrapper` → `geometry-encoder.onnx`
  - `DecoderWrapper` → `decoder.onnx`
- All exports use **opset 17** and **dynamic batch / prompt dimensions**, compatible with TensorRT.

---

## 3. Build TensorRT Engines

Once you have the ONNX models in `Onnx-Models`, build TensorRT engines using `Build_Engines.py`.

```bash
python Build_Engines.py --onnx "Onnx-Models" --engine "Engines"
```

Arguments:

- `--base` (optional): base directory (default: current working directory).
- `--onnx`: directory containing `.onnx` models (default: `BASE/Onnx-Models`).
- `--engine`: output directory for `.engine` files (default: `BASE/Engines`).

The script:

- Runs `trtexec` with **FP16** and appropriate **min/opt/max shapes** for each module:
  - `vision-encoder`
  - `text-encoder`
  - `geometry-encoder`
  - `decoder`
- Skips engines that already exist.

After building:

- Ensure `Engines` contains `vision-encoder.engine`, `text-encoder.engine`, `geometry-encoder.engine` (if used) and `decoder.engine`.
- **Copy `tokenizer.json`** from `Onnx-Models` (or model download dir) into `Engines`:

```bash
cp Onnx-Models/tokenizer.json Engines/
```

On Windows, use the equivalent `copy` command in PowerShell or CMD.

---

## 4. Verify System & TensorRT Installation

Use `Check.py` to audit your environment:

```bash
python Check.py
```

It reports:

- **GPU hardware and driver** via `nvidia-smi`
- **CUDA Python (`cuda-python`) and CUDART** status
- **NVCC** presence and version
- **PyTorch**, CUDA version, and **ONNX Runtime**
- **TensorRT Python bindings** and builder creation
- **`trtexec`** availability
- Available **ONNX Runtime providers** (CUDA / TensorRT, etc.)

Run this once after setup to confirm everything is wired correctly.

---

## 5. Run TensorRT Inference

With engines and tokenizer in place, run the end‑to‑end inference script:

```bash
python SAM3_TensorRT_Inference.py \
  --input "Assets/Test.jpg" \
  --prompt "person" \
  --conf 0.7 \
  --output result.jpg \
  --models ./Engines
```

Arguments:

- `--input`: path to input image file.
- `--prompt`: text prompt (e.g., `"person"`, `"car"`, `"dog"`).
- `--conf`: confidence threshold \(0.0–1.0\) applied on box scores.
- `--output`: path to save the annotated image.
- `--models`: directory containing `.engine` files and `tokenizer.json` (typically `Engines`).

What the script does:

- Wraps each engine with `TRTModule` for efficient execution using PyTorch CUDA tensors.
- Preprocesses the input image:
  - Resize to `1008 × 1008`
  - Normalize to \([-1, 1]\)
- Runs:
  - Vision encoder → FPN features + positional encodings
  - Text encoder → token embeddings + masks (via `tokenizers` and `tokenizer.json`)
  - Decoder → predicted boxes, logits, presence logits, and masks
- Computes combined scores from logits and presence logits, filters by `--conf`, denormalizes boxes, and draws them onto the original image.

Output:

- An image with bounding boxes and scores, saved to `--output`.

---

## 6. Files Overview

- `SAM3_PyTorch_To_Onnx.py`  
  TensorRT‑friendly ONNX exporter for SAM3:
  - Custom wrappers (`VisionEncoderWrapper`, `TextEncoderWrapper`, `GeometryEncoderWrapper`, `DecoderWrapper`)
  - Proper dynamic axes and shapes for TensorRT.

- `Build_Engines.py`  
  Converts ONNX models into TensorRT `.engine` files using `trtexec` with FP16 and explicit shapes.

- `SAM3_TensorRT_Inference.py`  
  Runs **standalone TensorRT inference** with text prompts and saves visualized detections.

- `Check.py`  
  System diagnostic script for GPU, CUDA, TensorRT, and ONNX Runtime.

- `Requirements_Install_Commands.txt`  
  Reference commands for Python and TensorRT stack installation (primarily Linux).

- `Export Instrcutions.txt`  
  Quick reference for downloading/exporting ONNX, building engines, and running inference.

- `Assets/Test.jpg`  
  Example input image for testing the pipeline.

---

## 7. Credits

This work is adapted from and inspired by:

- `https://github.com/jamjamjon/usls/tree/main/scripts/sam3-image`
- `https://github.com/facebookresearch/sam3`

