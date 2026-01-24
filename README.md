## SAM3 TensorRT Pipeline

This project provides a complete pipeline to run **SAM3** (Segment Anything Model 3) with **TensorRT**:

- **System audit** for CUDA / TensorRT readiness (`Check.py`)
- **ONNX export** of the SAM3 submodules (`SAM3_PyTorch_To_Onnx.py`)
- **TensorRT engine building** from ONNX (`Build_Engines.py`)
- **High‑performance inference** with text prompts (`SAM3_TensorRT_Inference.py`)
- **Interactive web UI** for easy testing (`ui_gradio.py`)

The workflow is designed around FP16 TensorRT engines with dynamic shapes and explicit batch, supporting both **bounding box detection** and **mask segmentation** modes.

![SAM3 UI](SAM3_UI.png)
![WORKFLOW](WorkFlow.png)

---

## 🚀 Performance Benchmarks

By migrating from native PyTorch to TensorRT (FP16), this pipeline delivers massive efficiency gains.

| Metric | Original PyTorch | TensorRT (FP16) | Improvement |
| :--- | :--- | :--- | :--- |
| **VRAM Usage** | ~6-7 GB | **~2.4 GB** | **~65% Reduction** |
| **Inference Time** (T4 GPU) | ~1.6 sec | **~0.6 sec** | **~2.5x Speedup** |

*Note: Benchmarks tested on NVIDIA T4 GPU. Performance may vary based on hardware.*

---

## Quick Start

1. **Install dependencies** (see section 1 for details):
   ```bash
   Refer Section 1
   ```

2. **Download pre-exported ONNX models**:
   ```bash
   hf download --local-dir "Onnx-Models" kishanstar2003/SAM3_ONNX_FP16
   ```

3. **Build TensorRT engines**:
   ```bash
   python3 Build_Engines.py --onnx "Onnx-Models" --engine "Engines"
   ```

4. **Run inference** (choose one):
   - **Web UI**: `python3 ui_gradio.py`
   - **Command line**: `python3 SAM3_TensorRT_Inference.py --input "Assets/Test.jpg" --prompt "person" --output result.jpg --segment`

---

## 1. Environment Setup

**Python packages**

Use the provided commands in `Requirements_Install_Commands.txt` (Linux‑oriented).

**TensorRT**

- Install TensorRT that matches your CUDA toolkit.
- Ensure `trtexec` is on your `PATH` and TensorRT libraries are visible in your library path.
- For Linux, `Requirements_Install_Commands.txt` includes an example `apt-get` + `pip` install and `PATH` / `LD_LIBRARY_PATH` exports.
- REMEMBER TO MATCH CUDA-PYTHON VERSION <= TO CUDA-TOOLKIT VERSION INSTALLED
---

## 2. Download or Export ONNX Models

You have two options: **download pre‑exported ONNX** or **export from PyTorch yourself**.

### 2.1. Download pre‑exported ONNX (recommended)

From `Instrcutions.txt`:

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
python3 SAM3_PyTorch_To_Onnx.py --all \
  --model-path "sam3" \
  --output-dir "Onnx-Models" \
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
python3 Build_Engines.py --onnx "Onnx-Models" --engine "Engines"
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

---

## 4. Verify System & TensorRT Installation

Use `Check.py` to audit your environment:

```bash
python3 Check.py
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

With engines and tokenizer in place, you can run inference in two ways: **command line** or **interactive web UI**.

### 5.1. Command Line Inference

Run the end‑to‑end inference script:

**Bounding Box Detection Mode:**
```bash
python3 SAM3_TensorRT_Inference.py \
  --input "Assets/Test.jpg" \
  --prompt "person" \
  --conf 0.7 \
  --output result.jpg \
  --models "Engines"
```

**Mask Segmentation Mode:**
```bash
python3 SAM3_TensorRT_Inference.py \
  --input "Assets/Test.jpg" \
  --prompt "person" \
  --conf 0.7 \
  --output result.jpg \
  --models "Engines" \
  --segment 
```

Arguments:

- `--input`: path to input image file.
- `--prompt`: text prompt (e.g., `"person"`, `"car"`, `"dog"`).
- `--conf`: confidence threshold \(0.0–1.0\) applied on box scores.
- `--output`: path to save the annotated image.
- `--models`: directory containing `.engine` files and `tokenizer.json` (typically `Engines`).
- `--segment`: (optional) enable mask segmentation mode. If omitted, uses bounding box detection.

### 5.2. Interactive Web UI

For easier testing and experimentation, use the Gradio web interface:

```bash
python ui_gradio.py
```

This launches a web interface where you can:
- Upload images directly
- Enter text prompts interactively
- Adjust confidence thresholds with sliders
- Toggle between bounding box and segmentation modes
- View results instantly with performance metrics

The UI automatically loads the TensorRT engines from the `Engines` directory and provides real-time inference.

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
- **Bounding Box Mode**: Draws rectangular boxes around detected objects
- **Segmentation Mode**: Generates and overlays pixel-accurate masks for detected objects

Output:

- An image with bounding boxes/masks and scores, saved to `--output`.

---

## 6. Files Overview

- `SAM3_PyTorch_To_Onnx.py`  
  TensorRT‑friendly ONNX exporter for SAM3:
  - Custom wrappers (`VisionEncoderWrapper`, `TextEncoderWrapper`, `GeometryEncoderWrapper`, `DecoderWrapper`)
  - Proper dynamic axes and shapes for TensorRT.

- `Build_Engines.py`  
  Converts ONNX models into TensorRT `.engine` files using `trtexec` with FP16 and explicit shapes.

- `SAM3_TensorRT_Inference.py`  
  Runs **standalone TensorRT inference** with text prompts and saves visualized detections. Supports both bounding box and mask segmentation modes.

- `ui_gradio.py`  
  **Interactive web interface** built with Gradio for easy testing and experimentation. Provides real-time inference with adjustable parameters.

- `Check.py`  
  System diagnostic script for GPU, CUDA, TensorRT, and ONNX Runtime.

- `Requirements_Install_Commands.txt`  
  Reference commands for Python and TensorRT stack installation (primarily Linux).

- `Instrcutions.txt`  
  Quick reference for downloading/exporting ONNX, building engines, and running inference commands.

- `Assets/Test.jpg`  
  Example input image for testing the pipeline.

---

## 7. Troubleshooting

**Common Issues:**

- **"CUDA out of memory"**: Reduce batch size or use smaller input images
- **"TensorRT engine not found"**: Ensure you've run `Build_Engines.py` and copied `tokenizer.json`
- **"trtexec not found"**: Add TensorRT bin directory to your PATH
- **Import errors**: Run `python3 Check.py` to verify your environment setup
- **Slow inference**: Ensure you're using GPU-enabled ONNX Runtime and TensorRT engines

---

## 8. Credits

This work is adapted from and inspired by:

- **Original SAM3 implementation**: [facebook/sam3](https://github.com/facebookresearch/sam3)
- **TensorRT optimization techniques**: [jamjamjon/usls SAM3 scripts](https://github.com/jamjamjon/usls/tree/main/scripts/sam3-image)

Special thanks to the contributors of these projects for their foundational work in making SAM3 accessible and optimized for deployment.

