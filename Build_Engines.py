import os
import subprocess
import sys
import argparse
import shutil
import onnx  # Required to inspect the model

PATCH_SIZE_DIVISOR = 14  # ViT patch size
TOKENIZER_ASSETS = ("vocab.json", "merges.txt", "special_tokens_map.json")

def get_onnx_resolution(onnx_path):
    """
    Loads the ONNX model and attempts to parse the input resolution (H, W).
    Assumes standard NCHW format for the 'images' input.
    """
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"Vision encoder not found at: {onnx_path}")

    print(f"[INFO] Inspecting {onnx_path} for resolution...")
    model = onnx.load(onnx_path)
    
    # Look for the input tensor named 'images'
    target_input = None
    for inp in model.graph.input:
        if inp.name == "images":
            target_input = inp
            break
    
    # If explicit 'images' not found, try the first input (fallback)
    if target_input is None:
        target_input = model.graph.input[0]
        print(f"[WARN] Input 'images' not found. Using first input: '{target_input.name}'")

    try:
        # Access tensor dimensions: [Batch, Channels, Height, Width]
        # dim[2] is Height, dim[3] is Width
        dims = target_input.type.tensor_type.shape.dim
        h = dims[2].dim_value
        w = dims[3].dim_value
        
        if h <= 0 or w <= 0:
            raise ValueError(f"Dimensions found ({w}x{h}) are dynamic or invalid. Export ONNX with static shapes.")
            
        if h != w:
            print(f"[WARN] Non-square input detected ({w}x{h}). Using Height ({h}) as base size.")
            
        return h
    except IndexError:
        raise ValueError("Could not parse input dimensions. Ensure ONNX has NCHW format.")

def build_engines(onnx_dir, engine_dir, size):
    """Builds TensorRT engines from ONNX models for a given resolution."""

    feat = size // PATCH_SIZE_DIVISOR
    feat0 = feat * 4
    feat1 = feat * 2
    feat2 = feat

    models = {
        "vision-encoder": {
            "onnx": os.path.join(onnx_dir, "vision-encoder.onnx"),
            "flags": [
                "--fp16",
                f"--minShapes=images:1x3x{size}x{size}",
                f"--optShapes=images:1x3x{size}x{size}",
                f"--maxShapes=images:1x3x{size}x{size}",
            ],
        },
        "text-encoder": {
            "onnx": os.path.join(onnx_dir, "text-encoder.onnx"),
            "flags": [
                "--fp16",
                "--minShapes=input_ids:1x32,attention_mask:1x32",
                "--optShapes=input_ids:1x32,attention_mask:1x32",
                "--maxShapes=input_ids:1x32,attention_mask:1x32",
            ],
        },
        "geometry-encoder": {
            "onnx": os.path.join(onnx_dir, "geometry-encoder.onnx"),
            "flags": [
                "--fp16",
                f"--minShapes=input_boxes:1x1x4,input_boxes_labels:1x1,"
                f"fpn_feat_2:1x256x{feat2}x{feat2},fpn_pos_2:1x256x{feat2}x{feat2}",
                f"--optShapes=input_boxes:1x5x4,input_boxes_labels:1x5,"
                f"fpn_feat_2:1x256x{feat2}x{feat2},fpn_pos_2:1x256x{feat2}x{feat2}",
                f"--maxShapes=input_boxes:1x200x4,input_boxes_labels:1x200,"
                f"fpn_feat_2:1x256x{feat2}x{feat2},fpn_pos_2:1x256x{feat2}x{feat2}",
            ],
        },
        "decoder": {
            "onnx": os.path.join(onnx_dir, "decoder.onnx"),
            "flags": [
                "--fp16",
                f"--minShapes=fpn_feat_0:1x256x{feat0}x{feat0},"
                f"fpn_feat_1:1x256x{feat1}x{feat1},"
                f"fpn_feat_2:1x256x{feat2}x{feat2},"
                f"fpn_pos_2:1x256x{feat2}x{feat2},"
                f"prompt_features:1x1x256,prompt_mask:1x1",
                f"--optShapes=fpn_feat_0:1x256x{feat0}x{feat0},"
                f"fpn_feat_1:1x256x{feat1}x{feat1},"
                f"fpn_feat_2:1x256x{feat2}x{feat2},"
                f"fpn_pos_2:1x256x{feat2}x{feat2},"
                f"prompt_features:1x40x256,prompt_mask:1x40",
                f"--maxShapes=fpn_feat_0:1x256x{feat0}x{feat0},"
                f"fpn_feat_1:1x256x{feat1}x{feat1},"
                f"fpn_feat_2:1x256x{feat2}x{feat2},"
                f"fpn_pos_2:1x256x{feat2}x{feat2},"
                f"prompt_features:1x250x256,prompt_mask:1x250",
            ],
        },
    }

    os.makedirs(engine_dir, exist_ok=True)

    # Copy tokenizer assets if present
    for asset_name in TOKENIZER_ASSETS:
        src = os.path.join(onnx_dir, asset_name)
        dst = os.path.join(engine_dir, asset_name)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)

    for name, cfg in models.items():
        engine_path = os.path.join(engine_dir, f"{name}.engine")
        if os.path.exists(engine_path):
            print(f"[SKIP] {engine_path} exists")
            continue

        print(f"[BUILD] {name} ({size}px)")
        cmd = ["trtexec", f"--onnx={cfg['onnx']}", f"--saveEngine={engine_path}"] + cfg["flags"]

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            print(f"[ERROR] Failed building {name}")
            sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Build TensorRT engines for SAM-3")
    parser.add_argument("--base", default=".")
    parser.add_argument("--onnx", help="ONNX directory (default: BASE/Onnx-Models)")
    parser.add_argument("--engine", help="Engine directory (default: BASE/Engines/SIZE)")
    
    # Removed explicit --size argument
    
    args = parser.parse_args()

    onnx_dir = args.onnx or os.path.join(args.base, "Onnx-Models")
    
    # 1. Auto-detect size
    try:
        vision_path = os.path.join(onnx_dir, "vision-encoder.onnx")
        detected_size = get_onnx_resolution(vision_path)
    except Exception as e:
        print(f"[ERROR] Failed to auto-detect resolution: {e}")
        sys.exit(1)

    # 2. Set engine dir based on detected size
    engine_dir = args.engine or os.path.join(args.base, "Engines", str(detected_size))

    print("---- TensorRT Build ----")
    print(f"Detected Size : {detected_size} px")
    print(f"ONNX Dir      : {onnx_dir}")
    print(f"Engine Dir    : {engine_dir}")
    print("------------------------")

    build_engines(onnx_dir, engine_dir, detected_size)
