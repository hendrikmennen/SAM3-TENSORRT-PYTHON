import os
import subprocess
import sys
import argparse
import shutil

def build_engines(onnx_dir, engine_dir):
    """Builds TensorRT engines from ONNX models found in onnx_dir."""
    
    # Define models and their specific trtexec flags
    models = {
        "vision-encoder": {
            "onnx": os.path.join(onnx_dir, "vision-encoder.onnx"),
            "flags": ["--fp16", "--minShapes=images:1x3x1008x1008", "--optShapes=images:1x3x1008x1008", "--maxShapes=images:1x3x1008x1008"]
        },
        "text-encoder": {
            "onnx": os.path.join(onnx_dir, "text-encoder.onnx"),
            "flags": ["--fp16", "--minShapes=input_ids:1x32,attention_mask:1x32", "--optShapes=input_ids:1x32,attention_mask:1x32", "--maxShapes=input_ids:1x32,attention_mask:1x32"]
        },
        "geometry-encoder": {
            "onnx": os.path.join(onnx_dir, "geometry-encoder.onnx"),
            "flags": ["--fp16", "--minShapes=input_boxes:1x1x4,input_boxes_labels:1x1,fpn_feat_2:1x256x72x72,fpn_pos_2:1x256x72x72", "--optShapes=input_boxes:1x5x4,input_boxes_labels:1x5,fpn_feat_2:1x256x72x72,fpn_pos_2:1x256x72x72", "--maxShapes=input_boxes:1x200x4,input_boxes_labels:1x200,fpn_feat_2:1x256x72x72,fpn_pos_2:1x256x72x72"]
        },
        "decoder": {
            "onnx": os.path.join(onnx_dir, "decoder.onnx"),
            "flags": ["--fp16", "--minShapes=fpn_feat_0:1x256x288x288,fpn_feat_1:1x256x144x144,fpn_feat_2:1x256x72x72,fpn_pos_2:1x256x72x72,prompt_features:1x1x256,prompt_mask:1x1", "--optShapes=fpn_feat_0:1x256x288x288,fpn_feat_1:1x256x144x144,fpn_feat_2:1x256x72x72,fpn_pos_2:1x256x72x72,prompt_features:1x40x256,prompt_mask:1x40", "--maxShapes=fpn_feat_0:1x256x288x288,fpn_feat_1:1x256x144x144,fpn_feat_2:1x256x72x72,fpn_pos_2:1x256x72x72,prompt_features:1x250x256,prompt_mask:1x250"]
        }
    }

    if not os.path.exists(engine_dir):
        print(f"[INIT] Creating output directory: {engine_dir}")
        os.makedirs(engine_dir)
        
    # --- Task: Copy tokenizer.json ---
    tokenizer_src = os.path.join(onnx_dir, "tokenizer.json")
    tokenizer_dst = os.path.join(engine_dir, "tokenizer.json")

    # Only copy if source exists
    if os.path.exists(tokenizer_src):
        # Only copy if destination does NOT exist (avoid overwriting)
        if not os.path.exists(tokenizer_dst):
            print(f"[COPY] Copying tokenizer.json to {engine_dir}...")
            try:
                shutil.copy2(tokenizer_src, tokenizer_dst)
                print("[SUCCESS] tokenizer.json copied.")
            except Exception as e:
                print(f"[ERROR] Failed to copy tokenizer.json: {e}")
        else:
            print(f"[SKIP] tokenizer.json already exists in destination.")
    else:
        print(f"[WARNING] tokenizer.json not found in {onnx_dir}, skipping copy.")

    # --- Task: Build Engines ---
    for name, config in models.items():
        engine_path = os.path.join(engine_dir, f"{name}.engine")

        # CHECK: If engine already exists, skip re-exporting
        if os.path.exists(engine_path):
            print(f"[SKIP] {name}.engine already exists.")
            continue

        print(f"\n[BUILD] Exporting {name}...")
        cmd = ["trtexec", f"--onnx={config['onnx']}", f"--saveEngine={engine_path}"] + config['flags']
        
        try:
            subprocess.run(cmd, check=True)
            print(f"[SUCCESS] Saved to {engine_path}")
        except subprocess.CalledProcessError:
            print(f"[ERROR] Failed to build {name}")
            sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export ONNX models to TensorRT engines.")
    
    # Arguments
    parser.add_argument("--base", default=".", help="Base directory (default: current)")
    parser.add_argument("--onnx", help="Directory containing ONNX models (default: BASE/Onnx-Models)")
    parser.add_argument("--engine", help="Directory to save Engines (default: BASE/Engines)")

    args = parser.parse_args()

    # Resolve paths: CLI args take priority, otherwise use BASE_DIR logic
    final_onnx_dir = args.onnx if args.onnx else os.path.join(args.base, "Onnx-Models")
    final_engine_dir = args.engine if args.engine else os.path.join(args.base, "Engines")

    print(f"--- Configuration ---")
    print(f"ONNX Dir:   {final_onnx_dir}")
    print(f"Engine Dir: {final_engine_dir}")
    print(f"----------------------")

    build_engines(final_onnx_dir, final_engine_dir)