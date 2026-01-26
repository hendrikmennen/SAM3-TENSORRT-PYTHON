import cv2
import time
import argparse
import numpy as np
import torch
import tensorrt as trt
from pathlib import Path
from typing import Dict, Tuple
from tokenizers import Tokenizer
from PIL import Image as PILImage
from tabulate import tabulate  

# --- Constants ---
PURPLE_COLOR = (255, 71, 151) 

class TRTModule:
    """Wrapper for TensorRT execution using PyTorch for memory management."""
    def __init__(self, engine_path: str):
        self.logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self.logger, "")

        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        if self.engine is None:
            raise RuntimeError(f"Failed to load engine: {engine_path}")

        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        
        # Parse IO info
        self.io_info = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            self.io_info[name] = {
                "mode": self.engine.get_tensor_mode(name),
                "dtype": self.engine.get_tensor_dtype(name),
                "shape": self.engine.get_tensor_shape(name) 
            }

    def _trt_dtype_to_torch(self, trt_dtype):
        mapping = {trt.float32: torch.float32, trt.float16: torch.float16,
                   trt.int32: torch.int32, trt.int64: torch.int64, trt.bool: torch.bool}
        return mapping.get(trt_dtype, torch.float32)

    def get_tensor_shape(self, name: str) -> Tuple:
        if name in self.io_info:
            return tuple(self.io_info[name]["shape"])
        raise KeyError(f"Tensor '{name}' not found in engine.")

    def __call__(self, **inputs: np.ndarray) -> Dict[str, np.ndarray]:
        torch_outputs = {}
        for name, data in inputs.items():
            if name in self.io_info and self.io_info[name]["mode"] == trt.TensorIOMode.INPUT:
                dtype = self._trt_dtype_to_torch(self.io_info[name]["dtype"])
                tensor = torch.from_numpy(data).cuda().to(dtype).contiguous()
                self.context.set_input_shape(name, tuple(tensor.shape))
                self.context.set_tensor_address(name, int(tensor.data_ptr()))

        for name, info in self.io_info.items():
            if info["mode"] == trt.TensorIOMode.OUTPUT:
                shape = tuple(self.context.get_tensor_shape(name))
                dtype = self._trt_dtype_to_torch(info["dtype"])
                out_tensor = torch.empty(shape, dtype=dtype, device='cuda')
                torch_outputs[name] = out_tensor
                self.context.set_tensor_address(name, int(out_tensor.data_ptr()))

        self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        self.stream.synchronize()
        return {k: v.cpu().numpy() for k, v in torch_outputs.items()}

class Sam3Inference:
    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        props = torch.cuda.get_device_properties(0)
        suffix = f"_sm{props.major}{props.minor}_trt{trt.__version__}"
        
        print(f"[INFO] Loading engines from {self.model_dir}...")
        self.vision_encoder = TRTModule(str(self._find("vision-encoder", suffix)))
        self.text_encoder = TRTModule(str(self._find("text-encoder", suffix)))
        self.decoder = TRTModule(str(self._find("decoder", suffix)))

        # --- AUTO-DETECT RESOLUTION ---
        try:
            input_shape = self.vision_encoder.get_tensor_shape("images")
            if len(input_shape) == 4:
                self.target_h = input_shape[2]
                self.target_w = input_shape[3]
            else:
                print(f"[WARN] Unexpected input shape {input_shape}. Defaulting to 1024.")
                self.target_h = self.target_w = 1024
            print(f"[INFO] Auto-detected input resolution: {self.target_w}x{self.target_h}")
        except KeyError:
            print("[WARN] Could not find tensor 'images' to determine resolution. Defaulting to 1024.")
            self.target_h = self.target_w = 1024
        # ------------------------------

        tokenizer_path = self.model_dir / "tokenizer.json"
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"tokenizer.json missing in {model_dir}")
            
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer.enable_padding(length=32, pad_id=49407)
        self.tokenizer.enable_truncation(max_length=32)

    def _find(self, name, suffix):
        p_spec = self.model_dir / f"{name}{suffix}.engine"
        p_gen = self.model_dir / f"{name}.engine"
        target = p_spec if p_spec.exists() else p_gen
        if not target.exists():
            raise FileNotFoundError(f"Could not find engine for {name} in {self.model_dir}")
        return target

    def run(self, img_path, prompt, conf, out_path, segment=False):
        # 1. Preprocess
        orig = cv2.imread(img_path)
        if orig is None:
            raise ValueError(f"Could not read image: {img_path}")
        h, w = orig.shape[:2]
        rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        
        pil_img = PILImage.fromarray(rgb).resize((self.target_w, self.target_h), PILImage.BILINEAR)
        pixel_values = (np.array(pil_img).astype(np.float32) / 127.5 - 1.0).transpose(2,0,1)[None]

        # Warmup / Sync
        torch.cuda.synchronize()
        
        # --- START TIMING ---
        t0 = time.time()

        # 2. Forward Pass
        v_feats = self.vision_encoder(images=pixel_values)
        tokens = self.tokenizer.encode(prompt)
        t_feats = self.text_encoder(
            input_ids=np.array([tokens.ids], dtype=np.int64), 
            attention_mask=np.array([tokens.attention_mask], dtype=np.int64)
        )
        
        out = self.decoder(
            fpn_feat_0=v_feats["fpn_feat_0"], fpn_feat_1=v_feats["fpn_feat_1"],
            fpn_feat_2=v_feats["fpn_feat_2"], fpn_pos_2=v_feats["fpn_pos_2"],
            prompt_features=t_feats["text_features"], prompt_mask=t_feats["text_mask"]
        )

        # 3. Post-process
        scores = (1 / (1 + np.exp(-out["pred_logits"][0]))) * (1 / (1 + np.exp(-out["presence_logits"][0, 0])))
        keep = scores > conf
        boxes = out["pred_boxes"][0][keep]

        masks = None
        if segment:
            if "pred_masks" not in out:
                print("[WARNING] Engine missing 'pred_masks'. Switching to BBox mode.")
                segment = False
            else:
                masks = out["pred_masks"][0][keep]

        # --- END TIMING ---
        torch.cuda.synchronize()
        t1 = time.time()
        
        inference_time = (t1 - t0) * 1000

        # 4. Visualization
        print(f"[INFO] Found: {len(boxes)} objects")
        color = PURPLE_COLOR

        if segment:
            overlay = orig.copy()
            for i, box in enumerate(boxes):
                mask_prob = 1 / (1 + np.exp(-masks[i]))
                if mask_prob.ndim == 3: mask_prob = mask_prob[0]
                mask_bin = (mask_prob > 0.5).astype(np.uint8)
                mask_resized = cv2.resize(mask_bin, (w, h), interpolation=cv2.INTER_NEAREST)
                overlay[mask_resized == 1] = color

            cv2.addWeighted(overlay, 0.4, orig, 0.6, 0, orig)

            for box in boxes:
                x1, y1, x2, y2 = (box * [w, h, w, h]).astype(int)
                self._draw_label(orig, prompt, x1, y1, color)
        else:
            for box in boxes:
                x1, y1, x2, y2 = (box * [w, h, w, h]).astype(int)
                cv2.rectangle(orig, (x1, y1), (x2, y2), color, 2)
                self._draw_label(orig, prompt, x1, y1, color)

        cv2.imwrite(out_path, orig)
  
        table_data = [
            ["Input Res", f"{self.target_w}x{self.target_h}"],
            ["Inference Time", f"{inference_time:.2f} ms"],
            ["Objects Found", len(boxes)]
        ]
        
        print(tabulate(table_data, headers=["Metric", "Value"], tablefmt="fancy_grid"))
        print(f"[SUCCESS] Saved to {out_path}")

        return inference_time

    def _draw_label(self, img, text, x, y, color):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        lbl_x = x
        lbl_y = y - 10 if y - 10 > text_h else y + text_h + 10
        cv2.rectangle(img, (lbl_x, lbl_y - text_h - 4), (lbl_x + text_w + 10, lbl_y + baseline - 2), color, -1)
        cv2.putText(img, text, (lbl_x + 5, lbl_y - 2), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM3 Standalone Inference CLI")
    parser.add_argument("--input", required=True, help="Path to input image file")
    parser.add_argument("--prompt", default="object", help="Text prompt for detection")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--output", default="output.jpg", help="Path to save the output image")
    parser.add_argument("--models", default="Engines", help="Directory containing .engine files")
    parser.add_argument("--segment", action="store_true", help="Use segmentation masks.")
    
    args = parser.parse_args()

    try:
        engine = Sam3Inference(args.models)
        engine.run(args.input, args.prompt, args.conf, args.output, segment=args.segment)
    except Exception as e:
        print(f"[ERROR] {e}")