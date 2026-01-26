import os
import cv2
import gradio as gr
import tempfile
import numpy as np
import subprocess
import atexit  # Added for exit cleanup
from pathlib import Path
from typing import Tuple, Optional, Union
from gradio.themes.utils.fonts import GoogleFont

# Import the class from your existing file
try:
    from SAM3_TensorRT_Inference import Sam3Inference
except ImportError:
    print("Warning: Could not import 'Sam3Inference'. Ensure the file exists.")
    Sam3Inference = None

# --- Configuration ---
MODEL_DIR = Path("Engines")
DEFAULT_OUTPUT = ("0.00 ms", "0.00 GB (Total)")

# --- Global State ---
sam3_engine: Optional[Sam3Inference] = None
temp_files_registry = set()  # Tracks files for cleanup on exit

# --- Cleanup Handler ---
def cleanup_on_exit():
    """Called automatically when the script terminates to remove leftover files."""
    if temp_files_registry:
        print(f"Cleaning up {len(temp_files_registry)} temporary files...")
        for path in list(temp_files_registry):
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass
        temp_files_registry.clear()

# Register the cleanup function
atexit.register(cleanup_on_exit)

def get_total_vram() -> float:
    """Queries nvidia-smi to get the total memory used by the GPU (in GB)."""
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
            encoding="utf-8"
        )
        mem_used_mib = float(result.strip().split('\n')[0])
        return mem_used_mib / 1024.0
    except Exception as e:
        print(f"Warning: Could not query VRAM: {e}")
        return 0.0

def load_model():
    """Initializes the TensorRT engine into global memory."""
    global sam3_engine
    if sam3_engine is None:
        if not MODEL_DIR.exists():
            print(f"Error: Model directory '{MODEL_DIR}' not found.")
            return
        
        print(f"Loading SAM3 TensorRT Engine from {MODEL_DIR}...")
        try:
            sam3_engine = Sam3Inference(str(MODEL_DIR))
            print("Engine loaded successfully!")
        except Exception as e:
            print(f"Error loading engine: {e}")

def inference_wrapper(
    image: np.ndarray, 
    prompt: str, 
    conf: float, 
    segment_mode: bool
) -> Tuple[Optional[np.ndarray], str, str]:
    """Wrapper to handle image I/O, run inference, and cleanup temp files."""
    if image is None:
        return (None, *DEFAULT_OUTPUT)
    
    if sam3_engine is None:
        return (image, "Model Not Loaded", "N/A")

    # Setup Temporary Files
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
        input_path = tmp_file.name
    output_path = input_path.replace(".jpg", "_out.jpg")

    # Register files immediately so they are tracked if the script crashes here
    temp_files_registry.add(input_path)
    temp_files_registry.add(output_path)

    try:
        # Save Input
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(input_path, img_bgr)

        # Run Inference
        metrics = sam3_engine.run(
            img_path=input_path,
            prompt=prompt,
            conf=conf,
            out_path=output_path,
            segment=segment_mode
        )

        # Process Metrics (Handle both float and tuple returns)
        inf_time = 0.0
        
        if isinstance(metrics, (float, int)):
            inf_time = float(metrics)
        elif isinstance(metrics, tuple) and len(metrics) >= 1:
            inf_time = float(metrics[0])
        
        # Fetch Global VRAM independently
        total_vram = get_total_vram()
        
        time_str = f"{inf_time:.2f} ms"
        mem_str = f"{total_vram:.2f} GB (Total)"

        # Load Result
        if os.path.exists(output_path):
            result_bgr = cv2.imread(output_path)
            if result_bgr is not None:
                result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
            else:
                result_rgb = image
        else:
            result_rgb = image 

        return result_rgb, time_str, mem_str

    except Exception as e:
        print(f"Critical error during inference: {e}")
        return image, "Error", "Error"

    finally:
        # cleanup this specific run's files immediately
        for path in [input_path, output_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
            # Remove from global registry since we just deleted them
            if path in temp_files_registry:
                temp_files_registry.discard(path)

# --- UI Styling & Layout ---

custom_css = """
.container { max-width: 1200px; margin: auto; }
.header-text { 
    text-align: center; 
    font-weight: 800; 
    font-size: 2.5rem; 
    background: -webkit-linear-gradient(45deg, #2563eb, #7c3aed); 
    -webkit-background-clip: text; 
    -webkit-text-fill-color: transparent; 
    margin-bottom: 5px;
}
.subtitle-text {
    text-align: center;
    font-size: 1.1rem;
    color: #4b5563;
    margin-bottom: 30px;
    font-weight: 400;
}
.fa { margin-right: 8px; }
"""

theme = gr.themes.Soft(
    primary_hue="blue",
    neutral_hue="slate",
    text_size="lg",
    font=[GoogleFont("Inter"), "ui-sans-serif", "system-ui"]
).set(
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_700",
    block_shadow="*shadow_drop_lg",
    block_title_text_weight="600"
)

# 3. Build the UI
with gr.Blocks(title="SAM3 TensorRT") as demo:
    
    with gr.Column(elem_classes=["container"]):
        gr.HTML(
            """
            <div style="text-align:center;">
                <h1 class='header-text'>
                    <i class="fa-solid fa-layer-group"></i> SAM3 TensorRT
                </h1>
                <p class='subtitle-text'>
                    High-performance segmentation powered by <i class="fa-brands fa-nvidia"></i> NVIDIA TensorRT
                </p>
            </div>
            """
        )

        with gr.Row(equal_height=False):
            
            # --- Left Column: Control Panel ---
            with gr.Column(scale=4, variant="panel"):
                gr.HTML("<h3><i class='fa-solid fa-sliders'></i> Configuration</h3>")
                
                input_image = gr.Image(
                    label="Input Source", 
                    type="numpy", 
                    height=320,
                    sources=['upload', 'clipboard'],
                    interactive=True
                )
                
                text_prompt = gr.Textbox(
                    label="Text Prompt", 
                    placeholder="e.g. cat, car, person", 
                    lines=1
                )

                with gr.Accordion("Advanced Parameters", open=False):
                    conf_slider = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.4, step=0.05, 
                        label="Confidence Threshold"
                    )
                    segment_check = gr.Checkbox(label="Generate Mask", value=True)

                run_btn = gr.Button("Run Inference", variant="primary", size="lg")

            # --- Right Column: Results Panel ---
            with gr.Column(scale=5, variant="panel"):
                gr.HTML("<h3><i class='fa-solid fa-eye'></i> Results</h3>")
                
                output_image = gr.Image(
                    label="Segmented Output", 
                    height=380,
                    interactive=False
                )
                
                # Metrics
                with gr.Row():
                    time_output = gr.Textbox(
                        label="Processing Time", 
                        value="0.00 ms", 
                        interactive=False,
                        scale=1
                    )
                    mem_output = gr.Textbox(
                        label="Total GPU Usage", 
                        value="0.00 GB (Total)", 
                        interactive=False,
                        scale=1
                    )

    # --- Bind Functions ---
    run_btn.click(
        fn=inference_wrapper,
        inputs=[input_image, text_prompt, conf_slider, segment_check],
        outputs=[output_image, time_output, mem_output]
    )

if __name__ == "__main__":
    try:
        load_model()
        print("Launching Professional UI...")
        
        font_awesome_cdn = """
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        """
        
        # 1. wrap launch in a try block
        demo.launch(
            allowed_paths=[tempfile.gettempdir()],
            css=custom_css,
            theme=theme,
            head=font_awesome_cdn
        )
        
    except KeyboardInterrupt:
        print("\n[INFO] User stopped the application.")
    except Exception as e:
        print(f"\n[ERROR] Failed to start application: {e}")
    finally:
        # 2. This block ALWAYS runs, even if you crash or Ctrl+C
        cleanup_on_exit()
        print("[INFO] Cleanup complete. Exiting.")