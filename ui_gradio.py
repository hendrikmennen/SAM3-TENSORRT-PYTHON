import os
import cv2
import gradio as gr
import tempfile
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from gradio.themes.utils.fonts import GoogleFont

# Import the class from your existing file
try:
    from SAM3_TensorRT_Inference import Sam3Inference
except ImportError:
    print("Warning: Could not import 'Sam3Inference'. Ensure the file exists.")
    Sam3Inference = None

# --- Configuration ---
MODEL_DIR = Path("Engines")
DEFAULT_OUTPUT = ("0.00 ms", "0.00 GB")

# --- Global Model State ---
sam3_engine: Optional[Sam3Inference] = None

def load_model():
    """Initializes the TensorRT engine into global memory."""
    global sam3_engine
    if sam3_engine is None:
        if not MODEL_DIR.exists():
            raise FileNotFoundError(f"Model directory '{MODEL_DIR}' not found.")
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

        # Process Metrics
        if metrics:
            inf_time, gpu_mem = metrics
            time_str = f"{inf_time:.2f} ms"
            mem_str = f"{gpu_mem:.3f} GB"
        else:
            time_str = "N/A"
            mem_str = "N/A"

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
        print(f"Critical error: {e}")
        return image, "Error", "Error"

    finally:
        for path in [input_path, output_path]:
            if os.path.exists(path) and os.path.isfile(path):
                try:
                    os.remove(path)
                except OSError:
                    pass

# --- UI Styling & Layout ---

# 1. Custom CSS for title, alignment, and icon spacing
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
.fa { margin-right: 8px; } /* Spacing for icons */
"""

# 2. Define a Professional Theme (Slate + Blue)
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
        # Header with Font Awesome Icons
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
                # Using HTML for section headers to include icons
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
                    info="Describe the object to detect",
                    lines=1
                )

                # Accordion for advanced settings
                with gr.Accordion("Advanced Parameters", open=False):
                    conf_slider = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.4, step=0.05, 
                        label="Confidence Threshold"
                    )
                    segment_check = gr.Checkbox(
                        label="Generate Mask", value=True
                    )

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
                        scale=1,
                        elem_id="time-box"
                    )
                    mem_output = gr.Textbox(
                        label="GPU VRAM", 
                        value="0.00 GB", 
                        interactive=False,
                        scale=1
                    )

    # --- Bind Functions ---
    run_btn.click(
        fn=inference_wrapper,
        inputs=[input_image, text_prompt, conf_slider, segment_check],
        outputs=[output_image, time_output, mem_output]
    )

# --- Launch with Font Awesome CDN ---
if __name__ == "__main__":
    try:
        load_model()
        print("Launching Professional UI...")
        
        # We inject the Font Awesome CDN into the <head> of the page
        font_awesome_cdn = """
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        """
        
        demo.launch(
            allowed_paths=[tempfile.gettempdir()],
            css=custom_css,
            theme=theme,
            head=font_awesome_cdn  # This loads the icons
        )
        
    except Exception as e:
        print(f"Failed to start application: {e}")