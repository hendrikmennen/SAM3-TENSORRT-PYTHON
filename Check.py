import torch
import sys
import subprocess
import os
import onnxruntime as ort

# New imports for CUDA Python test
try:
    from cuda import cudart
    CUDA_PYTHON_AVAILABLE = True
except ImportError:
    CUDA_PYTHON_AVAILABLE = False

def display_spec_header(title):
    print(f"\n{('=' * 60)}")
    print(f"{title.center(60)}")
    print(f"{('=' * 60)}")

def run_shell(command):
    """Helper to run a shell command and return the first line of output."""
    try:
        # We explicitly inherit the current environment to ensure paths are respected
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True, env=os.environ)
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except Exception:
        return None

def audit_system_specs():
    display_spec_header("SYSTEM SPECIFICATIONS AUDIT")

    # 1. Hardware & Driver (nvidia-smi)
    print(f"--- GPU Hardware & Driver ---")
    smi_out = run_shell("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits")
    if smi_out:
        name, driver, mem = smi_out.split(',')
        print(f"GPU Model:          {name.strip()}")
        print(f"Driver Version:     {driver.strip()}")
        print(f"Total VRAM:         {mem.strip()} MB")
    else:
        print("nvidia-smi:         ❌ Failed to query (Driver likely missing)")


    # 2. Compiler (NVCC)
    print(f"\n--- Compiler (NVCC) ---")
    # Try 1: Standard command
    nvcc_out = run_shell("nvcc --version")
    
    # Try 2: Default installation path (Common fix for Jupyter)
    if not nvcc_out:
        nvcc_out = run_shell("/usr/local/cuda/bin/nvcc --version")

    if nvcc_out:
        ver_line = [line for line in nvcc_out.split('\n') if "release" in line]
        print(f"NVCC Version:       {ver_line[0].strip() if ver_line else 'Found'}")
    else:
        print("NVCC Status:        ❌ Not found (Tried 'nvcc' and '/usr/local/cuda/bin/nvcc')")

    # 3. Python & PyTorch Stack
    print(f"\n--- Core Python Stack ---")
    print(f"Python:             {sys.version.split()[0]}")
    print(f"PyTorch:            {torch.__version__}")
    print(f"PyTorch CUDA Ver:   {torch.version.cuda}")
    print(f"ONNX Runtime:       {ort.__version__}")

    # 5. TensorRT Python Check
    print(f"\n--- TensorRT (Python) ---")
    try:
        import tensorrt as trt
        print(f"TensorRT Version:   {trt.__version__}")
        logger = trt.Logger(trt.Logger.ERROR)
        builder = trt.Builder(logger)
        print(f"TRT Builder Status: ✅ Functional")
    except ImportError:
        print(f"TensorRT Version:   ❌ Not installed in Python environment")
    except Exception as e:
        print(f"TensorRT Status:    ⚠️ Error: {e}")

    # 6. TensorRT CLI (trtexec) - NEW CHECK
    print(f"\n--- TensorRT CLI (trtexec) ---")
    # trtexec doesn't always have a --version flag, so we check if --help works
    # Try 1: Standard Path
    trtexec_path = "trtexec"
    trtexec_out = run_shell(f"{trtexec_path} --help")
    
    # Try 2: Common installation path
    if not trtexec_out:
        trtexec_path = "/usr/src/tensorrt/bin/trtexec"
        trtexec_out = run_shell(f"{trtexec_path} --help")
    
    # Try 3: User bin path (sometimes here on debian installs)
    if not trtexec_out:
        trtexec_path = "/usr/bin/trtexec"
        trtexec_out = run_shell(f"{trtexec_path} --help")

    if trtexec_out:
        print(f"trtexec Status:     ✅ Found at '{trtexec_path}'")
    else:
        print("trtexec Status:     ❌ Not found (Checked global PATH and /usr/src/tensorrt/bin/)")

    # 7. Compute Capabilities (PyTorch)
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        print(f"\nCompute Capability (PT): {major}.{minor}")
    else:
        print("\n❌ PyTorch cannot see CUDA-capable GPU.")

def verify_ort_providers():
    display_spec_header("ONNX RUNTIME PROVIDERS")
    providers = ort.get_available_providers()
    print("Available Providers:")
    for p in providers:
        status = "✅" if "Tensorrt" in p or "CUDA" in p else "  "
        print(f"{status} {p}")

if __name__ == "__main__":
    audit_system_specs()
    verify_ort_providers()
