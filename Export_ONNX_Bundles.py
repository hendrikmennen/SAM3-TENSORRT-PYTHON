#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def run_export(script_path: Path, model_path: Path, output_dir: Path, device: str, size: int, mode: str) -> None:
    cmd = [
        sys.executable,
        str(script_path),
        "--all",
        "--model-path",
        str(model_path),
        "--output-dir",
        str(output_dir),
        "--device",
        device,
        "--size",
        str(size),
    ]

    if mode == "fp16":
        cmd.append("--fp16")
    elif mode == "int8":
        cmd.append("--int8")
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    print(f"[EXPORT] {size} {mode.upper()} -> {output_dir}")
    subprocess.run(cmd, check=True)


def zip_directory_contents(source_dir: Path, zip_path: Path) -> None:
    print(f"[ZIP] {zip_path}")
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as zf:
        for file_path in sorted(source_dir.rglob("*")):
            if file_path.is_file():
                arcname = file_path.relative_to(source_dir)
                zf.write(file_path, arcname)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export SAM3 ONNX variants (644/1008 x FP16/INT8) and zip each output folder."
    )
    parser.add_argument("--model-path", required=True, help="Path to SAM3 model directory.")
    parser.add_argument("--device", default="cuda", help="Export device (default: cuda).")
    parser.add_argument(
        "--output-root",
        default="export/onnx-bundles",
        help="Root folder for all exports (default: export/onnx-bundles).",
    )
    parser.add_argument(
        "--export-script",
        default="SAM3_PyTorch_To_Onnx.py",
        help="Path to the export script (default: SAM3_PyTorch_To_Onnx.py).",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    script_path = Path(args.export_script)
    if not script_path.exists():
        raise FileNotFoundError(f"Export script not found: {script_path}")

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    variants = [
        (644, "fp16"),
        (644, "int8"),
        (1008, "fp16"),
        (1008, "int8"),
    ]

    for size, mode in variants:
        folder_name = f"onnx_{size}_{mode}"
        out_dir = output_root / folder_name
        out_dir.mkdir(parents=True, exist_ok=True)

        run_export(
            script_path=script_path,
            model_path=model_path,
            output_dir=out_dir,
            device=args.device,
            size=size,
            mode=mode,
        )

        zip_path = output_root / f"{folder_name}.zip"
        zip_directory_contents(out_dir, zip_path)

    print("\n[SUCCESS] All exports and zip bundles created:")
    for size, mode in variants:
        folder_name = f"onnx_{size}_{mode}"
        print(f"  - {output_root / folder_name}")
        print(f"  - {output_root / f'{folder_name}.zip'}")


if __name__ == "__main__":
    main()
