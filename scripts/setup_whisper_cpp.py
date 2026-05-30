#!/usr/bin/env python3
#  Copyright (C) 2025. Hao Zheng
#  All rights reserved.

"""Prepare the vendored whisper.cpp CLI and default model files."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VENDOR_DIR = REPO_ROOT / "vendor" / "whisper.cpp"
DEFAULT_MODEL = "base"
DEFAULT_VAD_MODEL = "silero-v6.2.0"


def default_model_dir() -> Path:
    return Path.home() / "Library" / "Application Support" / "OpenLRC" / "models"


def run(cmd: list[str], cwd: Path = REPO_ROOT) -> None:
    print(f"+ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def ensure_submodule() -> None:
    if (VENDOR_DIR / "CMakeLists.txt").exists():
        return
    run(["git", "submodule", "update", "--init", "--recursive", "vendor/whisper.cpp"])
    if not (VENDOR_DIR / "CMakeLists.txt").exists():
        raise RuntimeError("vendor/whisper.cpp is missing after submodule initialization.")


def build_whisper_cpp() -> Path:
    build_dir = VENDOR_DIR / "build"
    run(["cmake", "-S", ".", "-B", str(build_dir), "-DCMAKE_BUILD_TYPE=Release"], cwd=VENDOR_DIR)
    run(["cmake", "--build", str(build_dir), "--config", "Release", "--parallel", str(os.cpu_count() or 1)])

    cli_path = build_dir / "bin" / "whisper-cli"
    if not cli_path.exists():
        raise RuntimeError(f"Build completed, but whisper-cli was not found at {cli_path}")
    return cli_path


def download_models(model_dir: Path, model: str, vad_model: str) -> tuple[Path, Path]:
    model_dir.mkdir(parents=True, exist_ok=True)
    run(["sh", "models/download-ggml-model.sh", model, str(model_dir)], cwd=VENDOR_DIR)
    run(["sh", "models/download-vad-model.sh", vad_model, str(model_dir)], cwd=VENDOR_DIR)

    whisper_model = model_dir / f"ggml-{model}.bin"
    vad_model_path = model_dir / f"ggml-{vad_model}.bin"
    missing = [str(path) for path in (whisper_model, vad_model_path) if not path.exists()]
    if missing:
        raise RuntimeError(f"Model download finished, but expected files are missing: {', '.join(missing)}")
    return whisper_model, vad_model_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build vendored whisper.cpp and download default OpenLRC models.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"whisper.cpp model name, default: {DEFAULT_MODEL}")
    parser.add_argument(
        "--vad-model",
        default=DEFAULT_VAD_MODEL,
        help=f"whisper.cpp VAD model name, default: {DEFAULT_VAD_MODEL}",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=default_model_dir(),
        help="Directory for downloaded model files.",
    )
    parser.add_argument("--skip-build", action="store_true", help="Initialize the submodule but skip CMake build.")
    parser.add_argument("--skip-models", action="store_true", help="Skip model downloads.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_submodule()

    cli_path: Path | None = None
    if not args.skip_build:
        cli_path = build_whisper_cpp()

    whisper_model: Path | None = None
    vad_model: Path | None = None
    if not args.skip_models:
        whisper_model, vad_model = download_models(args.model_dir.expanduser(), args.model, args.vad_model)

    print("\nwhisper.cpp setup complete.")
    if cli_path:
        print(f"CLI: {cli_path}")
    if whisper_model:
        print(f"Whisper model: {whisper_model}")
    if vad_model:
        print(f"VAD model: {vad_model}")
    print("\nRuntime overrides:")
    print("  OPENLRC_WHISPER_CLI=/path/to/whisper-cli")
    print("  OPENLRC_WHISPER_MODEL=/path/to/ggml-base.bin")
    print("  OPENLRC_WHISPER_VAD_MODEL=/path/to/ggml-silero-v6.2.0.bin")
    print("  OPENLRC_WHISPER_MODEL_DIR=/path/to/models")


if __name__ == "__main__":
    main()
