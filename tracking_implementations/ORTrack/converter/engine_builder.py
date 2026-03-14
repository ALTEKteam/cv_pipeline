from __future__ import annotations
"""
TensorRT engine builder for ORTrack ONNX models.

Quick usage examples:
    python converter/engine_builder.py
    python converter/engine_builder.py --onnx output/onnx/ortrack.onnx --engine output/ortrack_fp16.engine --fp16
    python converter/engine_builder.py --onnx output/onnx/ortrack.onnx --engine output/ortrack_fp32.engine --no-fp16 --workspace-gb 2.0

Notes:
- Run this script from the ORTrack root directory for default relative paths.
- `--workspace-gb` controls TensorRT build-time workspace memory.
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import tensorrt as trt


LOGGER = logging.getLogger(__name__)


@dataclass
class EngineBuildOptions:
    """Configuration container for TensorRT engine build options.

    Attributes:
        onnx_path: Source ONNX model path.
        engine_path: Destination path for the serialized TensorRT engine.
        fp16: Enables FP16 precision when available.
        workspace_gb: Build workspace memory limit in GB.
    """
    onnx_path: Path
    engine_path: Path
    fp16: bool = True
    workspace_gb: float = 1.0


def _trt_severity_from_logger() -> trt.Logger.Severity:
    """Maps Python logging level to TensorRT logger severity.

    This keeps TensorRT verbosity consistent with the active project logger.
    """
    level = LOGGER.getEffectiveLevel()
    if level <= logging.DEBUG:
        return trt.Logger.VERBOSE
    if level <= logging.INFO:
        return trt.Logger.INFO
    if level <= logging.WARNING:
        return trt.Logger.WARNING
    if level <= logging.ERROR:
        return trt.Logger.ERROR
    return trt.Logger.INTERNAL_ERROR


def _workspace_bytes(workspace_gb: float) -> int:
    """Converts workspace size in GB to bytes with input validation."""
    if workspace_gb <= 0:
        raise ValueError("workspace_gb must be greater than 0")
    return int(workspace_gb * (1 << 30))


def build_engine(options: EngineBuildOptions) -> Path:
    """Builds and saves a TensorRT engine from an ONNX model.

    Args:
        options: Engine build configuration.

    Returns:
        Path to the generated engine file.

    Raises:
        FileNotFoundError: If the ONNX file is missing.
        RuntimeError: If ONNX parsing or engine build fails.
    """
    if not options.onnx_path.exists():
        raise FileNotFoundError(f"ONNX file not found: {options.onnx_path}")

    # Initialize TensorRT components (builder, network, parser)
    trt_logger = trt.Logger(_trt_severity_from_logger())
    builder = trt.Builder(trt_logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, trt_logger)

    LOGGER.info("Loading ONNX model: %s", options.onnx_path)
    with open(options.onnx_path, "rb") as file_obj:
        if not parser.parse(file_obj.read()):
            LOGGER.error("Failed to parse ONNX model.")
            for index in range(parser.num_errors):
                LOGGER.error("[TRT] %s", parser.get_error(index))
            raise RuntimeError("ONNX parsing failed")

    config = builder.create_builder_config()
    if options.fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        LOGGER.info("FP16 mode enabled.")
    else:
        LOGGER.info("FP16 mode disabled.")

    workspace_limit = _workspace_bytes(options.workspace_gb)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_limit)
    LOGGER.info("Workspace memory limit: %.2f GB", options.workspace_gb)

    # Build serialized engine blob
    LOGGER.info("Building TensorRT engine...")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("TensorRT engine build returned None")

    # Ensure output directory exists, then write engine to disk
    options.engine_path.parent.mkdir(parents=True, exist_ok=True)
    with open(options.engine_path, "wb") as file_obj:
        file_obj.write(serialized_engine)

    LOGGER.info("Engine successfully created: %s", options.engine_path)
    return options.engine_path


def parse_args() -> argparse.Namespace:
    """Parses CLI arguments for ONNX-to-TensorRT conversion.

    Returns:
        Parsed command-line arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Build a TensorRT engine from an ONNX model.")
    parser.add_argument(
        "--onnx",
        type=Path,
        default=Path("output/onnx/ortrack.onnx"),
        help="Path to the ONNX model file.",
    )
    parser.add_argument(
        "--engine",
        type=Path,
        default=Path("output/ortrack_fp16.engine"),
        help="Output path for the serialized TensorRT engine.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable FP16 precision.",
    )
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable FP16 precision.",
    )
    parser.add_argument(
        "--workspace-gb",
        type=float,
        default=1.0,
        help="Workspace memory limit in GB.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint.

    Resolves runtime options from CLI arguments and triggers engine build.
    """
    # Keep terminal output readable and consistent with other project scripts
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()

    # Default is FP16 enabled unless explicitly disabled
    fp16_enabled = True
    if args.no_fp16:
        fp16_enabled = False
    elif args.fp16:
        fp16_enabled = True

    options = EngineBuildOptions(
        onnx_path=args.onnx,
        engine_path=args.engine,
        fp16=fp16_enabled,
        workspace_gb=args.workspace_gb,
    )
    build_engine(options)


if __name__ == "__main__":
    main()
