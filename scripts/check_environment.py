"""Environment doctor: verifies that this PC can run the pipeline (useful after moving machines)."""
import _bootstrap  # noqa: F401  (adds src/ to sys.path)

import argparse
import importlib
import platform
import shutil
import sys
from typing import Callable, List, Optional, Tuple

from turnaround.config.loader import ConfigError, load_config

Result = Tuple[str, bool, str]


def check_import(module: str, critical: bool = True) -> Result:
    """Try importing a module and report its version.

    Args:
        module: Import name (e.g. ``cv2``).
        critical: Whether a failure should fail the whole check.
    Returns:
        (label, ok, detail) where ok is True for success and for non-critical failures.
    """
    try:
        mod = importlib.import_module(module)
        return (f"import {module}", True, str(getattr(mod, "__version__", "ok")))
    except Exception as exc:
        return (f"import {module}", not critical, f"MISSING ({exc})")


def check_cuda() -> Result:
    """Report CUDA availability through torch.

    Returns:
        (label, ok, detail); ok is always True because the pipeline falls back to CPU.
    """
    try:
        import torch

        if torch.cuda.is_available():
            return ("CUDA", True, f"{torch.cuda.get_device_name(0)} (torch {torch.__version__}, CUDA {torch.version.cuda})")
        return ("CUDA", True, "not available - pipeline will run on CPU (slow)")
    except ImportError:
        return ("CUDA", True, "torch not installed")


def check_opencv_ffmpeg() -> Result:
    """Check that OpenCV was built with FFmpeg (needed for HLS/YouTube streams).

    Returns:
        (label, ok, detail).
    """
    import cv2

    info = cv2.getBuildInformation()
    ok = any("FFMPEG:" in line and "YES" in line for line in info.splitlines())
    return ("OpenCV FFmpeg backend", ok, "yes" if ok else "NO - live streams will not open")


def check_config(path: Optional[str], load_model: bool, probe_source: bool) -> List[Result]:
    """Validate the config, weights, optional model/class mapping and optional source probe.

    Args:
        path: Config path or None.
        load_model: Also load the YOLO model and validate the class mapping.
        probe_source: Also try to open the configured source and read one frame.
    Returns:
        List of check results.
    Logic:
        Load the config (a ConfigError yields one failed result). Report the config path and whether the weights
        file exists. With ``load_model`` and existing weights: build UltralyticsTracker, load it and validate the
        class mapping, reporting device and class names or the exception. With ``probe_source``: build the source,
        start it, poll ``read`` up to 20 times for a frame, stop it and report success or failure.
    """
    results: List[Result] = []
    try:
        cfg = load_config(path)
    except ConfigError as exc:
        return [("config", False, str(exc))]
    results.append(("config", True, cfg.config_path or "default"))
    from pathlib import Path

    weights_ok = Path(cfg.model.weights_path).is_file()
    results.append(("weights file", weights_ok, cfg.model.weights_path))
    if load_model and weights_ok:
        try:
            from turnaround.detection.class_mapper import ClassMapper
            from turnaround.detection.ultralytics_tracker import UltralyticsTracker

            tracker = UltralyticsTracker(cfg.model.weights_path, cfg.model.device, cfg.model.half, cfg.model.imgsz,
                                         cfg.model.conf, cfg.model.iou, cfg.model.tracker_config, cfg.model.max_det)
            tracker.load()
            ClassMapper(cfg.classes).validate(tracker.class_names)
            results.append(("model + class map", True, f"{tracker.device_description}; classes={tracker.class_names}"))
        except Exception as exc:
            results.append(("model + class map", False, str(exc)))
    if probe_source:
        try:
            from turnaround.ingestion.factory import build_source

            src = build_source(cfg.source)
            src.start()
            frame = None
            for _ in range(20):
                packet = src.read(1.0)
                if packet is not None:
                    frame = packet.frame
                    break
            src.stop()
            results.append(("source", frame is not None, f"{cfg.source.kind}: {cfg.source.uri}" + ("" if frame is not None else " (no frame in 20 s)")))
        except Exception as exc:
            results.append(("source", False, str(exc)))
    return results


def main(argv: Optional[List[str]] = None) -> int:
    """Run all checks and print a table.

    Args:
        argv: Argument list (None uses ``sys.argv[1:]``).
    Returns:
        0 when every critical check passed, 1 otherwise.
    Logic:
        Parse flags; collect results: python version, imports (critical: numpy, cv2, yaml, ultralytics, torch;
        optional: yt_dlp, streamlit, pandas, plotly), ffmpeg binary (informational), OpenCV FFmpeg backend, CUDA and
        the config checks. Print an aligned OK/FAIL table and return 1 if any check failed.
    """
    p = argparse.ArgumentParser(description="Check the environment")
    p.add_argument("--config", default=None)
    p.add_argument("--load-model", action="store_true", help="Load the YOLO model and validate class names")
    p.add_argument("--probe-source", action="store_true", help="Open the configured source and read one frame")
    args = p.parse_args(argv)
    results: List[Result] = [("python", sys.version_info >= (3, 9), f"{platform.python_version()} on {platform.system()}")]
    for module, critical in [("numpy", True), ("cv2", True), ("yaml", True), ("ultralytics", True), ("torch", True),
                             ("yt_dlp", False), ("streamlit", False), ("pandas", False), ("plotly", False)]:
        results.append(check_import(module, critical))
    results.append(("ffmpeg binary", True, shutil.which("ffmpeg") or "not on PATH (optional; OpenCV bundles its own)"))
    try:
        results.append(check_opencv_ffmpeg())
    except Exception as exc:
        results.append(("OpenCV FFmpeg backend", False, str(exc)))
    results.append(check_cuda())
    results.extend(check_config(args.config, args.load_model, args.probe_source))
    width = max(len(r[0]) for r in results)
    failed = 0
    for label, ok, detail in results:
        failed += 0 if ok else 1
        print(f"[{'OK' if ok else 'FAIL'}] {label.ljust(width)}  {detail}")
    print("\nAll critical checks passed." if not failed else f"\n{failed} check(s) failed.")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
