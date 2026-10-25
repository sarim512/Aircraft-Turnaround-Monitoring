"""Ultralytics YOLOv8 + ByteTrack adapter implementing ObjectTracker."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from turnaround.detection.base import ObjectTracker
from turnaround.domain.types import BoundingBox, RawDetection

logger = logging.getLogger(__name__)


def _to_numpy(value: Any) -> np.ndarray:
    """Convert a torch tensor or array-like to a numpy array.

    Args:
        value: Object with ``.cpu().numpy()`` (torch) or anything ``np.asarray`` accepts.
    Returns:
        numpy array.
    """
    if hasattr(value, "cpu"):
        return value.cpu().numpy()
    return np.asarray(value)


def resolve_device(requested: str, half: Optional[bool]) -> Tuple[str, bool]:
    """Pick the inference device and FP16 flag with graceful CPU fallback.

    Args:
        requested: ``"auto"``, ``"cpu"``, ``"cuda:N"`` or a bare GPU index like ``"0"``.
        half: Requested FP16 setting; None means automatic.
    Returns:
        Tuple (device string for Ultralytics, use_half).
    Logic:
        1. Import torch lazily; if unavailable fall back to CPU.
        2. ``auto`` -> ``cuda:0`` if CUDA is available else ``cpu``.
        3. An explicit CUDA request without CUDA falls back to CPU with a warning.
        4. FP16 defaults to True only on CUDA; an explicit True on CPU is switched off with a warning.
    """
    try:
        import torch

        cuda_ok = bool(torch.cuda.is_available())
    except ImportError:
        cuda_ok = False
    wants_cuda = requested.startswith("cuda") or requested.isdigit()
    if requested == "auto":
        device = "cuda:0" if cuda_ok else "cpu"
    elif wants_cuda and not cuda_ok:
        logger.warning("CUDA device %r requested but unavailable; falling back to CPU", requested)
        device = "cpu"
    else:
        device = requested
    on_cuda = device != "cpu"
    if half is None:
        use_half = on_cuda
    elif half and not on_cuda:
        logger.warning("FP16 requested on CPU; disabling half precision")
        use_half = False
    else:
        use_half = bool(half)
    return device, use_half


class UltralyticsTracker(ObjectTracker):
    """YOLOv8 detector with Ultralytics' built-in ByteTrack (``model.track(persist=True)``)."""

    def __init__(
        self,
        weights_path: str,
        device: str = "auto",
        half: Optional[bool] = None,
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.5,
        tracker_config: str = "bytetrack.yaml",
        max_det: int = 100,
    ) -> None:
        """Store settings; the model is loaded by ``load``.

        Args:
            weights_path: Path to the ``.pt`` file.
            device: Requested device (see ``resolve_device``).
            half: Requested FP16 flag or None for auto.
            imgsz: Inference size.
            conf: Detector confidence floor.
            iou: NMS IoU threshold.
            tracker_config: ByteTrack YAML path (or the Ultralytics built-in name).
            max_det: Maximum detections per frame.
        """
        self._weights = weights_path
        self._requested_device = device
        self._requested_half = half
        self._imgsz = imgsz
        self._conf = conf
        self._iou = iou
        self._tracker_cfg = tracker_config
        self._max_det = max_det
        self._model: Any = None
        self._device = "cpu"
        self._half = False

    def load(self) -> None:
        """Load YOLO weights and warm up the tracking path.

        Raises:
            FileNotFoundError: Weights file does not exist.
            ImportError: ultralytics is not installed.
        Logic:
            1. Check the weights file; resolve device/FP16.
            2. Import ultralytics lazily and create ``YOLO``.
            3. Run one ``track`` call on a blank frame (initialises CUDA, predictor and tracker), then
               ``reset`` so the warm-up leaves no track state.
        Side effects:
            Reads weights from disk; may allocate GPU memory.
        """
        path = Path(self._weights)
        if not path.is_file():
            raise FileNotFoundError(f"Model weights not found: {path} (set model.weights_path in the config)")
        self._device, self._half = resolve_device(self._requested_device, self._requested_half)
        from ultralytics import YOLO  # lazy: keeps non-ML modules importable without torch

        self._model = YOLO(str(path))
        blank = np.zeros((self._imgsz, self._imgsz, 3), dtype=np.uint8)
        self._run_track(blank)
        self.reset()
        logger.info("Model loaded: %s on %s (half=%s)", path.name, self._device, self._half)

    def _run_track(self, frame: np.ndarray) -> Any:
        """Invoke ``model.track`` with the configured parameters.

        Args:
            frame: BGR image.
        Returns:
            The Ultralytics results list.
        """
        return self._model.track(
            frame,
            persist=True,
            tracker=self._tracker_cfg,
            conf=self._conf,
            iou=self._iou,
            imgsz=self._imgsz,
            device=self._device,
            half=self._half,
            max_det=self._max_det,
            verbose=False,
        )

    def track(self, frame: np.ndarray) -> List[RawDetection]:
        """Detect + track in one frame.

        Args:
            frame: BGR uint8 image.
        Returns:
            Parsed detections (empty list when nothing is found).
        Raises:
            RuntimeError: If ``load`` was not called.
        """
        if self._model is None:
            raise RuntimeError("UltralyticsTracker.load() must be called before track()")
        results = self._run_track(frame)
        if not results:
            return []
        return self._parse(results[0])

    def _parse(self, result: Any) -> List[RawDetection]:
        """Convert one Ultralytics ``Results`` object to RawDetections.

        Args:
            result: Object exposing ``boxes`` (xyxy, conf, cls, optional id) and ``names``.
        Returns:
            List of RawDetection; track_id is None for every box when the tracker returned no ids.
        Logic:
            Return [] if there are no boxes; convert tensors to numpy; resolve the label from
            ``result.names`` (fallback ``model.names``); read ids only if ``boxes.id`` is not None.
        """
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = _to_numpy(boxes.xyxy)
        conf = _to_numpy(boxes.conf)
        cls = _to_numpy(boxes.cls).astype(int)
        ids = getattr(boxes, "id", None)
        ids_np = _to_numpy(ids).astype(int) if ids is not None else None
        names = getattr(result, "names", None) or self._model.names
        detections: List[RawDetection] = []
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = (float(v) for v in xyxy[i])
            class_id = int(cls[i])
            detections.append(
                RawDetection(
                    box=BoundingBox(x1, y1, x2, y2),
                    confidence=float(conf[i]),
                    class_id=class_id,
                    label=str(names[class_id]),
                    track_id=int(ids_np[i]) if ids_np is not None else None,
                )
            )
        return detections

    def reset(self) -> None:
        """Best-effort reset of ByteTrack state (no-op if the predictor has no trackers yet).

        Logic:
            Look up ``model.predictor.trackers`` and call ``reset()`` on each tracker that has it;
            swallow exceptions because Ultralytics internals vary between versions.
        """
        if self._model is None:
            return
        try:
            trackers = getattr(getattr(self._model, "predictor", None), "trackers", None) or []
            for tracker in trackers:
                if hasattr(tracker, "reset"):
                    tracker.reset()
        except Exception:  # pragma: no cover - version dependent
            logger.warning("Tracker reset failed", exc_info=True)

    @property
    def class_names(self) -> Dict[int, str]:
        """Model vocabulary (class id -> name).

        Raises:
            RuntimeError: If the model is not loaded.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded")
        return {int(k): str(v) for k, v in self._model.names.items()}

    @property
    def device_description(self) -> str:
        """Device string such as ``cuda:0 (fp16)`` or ``cpu (fp32)``."""
        return f"{self._device} ({'fp16' if self._half else 'fp32'})"
