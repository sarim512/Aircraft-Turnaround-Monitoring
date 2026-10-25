"""Draws ROI polygons, aircraft, accepted/rejected equipment and service states onto a frame."""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

from turnaround.domain.types import ALL_SERVICES, ServiceType
from turnaround.spatial.roi import RoiSet
from turnaround.temporal.engine import FrameResult

Color = Tuple[int, int, int]

SERVICE_COLORS: Dict[ServiceType, Color] = {
    ServiceType.AEROBRIDGE: (230, 160, 40),
    ServiceType.CATERING: (60, 190, 90),
    ServiceType.CARGO: (40, 140, 240),
    ServiceType.PUSHBACK: (70, 70, 230),
}
AIRCRAFT_COLOR: Color = (235, 235, 235)
REJECTED_COLOR: Color = (130, 130, 130)
ROI_COLOR: Color = (0, 215, 255)


class FrameAnnotator:
    """Stateless renderer producing an annotated copy of a frame."""

    def __init__(self, draw_rejected: bool = False, draw_roi: bool = True) -> None:
        """Configure what gets drawn.

        Args:
            draw_rejected: Also draw rejected equipment boxes (gray, with the rejection reason).
            draw_roi: Draw the ROI polygons.
        """
        self._draw_rejected = draw_rejected
        self._draw_roi = draw_roi

    def render(
        self,
        frame: np.ndarray,
        result: Optional[FrameResult] = None,
        roi: Optional[RoiSet] = None,
        banner: Optional[str] = None,
        hud_lines: Sequence[str] = (),
    ) -> np.ndarray:
        """Produce the annotated frame.

        Args:
            frame: Source BGR frame (not modified).
            result: Engine result for the frame, or None (e.g. analysis paused).
            roi: ROI set whose bound polygons are drawn, or None.
            banner: Optional red warning bar text at the top (e.g. scene change).
            hud_lines: Extra text lines drawn bottom-left (fps, latency, ...).
        Returns:
            A new annotated image.
        Logic:
            Copy the frame; draw ROI polygons; aircraft boxes labelled ``A<id>`` (``coasting`` when not
            observed); accepted equipment boxes in the service color labelled ``service A<id> conf``;
            optionally rejected boxes in gray; a per-aircraft list of active services at the top-left;
            HUD lines; finally the banner.
        """
        canvas = frame.copy()
        if roi is not None and self._draw_roi:
            for key, poly in roi.pixel_polygons().items():
                pts = np.array(poly.points, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(canvas, [pts], True, ROI_COLOR, 2 if key == "analysis" else 1)
        if result is not None:
            for ac in result.aircraft:
                label = f"A{ac.aircraft_id}" + ("" if ac.observed else " coasting")
                self._box(canvas, ac.box.as_int_tuple(), AIRCRAFT_COLOR, label, thickness=2)
            for acc in result.association.accepted:
                text = f"{acc.service.value} A{acc.aircraft_id} {acc.detection.confidence:.2f}"
                self._box(canvas, acc.detection.box.as_int_tuple(), SERVICE_COLORS[acc.service], text, thickness=2)
            if self._draw_rejected:
                for rej in result.association.rejected:
                    self._box(canvas, rej.detection.box.as_int_tuple(), REJECTED_COLOR, rej.reason.value, thickness=1)
            y = 22
            for aircraft_id, services in sorted(result.active_services.items()):
                names = ", ".join(s.value for s in ALL_SERVICES if s in services) or "-"
                self._text(canvas, f"A{aircraft_id}: {names}", (10, y), AIRCRAFT_COLOR)
                y += 24
        h = canvas.shape[0]
        for i, line in enumerate(reversed(list(hud_lines))):
            self._text(canvas, line, (10, h - 12 - 22 * i), (200, 200, 200))
        if banner:
            cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 34), (30, 30, 200), -1)
            cv2.putText(canvas, banner, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        return canvas

    @staticmethod
    def _text(canvas: np.ndarray, text: str, org: Tuple[int, int], color: Color) -> None:
        """Draw text on a dark backing rectangle for legibility.

        Args:
            canvas: Image drawn on in place.
            text: Text to draw.
            org: Bottom-left text origin (x, y).
            color: Text BGR color.
        """
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        x, y = org
        x = max(3, min(x, canvas.shape[1] - tw - 6))  # keep the label fully inside the image
        cv2.rectangle(canvas, (x - 3, y - th - 4), (x + tw + 3, y + base), (20, 20, 20), -1)
        cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    def _box(self, canvas: np.ndarray, box: Tuple[int, int, int, int], color: Color, label: str, thickness: int) -> None:
        """Draw a labelled rectangle.

        Args:
            canvas: Image drawn on in place.
            box: (x1, y1, x2, y2) integer corners.
            color: BGR color.
            label: Text drawn above the box (kept inside the image).
            thickness: Line thickness.
        """
        x1, y1, x2, y2 = box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        self._text(canvas, label, (x1, max(16, y1 - 6)), color)
