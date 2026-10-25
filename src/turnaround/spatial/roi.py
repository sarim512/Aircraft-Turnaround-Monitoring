"""Manually defined region-of-interest polygons and membership tests."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from turnaround.config.schema import RoiConfig
from turnaround.domain.types import BoundingBox, ServiceType

Point = Tuple[float, float]
ANALYSIS_KEY = "analysis"


@dataclass(frozen=True)
class Polygon:
    """Simple polygon in pixel space.

    Attributes:
        points: Ordered vertices (x, y); the polygon is implicitly closed.
    """

    points: Tuple[Point, ...]

    def contains(self, x: float, y: float) -> bool:
        """Point-in-polygon test (even-odd ray casting).

        Args:
            x: Point x coordinate.
            y: Point y coordinate.
        Returns:
            True if the point is inside (points exactly on an edge may fall either way).
        Logic:
            Cast a horizontal ray to +x; toggle ``inside`` for every polygon edge whose y-span straddles
            ``y`` and whose crossing x is to the right of the point.
        """
        inside = False
        pts = self.points
        j = len(pts) - 1
        for i in range(len(pts)):
            xi, yi = pts[i]
            xj, yj = pts[j]
            if (yi > y) != (yj > y):
                x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
                if x < x_cross:
                    inside = not inside
            j = i
        return inside

    def scaled(self, sx: float, sy: float) -> "Polygon":
        """Return a copy with x scaled by ``sx`` and y by ``sy``.

        Args:
            sx: Horizontal factor.
            sy: Vertical factor.
        Returns:
            New Polygon.
        """
        return Polygon(tuple((x * sx, y * sy) for x, y in self.points))


def _to_polygon(points: List[List[float]]) -> Polygon:
    """Build a Polygon from YAML-style [[x, y], ...].

    Args:
        points: List of two-element lists.
    Returns:
        Polygon with float tuples.
    """
    return Polygon(tuple((float(x), float(y)) for x, y in points))


class RoiSet:
    """All configured ROI polygons, resolved to the pixel size of the current frame.

    ``bind`` must be called with the frame size before membership queries when ROI is enabled.
    The engine does this on every frame; it is cheap because results are cached per size.
    """

    def __init__(self, cfg: RoiConfig) -> None:
        """Parse the configured polygons (kept in their config coordinate system).

        Args:
            cfg: ROI configuration (already validated).
        """
        self._cfg = cfg
        self._raw: Dict[str, Polygon] = {}
        if cfg.enabled:
            if cfg.analysis_zone:
                self._raw[ANALYSIS_KEY] = _to_polygon(cfg.analysis_zone)
            for service, pts in cfg.service_zones.items():
                self._raw[service] = _to_polygon(pts)
        self._bound_size: Optional[Tuple[int, int]] = None
        self._pixel: Dict[str, Polygon] = {}

    @property
    def enabled(self) -> bool:
        """True if ROI filtering is switched on in the configuration."""
        return self._cfg.enabled

    def bind(self, frame_width: int, frame_height: int) -> None:
        """Convert polygons to pixel coordinates of the given frame size.

        Args:
            frame_width: Frame width in px.
            frame_height: Frame height in px.
        Logic:
            No-op if already bound to this size. Normalized mode multiplies by (W, H); pixel mode scales
            by (W / ref_w, H / ref_h) using ``reference_resolution``.
        """
        if self._bound_size == (frame_width, frame_height):
            return
        if self._cfg.coordinate_mode == "normalized":
            sx, sy = float(frame_width), float(frame_height)
        else:
            ref_w, ref_h = self._cfg.reference_resolution  # type: ignore[misc]
            sx, sy = frame_width / float(ref_w), frame_height / float(ref_h)
        self._pixel = {key: poly.scaled(sx, sy) for key, poly in self._raw.items()}
        self._bound_size = (frame_width, frame_height)

    def pixel_polygons(self) -> Dict[str, Polygon]:
        """Polygons in the bound frame's pixel space, keyed by ``"analysis"`` or service name (empty if unbound)."""
        return dict(self._pixel)

    def _inside(self, key: str, box: BoundingBox) -> bool:
        """Test the box anchor against one named polygon.

        Args:
            key: ``"analysis"`` or a service name.
            box: Detection box.
        Returns:
            True when ROI is disabled, the polygon is not defined, or the anchor lies inside it.
        Raises:
            RuntimeError: If ROI is enabled and a polygon exists but ``bind`` was never called.
        """
        if not self._cfg.enabled or key not in self._raw:
            return True
        if self._bound_size is None:
            raise RuntimeError("RoiSet.bind() must be called before membership queries")
        x, y = box.anchor(self._cfg.anchor)
        return self._pixel[key].contains(x, y)

    def allows_aircraft(self, box: BoundingBox) -> bool:
        """Whether an aircraft box passes the analysis zone.

        Args:
            box: Aircraft detection box.
        Returns:
            True if ``apply_to_aircraft`` is False or the box anchor is inside the analysis zone.
        """
        if not self._cfg.apply_to_aircraft:
            return True
        return self._inside(ANALYSIS_KEY, box)

    def allows_equipment(self, box: BoundingBox) -> bool:
        """Whether an equipment box lies in the analysis zone.

        Args:
            box: Equipment detection box.
        Returns:
            True if the anchor is inside the analysis zone (or no zone is configured).
        """
        return self._inside(ANALYSIS_KEY, box)

    def allows_service(self, service: ServiceType, box: BoundingBox) -> bool:
        """Whether an equipment box lies in the zone dedicated to its service.

        Args:
            service: Equipment service type.
            box: Equipment detection box.
        Returns:
            True if the service has no dedicated zone or the anchor is inside it.
        """
        return self._inside(service.value, box)
