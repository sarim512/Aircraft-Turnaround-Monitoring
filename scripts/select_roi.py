"""Interactive helper to draw the manual ROI polygon and print a ready-to-paste YAML snippet."""
import _bootstrap  # noqa: F401  (adds src/ to sys.path)

import argparse
import sys
from typing import List, Optional, Tuple

import cv2
import numpy as np

from turnaround.config.loader import load_config
from turnaround.ingestion.factory import build_source


def grab_frame(cfg_path: Optional[str], source: Optional[str], kind: Optional[str]) -> np.ndarray:
    """Fetch one frame from the configured (or overridden) source.

    Args:
        cfg_path: Config file path or None for the default.
        source: Optional override of source.uri.
        kind: Optional override of source.kind.
    Returns:
        A BGR frame.
    Raises:
        RuntimeError: If no frame could be read within ~30 s.
    Logic:
        Build the source via the factory, start it and poll ``read`` up to 30 times.
    Side effects:
        Opens (and then closes) the video source / network stream.
    """
    cfg = load_config(cfg_path, {"source.uri": source, "source.kind": kind})
    src = build_source(cfg.source)
    src.start()
    try:
        for _ in range(30):
            packet = src.read(1.0)
            if packet is not None:
                return packet.frame
    finally:
        src.stop()
    raise RuntimeError("Could not read a frame from the source")


def pick_polygon(frame: np.ndarray) -> List[Tuple[float, float]]:
    """Let the user click polygon vertices on the frame.

    Args:
        frame: Frame to display.
    Returns:
        Normalized (x, y) vertices in [0, 1].
    Logic:
        Left click adds a vertex, ``u`` removes the last, ``c`` clears, Enter/Space accepts, Esc aborts
        (returns an empty list). The polygon preview is redrawn after each change.
    """
    h, w = frame.shape[:2]
    points: List[Tuple[int, int]] = []
    window = "ROI: click vertices | u=undo c=clear Enter=accept Esc=abort"

    def on_mouse(event, x, y, *_):
        """Append a vertex on left click.

        Args:
            event: OpenCV mouse event code.
            x: Cursor x in image pixels.
            y: Cursor y in image pixels.
            *_: Ignored extra arguments (flags, user data).
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)
    while True:
        canvas = frame.copy()
        for pt in points:
            cv2.circle(canvas, pt, 4, (0, 215, 255), -1)
        if len(points) >= 2:
            cv2.polylines(canvas, [np.array(points, np.int32)], len(points) >= 3, (0, 215, 255), 2)
        cv2.imshow(window, canvas)
        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            points.clear()
            break
        if key in (13, 32) and len(points) >= 3:
            break
        if key == ord("u") and points:
            points.pop()
        if key == ord("c"):
            points.clear()
    cv2.destroyAllWindows()
    return [(round(x / w, 4), round(y / h, 4)) for x, y in points]


def main(argv: Optional[List[str]] = None) -> int:
    """Run the interactive ROI selection and print the YAML snippet.

    Args:
        argv: Argument list (None uses ``sys.argv[1:]``).
    Returns:
        0 on success, 1 when aborted.
    Side effects:
        Opens an OpenCV GUI window; prints to stdout.
    """
    p = argparse.ArgumentParser(description="Draw the analysis-zone polygon")
    p.add_argument("--config", default=None)
    p.add_argument("--source", default=None)
    p.add_argument("--kind", choices=["youtube", "stream", "file"], default=None)
    args = p.parse_args(argv)
    frame = grab_frame(args.config, args.source, args.kind)
    polygon = pick_polygon(frame)
    if len(polygon) < 3:
        print("Aborted: need at least 3 points", file=sys.stderr)
        return 1
    print("\nPaste into configs/default.yaml:\n")
    print("roi:\n  enabled: true\n  coordinate_mode: normalized\n  analysis_zone:")
    for x, y in polygon:
        print(f"    - [{x}, {y}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
