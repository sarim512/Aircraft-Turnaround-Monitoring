"""Frame buffer, YouTube URL selection, file replay and the resilient stream source."""
import threading
import time

import cv2
import numpy as np
import pytest

from turnaround.config.schema import SourceConfig
from turnaround.ingestion.factory import build_source
from turnaround.ingestion.file_source import VideoFileSource
from turnaround.ingestion.frame_buffer import LatestFrameBuffer
from turnaround.ingestion.stream_source import StreamFrameSource
from turnaround.ingestion.youtube import StreamResolutionError, pick_stream_url


def make_video(path, n_frames=100, fps=10, size=(64, 48)):
    """Write a small MJPG test video of constant-color frames."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    for i in range(n_frames):
        frame = np.full((size[1], size[0], 3), i % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return str(path)


def test_buffer_drops_oldest_and_counts():
    buf = LatestFrameBuffer(2)
    assert [buf.put(i) for i in range(4)] == [False, False, True, True]
    assert buf.dropped == 2 and buf.get(0.1) == 2 and buf.get(0.1) == 3 and buf.get(0.05) is None


def test_buffer_get_wakes_on_put_and_close_unblocks():
    buf = LatestFrameBuffer(1)
    threading.Timer(0.1, buf.put, args=("x",)).start()
    assert buf.get(2.0) == "x"
    threading.Timer(0.1, buf.close).start()
    start = time.monotonic()
    assert buf.get(5.0) is None and time.monotonic() - start < 2.0
    with pytest.raises(ValueError):
        LatestFrameBuffer(0)


def test_pick_stream_url_precedence_and_failure():
    assert pick_stream_url({"url": "u", "formats": [{"url": "f"}]}) == "u"
    assert pick_stream_url({"requested_formats": [{"url": None}, {"url": "r"}]}) == "r"
    assert pick_stream_url({"formats": [{"url": "worst"}, {"url": "best"}]}) == "best"
    with pytest.raises(StreamResolutionError):
        pick_stream_url({"formats": [{}]})


def test_file_source_stride_timestamps_and_exhaustion(tmp_path):
    path = make_video(tmp_path / "v.avi", n_frames=100, fps=10)
    src = VideoFileSource(path, target_fps=5.0, start_epoch=1000.0)
    src.start()
    packets = []
    while True:
        p = src.read()
        if p is None:
            break
        packets.append(p)
    assert src.exhausted and len(packets) == 50
    assert packets[0].timestamp == pytest.approx(1000.0) and packets[1].timestamp == pytest.approx(1000.2)
    assert packets[-1].media_time_s == pytest.approx(9.8)
    assert [p.index for p in packets[:3]] == [1, 2, 3]
    src.stop()
    assert src.read() is None


def test_file_source_loop_keeps_timeline_increasing(tmp_path):
    path = make_video(tmp_path / "v.avi", n_frames=20, fps=10)
    src = VideoFileSource(path, target_fps=10.0, loop=True, start_epoch=0.0)
    src.start()
    ts = [src.read().timestamp for _ in range(45)]
    src.stop()
    assert all(b > a for a, b in zip(ts, ts[1:])) and ts[20] == pytest.approx(2.0)


def test_file_source_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        VideoFileSource(str(tmp_path / "none.avi"), 5.0).start()


def test_factory_builds_each_kind(tmp_path):
    assert isinstance(build_source(SourceConfig(kind="file", uri=str(tmp_path / "a.avi"))), VideoFileSource)
    assert isinstance(build_source(SourceConfig(kind="youtube", uri="https://youtu.be/x")), StreamFrameSource)
    assert isinstance(build_source(SourceConfig(kind="stream", uri="rtsp://h/s")), StreamFrameSource)
    with pytest.raises(ValueError):
        build_source(SourceConfig(kind="nope", uri="x"))


class FakeCapture:
    """Capture yielding ``n`` frames, then failing grab()."""

    def __init__(self, n):
        """Create a capture that yields ``n`` frames."""
        self.n, self.i, self.released = n, 0, False

    def isOpened(self):
        """Always True."""
        return True

    def grab(self):
        """True for the first ``n`` calls, then False (simulates a stream failure)."""
        self.i += 1
        return self.i <= self.n

    def retrieve(self):
        """Return a black 4x4 frame."""
        return True, np.zeros((4, 4, 3), np.uint8)

    def release(self):
        """Record that the capture was released."""
        self.released = True


def test_stream_source_reconnects_and_reresolves_url():
    urls = []
    caps = []

    def resolver():
        urls.append(len(urls))
        return f"url{len(urls)}"

    def factory(url):
        cap = FakeCapture(5)
        caps.append(cap)
        return cap

    src = StreamFrameSource(resolver, target_fps=1000.0, buffer_size=50, reconnect_initial_s=0.05, reconnect_max_s=0.1, capture_factory=factory)
    src.start()
    got = []
    deadline = time.monotonic() + 5.0
    while len(got) < 10 and time.monotonic() < deadline:
        p = src.read(0.5)
        if p:
            got.append(p)
    src.stop()
    assert len(got) >= 10 and len(urls) >= 2  # resolved again after the first capture ended
    assert src.stats()["reconnects"] >= 1 and not src.exhausted
    assert [p.index for p in got] == sorted(p.index for p in got)
    assert all(c.released for c in caps)


def test_stream_source_backoff_and_failed_open():
    src = StreamFrameSource(lambda: "u", 5.0, reconnect_initial_s=1.0, reconnect_max_s=5.0, capture_factory=lambda u: None)
    assert [src._backoff_delay(a) for a in (1, 2, 3, 4, 5)] == [1.0, 2.0, 4.0, 5.0, 5.0]
    src2 = StreamFrameSource(lambda: (_ for _ in ()).throw(RuntimeError("boom")), 5.0, reconnect_initial_s=0.01,
                             capture_factory=lambda u: FakeCapture(1))
    src2.start()
    assert src2.read(0.2) is None
    src2.stop()
