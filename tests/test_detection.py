"""ClassMapper validation/resolution and Ultralytics result parsing."""
from types import SimpleNamespace

import numpy as np
import pytest

from turnaround.config.schema import ClassMapConfig
from turnaround.detection.class_mapper import ClassMapper, ClassMappingError, normalize_label
from turnaround.detection.ultralytics_tracker import UltralyticsTracker, resolve_device
from turnaround.domain.types import BoundingBox, DetectionKind, RawDetection, ServiceType

NAMES = {0: "Aircraft", 1: "Aerobridge Connected", 2: "catering-truck-connected", 3: "cargo_loader_connected", 4: "pushback_tug_connected", 5: "person"}


def raw(cid, label="x", track=1):
    """Build a RawDetection of the given class id."""
    return RawDetection(BoundingBox(0, 0, 10, 10), 0.9, cid, label, track)


def test_normalize_label():
    assert normalize_label(" Catering-Truck Connected ") == "catering_truck_connected"


def test_mapper_resolves_names_tolerantly_and_ignores_unmapped_classes():
    mapper = ClassMapper(ClassMapConfig())
    mapper.validate(NAMES)
    assert mapper.resolve(raw(0)).kind is DetectionKind.AIRCRAFT
    assert mapper.resolve(raw(2)).service is ServiceType.CATERING
    assert mapper.resolve(raw(5)) is None
    assert len(mapper.resolve_all([raw(0), raw(5), raw(4)])) == 2


def test_mapper_supports_integer_ids_and_reports_errors():
    mapper = ClassMapper(ClassMapConfig(aircraft=0, aerobridge=1, catering=2, cargo=3, pushback=4))
    mapper.validate(NAMES)
    assert mapper.resolve(raw(4)).service is ServiceType.PUSHBACK
    with pytest.raises(ClassMappingError):
        ClassMapper(ClassMapConfig(aircraft=99)).validate(NAMES)
    with pytest.raises(ClassMappingError):
        ClassMapper(ClassMapConfig(aerobridge="does_not_exist")).validate(NAMES)
    with pytest.raises(ClassMappingError):
        ClassMapper(ClassMapConfig(catering="aircraft")).validate(NAMES)  # two roles -> same id
    with pytest.raises(RuntimeError):
        ClassMapper(ClassMapConfig()).resolve(raw(0))


class FakeBoxes:
    """Mimics ultralytics ``Boxes``: array attributes and ``len``."""

    def __init__(self, with_ids):
        """Create two fixed boxes; ``with_ids`` toggles tracker ids."""
        self.xyxy = np.array([[1, 2, 30, 40], [5, 5, 9, 9]], dtype=float)
        self.conf = np.array([0.9, 0.5])
        self.cls = np.array([0.0, 3.0])
        self.id = np.array([7.0, 8.0]) if with_ids else None

    def __len__(self):
        """Number of boxes (always 2)."""
        return 2


def fake_result(with_ids=True):
    """Build an Ultralytics-like result with two boxes (with or without track ids)."""
    return SimpleNamespace(boxes=FakeBoxes(with_ids), names=NAMES)


def test_parse_result_with_and_without_track_ids():
    tracker = UltralyticsTracker("w.pt")
    dets = tracker._parse(fake_result())
    assert [(d.class_id, d.track_id, d.label) for d in dets] == [(0, 7, "Aircraft"), (3, 8, "cargo_loader_connected")]
    assert dets[0].box == BoundingBox(1.0, 2.0, 30.0, 40.0)
    assert [d.track_id for d in tracker._parse(fake_result(with_ids=False))] == [None, None]
    assert tracker._parse(SimpleNamespace(boxes=None, names=NAMES)) == []


def test_tracker_requires_load_and_missing_weights_raise(tmp_path):
    tracker = UltralyticsTracker(str(tmp_path / "missing.pt"))
    with pytest.raises(RuntimeError):
        tracker.track(np.zeros((8, 8, 3), np.uint8))
    with pytest.raises(FileNotFoundError):
        tracker.load()
    tracker.reset()  # no-op without a model


def test_resolve_device_falls_back_to_cpu_without_cuda(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def no_torch(name, *a, **k):
        if name == "torch":
            raise ImportError
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_torch)
    assert resolve_device("auto", None) == ("cpu", False)
    assert resolve_device("cuda:0", True) == ("cpu", False)
    assert resolve_device("cpu", None) == ("cpu", False)
