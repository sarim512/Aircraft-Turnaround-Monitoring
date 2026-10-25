"""ROI polygons and equipment association."""
import pytest

from turnaround.config.schema import AssociationConfig, RoiConfig
from turnaround.domain.types import AircraftSnapshot, BoundingBox, ServiceType
from turnaround.spatial.association import EquipmentAssociator, RejectionReason
from turnaround.spatial.roi import Polygon, RoiSet

from helpers import AIRCRAFT_BOX, EQUIP_BOX, FAR_BOX, equipment

SQUARE = Polygon(((0, 0), (10, 0), (10, 10), (0, 10)))


def snapshot(aid=1, box=AIRCRAFT_BOX):
    """Build an observed AircraftSnapshot."""
    return AircraftSnapshot(aid, box, 0.9, True, 0.0, 0.0)


def test_polygon_contains_and_concave_shape():
    assert SQUARE.contains(5, 5) and not SQUARE.contains(11, 5) and not SQUARE.contains(5, -1)
    concave = Polygon(((0, 0), (10, 0), (10, 10), (5, 4), (0, 10)))
    assert concave.contains(2, 2) and not concave.contains(5, 8)


def test_normalized_and_pixel_polygons_scale_to_frame():
    norm = RoiSet(RoiConfig(enabled=True, analysis_zone=[[0, 0], [0.5, 0], [0.5, 1], [0, 1]]))
    norm.bind(640, 360)
    assert norm.allows_equipment(BoundingBox(100, 100, 140, 140)) and not norm.allows_equipment(BoundingBox(400, 100, 440, 140))
    pix = RoiSet(RoiConfig(enabled=True, coordinate_mode="pixel", reference_resolution=[1280, 720],
                           analysis_zone=[[0, 0], [640, 0], [640, 720], [0, 720]]))
    pix.bind(640, 360)  # half resolution -> polygon covers x < 320
    assert pix.allows_equipment(BoundingBox(100, 100, 140, 140)) and not pix.allows_equipment(BoundingBox(400, 100, 440, 140))


def test_disabled_roi_allows_everything_and_unbound_enabled_roi_raises():
    assert RoiSet(RoiConfig(enabled=False)).allows_equipment(FAR_BOX)
    roi = RoiSet(RoiConfig(enabled=True, analysis_zone=[[0, 0], [1, 0], [1, 1]]))
    with pytest.raises(RuntimeError):
        roi.allows_equipment(FAR_BOX)


def test_bottom_center_anchor_changes_membership():
    cfg = RoiConfig(enabled=True, anchor="bottom_center", analysis_zone=[[0, 0], [1, 0], [1, 0.5], [0, 0.5]])
    roi = RoiSet(cfg)
    roi.bind(100, 100)
    assert roi.allows_equipment(BoundingBox(10, 10, 30, 40))  # bottom y=40 < 50
    assert not roi.allows_equipment(BoundingBox(10, 10, 30, 60))  # bottom y=60 > 50


def test_apply_to_aircraft_flag():
    cfg = RoiConfig(enabled=True, apply_to_aircraft=False, analysis_zone=[[0, 0], [0.1, 0], [0.1, 0.1]])
    roi = RoiSet(cfg)
    roi.bind(640, 360)
    assert roi.allows_aircraft(AIRCRAFT_BOX)


def make_assoc(roi_cfg=None, **kwargs):
    """Build an EquipmentAssociator with a bound RoiSet for a 640x360 frame."""
    roi = RoiSet(roi_cfg or RoiConfig())
    roi.bind(640, 360)
    return EquipmentAssociator(AssociationConfig(**kwargs), roi)


def test_accepts_equipment_on_aircraft_and_reports_presence():
    res = make_assoc().associate([equipment(ServiceType.AEROBRIDGE)], [snapshot()])
    assert len(res.accepted) == 1 and res.accepted[0].aircraft_id == 1
    presence = res.presence_for(1)
    assert presence[ServiceType.AEROBRIDGE] and not presence[ServiceType.CARGO]


@pytest.mark.parametrize("det_kwargs,reason", [
    (dict(conf=0.1), RejectionReason.LOW_CONFIDENCE),
    (dict(box=FAR_BOX), RejectionReason.NOT_ON_AIRCRAFT),
])
def test_rejections(det_kwargs, reason):
    res = make_assoc().associate([equipment(ServiceType.CATERING, **det_kwargs)], [snapshot()])
    assert res.accepted == [] and res.rejected[0].reason is reason


def test_no_aircraft_context_and_roi_rejections():
    res = make_assoc().associate([equipment(ServiceType.CATERING)], [])
    assert res.rejected[0].reason is RejectionReason.NO_AIRCRAFT_CONTEXT
    outside = RoiConfig(enabled=True, analysis_zone=[[0, 0], [0.1, 0], [0.1, 0.1], [0, 0.1]])
    res = make_assoc(outside).associate([equipment(ServiceType.CATERING)], [snapshot()])
    assert res.rejected[0].reason is RejectionReason.OUTSIDE_ANALYSIS_ZONE
    zone = RoiConfig(enabled=True, service_zones={"cargo": [[0, 0], [0.1, 0], [0.1, 0.1]]})
    res = make_assoc(zone).associate([equipment(ServiceType.CARGO)], [snapshot()])
    assert res.rejected[0].reason is RejectionReason.OUTSIDE_SERVICE_ZONE


def test_assigned_to_aircraft_with_largest_overlap():
    left = snapshot(1, BoundingBox(0, 100, 380, 300))  # overlaps equipment partially
    right = snapshot(2, AIRCRAFT_BOX)
    res = make_assoc().associate([equipment(ServiceType.PUSHBACK)], [left, right])
    assert res.accepted[0].aircraft_id == 2


def test_per_service_confidence_floor_is_respected():
    assoc = make_assoc(min_confidence={"aerobridge": 0.9, "catering": 0.1, "cargo": 0.1, "pushback": 0.1})
    res = assoc.associate([equipment(ServiceType.AEROBRIDGE, conf=0.8), equipment(ServiceType.CATERING, conf=0.2)], [snapshot()])
    assert [a.service for a in res.accepted] == [ServiceType.CATERING]
