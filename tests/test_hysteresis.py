"""HysteresisFilter: activation, flicker, occlusion, gap capping, back-dated boundaries."""
from turnaround.config.schema import ServiceRuleConfig
from turnaround.temporal.hysteresis import HysteresisFilter, TransitionKind

RULE = ServiceRuleConfig(on_window_s=3.0, on_min_hits=4, off_grace_s=8.0, min_interval_s=5.0)


def feed(flt, samples, dt=0.2, t0=0.0):
    """Feed booleans at fixed dt; return list of (index, transition)."""
    out = []
    for i, positive in enumerate(samples):
        tr = flt.update(t0 + i * dt, dt, positive)
        if tr:
            out.append((i, tr))
    return out


def test_activates_after_min_hits_and_backdates_start():
    flt = HysteresisFilter(RULE)
    out = feed(flt, [False, False, True, True, True, True, True])
    assert len(out) == 1
    idx, tr = out[0]
    assert tr.kind is TransitionKind.ACTIVATED
    assert idx == 5  # 4th positive
    assert tr.effective_ts == 2 * 0.2  # first positive, not the decision time
    assert tr.decided_ts > tr.effective_ts and flt.active


def test_sparse_positives_never_activate():
    flt = HysteresisFilter(RULE)
    samples = ([True] + [False] * 14) * 6  # one positive every 3 s
    assert feed(flt, samples) == []
    assert not flt.active


def test_flicker_does_not_release():
    flt = HysteresisFilter(RULE)
    feed(flt, [True] * 5)
    assert flt.active
    assert feed(flt, ([True, False, False, False] * 20), t0=1.0) == []
    assert flt.active


def test_occlusion_shorter_than_grace_keeps_state_and_longer_releases_at_last_positive():
    flt = HysteresisFilter(RULE)
    feed(flt, [True] * 5)  # last positive at t=0.8
    assert feed(flt, [False] * 30, t0=1.0) == []  # 6 s < 8 s grace
    out = feed(flt, [False] * 20, t0=7.0)  # absence reaches 8 s
    assert len(out) == 1
    tr = out[0][1]
    assert tr.kind is TransitionKind.DEACTIVATED
    assert tr.effective_ts == 4 * 0.2  # last positive, not the release time
    assert tr.start_ts == 0.0
    assert not flt.active


def test_positive_after_short_absence_resets_grace_counter():
    flt = HysteresisFilter(RULE)
    feed(flt, [True] * 5)
    feed(flt, [False] * 30, t0=1.0)  # 6 s absent
    feed(flt, [True], t0=7.0)
    assert feed(flt, [False] * 30, t0=7.2) == []  # another 6 s, counter restarted
    assert flt.active


def test_capped_dt_makes_long_stream_gap_harmless():
    flt = HysteresisFilter(RULE)
    feed(flt, [True] * 5)
    # a 60 s stall is credited as at most max_gap_s=3 s by the engine; filter receives dt=3
    assert flt.update(61.0, 3.0, False) is None
    assert flt.active


def test_force_close_returns_last_positive_and_resets():
    flt = HysteresisFilter(RULE)
    feed(flt, [True] * 5)
    tr = flt.force_close(10.0)
    assert tr is not None and tr.kind is TransitionKind.DEACTIVATED and tr.effective_ts == 0.8
    assert not flt.active and flt.force_close(11.0) is None


def test_can_reactivate_after_release():
    flt = HysteresisFilter(RULE)
    feed(flt, [True] * 5)
    feed(flt, [False] * 45, t0=1.0)
    assert not flt.active
    out = feed(flt, [True] * 5, t0=20.0)
    assert out and out[0][1].kind is TransitionKind.ACTIVATED
