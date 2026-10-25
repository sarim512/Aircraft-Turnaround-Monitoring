"""SceneGuard: stable view, change detection, recovery, lighting robustness."""
import cv2
import numpy as np

from turnaround.config.schema import SceneGuardConfig
from turnaround.spatial.scene_guard import SceneGuard, SceneState


def scene(seed, size=(320, 180)):
    """Reproducible blocky random scene image for a seed."""
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, (size[1] // 8, size[0] // 8, 3), dtype=np.uint8)
    return cv2.resize(small, size, interpolation=cv2.INTER_NEAREST)


def test_same_scene_stays_stable_even_with_brightness_change_and_noise():
    guard = SceneGuard(SceneGuardConfig())
    a = scene(1)
    for i in range(10):
        assert guard.evaluate(a) is SceneState.STABLE
    darker = np.clip(a.astype(np.float32) * 0.5, 0, 255).astype(np.uint8)
    noisy = np.clip(a.astype(np.int16) + np.random.default_rng(0).integers(-6, 6, a.shape), 0, 255).astype(np.uint8)
    for frame in (darker, noisy, darker, noisy):
        assert guard.evaluate(frame) is SceneState.STABLE


def test_change_requires_consecutive_frames_then_recovers():
    guard = SceneGuard(SceneGuardConfig(trigger_frames=3, recover_frames=3))
    a, b = scene(1), scene(2)
    guard.evaluate(a)
    assert guard.evaluate(b) is SceneState.STABLE and guard.evaluate(b) is SceneState.STABLE  # not yet 3 in a row
    assert guard.evaluate(a) is SceneState.STABLE  # streak broken
    states = [guard.evaluate(b) for _ in range(3)]
    assert states[-1] is SceneState.CHANGED
    assert guard.evaluate(a) is SceneState.CHANGED and guard.evaluate(a) is SceneState.CHANGED
    assert guard.evaluate(a) is SceneState.STABLE
    assert guard.last_score > 0.9


def test_reference_image_defines_expected_view(tmp_path):
    ref = tmp_path / "ref.png"
    cv2.imwrite(str(ref), scene(5))
    guard = SceneGuard(SceneGuardConfig(reference_image=str(ref), trigger_frames=1))
    assert guard.evaluate(scene(6)) is SceneState.CHANGED  # first frame already differs from the reference


def test_flat_frames_have_zero_similarity():
    guard = SceneGuard(SceneGuardConfig(trigger_frames=1))
    flat = np.zeros((90, 160, 3), np.uint8)
    guard.evaluate(flat)
    assert guard.evaluate(flat) is SceneState.CHANGED  # no structure -> cannot be validated
