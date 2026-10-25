"""Config loading, overrides, validation and path resolution."""
from pathlib import Path

import pytest
import yaml

from turnaround.config.loader import ConfigError, load_config, resolve_project_root


def write(tmp_path, data):
    """Write a dict as YAML into tmp_path and return the path."""
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_default_config_loads_with_absolute_paths():
    cfg = load_config()
    root = resolve_project_root()
    assert Path(cfg.model.weights_path) == root / "weights" / "best.pt"
    assert Path(cfg.storage.db_path).is_absolute() and Path(cfg.source.uri).is_absolute()
    assert cfg.association.min_confidence["cargo"] == 0.35
    assert cfg.config_path.endswith("default.yaml")


def test_project_root_env_override_relocates_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("TURNAROUND_HOME", str(tmp_path))
    cfg = load_config(Path(__file__).parents[1] / "configs" / "default.yaml")
    assert Path(cfg.storage.db_path) == (tmp_path / "data" / "turnaround.db").resolve()


def test_overrides_and_env_expansion(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_STREAM", "https://example.org/live.m3u8")
    path = write(tmp_path, {"source": {"kind": "stream", "uri": "${MY_STREAM}"}})
    cfg = load_config(path, {"model.device": "cpu", "source.target_fps": 2, "logging.level": None})
    assert cfg.source.uri == "https://example.org/live.m3u8" and cfg.model.device == "cpu" and cfg.source.target_fps == 2
    assert cfg.logging.level == "INFO"  # None override ignored


def test_partial_min_confidence_merges_with_defaults(tmp_path):
    cfg = load_config(write(tmp_path, {"source": {"uri": "x.mp4"}, "association": {"min_confidence": {"cargo": 0.6}}}))
    assert cfg.association.min_confidence["cargo"] == 0.6 and cfg.association.min_confidence["aerobridge"] == 0.35


def test_temporal_override_merges_into_rule(tmp_path):
    cfg = load_config(write(tmp_path, {"source": {"uri": "x.mp4"}, "temporal": {"overrides": {"pushback": {"off_grace_s": 2.5}}}}))
    assert cfg.temporal.rule_for("pushback").off_grace_s == 2.5 and cfg.temporal.rule_for("cargo").off_grace_s == 8.0


@pytest.mark.parametrize("data,fragment", [
    ({"source": {"uri": "x", "bogus": 1}}, "Unknown key"),
    ({"bogus": {}}, "Unknown key"),
    ({"source": {"kind": "ftp", "uri": "x"}}, "source.kind"),
    ({"source": {"uri": ""}}, "source.uri"),
    ({"model": {"conf": 0}, "source": {"uri": "x"}}, "model.conf"),
    ({"source": {"uri": "x"}, "roi": {"enabled": True, "analysis_zone": [[0, 0], [1, 1]]}}, "at least 3 points"),
    ({"source": {"uri": "x"}, "roi": {"enabled": True, "analysis_zone": [[0, 0], [2, 0], [1, 1]]}}, "outside [0, 1]"),
    ({"source": {"uri": "x"}, "roi": {"enabled": True, "coordinate_mode": "pixel", "analysis_zone": [[0, 0], [9, 0], [1, 1]]}}, "reference_resolution"),
    ({"source": {"uri": "x"}, "roi": {"enabled": True, "service_zones": {"tractor": [[0, 0], [1, 0], [1, 1]]}}}, "service_zones"),
    ({"source": {"uri": "x"}, "temporal": {"overrides": {"pushback": {"nope": 1}}}}, "unknown keys"),
    ({"source": {"uri": "x"}, "temporal": {"defaults": {"on_min_hits": 0}}}, "on_min_hits"),
    ({"source": {"uri": "x"}, "scene_guard": {"change_threshold": 0.9, "recover_threshold": 0.5}}, "scene_guard"),
    ({"source": {"uri": "x"}, "logging": {"level": "LOUD"}}, "logging.level"),
])
def test_invalid_configs_raise_helpful_errors(tmp_path, data, fragment):
    with pytest.raises(ConfigError) as exc:
        load_config(write(tmp_path, data))
    assert fragment in str(exc.value)


def test_missing_file_and_non_mapping(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "missing.yaml")
    bad = tmp_path / "list.yaml"
    bad.write_text("- 1\n- 2\n")
    with pytest.raises(ConfigError):
        load_config(bad)
