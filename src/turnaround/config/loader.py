"""YAML configuration loading, override application, validation and path resolution."""
from __future__ import annotations

import copy
import dataclasses
import logging
import os
import typing
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

import yaml

from turnaround.config.schema import AppConfig, AssociationConfig, ServiceRuleConfig
from turnaround.domain.types import ANCHOR_MODES, SERVICE_NAMES

logger = logging.getLogger(__name__)

ENV_HOME = "TURNAROUND_HOME"
DEFAULT_CONFIG_RELATIVE = "configs/default.yaml"
SOURCE_KINDS = ("youtube", "stream", "file")
ROI_MODES = ("normalized", "pixel")
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


class ConfigError(ValueError):
    """Raised when the configuration file is missing, malformed or invalid."""


def resolve_project_root() -> Path:
    """Locate the project root used to resolve relative paths.

    Returns:
        ``$TURNAROUND_HOME`` if set, else the repository root inferred from this file's location
        (``src/turnaround/config/loader.py`` -> three levels up).
    """
    env = os.environ.get(ENV_HOME)
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[3]


def load_config(
    path: Optional[Union[str, Path]] = None,
    overrides: Optional[Mapping[str, Any]] = None,
) -> AppConfig:
    """Load, override, validate and path-resolve the configuration.

    Args:
        path: YAML file path; None uses ``<project_root>/configs/default.yaml``.
        overrides: Dotted-key overrides applied after parsing, e.g. ``{"source.uri": "x.mp4"}``.
    Returns:
        A fully validated AppConfig whose filesystem paths are absolute.
    Raises:
        ConfigError: File not found, not a mapping, unknown key, or any validation failure.
    Logic:
        1. Locate the file (``_locate_config``).
        2. ``yaml.safe_load``; require a mapping (empty file -> empty mapping).
        3. Expand ``${VAR}`` / ``$VAR`` in every string (``_expand_env``).
        4. Apply dotted overrides (``_apply_overrides``).
        5. Build nested dataclasses strictly (``_build_dataclass``): unknown keys raise; partial
           ``association.min_confidence`` maps are merged over the defaults.
        6. ``validate_config`` aggregates all rule violations into one ConfigError.
        7. ``_resolve_paths`` turns relative paths absolute against the project root.
        8. Store the YAML location in ``config_path``.
    Side effects:
        Reads the YAML file; reads environment variables.
    """
    root = resolve_project_root()
    cfg_path = _locate_config(path, root)
    try:
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {cfg_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"Top level of {cfg_path} must be a mapping")
    raw = _expand_env(raw)
    if overrides:
        raw = _apply_overrides(raw, overrides)
    cfg = _build_dataclass(AppConfig, raw, "")
    cfg.association.min_confidence = {**AssociationConfig().min_confidence, **cfg.association.min_confidence}
    validate_config(cfg)
    _resolve_paths(cfg, root)
    cfg.config_path = str(cfg_path)
    return cfg


def _locate_config(path: Optional[Union[str, Path]], root: Path) -> Path:
    """Find the YAML file.

    Args:
        path: User supplied path or None.
        root: Project root.
    Returns:
        Existing absolute path.
    Raises:
        ConfigError: If no candidate exists.
    Logic:
        1. None -> ``root/configs/default.yaml``.
        2. Absolute path -> used as is.
        3. Relative path -> try the current working directory, then the project root.
    """
    if path is None:
        candidate = root / DEFAULT_CONFIG_RELATIVE
        candidates = [candidate]
    else:
        p = Path(path).expanduser()
        candidates = [p] if p.is_absolute() else [Path.cwd() / p, root / p]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise ConfigError(f"Config file not found. Tried: {', '.join(str(c) for c in candidates)}")


def _expand_env(obj: Any) -> Any:
    """Recursively expand environment variables in all strings.

    Args:
        obj: Parsed YAML value (dict, list, str or scalar).
    Returns:
        Same structure with ``os.path.expandvars`` applied to each string (unknown variables stay literal).
    """
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    if isinstance(obj, list):
        return [_expand_env(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    return obj


def _apply_overrides(raw: Dict[str, Any], overrides: Mapping[str, Any]) -> Dict[str, Any]:
    """Apply dotted-key overrides to a raw config dict.

    Args:
        raw: Parsed YAML mapping.
        overrides: Mapping such as ``{"model.device": "cpu"}``; values of None are ignored.
    Returns:
        A deep copy of ``raw`` with overrides applied (intermediate dicts are created).
    """
    result = copy.deepcopy(raw)
    for dotted, value in overrides.items():
        if value is None:
            continue
        parts = dotted.split(".")
        node = result
        for part in parts[:-1]:
            nxt = node.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                node[part] = nxt
            node = nxt
        node[parts[-1]] = value
    return result


def _build_dataclass(cls: type, data: Any, where: str) -> Any:
    """Strictly build a (possibly nested) dataclass from a mapping.

    Args:
        cls: Dataclass type to instantiate.
        data: Mapping of field values (None is treated as empty).
        where: Dotted path used in error messages.
    Returns:
        Instance of ``cls`` with defaults for missing keys.
    Raises:
        ConfigError: If ``data`` is not a mapping or contains unknown keys.
    Logic:
        1. Resolve field type hints.
        2. Reject keys that are not dataclass fields.
        3. Recurse for fields whose type is itself a dataclass; copy other values verbatim.
    """
    if data is None:
        data = {}
    label = where or "<root>"
    if not isinstance(data, Mapping):
        raise ConfigError(f"'{label}' must be a mapping, got {type(data).__name__}")
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - set(fields))
    if unknown:
        raise ConfigError(f"Unknown key(s) under '{label}': {', '.join(unknown)}. Valid: {', '.join(sorted(fields))}")
    hints = typing.get_type_hints(cls)
    kwargs: Dict[str, Any] = {}
    for name in fields:
        if name not in data:
            continue
        hint = hints[name]
        value = data[name]
        if dataclasses.is_dataclass(hint):
            value = _build_dataclass(hint, value, f"{where}.{name}" if where else name)
        kwargs[name] = value
    return cls(**kwargs)


def _polygon_errors(points: Any, name: str, mode: str) -> List[str]:
    """Validate one ROI polygon.

    Args:
        points: Candidate polygon (list of [x, y]).
        name: Label for messages.
        mode: ``"normalized"`` or ``"pixel"``.
    Returns:
        List of human-readable problems (empty when valid).
    """
    errors: List[str] = []
    if not isinstance(points, list) or len(points) < 3:
        return [f"roi polygon '{name}' needs at least 3 points"]
    for pt in points:
        if not (isinstance(pt, (list, tuple)) and len(pt) == 2 and all(isinstance(v, (int, float)) for v in pt)):
            errors.append(f"roi polygon '{name}' has an invalid point {pt!r}; expected [x, y]")
            continue
        if mode == "normalized" and not all(0.0 <= v <= 1.0 for v in pt):
            errors.append(f"roi polygon '{name}' point {pt!r} outside [0, 1] for normalized mode")
    return errors


def validate_config(cfg: AppConfig) -> None:
    """Check every value range / enum / cross-field rule.

    Args:
        cfg: Configuration to validate (not modified).
    Raises:
        ConfigError: One error listing all violations.
    Logic:
        Collect violations for source, model, ROI (modes, resolution, polygons, zone keys), scene guard
        thresholds, aircraft, association (keys and ranges), temporal (global caps, default rule and every
        override key/value), output and logging; raise once if the list is non-empty.
    """
    errs: List[str] = []

    def check(cond: bool, msg: str) -> None:
        """Record ``msg`` as a violation when ``cond`` is False.

        Args:
            cond: Rule outcome (True = rule satisfied).
            msg: Message appended to the shared error list when the rule fails.
        """
        if not cond:
            errs.append(msg)

    s = cfg.source
    check(s.kind in SOURCE_KINDS, f"source.kind must be one of {SOURCE_KINDS}, got {s.kind!r}")
    check(bool(str(s.uri).strip()), "source.uri must not be empty")
    check(s.target_fps > 0, "source.target_fps must be > 0")
    check(s.max_height > 0, "source.max_height must be > 0")
    check(s.buffer_size >= 1, "source.buffer_size must be >= 1")
    check(s.reconnect_initial_s > 0 and s.reconnect_max_s >= s.reconnect_initial_s, "source reconnect delays invalid")

    m = cfg.model
    check(m.imgsz > 0, "model.imgsz must be > 0")
    check(0.0 < m.conf <= 1.0, "model.conf must be in (0, 1]")
    check(0.0 < m.iou <= 1.0, "model.iou must be in (0, 1]")
    check(m.max_det >= 1, "model.max_det must be >= 1")

    r = cfg.roi
    check(r.coordinate_mode in ROI_MODES, f"roi.coordinate_mode must be one of {ROI_MODES}")
    check(r.anchor in ANCHOR_MODES, f"roi.anchor must be one of {ANCHOR_MODES}")
    if r.enabled:
        if r.coordinate_mode == "pixel":
            ok = isinstance(r.reference_resolution, list) and len(r.reference_resolution) == 2 and all(
                isinstance(v, (int, float)) and v > 0 for v in r.reference_resolution)
            check(ok, "roi.reference_resolution must be [width, height] > 0 in pixel mode")
        if r.analysis_zone:
            errs.extend(_polygon_errors(r.analysis_zone, "analysis_zone", r.coordinate_mode))
        for svc, poly in r.service_zones.items():
            check(svc in SERVICE_NAMES, f"roi.service_zones key {svc!r} is not one of {SERVICE_NAMES}")
            errs.extend(_polygon_errors(poly, f"service_zones.{svc}", r.coordinate_mode))

    g = cfg.scene_guard
    check(len(g.thumbnail_size) == 2 and all(v >= 8 for v in g.thumbnail_size), "scene_guard.thumbnail_size must be [w>=8, h>=8]")
    check(0.0 <= g.change_threshold <= g.recover_threshold <= 1.0, "scene_guard needs 0 <= change_threshold <= recover_threshold <= 1")
    check(g.trigger_frames >= 1 and g.recover_frames >= 1, "scene_guard trigger/recover frames must be >= 1")
    check(0.0 <= g.reference_blend <= 1.0, "scene_guard.reference_blend must be in [0, 1]")

    a = cfg.aircraft
    check(0.0 < a.min_confidence <= 1.0, "aircraft.min_confidence must be in (0, 1]")
    check(a.confirm_hits >= 1, "aircraft.confirm_hits must be >= 1")
    check(a.tentative_timeout_s > 0 and a.lost_timeout_s > 0, "aircraft timeouts must be > 0")
    check(0.0 <= a.reid_iou_threshold <= 1.0, "aircraft.reid_iou_threshold must be in [0, 1]")
    check(0.0 < a.duplicate_iou_threshold <= 1.0, "aircraft.duplicate_iou_threshold must be in (0, 1]")

    c = cfg.association
    check(c.aircraft_box_margin >= 0.0, "association.aircraft_box_margin must be >= 0")
    check(0.0 <= c.min_equipment_overlap <= 1.0, "association.min_equipment_overlap must be in [0, 1]")
    for svc in SERVICE_NAMES:
        check(svc in c.min_confidence, f"association.min_confidence missing service {svc!r}")
    for svc, val in c.min_confidence.items():
        check(svc in SERVICE_NAMES, f"association.min_confidence key {svc!r} is not a service")
        check(isinstance(val, (int, float)) and 0.0 <= val <= 1.0, f"association.min_confidence.{svc} must be in [0, 1]")

    t = cfg.temporal
    check(t.max_gap_s > 0, "temporal.max_gap_s must be > 0")
    check(t.partial_start_window_s >= 0, "temporal.partial_start_window_s must be >= 0")
    rule_fields = {f.name for f in dataclasses.fields(ServiceRuleConfig)}
    for svc, override in t.overrides.items():
        check(svc in SERVICE_NAMES, f"temporal.overrides key {svc!r} is not a service")
        if not isinstance(override, dict):
            errs.append(f"temporal.overrides.{svc} must be a mapping")
            continue
        bad = sorted(set(override) - rule_fields)
        check(not bad, f"temporal.overrides.{svc} has unknown keys {bad}; valid: {sorted(rule_fields)}")
    if not errs:
        for svc in SERVICE_NAMES:
            rule = t.rule_for(svc)
            check(rule.on_window_s > 0, f"temporal rule for {svc}: on_window_s must be > 0")
            check(rule.on_min_hits >= 1, f"temporal rule for {svc}: on_min_hits must be >= 1")
            check(rule.off_grace_s > 0, f"temporal rule for {svc}: off_grace_s must be > 0")
            check(rule.min_interval_s >= 0, f"temporal rule for {svc}: min_interval_s must be >= 0")

    o = cfg.output
    check(o.publish_interval_s >= 0, "output.publish_interval_s must be >= 0")
    check(1 <= o.jpeg_quality <= 100, "output.jpeg_quality must be in [1, 100]")
    check(str(cfg.logging.level).upper() in LOG_LEVELS, f"logging.level must be one of {LOG_LEVELS}")

    if errs:
        raise ConfigError("Invalid configuration:\n  - " + "\n  - ".join(errs))


def _abs(value: str, root: Path) -> str:
    """Make a path absolute.

    Args:
        value: Absolute or relative path string (``~`` allowed).
        root: Base directory for relative paths.
    Returns:
        Absolute path string.
    """
    p = Path(value).expanduser()
    return str(p if p.is_absolute() else (root / p).resolve())


def _resolve_paths(cfg: AppConfig, root: Path) -> None:
    """Rewrite every filesystem path of the config to an absolute path (in place).

    Args:
        cfg: Configuration to modify.
        root: Project root for relative paths.
    Logic:
        Resolve model.weights_path, model.tracker_config, storage.db_path, output.runtime_dir, and the
        optional scene_guard.reference_image, output.record_path, logging.file; resolve source.uri only
        when source.kind == "file".
    """
    cfg.model.weights_path = _abs(cfg.model.weights_path, root)
    cfg.model.tracker_config = _abs(cfg.model.tracker_config, root)
    cfg.storage.db_path = _abs(cfg.storage.db_path, root)
    cfg.output.runtime_dir = _abs(cfg.output.runtime_dir, root)
    if cfg.scene_guard.reference_image:
        cfg.scene_guard.reference_image = _abs(cfg.scene_guard.reference_image, root)
    if cfg.output.record_path:
        cfg.output.record_path = _abs(cfg.output.record_path, root)
    if cfg.logging.file:
        cfg.logging.file = _abs(cfg.logging.file, root)
    if cfg.source.kind == "file":
        cfg.source.uri = _abs(cfg.source.uri, root)
