"""Translates model class names/ids into pipeline roles (aircraft vs. which service equipment)."""
from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Mapping, Optional, Tuple, Union

from turnaround.config.schema import ClassMapConfig
from turnaround.domain.types import Detection, DetectionKind, RawDetection, ServiceType

logger = logging.getLogger(__name__)

Role = Tuple[DetectionKind, Optional[ServiceType]]


class ClassMappingError(ValueError):
    """Raised when the configured classes cannot be matched against the model's class names."""


def normalize_label(label: Union[str, int]) -> str:
    """Normalise a class label for tolerant comparison.

    Args:
        label: Class name (or id, which is stringified).
    Returns:
        Lowercase string with spaces and hyphens replaced by underscores and trimmed.
    """
    return str(label).strip().lower().replace(" ", "_").replace("-", "_")


class ClassMapper:
    """Maps RawDetection -> Detection using the YAML class map and the loaded model's names."""

    def __init__(self, cfg: ClassMapConfig) -> None:
        """Store the role targets (resolution to ids happens in ``validate``).

        Args:
            cfg: Class-map configuration.
        """
        self._targets: List[Tuple[Union[str, int], Role]] = [
            (cfg.aircraft, (DetectionKind.AIRCRAFT, None)),
            (cfg.aerobridge, (DetectionKind.EQUIPMENT, ServiceType.AEROBRIDGE)),
            (cfg.catering, (DetectionKind.EQUIPMENT, ServiceType.CATERING)),
            (cfg.cargo, (DetectionKind.EQUIPMENT, ServiceType.CARGO)),
            (cfg.pushback, (DetectionKind.EQUIPMENT, ServiceType.PUSHBACK)),
        ]
        self._by_id: Optional[Dict[int, Role]] = None

    def validate(self, class_names: Mapping[int, str]) -> None:
        """Resolve every configured role to a model class id (fail fast on mismatch).

        Args:
            class_names: Model vocabulary, class id -> name.
        Raises:
            ClassMappingError: A configured name/id is missing, ambiguous, or mapped twice.
        Logic:
            For each target: an int must be a key of ``class_names``; a str is compared after
            ``normalize_label`` with every model name and must match exactly one. Two roles may not
            resolve to the same class id. On success ``_by_id`` is populated.
        """
        normalized = {cid: normalize_label(name) for cid, name in class_names.items()}
        by_id: Dict[int, Role] = {}
        for target, role in self._targets:
            if isinstance(target, int):
                if target not in class_names:
                    raise ClassMappingError(f"Class id {target} not in model classes {dict(class_names)}")
                cid = target
            else:
                matches = [cid for cid, name in normalized.items() if name == normalize_label(target)]
                if len(matches) != 1:
                    raise ClassMappingError(
                        f"Config class {target!r} matched {len(matches)} model classes; model classes are {dict(class_names)}"
                    )
                cid = matches[0]
            if cid in by_id:
                raise ClassMappingError(f"Model class id {cid} is mapped to two roles")
            by_id[cid] = role
        self._by_id = by_id
        logger.info("Class map resolved: %s", {cid: (r[0].value, r[1].value if r[1] else None) for cid, r in by_id.items()})

    def resolve(self, raw: RawDetection) -> Optional[Detection]:
        """Convert one raw detection.

        Args:
            raw: Detection in model vocabulary.
        Returns:
            The mapped Detection, or None when the class is not part of the role map (ignored class).
        Raises:
            RuntimeError: If ``validate`` has not been called.
        """
        if self._by_id is None:
            raise RuntimeError("ClassMapper.validate() must be called before resolve()")
        role = self._by_id.get(raw.class_id)
        if role is None:
            return None
        kind, service = role
        return Detection(
            box=raw.box,
            confidence=raw.confidence,
            class_id=raw.class_id,
            label=raw.label,
            track_id=raw.track_id,
            kind=kind,
            service=service,
        )

    def resolve_all(self, raws: Iterable[RawDetection]) -> List[Detection]:
        """Convert many detections, dropping unmapped classes.

        Args:
            raws: Raw detections of one frame.
        Returns:
            Mapped detections in input order.
        """
        out: List[Detection] = []
        for raw in raws:
            det = self.resolve(raw)
            if det is not None:
                out.append(det)
        return out
