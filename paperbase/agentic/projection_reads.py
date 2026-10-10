"""Deterministic multi-projection reads and exact semantic identity; no judgement."""
from dataclasses import asdict
import json

from .contracts import FactProjection


def iter_projections(record):
    projections = tuple(getattr(record, "projections", ()) or ())
    if projections:
        return projections
    # Explicit compatibility for historical DTO-like audit objects.
    projection = getattr(record, "projection", None)
    return (projection,) if projection is not None else ()


def iter_projection_parts(record):
    return tuple(part for projection in iter_projections(record)
                 for part in (projection.facts if isinstance(projection, FactProjection) else (projection,)))


def projection_support_anchors(record):
    return tuple(dict.fromkeys(anchor for part in iter_projection_parts(record)
                              for anchor in getattr(part, "support_anchors", ())))


def projection_identity(projection):
    # Preserve ordered semantic payloads (e.g. mechanism steps) and provenance.
    return json.dumps(asdict(projection), ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), default=lambda item: item.value)


def dedupe_projections(projections):
    by_identity = {}
    for projection in projections:
        by_identity.setdefault(projection_identity(projection), projection)
    return tuple(by_identity.values())
