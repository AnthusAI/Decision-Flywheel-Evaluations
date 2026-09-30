"""Strict text-free JSON boundaries for a development capability pilot."""
from __future__ import annotations

from pathlib import Path

from .pilot import PilotPlan
from .serialization import (_cell, _cell_payload, _keys, _mapping, _read,
                            _sequence, _write)


_SCHEMA = "decision-flywheel-evaluations/pilot-plan/v1"


def write_pilot(path: str | Path, plan: PilotPlan) -> None:
    """Persist frozen request identities, never source state or credentials."""
    if not isinstance(plan, PilotPlan):
        raise ValueError("pilot plan must be typed")
    plan.validate()
    _write(path, {"schema": _SCHEMA, "pilot": {
        "protocol_identity": plan.protocol_identity,
        "manifest_sha256": plan.manifest_sha256,
        "source_preflight_checksum": plan.source_preflight_checksum,
        "cells": [_cell_payload(cell) for cell in plan.cells],
        "checksum": plan.checksum,
    }})


def read_pilot(path: str | Path) -> PilotPlan:
    """Reject unknown fields and corrupt hashes before a collector sees a plan."""
    document = _read(path, _SCHEMA, {"schema", "pilot"})
    payload = _mapping(document["pilot"], "pilot plan")
    _keys(payload, {"protocol_identity", "manifest_sha256", "source_preflight_checksum",
                    "cells", "checksum"}, "pilot plan")
    cells = tuple(_cell(_mapping(value, "request cell"))
                  for value in _sequence(payload["cells"], "pilot cells"))
    plan = PilotPlan(payload["protocol_identity"], payload["manifest_sha256"],
                     payload["source_preflight_checksum"], cells, payload["checksum"])
    plan.validate()
    return plan
