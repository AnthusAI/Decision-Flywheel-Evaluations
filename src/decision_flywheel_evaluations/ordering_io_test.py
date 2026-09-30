import json

import pytest

from .ordering import plan_ordering
from .ordering_io import read_ordering, write_ordering
from .ordering_test import _initial_inputs


def _plan():
    protocol, manifest, rows, initial, observations, treatments = _initial_inputs()
    return plan_ordering(protocol, manifest, rows, initial, observations,
                         "artifacts/initial-result.json", treatments), rows


def test_an_ordering_plan_round_trips_without_source_text(tmp_path):
    plan, rows = _plan()
    path = tmp_path / "ordering.json"
    write_ordering(path, plan)

    assert read_ordering(path) == plan
    assert rows[0].text not in path.read_text(encoding="utf-8")
    assert json.loads(path.read_text(encoding="utf-8"))["schema"] == "decision-flywheel-evaluations/ordering-plan/v1"


def test_an_ordering_reader_rejects_unknown_root_inner_cell_treatment_and_request_fields(tmp_path):
    plan, _ = _plan()
    path = tmp_path / "ordering.json"
    write_ordering(path, plan)
    document = json.loads(path.read_text(encoding="utf-8"))
    mutations = (
        (document, "unexpected", "root"),
        (document["ordering"], "unexpected", "inner"),
        (document["ordering"]["cells"][0], "unexpected", "cell"),
        (document["ordering"]["treatments"][0], "unexpected", "treatment"),
        (document["ordering"]["cells"][0]["request"], "unexpected", "request"),
    )
    for payload, key, _name in mutations:
        payload[key] = "untrusted"
        path.write_text(json.dumps(document), encoding="utf-8")
        with pytest.raises(ValueError, match="fields"):
            read_ordering(path)
        del payload[key]


def test_an_ordering_reader_rejects_tampered_checksums_and_malformed_immutable_types(tmp_path):
    plan, _ = _plan()
    path = tmp_path / "ordering.json"
    write_ordering(path, plan)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["ordering"]["checksum"] = "0" * 64
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        read_ordering(path)

    document["ordering"]["checksum"] = plan.checksum
    document["ordering"]["treatments"] = {"not": "an immutable array"}
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="array"):
        read_ordering(path)
