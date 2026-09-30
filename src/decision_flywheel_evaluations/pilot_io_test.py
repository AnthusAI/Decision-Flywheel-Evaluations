import json
from dataclasses import replace
from pathlib import Path

import pytest

from .pilot import plan_jev_pilot
from .pilot_io import read_pilot, write_pilot
from .preflight import preflight
from .preflight_test import _protocol, _rows
from .manifests import read_manifest


def _fixture():
    manifest, rows = _rows()
    base = _protocol(manifest, rows, artifact=False)
    protocol = replace(base, models=(base.models[0],))
    source = preflight(protocol, manifest=manifest, rows=rows, stage="optimization")
    return plan_jev_pilot(protocol, manifest, rows, source), rows


def test_pilot_json_round_trips_only_frozen_request_metadata_without_source_text(tmp_path):
    plan, rows = _fixture()
    path = tmp_path / "pilot.json"

    write_pilot(path, plan)

    assert read_pilot(path) == plan
    content = path.read_text(encoding="utf-8")
    assert all(row.text not in content for row in rows)
    assert '"text":' not in content


def test_pilot_json_rejects_extra_fields_and_changed_plan_hashes(tmp_path):
    plan, _rows_unused = _fixture()
    path = tmp_path / "pilot.json"
    write_pilot(path, plan)
    original = json.loads(path.read_text(encoding="utf-8"))

    for document in (
        {**original, "credentials": "secret sentinel"},
        {**original, "pilot": {**original["pilot"], "checksum": "0" * 64}},
        {**original, "pilot": {**original["pilot"], "state": {"text": "source sentinel"}}},
    ):
        path.write_text(json.dumps(document), encoding="utf-8")
        with pytest.raises(ValueError):
            read_pilot(path)


def test_pilot_json_refuses_unknown_cell_fields_before_returning_a_plan(tmp_path):
    plan, _rows_unused = _fixture()
    path = tmp_path / "pilot.json"
    write_pilot(path, plan)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["pilot"]["cells"][0]["raw_provider_text"] = "private sentinel"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="request cell"):
        read_pilot(path)


@pytest.mark.parametrize("dataset", ("ag_news", "emotion"))
def test_the_committed_pilot_matches_the_initial_design_and_only_development_targets(dataset):
    root = Path(__file__).parents[2]
    plan = read_pilot(root / "studies" / "pilots" / f"{dataset}.plan.json")
    manifest = read_manifest(root / "studies" / "manifests" / f"{dataset}.json")
    record = json.loads((root / "studies" / "INITIAL_JEV_PREFLIGHT.json").read_text())["studies"][dataset]
    registration = (root / "studies" / "INITIAL_JEV_PILOT.md").read_text()

    assert plan.protocol_identity == record["proposed_protocol_identity"]
    assert plan.source_preflight_checksum == record["preflight_checksum"]
    assert len(plan.cells) == plan.physical_count == 21
    assert {cell.target_id for cell in plan.cells} <= {row.id for row in manifest.development}
    assert {item for cell in plan.cells for item in cell.example_ids} <= {row.id for row in manifest.candidate}
    assert all(cell.display_order == "canonical" for cell in plan.cells)
    assert all(value in registration for value in (plan.protocol_identity, plan.source_preflight_checksum, plan.checksum))
