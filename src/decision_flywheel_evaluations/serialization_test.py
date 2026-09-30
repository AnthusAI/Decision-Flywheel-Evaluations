import json
from dataclasses import replace

from .metrics import Observation
from .preflight import preflight
from .preflight_test import _protocol, _rows
from .protocol import transport_config_fingerprint
from .serialization import (read_observations, read_preflight, read_protocol, read_rows_fixture,
                            write_observations, write_preflight, write_protocol, write_rows_fixture)


def test_a_protocol_and_preflight_round_trip_without_dataset_text(tmp_path):
    manifest, rows = _rows()
    protocol = _protocol(manifest, rows)
    plan = preflight(protocol, manifest=manifest, rows=rows)
    protocol_path = tmp_path / "protocol.json"
    preflight_path = tmp_path / "preflight.json"
    rows_path = tmp_path / "rows.fixture.json"

    write_protocol(protocol_path, protocol)
    write_preflight(preflight_path, plan)
    write_rows_fixture(rows_path, rows)

    assert read_protocol(protocol_path) == protocol
    assert read_preflight(preflight_path) == plan
    assert read_rows_fixture(rows_path) == rows
    metadata = protocol_path.read_text() + preflight_path.read_text()
    assert rows[0].text not in metadata
    assert "\"schema\":\"decision-flywheel-evaluations/preflight/v1\"" in preflight_path.read_text()
    assert json.loads(rows_path.read_text())["schema"] == "decision-flywheel-evaluations/rows-fixture/v1"


def test_protocol_loader_rejects_an_unknown_or_tampered_schema_before_use(tmp_path):
    manifest, rows = _rows()
    path = tmp_path / "protocol.json"
    write_protocol(path, _protocol(manifest, rows))
    document = json.loads(path.read_text())
    document["schema"] = "unknown"
    path.write_text(json.dumps(document), encoding="utf-8")

    try:
        read_protocol(path)
    except ValueError as error:
        assert "schema" in str(error)
    else:
        raise AssertionError("unknown protocol serialization was accepted")


def test_protocol_serialization_preserves_the_public_transport_fingerprint(tmp_path):
    manifest, rows = _rows()
    protocol = _protocol(manifest, rows, artifact=False)
    transport = transport_config_fingerprint(base_url="https://gateway.example/v1", timeout_seconds=10,
                                             retries=0, adapter_revision="jev-adapter-1")
    protocol = replace(protocol, models=(replace(protocol.models[0], transport_fingerprint=transport), protocol.models[1]))
    path = tmp_path / "protocol.json"
    write_protocol(path, protocol)

    assert read_protocol(path).models[0].transport_fingerprint == transport


def test_observation_confidence_round_trips_and_legacy_rows_default_to_none(tmp_path):
    path = tmp_path / "observations.json"
    row = Observation("r", "t", "c", 0, "canonical", "yes", "yes", "completed",
                      {"yes": .9, "no": .1}, confidence=.37)
    write_observations(path, (row,))

    assert read_observations(path) == (row,)
    document = json.loads(path.read_text())
    del document["observations"][0]["confidence"]
    path.write_text(json.dumps(document), encoding="utf-8")
    assert read_observations(path)[0].confidence is None


def test_observation_reader_rejects_unknown_or_invalid_confidence_fields(tmp_path):
    path = tmp_path / "observations.json"
    row = Observation("r", "t", "c", 0, "canonical", "yes", "yes", "completed")
    write_observations(path, (row,))
    document = json.loads(path.read_text())
    document["observations"][0]["confidence"] = True
    path.write_text(json.dumps(document), encoding="utf-8")
    try:
        read_observations(path)
    except ValueError as error:
        assert "confidence" in str(error)
    else:
        raise AssertionError("boolean confidence was accepted")
    document["observations"][0]["confidence"] = .37
    document["observations"][0]["unexpected"] = "provider payload"
    path.write_text(json.dumps(document), encoding="utf-8")
    try:
        read_observations(path)
    except ValueError as error:
        assert "fields" in str(error)
    else:
        raise AssertionError("unknown observation payload field was accepted")
