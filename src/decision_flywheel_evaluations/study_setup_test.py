import hashlib
import importlib.util
import json
import re
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from .datasets import AG_NEWS, EMOTION
from .manifests import (HISTORICAL_AG_NEWS_COMMIT, DatasetManifest, ExposureStatus,
                        ManifestRecord, PreparationMetadata, read_manifest, write_manifest)
from .protocol import Selector
from .serialization import read_protocol
from .study_setup import PublicJevConfiguration, freeze_initial_jev_protocol


def _manifest(spec):
    per_label = 512 if spec is AG_NEWS else 256
    development_per_label = 100
    scoreboard_per_label = 500 if spec is AG_NEWS else None
    scoreboard_count = 2000
    records = []
    for role, count in (("candidate", per_label * len(spec.labels)),
                        ("development", development_per_label * len(spec.labels)),
                        ("scoreboard", scoreboard_count)):
        for index in range(count):
            identifier = f"{role}-{index}"
            source_index = index + (per_label * len(spec.labels) if role == "development" else 0)
            records.append(ManifestRecord(identifier, "train" if role != "scoreboard" else "test", source_index,
                                          spec.labels[index % len(spec.labels)], hashlib.sha256(identifier.encode()).hexdigest(), role))
    configuration = {"candidate_per_label": per_label, "development_per_label": development_per_label,
                     "scoreboard_per_label": scoreboard_per_label,
                     "natural_official_scoreboard": spec is EMOTION, "ladder_per_label": 64,
                     "historical_source_commit": HISTORICAL_AG_NEWS_COMMIT if spec is AG_NEWS else None}
    preparation = PreparationMetadata(configuration,
        {"candidate": per_label * len(spec.labels), "development": development_per_label * len(spec.labels),
         "scoreboard": scoreboard_count, "official_history_excluded": 2000 if spec is AG_NEWS else 0,
         "dedup_excluded": 47 if spec is AG_NEWS else 60, "leakage_excluded": 3 if spec is AG_NEWS else 11},
        "3418a9c273e6ba14508c29ebf911d05dae95db696bf8ab3b1b2941f9754a49d7" if spec is AG_NEWS else None)
    return DatasetManifest(spec.name, spec.revision, 20261001,
                           {"candidate": per_label * len(spec.labels), "development": development_per_label * len(spec.labels),
                            "scoreboard": scoreboard_count},
                           ExposureStatus.CONFIRMATORY_FRESH if spec is AG_NEWS else ExposureStatus.EXPLORATORY,
                           tuple(records), (), preparation)


def _configuration():
    return PublicJevConfiguration("jev-1.13.0", "https://gateway.example/v1", 20.0,
                                  "jev-adapter-1", "typesafe-sdk-0.7", 64, 8_250)


def test_an_initial_jev_protocol_uses_fixed_native_templates_and_declared_selection_matrix():
    protocol, transport = freeze_initial_jev_protocol(_manifest(AG_NEWS), _configuration())

    assert protocol.task.instructions == (
        "Choose exactly one AG News topic. Use labeled_examples as examples of the intended categories. "
        "Classify only target.text; do not classify the examples themselves."
    )
    assert protocol.task.labels == AG_NEWS.labels
    assert protocol.display_rule == "canonical"
    assert protocol.per_label_sizes == (0, 1, 4, 16, 64)
    assert protocol.random_draw_seeds == (0, 1, 2, 3, 4)
    assert protocol.optimization.selectors == (Selector.RANDOM,)
    assert protocol.optimization.per_label_sizes == (1, 4, 16, 64)
    assert protocol.optimization.draw_seeds == (0, 1, 2, 3, 4)
    assert protocol.optimization.max_model_calls == 8_250
    assert protocol.models[0].transport_fingerprint == transport["transport_fingerprint"]
    assert protocol.selector_search_artifact is None


def test_an_initial_emotion_protocol_keeps_its_fixed_template_and_requires_an_explicit_ceiling():
    protocol, _ = freeze_initial_jev_protocol(
        _manifest(EMOTION),
        PublicJevConfiguration("jev-1.13.0", "https://gateway.example/v1", 20.0,
                               "jev-adapter-1", "typesafe-sdk-0.7", 64, 12_100),
    )

    assert protocol.task.instructions == (
        "Choose exactly one emotion label. Use labeled_examples as examples of the intended categories. "
        "Classify only target.text; do not classify the examples themselves."
    )
    assert protocol.task.labels == EMOTION.labels
    assert protocol.optimization.max_model_calls == 12_100
    with pytest.raises(ValueError, match="ceiling"):
        freeze_initial_jev_protocol(
            _manifest(EMOTION),
            PublicJevConfiguration("jev-1.13.0", "https://gateway.example/v1", 20.0,
                                   "jev-adapter-1", "typesafe-sdk-0.7", 64, 11_999),
        )


def test_setup_requires_an_exact_prepared_exposure_safe_manifest_and_public_nonsecret_configuration():
    manifest = _manifest(EMOTION)
    with pytest.raises(ValueError, match="prepared"):
        freeze_initial_jev_protocol(manifest.__class__(**{**manifest.__dict__, "preparation": None}), _configuration())
    with pytest.raises(ValueError, match="capability"):
        freeze_initial_jev_protocol(manifest, PublicJevConfiguration("jev-1.0.0", "https://gateway.example", 10,
                                                                       "adapter", "sdk", 63, 12_000))
    with pytest.raises(ValueError, match="version"):
        freeze_initial_jev_protocol(manifest, PublicJevConfiguration("latest", "https://gateway.example", 10,
                                                                       "adapter", "sdk", 64, 12_000))
    with pytest.raises(ValueError, match="transport") as rejected:
        freeze_initial_jev_protocol(manifest, PublicJevConfiguration("jev-1.0.0", "https://key:secret@gateway.example", 10,
                                                                       "adapter", "sdk", 64, 12_000))
    assert "secret" not in str(rejected.value)
    with pytest.raises(ValueError, match="SDK root"):
        freeze_initial_jev_protocol(manifest, PublicJevConfiguration("jev-1.0.0", "https://gateway.example/v1/systemone", 10,
                                                                       "adapter", "sdk", 64, 12_000))
    with pytest.raises(ValueError, match="revision"):
        freeze_initial_jev_protocol(replace(_manifest(AG_NEWS), revision=EMOTION.revision), _configuration())
    insufficient_history = replace(
        _manifest(AG_NEWS),
        preparation=replace(_manifest(AG_NEWS).preparation,
                            sample_counts={**_manifest(AG_NEWS).preparation.sample_counts,
                                           "official_history_excluded": 1999}),
    )
    with pytest.raises(ValueError, match="history"):
        freeze_initial_jev_protocol(insufficient_history, _configuration())


def _authoring_script():
    script_path = Path(__file__).parents[2] / "scripts" / "freeze_jev_study.py"
    specification = importlib.util.spec_from_file_location("freeze_jev_study_test", script_path)
    assert specification is not None and specification.loader is not None
    script = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(script)
    return script


def _authoring_arguments(manifest_path, output):
    return ["--manifest", str(manifest_path), "--output", str(output), "--model-version", "jev-1.13.0",
            "--timeout-seconds", "30", "--adapter-revision", "adapter-1", "--package-revision", "sdk-1",
            "--max-per-label", "64", "--max-model-calls", "8000"]


def test_the_authoring_script_never_overwrites_or_creates_output_for_a_sensitive_endpoint(tmp_path, capsys, monkeypatch):
    script = _authoring_script()
    monkeypatch.setattr(script, "_is_committed", lambda *_: True)
    manifest_path = tmp_path / "synthetic-manifest.json"
    write_manifest(manifest_path, _manifest(AG_NEWS))
    output = tmp_path / "protocol.json"
    output.write_text("must not be replaced", encoding="utf-8")
    common = _authoring_arguments(manifest_path, output)

    assert script.main([*common, "--base-url", "https://api.typesafe.ai"]) == 1
    assert output.read_text(encoding="utf-8") == "must not be replaced"
    safe_output = tmp_path / "safe-output.json"
    assert script.main([*common[:3], str(safe_output), *common[4:], "--base-url", "https://key:sentinel@gateway.example"]) == 1
    assert not safe_output.exists()
    assert "sentinel" not in capsys.readouterr().out

    output.unlink()
    assert script.main([*common, "--base-url", "https://api.typesafe.ai"]) == 0
    assert read_protocol(output).optimization.max_model_calls == 8000
    assert '"status": "authored_unapproved"' in capsys.readouterr().out


def test_the_authoring_script_rejects_an_uncommitted_manifest_before_creating_output(tmp_path, capsys, monkeypatch):
    script = _authoring_script()
    monkeypatch.setattr(script, "_is_committed", lambda *_: False)
    manifest_path = tmp_path / "synthetic-manifest.json"
    output = tmp_path / "protocol.json"
    write_manifest(manifest_path, _manifest(AG_NEWS))

    assert script.main([*_authoring_arguments(manifest_path, output), "--base-url", "https://api.typesafe.ai"]) == 1
    assert not output.exists()
    assert capsys.readouterr().out == '{"reason": "ValueError", "status": "failed"}\n'


def test_the_committed_manifest_checker_compares_the_local_head_blob_digest_without_repository_history(tmp_path, monkeypatch):
    script = _authoring_script()
    repository = tmp_path / "repository"
    manifest_path = repository / "studies" / "manifests" / "manifest.json"
    committed = b'{"text_free":true}\n'
    calls = []

    def fake_run(arguments, **_kwargs):
        calls.append(arguments)
        if arguments[1:3] == ["rev-parse", "--show-toplevel"]:
            return SimpleNamespace(stdout=f"{repository}\n".encode())
        assert arguments == ["git", "show", "HEAD:studies/manifests/manifest.json"]
        return SimpleNamespace(stdout=committed)

    monkeypatch.setattr(script.subprocess, "run", fake_run)

    assert script._is_committed(manifest_path, hashlib.sha256(committed).hexdigest()) is True
    assert script._is_committed(manifest_path, "0" * 64) is False
    assert calls[0] == ["git", "rev-parse", "--show-toplevel"]


def test_the_initial_preflight_record_matches_committed_inputs_and_internal_estimates_without_generated_preflight_caches(monkeypatch):
    root = Path(__file__).parents[2]
    original_read_text = Path.read_text

    def reject_generated_preflight(path, *args, **kwargs):
        if ".data/proposals" in str(path):
            raise AssertionError("study setup specs must not read generated proposal caches")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", reject_generated_preflight)
    record = json.loads((root / "studies" / "INITIAL_JEV_PREFLIGHT.json").read_text(encoding="utf-8"))

    assert record["status"] == "unapproved"
    assert record["permission_status"] == "not_requested"
    assert record["live_results_status"] == "none"
    assert record["inference"] == "none; descriptive paired 95% intervals"
    assert record["token_estimate"]["counter_identity"] == "whitespace-request-estimate-v1"
    transport = record["public_transport"]
    assert transport == {
        "base_url": "https://api.typesafe.ai", "core_revision": "6137fa185a1a98afa84b5e6d5948d1780df5d56d",
        "max_per_label": 64, "sdk_retries": 0, "sdk_revision": "typesafe-sdk-0.7.1",
        "timeout_seconds": 30, "transport_fingerprint": "109e9d3978f49efd37d4d10beff240396895059a4e4cb7b72ce2bef1a7c06746",
        "version": "jev-1.13.0",
    }

    references = {
        "ag_news": {
            "preflight_checksum": "8fdd7cb6613c0cf29db31dddf4f5fe70bc07bc7c6e99e7476d0175743f262432",
            "proposed_protocol_identity": "914b7e251379553f00b2040c1d6b63c68dd8c41424041f9ed7eddeb070006d63",
        },
        "emotion": {
            "preflight_checksum": "e3d09d8089264ef8e3644ec9a03bc4a6512d5f2e9545a9145e87d2e96c69d052",
            "proposed_protocol_identity": "af1d49119eda713c08ecfa72c7d45134f219e3c6ff81c728217c606d3267297e",
        },
    }
    for dataset, ceiling in (("ag_news", 8400), ("emotion", 12600)):
        study = record["studies"][dataset]
        protocol, rebuilt_transport = freeze_initial_jev_protocol(
            read_manifest(root / "studies" / "manifests" / f"{dataset}.json"),
            PublicJevConfiguration(transport["version"], transport["base_url"], transport["timeout_seconds"],
                                   transport["core_revision"], transport["sdk_revision"],
                                   transport["max_per_label"], ceiling),
        )
        assert study["proposed_protocol_identity"] == protocol.identity
        assert protocol.primary_family_correction == record["inference"]
        assert rebuilt_transport["transport_fingerprint"] == transport["transport_fingerprint"]
        assert study["proposed_protocol_identity"] == references[dataset]["proposed_protocol_identity"]
        assert study["preflight_checksum"] == references[dataset]["preflight_checksum"]
        assert re.fullmatch(r"[0-9a-f]{64}", study["preflight_checksum"])
        assert study["outer_proposed_model_call_ceiling"] == ceiling
        assert protocol.optimization.per_label_sizes == (1, 4, 16, 64)
        assert protocol.optimization.draw_seeds == (0, 1, 2, 3, 4)
        expected_count = 20 * protocol.development_count
        assert study["logical_request_count"] == study["physical_request_count"] == expected_count
        assert set(study["levels"]) == {str(size) for size in protocol.optimization.per_label_sizes}
        assert isinstance(study["estimated_request_tokens"]["total"], int)
        assert isinstance(study["estimated_request_tokens"]["max"], int)
        assert study["estimated_request_tokens"]["total"] > 0
        assert study["estimated_request_tokens"]["max"] > 0
        assert sum(level["logical_request_count"] for level in study["levels"].values()) == study["logical_request_count"]
        assert sum(level["estimated_request_tokens"]["total"] for level in study["levels"].values()) == study["estimated_request_tokens"]["total"]
        assert max(level["estimated_request_tokens"]["max"] for level in study["levels"].values()) == study["estimated_request_tokens"]["max"]
        for level, values in study["levels"].items():
            assert int(level) in protocol.optimization.per_label_sizes
            assert values["logical_request_count"] == len(protocol.optimization.draw_seeds) * protocol.development_count
            assert isinstance(values["estimated_request_tokens"]["total"], int)
            assert isinstance(values["estimated_request_tokens"]["max"], int)
            assert values["estimated_request_tokens"]["total"] >= values["estimated_request_tokens"]["max"] > 0
