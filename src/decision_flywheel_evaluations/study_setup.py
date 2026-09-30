"""Offline authoring of the first typed JEV development-selection protocol."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping
from urllib.parse import urlsplit

from decision_flywheel.models import DecisionTask

from .datasets import AG_NEWS, EMOTION
from .manifests import DatasetManifest, ExposureStatus, HISTORICAL_AG_NEWS_COMMIT
from .preflight import manifest_sha256
from .protocol import Capability, FrozenProtocol, ModelIdentity, OptimizationSpec, Selector, transport_config_fingerprint
from .study import Engine


_PUBLIC_VERSION = re.compile(r"^jev-\d+\.\d+\.\d+$")
_TASKS = {
    AG_NEWS.name: ("ag-news", AG_NEWS.labels,
                    "Choose exactly one AG News topic. Use labeled_examples as examples of the intended categories. "
                    "Classify only target.text; do not classify the examples themselves.", 8_000),
    EMOTION.name: ("emotion", EMOTION.labels,
                   "Choose exactly one emotion label. Use labeled_examples as examples of the intended categories. "
                   "Classify only target.text; do not classify the examples themselves.", 12_000),
}
_EXPECTED = {
    AG_NEWS.name: {"counts": {"candidate": 2048, "development": 400, "scoreboard": 2000},
                   "exposure": ExposureStatus.CONFIRMATORY_FRESH, "candidate_per_label": 512,
                   "development_per_label": 100, "scoreboard_per_label": 500,
                   "natural_official_scoreboard": False, "history": True,
                   "revision": AG_NEWS.revision,
                   "history_fingerprint": "3418a9c273e6ba14508c29ebf911d05dae95db696bf8ab3b1b2941f9754a49d7"},
    EMOTION.name: {"counts": {"candidate": 1536, "development": 600, "scoreboard": 2000},
                   "exposure": ExposureStatus.EXPLORATORY, "candidate_per_label": 256,
                   "development_per_label": 100, "scoreboard_per_label": None,
                   "natural_official_scoreboard": True, "history": False,
                   "revision": EMOTION.revision},
}


@dataclass(frozen=True)
class PublicJevConfiguration:
    """Caller-supplied public routing identity; credentials are deliberately absent."""

    version: str
    base_url: str
    timeout_seconds: float
    adapter_revision: str
    package_revision: str
    max_per_label: int
    max_model_calls: int


def freeze_initial_jev_protocol(manifest: DatasetManifest, configuration: PublicJevConfiguration) -> tuple[FrozenProtocol, Mapping[str, object]]:
    """Create the initial unselected protocol; this performs no approval or collection."""
    _validate_manifest(manifest)
    task_name, labels, instructions, minimum_model_calls = _TASKS[manifest.dataset]
    _validate_public_configuration(configuration, minimum_model_calls)
    _validate_sdk_base_url(configuration.base_url)
    transport_fingerprint = transport_config_fingerprint(
        base_url=configuration.base_url, timeout_seconds=configuration.timeout_seconds, retries=0,
        adapter_revision=configuration.adapter_revision, package_revision=configuration.package_revision,
    )
    task = DecisionTask(task_name, labels, instructions)
    protocol = FrozenProtocol(
        name=f"{task_name}-jev-development-selection", dataset=manifest.dataset,
        dataset_manifest_sha256=manifest_sha256(manifest), candidate_count=manifest.counts["candidate"],
        development_count=manifest.counts["development"], scoreboard_count=manifest.counts["scoreboard"], task=task,
        models=(ModelIdentity(Engine.JEV, configuration.version,
                              frozenset({Capability.ZERO_SHOT, Capability.FEW_SHOT}), configuration.max_per_label,
                              transport_fingerprint),),
        selectors=(Selector.ZERO, Selector.RANDOM, Selector.DEVELOPMENT_SELECTED_GLOBAL,
                   Selector.PROTOTYPE, Selector.RETRIEVAL),
        per_label_sizes=(0, 1, 4, 16, 64), random_draw_seeds=(0, 1, 2, 3, 4),
        display_rule="canonical", selector_configuration_version="native-context-policy-v1",
        selector_search_artifact=None,
        optimization=OptimizationSpec((Selector.RANDOM,), (1, 4, 16, 64), (0, 1, 2, 3, 4),
                                      configuration.max_model_calls),
        metric="accuracy" if manifest.dataset == AG_NEWS.name else "macro_f1",
    )
    protocol.validate()
    return protocol, {"base_url": configuration.base_url, "timeout_seconds": configuration.timeout_seconds,
                      "adapter_revision": configuration.adapter_revision, "package_revision": configuration.package_revision,
                      "retries": 0, "transport_fingerprint": transport_fingerprint}


def _validate_public_configuration(configuration: PublicJevConfiguration, minimum_model_calls: int) -> None:
    if not isinstance(configuration, PublicJevConfiguration):
        raise ValueError("public JEV configuration is required")
    if not isinstance(configuration.version, str) or not _PUBLIC_VERSION.fullmatch(configuration.version):
        raise ValueError("public JEV version must use the exact jev-N.N.N form")
    if (not isinstance(configuration.adapter_revision, str) or not configuration.adapter_revision
            or not isinstance(configuration.package_revision, str) or not configuration.package_revision):
        raise ValueError("public adapter and package revisions must be explicit safe values")
    if (not isinstance(configuration.max_per_label, int) or isinstance(configuration.max_per_label, bool)
            or configuration.max_per_label < 64):
        raise ValueError("JEV capability max_per_label must cover the fixed 64-example ladder")
    if (not isinstance(configuration.max_model_calls, int) or isinstance(configuration.max_model_calls, bool)
            or configuration.max_model_calls < minimum_model_calls):
        raise ValueError("declared model-call ceiling is below the fixed initial study minimum")


def _validate_sdk_base_url(base_url: str) -> None:
    """The SDK appends its own System One route to this public base URL."""
    path = urlsplit(base_url).path.rstrip("/").casefold() if isinstance(base_url, str) else ""
    if path.endswith("/v1/systemone") or path.endswith("/systemone"):
        raise ValueError("transport base URL must be the SDK root, not the System One endpoint")


def _validate_manifest(manifest: DatasetManifest) -> None:
    if not isinstance(manifest, DatasetManifest):
        raise ValueError("a typed prepared manifest is required")
    manifest.validate()
    expected = _EXPECTED.get(manifest.dataset)
    if expected is None or manifest.revision != expected["revision"]:
        raise ValueError("manifest dataset revision is not the declared initial benchmark")
    preparation = manifest.preparation
    if preparation is None or manifest.seed != 20261001 or manifest.counts != expected["counts"]:
        raise ValueError("manifest is not the declared prepared benchmark")
    configuration = preparation.configuration
    if (manifest.exposure_status is not expected["exposure"]
            or configuration.get("candidate_per_label") != expected["candidate_per_label"]
            or configuration.get("development_per_label") != expected["development_per_label"]
            or configuration.get("scoreboard_per_label") != expected["scoreboard_per_label"]
            or configuration.get("natural_official_scoreboard") is not expected["natural_official_scoreboard"]
            or configuration.get("ladder_per_label") != 64
            or preparation.sample_counts.get("candidate") != manifest.counts["candidate"]
            or preparation.sample_counts.get("development") != manifest.counts["development"]
            or preparation.sample_counts.get("scoreboard") != manifest.counts["scoreboard"]):
        raise ValueError("manifest preparation assertions do not match the declared benchmark")
    if expected["history"]:
        if (configuration.get("historical_source_commit") != HISTORICAL_AG_NEWS_COMMIT
                or preparation.exposure_history_fingerprint != expected["history_fingerprint"]
                or preparation.sample_counts.get("official_history_excluded", 0) < 2000):
            raise ValueError("AG News manifest lacks required history/exposure provenance")
    elif configuration.get("historical_source_commit") is not None or preparation.exposure_history_fingerprint is not None:
        raise ValueError("Emotion manifest has unexpected historical exposure provenance")
