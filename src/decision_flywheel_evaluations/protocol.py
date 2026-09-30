"""Frozen, typed contracts for offline Decision Flywheel preflight."""
from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping
from urllib.parse import urlsplit, urlunsplit

from decision_flywheel.artifacts import ArtifactValidationError, create_artifact, load_artifact
from decision_flywheel.budget import ContextBudget, build_context_plan
from decision_flywheel.context import RandomBalanced
from decision_flywheel.models import Item, LabeledItem
from decision_flywheel.models import DecisionTask
from decision_flywheel.optimizer import OptimizationResult

from .datasets import AG_NEWS, EMOTION
from .study import Engine

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ORDERS = {"canonical", "reversed", "interleaved", "shuffled"}
_TRANSPORT_REVISION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,127}$")


def transport_config_fingerprint(*, base_url: str, timeout_seconds: float, retries: int,
                                 adapter_revision: str | None = None,
                                 package_revision: str | None = None) -> str:
    """Hash only allowlisted, public transport settings; credentials never enter provenance."""
    if not isinstance(base_url, str) or not base_url or base_url != base_url.strip():
        raise ValueError("transport base URL is invalid")
    try:
        parsed = urlsplit(base_url)
        port = parsed.port
    except ValueError as error:
        raise ValueError("transport base URL is invalid") from error
    if (parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname or parsed.username is not None
            or parsed.password is not None or parsed.query or parsed.fragment):
        raise ValueError("transport base URL is invalid")
    if (not isinstance(timeout_seconds, (int, float)) or isinstance(timeout_seconds, bool)
            or not math.isfinite(float(timeout_seconds)) or float(timeout_seconds) <= 0
            or not isinstance(retries, int) or isinstance(retries, bool) or retries < 0):
        raise ValueError("transport timeout or retry settings are invalid")
    for revision in (adapter_revision, package_revision):
        if revision is not None and (not isinstance(revision, str) or not _TRANSPORT_REVISION.fullmatch(revision)):
            raise ValueError("transport revision is invalid")
    hostname = parsed.hostname.lower()
    host = f"[{hostname}]" if ":" in hostname else hostname
    netloc = host if port is None else f"{host}:{port}"
    canonical_url = urlunsplit((parsed.scheme.lower(), netloc, parsed.path.rstrip("/"), "", ""))
    payload = {"adapter_revision": adapter_revision, "base_url": canonical_url,
               "package_revision": package_revision, "retries": retries,
               "timeout_seconds": float(timeout_seconds)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _split_fingerprint(task: DecisionTask, rows: Iterable[LabeledItem]) -> str:
    """Match the core optimizer's text-free split fingerprint exactly."""
    records = []
    for row in rows:
        text = row.item.values.get(task.input_field)
        if not isinstance(text, str):
            raise ValueError("optimizer provenance item lacks the frozen task input")
        normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
        text_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        records.append((row.item.id, task.validate_label(row.label), text_hash))
    return hashlib.sha256(json.dumps(sorted(records), ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def _selected_global_anchor(task: DecisionTask, candidates: Iterable[LabeledItem]) -> Item:
    values = {candidate.item.values.get(task.input_field) for candidate in candidates}
    index = 0
    while True:
        text = f"__selected_global_validation_anchor_{index}__"
        if text not in values:
            return Item(f"__selected_global_validation_anchor_{index}__", {task.input_field: text})
        index += 1


def _policy_seed(name: str, configuration: object) -> int | None:
    if name != "random-balanced":
        if isinstance(configuration, Mapping) and not configuration:
            return None
        raise ValueError("selected-global artifact policy configuration is unsupported")
    if not isinstance(configuration, Mapping) or set(configuration) != {"seed"}:
        raise ValueError("selected-global artifact random policy configuration is invalid")
    seed = configuration["seed"]
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("selected-global artifact random policy seed is invalid")
    return seed


def _core_search_fingerprint(protocol: "FrozenProtocol", model_fingerprint: str,
                             candidate_pool_fingerprint: str, development_split_fingerprint: str) -> str:
    """Mirror the core optimizer's documented provenance schema exactly."""
    trials = []
    for size in protocol.optimization.per_label_sizes:
        for seed in protocol.optimization.draw_seeds:
            policy = RandomBalanced(seed)
            budget = ContextBudget(per_label=size)
            trials.append({"name": f"{policy.name}:{policy.fingerprint[:12]}:{size}",
                           "policy_fingerprint": policy.fingerprint, "per_label": size,
                           "budget": {"max_tokens": budget.max_tokens, "per_label": budget.per_label,
                                      "provider_token_limit": budget.provider_token_limit}})
    payload = {"candidate_pool_fingerprint": candidate_pool_fingerprint, "declared_trials": trials,
               "development_split_fingerprint": development_split_fingerprint,
               "display_order": protocol.display_rule, "max_model_calls": protocol.optimization.max_model_calls,
               "model_fingerprint": model_fingerprint, "objective": _optimizer_objective(protocol.metric), "order_seed": 0,
               "presentation_label_order": protocol.task.labels, "task_fingerprint": protocol.task.fingerprint}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def _optimizer_objective(metric: str) -> str:
    return "macro-f1" if metric == "macro_f1" else metric


def _core_decision_fingerprint(model_fingerprint: str, task: DecisionTask, plan: object) -> str:
    """Mirror the optimizer's text-free checkpoint key for a completed decision."""
    payload = {"model_fingerprint": model_fingerprint, "task_fingerprint": task.fingerprint,
               "request_fingerprint": plan.token_accounting.serialized_request_fingerprint,
               "context_ids": plan.example_ids, "display_order": plan.display_order,
               "order_seed": plan.order_seed, "presentation_label_order": plan.presentation_label_order}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def _well_formed_trial_record(record: object) -> bool:
    if not isinstance(record, CompletedTrialRecord):
        return False
    if (not all(isinstance(value, str) and value for value in
                (record.trial_name, record.policy_name, record.policy_version, record.selection_mode, record.policy_fingerprint))
            or not _SHA256.fullmatch(record.policy_fingerprint)
            or not isinstance(record.policy_configuration, Mapping)
            or not isinstance(record.per_label, int) or isinstance(record.per_label, bool) or record.per_label < 0
            or not isinstance(record.order_seed, int) or isinstance(record.order_seed, bool) or record.order_seed < 0
            or record.display_order not in _ORDERS or record.status != "completed"
            or not isinstance(record.objective, (int, float)) or isinstance(record.objective, bool)
            or not math.isfinite(record.objective) or not isinstance(record.presentation_label_order, tuple)
            or not isinstance(record.decision_target_ids, tuple)
            or not isinstance(record.decision_request_fingerprints, tuple)
            or not isinstance(record.decision_predictions, tuple)
            or not _SHA256.fullmatch(record.development_split_fingerprint)
            or len(record.decision_target_ids) != len(record.decision_request_fingerprints)
            or len(record.decision_target_ids) != len(record.decision_predictions)):
        return False
    for value in (record.max_tokens, record.provider_token_limit):
        if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
            return False
    return (all(isinstance(value, str) and value for value in record.presentation_label_order)
            and all(isinstance(value, str) and value for value in record.decision_target_ids)
            and len(set(record.decision_target_ids)) == len(record.decision_target_ids)
            and all(isinstance(value, str) and _SHA256.fullmatch(value)
                    for value in record.decision_request_fingerprints)
            and all(isinstance(value, str) and value for value in record.decision_predictions))


def _development_score(task: DecisionTask, development: tuple[LabeledItem, ...], predictions: tuple[str, ...],
                       objective: str) -> float:
    actual = tuple(task.validate_label(item.label) for item in development)
    if objective == "accuracy":
        return sum(expected == observed for expected, observed in zip(actual, predictions)) / len(actual)
    if objective != "macro-f1":
        raise ValueError("selected-global artifact has an unsupported optimizer objective")
    f1s = []
    for label in task.labels:
        true_positive = sum(expected == label and observed == label for expected, observed in zip(actual, predictions))
        false_positive = sum(expected != label and observed == label for expected, observed in zip(actual, predictions))
        false_negative = sum(expected == label and observed != label for expected, observed in zip(actual, predictions))
        denominator = 2 * true_positive + false_positive + false_negative
        f1s.append(2 * true_positive / denominator if denominator else 0.0)
    return sum(f1s) / len(f1s)


class Capability(str, Enum):
    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"


class Selector(str, Enum):
    ZERO = "zero"
    RANDOM = "random"
    DEVELOPMENT_SELECTED_GLOBAL = "development_selected_global"
    PROTOTYPE = "prototype"
    RETRIEVAL = "retrieval"


@dataclass(frozen=True)
class ModelIdentity:
    engine: Engine
    version: str
    capabilities: frozenset[Capability]
    max_per_label: int
    transport_fingerprint: str = ""

    def validate(self) -> None:
        if not isinstance(self.engine, Engine) or not isinstance(self.version, str) or not self.version:
            raise ValueError("models need a typed engine and versioned identity")
        if not self.capabilities or any(not isinstance(item, Capability) for item in self.capabilities):
            raise ValueError("models need explicit capabilities")
        if not isinstance(self.max_per_label, int) or isinstance(self.max_per_label, bool) or self.max_per_label < 0:
            raise ValueError("model context limit must be a nonnegative integer")
        if not isinstance(self.transport_fingerprint, str) or (self.transport_fingerprint and not _SHA256.fullmatch(self.transport_fingerprint)):
            raise ValueError("model transport fingerprint must be empty or a SHA-256 value")

    @property
    def route_identity(self) -> str:
        self.validate()
        return f"{self.engine.value}:{self.version}"

    @property
    def semantic_identity(self) -> str:
        self.validate()
        return self.route_identity if not self.transport_fingerprint else f"{self.route_identity}@transport-{self.transport_fingerprint}"

    def supports_zero_shot(self) -> bool: return Capability.ZERO_SHOT in self.capabilities
    def supports_few_shot(self, per_label: int) -> bool: return Capability.FEW_SHOT in self.capabilities and per_label <= self.max_per_label


@dataclass(frozen=True)
class CompletedTrialRecord:
    """Text-free copy of one completed core optimizer trial and its development requests."""

    trial_name: str
    policy_name: str
    policy_version: str
    selection_mode: str
    policy_configuration: Mapping[str, object]
    policy_fingerprint: str
    per_label: int
    max_tokens: int | None
    provider_token_limit: int | None
    display_order: str
    order_seed: int
    presentation_label_order: tuple[str, ...]
    status: str
    objective: float
    development_split_fingerprint: str
    decision_target_ids: tuple[str, ...]
    decision_request_fingerprints: tuple[str, ...]
    decision_predictions: tuple[str, ...]

    @classmethod
    def from_core(cls, trial: object, development_split_fingerprint: str) -> "CompletedTrialRecord":
        metadata = trial.policy_metadata
        return cls(trial.trial_name, metadata.name, metadata.version, metadata.selection_mode,
                   dict(metadata.configuration), trial.policy_fingerprint, trial.per_label,
                   trial.budget.max_tokens, trial.budget.provider_token_limit, trial.display_order,
                   trial.order_seed, tuple(trial.presentation_label_order), trial.status, trial.objective,
                   development_split_fingerprint,
                   tuple(item.target_id for item in trial.decisions),
                   tuple(item.request_fingerprint for item in trial.decisions),
                   tuple(item.prediction for item in trial.decisions))

    def payload(self) -> dict[str, object]:
        return {"trial_name": self.trial_name, "policy_name": self.policy_name,
                "policy_version": self.policy_version, "selection_mode": self.selection_mode,
                "policy_configuration": dict(self.policy_configuration),
                "policy_fingerprint": self.policy_fingerprint, "per_label": self.per_label,
                "max_tokens": self.max_tokens, "provider_token_limit": self.provider_token_limit,
                "display_order": self.display_order, "order_seed": self.order_seed,
                "presentation_label_order": list(self.presentation_label_order), "status": self.status,
                "objective": self.objective, "development_split_fingerprint": self.development_split_fingerprint,
                "decision_target_ids": list(self.decision_target_ids),
                "decision_request_fingerprints": list(self.decision_request_fingerprints),
                "decision_predictions": list(self.decision_predictions)}


@dataclass(frozen=True)
class SelectedGlobalArtifact:
    """An integrity-checked record derived from a complete core optimizer winner.

    Its checksums detect accidental corruption; neither they nor the core
    artifact hash authenticate an author.
    """
    reference: str
    checksum: str
    task_fingerprint: str
    candidate_pool_fingerprint: str
    development_split_fingerprint: str
    search_fingerprint: str
    objective: str
    objective_value: float
    model_fingerprint: str
    winner_trial_name: str
    policy_name: str
    policy_fingerprint: str
    selection_seed: int | None
    completed_trial_names: tuple[str, ...]
    completed_trials_fingerprint: str
    completed_trials: tuple[CompletedTrialRecord, ...]
    context_policy_artifact: str
    ids_by_size: Mapping[int, tuple[str, ...]]

    @classmethod
    def from_optimization(cls, reference: str, task: DecisionTask, candidates: Iterable[LabeledItem],
                          pool_revision: str, optimization: OptimizationResult) -> "SelectedGlobalArtifact":
        """Freeze the core winner and derive every ladder membership from its policy."""
        pool = tuple(candidates)
        if not isinstance(optimization, OptimizationResult):
            raise ValueError("selected-global artifact needs a core OptimizationResult")
        if optimization.winner is None or any(trial.status != "completed" for trial in optimization.trials):
            raise ValueError("selected-global artifact requires every declared optimizer trial to complete")
        try:
            serialized = create_artifact(task, pool, pool_revision, optimization)
            deployed = load_artifact(serialized, task, pool, pool_revision)
        except ArtifactValidationError as error:
            raise ValueError("selected-global artifact cannot bind the core optimizer winner") from error
        metadata = deployed.policy.metadata
        if metadata.selection_mode != "fixed-global":
            raise ValueError("selected-global winner must use a fixed-global core context policy")
        seed = _policy_seed(metadata.name, metadata.configuration)
        anchor = _selected_global_anchor(task, pool)
        ids_by_size = {
            size: tuple(item.item.id for item in deployed.policy.select(task, anchor, pool, per_label=size))
            for size in (1, 4, 16, 64)
        }
        completed_trials = tuple(CompletedTrialRecord.from_core(trial, optimization.development_split_fingerprint)
                                 for trial in optimization.trials)
        completed_names = tuple(trial.trial_name for trial in completed_trials)
        trial_fingerprint = cls._trials_fingerprint(completed_trials)
        checksum = cls._checksum(reference, task.fingerprint, optimization.candidate_pool_fingerprint,
                                 optimization.development_split_fingerprint, optimization.search_fingerprint,
                                 optimization.objective, optimization.winner.objective, optimization.model_fingerprint,
                                 optimization.winner.trial_name, metadata.name, deployed.policy.fingerprint, seed,
                                 completed_names, trial_fingerprint, completed_trials, serialized, ids_by_size)
        return cls(reference, checksum, task.fingerprint, optimization.candidate_pool_fingerprint,
                   optimization.development_split_fingerprint, optimization.search_fingerprint,
                   optimization.objective, optimization.winner.objective, optimization.model_fingerprint,
                   optimization.winner.trial_name, metadata.name, deployed.policy.fingerprint, seed,
                   completed_names, trial_fingerprint, completed_trials, serialized, ids_by_size)

    @staticmethod
    def _checksum(reference: str, task_fingerprint: str, candidate_pool_fingerprint: str,
                  development_split_fingerprint: str, search_fingerprint: str, objective: str,
                  objective_value: float, model_fingerprint: str, winner_trial_name: str,
                  policy_name: str, policy_fingerprint: str, selection_seed: int | None,
                  completed_trial_names: tuple[str, ...], completed_trials_fingerprint: str,
                  completed_trials: tuple[CompletedTrialRecord, ...], context_policy_artifact: str,
                  ids_by_size: Mapping[int, tuple[str, ...]]) -> str:
        payload = {"reference": reference, "task_fingerprint": task_fingerprint,
                   "candidate_pool_fingerprint": candidate_pool_fingerprint,
                   "development_split_fingerprint": development_split_fingerprint,
                   "search_fingerprint": search_fingerprint, "objective": objective,
                   "objective_value": objective_value, "model_fingerprint": model_fingerprint,
                   "winner_trial_name": winner_trial_name, "policy_name": policy_name,
                   "policy_fingerprint": policy_fingerprint, "selection_seed": selection_seed,
                   "completed_trial_names": list(completed_trial_names),
                   "completed_trials_fingerprint": completed_trials_fingerprint,
                   "completed_trials": [trial.payload() for trial in completed_trials],
                   "context_policy_artifact": context_policy_artifact,
                   "ids_by_size": {str(size): list(ids) for size, ids in sorted(ids_by_size.items())},
                   }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    @staticmethod
    def _trials_fingerprint(trials: Iterable[CompletedTrialRecord]) -> str:
        return hashlib.sha256(json.dumps([trial.payload() for trial in trials], sort_keys=True,
                                         separators=(",", ":")).encode()).hexdigest()

    def payload(self) -> dict[str, object]:
        return {"reference": self.reference, "checksum": self.checksum, "task_fingerprint": self.task_fingerprint,
                "candidate_pool_fingerprint": self.candidate_pool_fingerprint,
                "development_split_fingerprint": self.development_split_fingerprint,
                "search_fingerprint": self.search_fingerprint, "objective": self.objective,
                "objective_value": self.objective_value, "model_fingerprint": self.model_fingerprint,
                "winner_trial_name": self.winner_trial_name, "policy_name": self.policy_name,
                "policy_fingerprint": self.policy_fingerprint, "selection_seed": self.selection_seed,
                "completed_trial_names": list(self.completed_trial_names),
                "completed_trials_fingerprint": self.completed_trials_fingerprint,
                "completed_trials": [trial.payload() for trial in self.completed_trials],
                "context_policy_artifact": self.context_policy_artifact,
                "ids_by_size": {str(size): list(ids) for size, ids in sorted(self.ids_by_size.items())},
                }

    def validate(self, task: DecisionTask) -> None:
        if not isinstance(self.reference, str) or not self.reference or not _SHA256.fullmatch(self.checksum):
            raise ValueError("selected-global artifact needs a reference and checksum")
        if self.task_fingerprint != task.fingerprint or not _SHA256.fullmatch(self.candidate_pool_fingerprint):
            raise ValueError("selected-global artifact must match the frozen task and candidate pool")
        if (not _SHA256.fullmatch(self.development_split_fingerprint) or not _SHA256.fullmatch(self.search_fingerprint)
                or not _SHA256.fullmatch(self.policy_fingerprint) or not _SHA256.fullmatch(self.completed_trials_fingerprint)
                or self.objective not in {"accuracy", "macro-f1"} or not isinstance(self.objective_value, (int, float))
                or isinstance(self.objective_value, bool) or not math.isfinite(self.objective_value)
                or not isinstance(self.model_fingerprint, str) or not self.model_fingerprint
                or not isinstance(self.winner_trial_name, str) or not self.winner_trial_name
                or not isinstance(self.policy_name, str) or not self.policy_name
                or not isinstance(self.context_policy_artifact, str) or not self.context_policy_artifact
                or not isinstance(self.completed_trial_names, tuple) or not self.completed_trial_names
                or len(set(self.completed_trial_names)) != len(self.completed_trial_names)
                or self.winner_trial_name not in self.completed_trial_names
                or not isinstance(self.completed_trials, tuple) or not self.completed_trials
                or (self.selection_seed is not None and (not isinstance(self.selection_seed, int)
                                                         or isinstance(self.selection_seed, bool)
                                                         or self.selection_seed < 0))
                or any(not isinstance(name, str) or not name for name in self.completed_trial_names)):
            raise ValueError("selected-global artifact has incomplete optimizer provenance")
        if any(not _well_formed_trial_record(record) for record in self.completed_trials):
            raise ValueError("selected-global artifact has invalid completed trial records")
        if self.checksum != self._checksum(self.reference, self.task_fingerprint, self.candidate_pool_fingerprint,
                                           self.development_split_fingerprint, self.search_fingerprint, self.objective,
                                           self.objective_value, self.model_fingerprint, self.winner_trial_name,
                                           self.policy_name, self.policy_fingerprint, self.selection_seed,
                                           self.completed_trial_names, self.completed_trials_fingerprint,
                                           self.completed_trials, self.context_policy_artifact, self.ids_by_size):
            raise ValueError("selected-global artifact checksum does not match its frozen membership")
        if (tuple(trial.trial_name for trial in self.completed_trials) != self.completed_trial_names
                or self._trials_fingerprint(self.completed_trials) != self.completed_trials_fingerprint):
            raise ValueError("selected-global artifact completed trial records do not match their integrity fields")
        if tuple(sorted(self.ids_by_size)) != (1, 4, 16, 64):
            raise ValueError("selected-global artifact must freeze every positive ladder size")
        if any(not isinstance(ids, tuple) or any(not isinstance(identifier, str) or not identifier for identifier in ids)
               or len(ids) != size * len(task.labels) or len(set(ids)) != len(ids)
               for size, ids in self.ids_by_size.items()):
            raise ValueError("selected-global artifact has an invalid balanced membership anchor")

    def validate_for(self, protocol: "FrozenProtocol", candidates: Iterable[LabeledItem],
                     development: Iterable[LabeledItem], pool_revision: str) -> None:
        """Rehydrate and verify the actual core winner before scoreboard use."""
        task = protocol.task
        self.validate(task)
        pool, development_rows = tuple(candidates), tuple(development)
        if _split_fingerprint(task, pool) != self.candidate_pool_fingerprint:
            raise ValueError("selected-global artifact does not match rehydrated candidate pool")
        if _split_fingerprint(task, development_rows) != self.development_split_fingerprint:
            raise ValueError("selected-global artifact does not match the frozen development split")
        try:
            deployed = load_artifact(self.context_policy_artifact, task, pool, pool_revision)
        except ArtifactValidationError as error:
            raise ValueError("selected-global artifact cannot rehydrate its core context policy") from error
        document = json.loads(self.context_policy_artifact)
        metadata = deployed.policy.metadata
        if (document["development"]["objective_name"] != self.objective
                or deployed.search_fingerprint != self.search_fingerprint or deployed.development_objective != self.objective_value
                or deployed.development_split_fingerprint != self.development_split_fingerprint
                or deployed.model_fingerprint != self.model_fingerprint or deployed.trial_name != self.winner_trial_name
                or metadata.name != self.policy_name or deployed.policy.fingerprint != self.policy_fingerprint
                or _policy_seed(metadata.name, metadata.configuration) != self.selection_seed):
            raise ValueError("selected-global artifact winner provenance does not match its core context policy")
        anchor = _selected_global_anchor(task, pool)
        expected = {size: tuple(item.item.id for item in deployed.policy.select(task, anchor, pool, per_label=size))
                    for size in (1, 4, 16, 64)}
        if dict(self.ids_by_size) != expected:
            raise ValueError("selected-global artifact membership does not match its winning policy")
        self._validate_declared_search(protocol, pool, development_rows)

    def _validate_declared_search(self, protocol: "FrozenProtocol", pool: tuple[LabeledItem, ...],
                                  development: tuple[LabeledItem, ...]) -> None:
        """Match every persisted core trial and request to this protocol's declared matrix."""
        expected = []
        for size in protocol.optimization.per_label_sizes:
            for seed in protocol.optimization.draw_seeds:
                policy = RandomBalanced(seed)
                budget = ContextBudget(per_label=size)
                expected.append((policy, size, budget))
        if len(self.completed_trials) != len(expected):
            raise ValueError("selected-global artifact does not cover every declared optimizer trial")
        development_ids = tuple(item.item.id for item in development)
        for record, (policy, size, budget) in zip(self.completed_trials, expected):
            expected_name = f"{policy.name}:{policy.fingerprint[:12]}:{size}"
            fingerprints = []
            for target in development:
                plan = build_context_plan(protocol.task, target.item, pool, policy, budget=budget,
                                          display_order=protocol.display_rule, order_seed=0,
                                          presentation_label_order=protocol.task.labels)
                fingerprints.append(_core_decision_fingerprint(self.model_fingerprint, protocol.task, plan))
            if (record.trial_name != expected_name or record.policy_name != policy.metadata.name
                    or record.policy_version != policy.metadata.version
                    or record.selection_mode != policy.metadata.selection_mode
                    or dict(record.policy_configuration) != dict(policy.metadata.configuration)
                    or record.policy_fingerprint != policy.fingerprint or record.per_label != size
                    or record.max_tokens != budget.max_tokens or record.provider_token_limit != budget.provider_token_limit
                    or record.display_order != protocol.display_rule or record.order_seed != 0
                    or record.presentation_label_order != protocol.task.labels or record.status != "completed"
                    or not isinstance(record.objective, (int, float)) or isinstance(record.objective, bool)
                    or not math.isfinite(record.objective) or not 0.0 <= record.objective <= 1.0
                    or record.development_split_fingerprint != self.development_split_fingerprint
                    or record.decision_target_ids != development_ids
                    or record.decision_request_fingerprints != tuple(fingerprints)):
                raise ValueError("selected-global artifact trial record does not match the declared optimizer search")
            try:
                predictions = tuple(protocol.task.validate_label(prediction) for prediction in record.decision_predictions)
            except ValueError as error:
                raise ValueError("selected-global artifact trial record has an invalid canonical prediction") from error
            if predictions != record.decision_predictions or not math.isclose(
                    _development_score(protocol.task, development, predictions, _optimizer_objective(protocol.metric)),
                    record.objective, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("selected-global artifact trial objective does not match its trusted development predictions")
        winner = next(record for record in self.completed_trials if record.trial_name == self.winner_trial_name)
        declared_winner = max(self.completed_trials, key=lambda record: (record.objective, record.trial_name))
        if (declared_winner.trial_name != self.winner_trial_name or winner.objective != self.objective_value
                or winner.policy_name != self.policy_name
                or winner.policy_fingerprint != self.policy_fingerprint):
            raise ValueError("selected-global artifact winner does not match its completed trial record")
        expected_search = _core_search_fingerprint(protocol, self.model_fingerprint,
                                                   self.candidate_pool_fingerprint,
                                                   self.development_split_fingerprint)
        if self.search_fingerprint != expected_search:
            raise ValueError("selected-global artifact search fingerprint does not match the declared optimizer search")


@dataclass(frozen=True)
class OptimizationSpec:
    selectors: tuple[Selector, ...]
    per_label_sizes: tuple[int, ...]
    draw_seeds: tuple[int, ...]
    max_model_calls: int
    # Kept only to read older draft objects.  It is not part of the frozen plan:
    # the preflight checksum is rebuilt from actual request metadata.
    artifact_checksum: str | None = None

    def validate(self) -> None:
        if self.selectors != (Selector.RANDOM,):
            raise ValueError("optimization declares the complete RandomBalanced search matrix")
        if self.per_label_sizes != (1, 4, 16, 64) or self.draw_seeds != (0, 1, 2, 3, 4):
            raise ValueError("declare complete positive ladder and random draw seeds for optimization")
        if not isinstance(self.max_model_calls, int) or isinstance(self.max_model_calls, bool) or self.max_model_calls < 0:
            raise ValueError("declare a non-negative optimizer call ceiling")
        if self.artifact_checksum is not None and not _SHA256.fullmatch(self.artifact_checksum):
            raise ValueError("legacy optimization checksum claims must be SHA-256 values")


@dataclass(frozen=True)
class FrozenProtocol:
    name: str
    dataset: str
    dataset_manifest_sha256: str
    candidate_count: int
    development_count: int
    scoreboard_count: int
    task: DecisionTask
    models: tuple[ModelIdentity, ...]
    selectors: tuple[Selector, ...]
    per_label_sizes: tuple[int, ...]
    random_draw_seeds: tuple[int, ...]
    display_rule: str
    selector_configuration_version: str
    selector_search_artifact: SelectedGlobalArtifact | None
    optimization: OptimizationSpec
    selection_transfer_source: str | None = None
    metric: str = "macro_f1"
    primary_family_correction: str = "Holm across two primary contrasts"

    @property
    def identity(self) -> str:
        payload = self.frozen_payload()
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def frozen_payload(self) -> dict[str, object]:
        """The actual executable configuration, excluding unverified checksum claims."""
        self.validate()
        return {
            "name": self.name, "dataset": self.dataset, "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "candidate_count": self.candidate_count, "development_count": self.development_count,
            "scoreboard_count": self.scoreboard_count, "task_fingerprint": self.task.fingerprint,
            "models": [{"route_identity": item.route_identity, "semantic_identity": item.semantic_identity,
                        "transport_fingerprint": item.transport_fingerprint,
                        "capabilities": sorted(capability.value for capability in item.capabilities),
                        "max_per_label": item.max_per_label} for item in self.models],
            "selectors": [item.value for item in self.selectors], "per_label_sizes": self.per_label_sizes,
            "random_draw_seeds": self.random_draw_seeds, "display_rule": self.display_rule,
            "selector_configuration_version": self.selector_configuration_version,
            "selected_global": self.selector_search_artifact.payload() if self.selector_search_artifact else None,
            "optimization": {"selectors": [item.value for item in self.optimization.selectors],
                             "per_label_sizes": self.optimization.per_label_sizes,
                             "draw_seeds": self.optimization.draw_seeds,
                             "max_model_calls": self.optimization.max_model_calls},
            "selection_transfer_source": self.selection_transfer_source,
            "metric": self.metric, "primary_family_correction": self.primary_family_correction,
        }

    @property
    def primary_contrasts(self) -> tuple[tuple[object, ...], tuple[object, ...]]:
        return ((Selector.RETRIEVAL, Selector.RANDOM, 16), (Selector.RANDOM, Selector.RANDOM, (64, 1)))

    def validate(self) -> None:
        if not isinstance(self.name, str) or not self.name or self.dataset not in {"dair-ai/emotion", "fancyzhx/ag_news"}:
            raise ValueError("protocol needs a supported dataset and name")
        if not _SHA256.fullmatch(self.dataset_manifest_sha256): raise ValueError("protocol needs a frozen dataset manifest SHA-256")
        if not isinstance(self.task, DecisionTask): raise ValueError("protocol needs a DecisionTask")
        expected_labels = EMOTION.labels if self.dataset == EMOTION.name else AG_NEWS.labels
        if self.task.labels != expected_labels:
            raise ValueError("protocol task must use the dataset's frozen label names and order")
        for count in (self.candidate_count, self.development_count, self.scoreboard_count):
            if not isinstance(count, int) or isinstance(count, bool) or count < 0: raise ValueError("declared partition counts must be nonnegative integers")
        if self.scoreboard_count == 0: raise ValueError("declare a protected scoreboard count")
        if not self.models: raise ValueError("protocol needs models")
        for item in self.models: item.validate()
        if len({item.semantic_identity for item in self.models}) != len(self.models): raise ValueError("protocol models must be unique")
        if self.selectors != (Selector.ZERO, Selector.RANDOM, Selector.DEVELOPMENT_SELECTED_GLOBAL, Selector.PROTOTYPE, Selector.RETRIEVAL): raise ValueError("freeze all selector conditions in canonical order")
        if self.per_label_sizes != (0, 1, 4, 16, 64) or self.random_draw_seeds != (0, 1, 2, 3, 4): raise ValueError("freeze ladder sizes and random draws")
        if self.display_rule not in _ORDERS: raise ValueError("display rule must match a core display order")
        if not self.selector_configuration_version: raise ValueError("freeze selector configuration version")
        self.optimization.validate()
        if self.selector_search_artifact is not None:
            if not isinstance(self.selector_search_artifact, SelectedGlobalArtifact):
                raise ValueError("selected-global artifact must be a typed optimizer artifact")
            self.selector_search_artifact.validate(self.task)
            eligible = {model.semantic_identity for model in self.models if model.supports_few_shot(64)}
            source = self.selector_search_artifact.model_fingerprint
            if source not in eligible and self.selection_transfer_source != source:
                raise ValueError("selected-global optimizer model must be few-shot eligible or explicitly declared as a transfer source")
        if self.selection_transfer_source is not None and (not isinstance(self.selection_transfer_source, str)
                                                           or not self.selection_transfer_source):
            raise ValueError("selection transfer source must be a non-empty model identity")
        expected = "macro_f1" if self.dataset == "dair-ai/emotion" else "accuracy"
        if self.metric != expected: raise ValueError(f"{self.dataset} requires primary metric {expected}")
        if self.primary_family_correction != "Holm across two primary contrasts": raise ValueError("state the Holm correction across the two primary contrasts")


@dataclass(frozen=True)
class OrderAnchor:
    initial_result_artifact: str
    frozen_context_artifact: SelectedGlobalArtifact
    dataset_manifest_sha256: str

    def validate(self, task: DecisionTask) -> None:
        if not self.initial_result_artifact or not _SHA256.fullmatch(self.dataset_manifest_sha256): raise ValueError("a completed initial result anchor is required")
        self.frozen_context_artifact.validate(task)


@dataclass(frozen=True)
class OrderTreatment:
    display_order: str
    seed: int

    def validate(self) -> None:
        if self.display_order not in _ORDERS or not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("order treatment needs a core order and nonnegative seed")


@dataclass(frozen=True)
class OrderProtocol:
    name: str
    dataset_manifest_sha256: str
    task: DecisionTask
    anchor: OrderAnchor | None
    treatments: tuple[OrderTreatment, ...]

    def validate(self) -> None:
        if not self.name or not _SHA256.fullmatch(self.dataset_manifest_sha256): raise ValueError("order protocol needs a frozen manifest")
        if self.anchor is None: raise ValueError("a completed initial result is required before freezing order study")
        self.anchor.validate(self.task)
        if self.anchor.dataset_manifest_sha256 != self.dataset_manifest_sha256: raise ValueError("order anchor must use the frozen manifest")
        for item in self.treatments: item.validate()
        required = {"canonical", "interleaved", "reversed"}
        if not required <= {item.display_order for item in self.treatments} or sum(item.display_order == "shuffled" for item in self.treatments) < 2:
            raise ValueError("freeze canonical, interleaved, reversed, and multiple shuffled treatments")
