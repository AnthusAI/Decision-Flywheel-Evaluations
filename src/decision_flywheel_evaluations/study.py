"""Pure study-plan structures; no model clients, data downloads, or secrets."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping


class Engine(str, Enum):
    JEV = "jev"
    KEV = "kev"
    LAYA = "laya"


@dataclass(frozen=True)
class SplitManifest:
    dataset: str
    revision: str
    candidate_ids: frozenset[str]
    development_ids: frozenset[str]
    scoreboard_ids: frozenset[str]

    def validate(self) -> None:
        if not self.scoreboard_ids:
            raise ValueError("a study needs a protected scoreboard")
        groups = (self.candidate_ids, self.development_ids, self.scoreboard_ids)
        if any(left & right for index, left in enumerate(groups) for right in groups[index + 1:]):
            raise ValueError("candidate, development, and scoreboard IDs must be disjoint")


@dataclass(frozen=True)
class StudyPlan:
    name: str
    manifest: SplitManifest
    engines: tuple[Engine, ...]
    policies: tuple[str, ...]
    context_budget: int
    primary_metric: str
    request_ceiling: int
    task_fingerprint: str
    engine_exclusions: Mapping[Engine, str]

    def validate(self) -> None:
        self.manifest.validate()
        if not self.engines: raise ValueError("a study needs at least one engine")
        if not self.policies: raise ValueError("a study needs at least one policy")
        if self.context_budget < 0: raise ValueError("context budget cannot be negative")
        if self.request_ceiling <= 0: raise ValueError("request ceiling must be positive")
        if not self.task_fingerprint: raise ValueError("freeze the task before collection")
        if set(self.engine_exclusions) - set(Engine): raise ValueError("unknown engine exclusion")
