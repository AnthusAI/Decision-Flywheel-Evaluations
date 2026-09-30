"""Offline installed-package provenance checks for the opt-in live JEV adapter.

Only package metadata is inspected here: this module never imports a provider,
reads an environment variable, or makes a network request.  Editable core
installs are intentionally rejected because their clean Git state cannot be
established solely from installed metadata; live collection requires a pinned
PEP 610 Git installation at the recorded commit.  Ordinary wheels do not carry
the required commit provenance and are rejected as unavailable provenance.
"""
from __future__ import annotations

import json
import re
from importlib import metadata
from typing import Callable


_SHA = re.compile(r"^[0-9a-f]{40}$")
_SDK_REVISION = re.compile(r"^typesafe-sdk-(\d+\.\d+\.\d+)$")


def validate_jev_runtime(
    core_revision: str,
    package_revision: str,
    *,
    package_version: Callable[[str], str] = metadata.version,
    distribution: Callable[[str], metadata.Distribution] = metadata.distribution,
) -> None:
    """Require the installed SDK and core package to match frozen provenance."""
    match = _SDK_REVISION.fullmatch(package_revision) if isinstance(package_revision, str) else None
    if not isinstance(core_revision, str) or not _SHA.fullmatch(core_revision) or match is None:
        raise ValueError("live JEV requires explicit pinned core and typesafe-sdk revisions")
    try:
        installed_sdk = package_version("typesafe-sdk")
    except Exception as error:
        raise ValueError("installed typesafe-sdk provenance is unavailable") from error
    if installed_sdk != match.group(1):
        raise ValueError("installed typesafe-sdk version does not match the frozen package revision")
    try:
        direct_url_text = distribution("decision-flywheel").read_text("direct_url.json")
        direct_url = json.loads(direct_url_text) if isinstance(direct_url_text, str) else None
    except Exception as error:
        raise ValueError("installed decision-flywheel provenance is unavailable") from error
    if not isinstance(direct_url, dict):
        raise ValueError("installed decision-flywheel provenance is unavailable")
    if isinstance(direct_url.get("dir_info"), dict) and direct_url["dir_info"].get("editable") is True:
        raise ValueError("live JEV requires a pinned PEP 610 Git decision-flywheel install, not an editable install")
    vcs_info = direct_url.get("vcs_info")
    if (not isinstance(vcs_info, dict) or vcs_info.get("vcs") != "git"
            or vcs_info.get("commit_id") != core_revision
            or vcs_info.get("requested_revision") != core_revision):
        raise ValueError("installed decision-flywheel VCS provenance does not match the frozen core revision")
