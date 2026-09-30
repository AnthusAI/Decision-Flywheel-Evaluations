import json
from importlib import metadata

import pytest

from .runtime_identity import validate_jev_runtime


_CORE = "6137fa185a1a98afa84b5e6d5948d1780df5d56d"


class _Distribution:
    def __init__(self, direct_url):
        self.direct_url = direct_url

    def read_text(self, name):
        assert name == "direct_url.json"
        return None if self.direct_url is None else json.dumps(self.direct_url)


def _metadata(*, sdk_version="0.7.1", direct_url=None):
    if direct_url is None:
        direct_url = {"url": "https://github.com/AnthusAI/Decision-Flywheel.git",
                      "vcs_info": {"vcs": "git", "commit_id": _CORE, "requested_revision": _CORE}}
    return (lambda name: sdk_version if name == "typesafe-sdk" else pytest.fail(f"unexpected version lookup {name}"),
            lambda name: _Distribution(direct_url) if name == "decision-flywheel" else pytest.fail(f"unexpected distribution lookup {name}"))


def test_a_pinned_typesafe_sdk_and_core_vcs_commit_pass_the_offline_runtime_gate():
    version, distribution = _metadata()

    validate_jev_runtime(_CORE, "typesafe-sdk-0.7.1", package_version=version, distribution=distribution)


def test_missing_or_mismatched_sdk_and_core_provenance_are_rejected_without_importing_a_provider():
    def missing_sdk(_name):
        raise metadata.PackageNotFoundError("typesafe-sdk")

    with pytest.raises(ValueError, match="typesafe-sdk"):
        validate_jev_runtime(_CORE, "typesafe-sdk-0.7.1", package_version=missing_sdk,
                             distribution=lambda _name: pytest.fail("core metadata must not be read without the SDK"))
    version, distribution = _metadata(sdk_version="0.7.0")
    with pytest.raises(ValueError, match="typesafe-sdk"):
        validate_jev_runtime(_CORE, "typesafe-sdk-0.7.1", package_version=version, distribution=distribution)
    version, distribution = _metadata(direct_url={"url": "https://github.com/AnthusAI/Decision-Flywheel.git",
                                                   "vcs_info": {"vcs": "git", "commit_id": "0" * 40,
                                                                "requested_revision": "0" * 40}})
    with pytest.raises(ValueError, match="decision-flywheel"):
        validate_jev_runtime(_CORE, "typesafe-sdk-0.7.1", package_version=version, distribution=distribution)
    version, distribution = _metadata(direct_url={})
    with pytest.raises(ValueError, match="decision-flywheel"):
        validate_jev_runtime(_CORE, "typesafe-sdk-0.7.1", package_version=version, distribution=distribution)


def test_an_editable_or_unknown_core_install_is_rejected_for_live_collection():
    version, distribution = _metadata(direct_url={"url": "file:///work/Decision-Flywheel", "dir_info": {"editable": True}})

    with pytest.raises(ValueError, match="pinned PEP 610 Git"):
        validate_jev_runtime(_CORE, "typesafe-sdk-0.7.1", package_version=version, distribution=distribution)
    version, _distribution = _metadata()
    with pytest.raises(ValueError, match="provenance is unavailable"):
        validate_jev_runtime(_CORE, "typesafe-sdk-0.7.1", package_version=version,
                             distribution=lambda _name: _Distribution(None))
