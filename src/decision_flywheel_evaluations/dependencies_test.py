"""Packaging dependency specifications."""

from importlib.metadata import requires


def test_yaml_bundle_serialization_declares_its_runtime_dependency():
    dependencies = requires("decision-flywheel-evaluations") or []

    assert any(dependency.lower().startswith("pyyaml") and "extra ==" not in dependency.lower()
               for dependency in dependencies)
