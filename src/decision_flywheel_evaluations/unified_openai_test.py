import pytest

litellm = pytest.importorskip("litellm")

from .unified_openai import allow_gpt6_token_parameter


def _mapped(model):
    return litellm.get_optional_params(model=model, custom_llm_provider="openai", max_tokens=123)


def test_gpt6_models_use_max_completion_tokens_once_the_shim_is_applied():
    allow_gpt6_token_parameter()
    params = _mapped("gpt-6-luna")
    assert params.get("max_completion_tokens") == 123
    assert "max_tokens" not in params


def test_the_shim_leaves_other_model_names_untouched_and_is_idempotent():
    allow_gpt6_token_parameter()
    allow_gpt6_token_parameter()
    assert _mapped("gpt-4o").get("max_tokens") == 123
    assert _mapped("gpt-5").get("max_completion_tokens") == 123


def test_the_shim_omits_the_default_temperature_for_gpt6_only():
    pytest.importorskip("tactus")
    import tactus.dspy.config as dspy_config
    import tactus.model_params as model_params

    reference = dspy_config.default_temperature_for_model("gpt-4o")
    allow_gpt6_token_parameter()
    allow_gpt6_token_parameter()
    for module in (dspy_config, model_params):
        assert module.default_temperature_for_model("gpt-6-luna") is None
        assert module.default_temperature_for_model("gpt-4o") == reference
