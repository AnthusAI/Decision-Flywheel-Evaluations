"""LiteLLM does not yet recognise the gpt-6 family, so it sends ``max_tokens`` (rejected by
these models) instead of ``max_completion_tokens``. Treat gpt-6 names like gpt-5 for that
mapping, and omit Tactus's default ``temperature`` (these models reject 0.0); every other model
name behaves exactly as before."""
from __future__ import annotations


def allow_gpt6_token_parameter() -> None:
    from litellm.llms.openai.chat.gpt_5_transformation import OpenAIGPT5Config

    if getattr(OpenAIGPT5Config, "_unified_gpt6_patched", False):
        return
    original = OpenAIGPT5Config.is_model_gpt_5_model.__func__

    def is_model_gpt_5_model(cls, model: str) -> bool:
        return original(cls, model) or model.split("/")[-1].startswith("gpt-6")

    OpenAIGPT5Config.is_model_gpt_5_model = classmethod(is_model_gpt_5_model)
    OpenAIGPT5Config._unified_gpt6_patched = True

    try:
        import tactus.dspy.config as dspy_config
        import tactus.model_params as model_params
    except ImportError:
        return
    for module in (dspy_config, model_params):
        default = module.default_temperature_for_model
        module.default_temperature_for_model = (
            lambda model, _default=default: None if str(model).split("/")[-1].startswith("gpt-6")
            else _default(model))
