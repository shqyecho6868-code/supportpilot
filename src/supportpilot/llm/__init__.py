from supportpilot.config import Settings
from supportpilot.llm.base import INTENTS, Intent, LLMClient, LLMTurn, TextCallback


def build_llm(settings: Settings) -> LLMClient:
    if settings.llm_provider == "rule_based":
        from supportpilot.llm.rule_based import RuleBasedLLM

        return RuleBasedLLM()
    from supportpilot.llm.anthropic_llm import AnthropicLLM

    return AnthropicLLM(
        model=settings.anthropic_model,
        api_key=settings.anthropic_api_key,
        fast_model=settings.fast_model,
        fallback_model=settings.fallback_model,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
        retry_base_delay=settings.llm_retry_base_delay,
        retry_max_delay=settings.llm_retry_max_delay,
    )


__all__ = ["INTENTS", "Intent", "LLMClient", "LLMTurn", "TextCallback", "build_llm"]
