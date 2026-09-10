"""Direct OpenAI-API inference provider (BeliefLens adapter).

Subclasses the OpenRouter provider -- OpenAI's /v1/chat/completions is
wire-compatible -- and adjusts the payload for OpenAI's stricter schema:
no OpenRouter-only fields, no `temperature`/`top_k` (rejected by gpt-5
reasoning endpoints), and `reasoning: {effort}` mapped to `reasoning_effort`.
"""

import os
from typing import Any, Dict, List

from .openrouter import OpenRouterInference


class OpenAIDirectInference(OpenRouterInference):
    API_URL = "https://api.openai.com/v1/chat/completions"

    def __init__(self, model: str, api_key: str = None, **kwargs):
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY required for provider=openai")
        super().__init__(model, api_key=key, **kwargs)

    def _build_payload(
        self,
        messages: List[Dict[str, Any]],
        sampling_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        payload = super()._build_payload(messages, sampling_params)
        payload.pop("usage", None)        # OpenRouter-only envelope field
        payload.pop("provider", None)     # OpenRouter routing config
        payload.pop("temperature", None)  # rejected by gpt-5 reasoning models
        payload.pop("top_k", None)        # not an OpenAI parameter
        reasoning = payload.pop("reasoning", None)
        if isinstance(reasoning, dict) and reasoning.get("effort"):
            payload["reasoning_effort"] = reasoning["effort"]
        return payload
