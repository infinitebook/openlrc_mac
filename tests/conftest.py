#  Copyright (C) 2026. Hao Zheng
#  All rights reserved.

"""Shared test configuration.

All values can be overridden via environment variables so that developers
can point tests at a local vLLM instance or use different models without
editing source code.

Environment variables:
    OPENLRC_TEST_LLM_BASE_URL   LLM API base URL (default: OpenRouter)
    OPENLRC_TEST_LLM_API_KEY    LLM API key
    OPENLRC_TEST_MODEL_GPT      Model name for GPT tests
    OPENLRC_TEST_MODEL_CLAUDE   Model name for Claude tests
    OPENLRC_TEST_MODEL_GEMINI   Model name for Gemini tests
    OPENLRC_TEST_LIVE_API       Set to "1" to enable live API tests
    OPENLRC_TEST_STRESS         Set to "1" to enable stress tests
"""

import os

from openlrc.models import ModelConfig, ModelProvider

TEST_LLM_BASE_URL = os.environ.get("OPENLRC_TEST_LLM_BASE_URL", "https://openrouter.ai/api/v1")
TEST_LLM_API_KEY = os.environ.get("OPENLRC_TEST_LLM_API_KEY")

_MODEL_NAMES = {
    "gpt": os.environ.get("OPENLRC_TEST_MODEL_GPT", "openai/gpt-5-nano"),
    "claude": os.environ.get("OPENLRC_TEST_MODEL_CLAUDE", "anthropic/claude-haiku-4.5"),
    "gemini": os.environ.get("OPENLRC_TEST_MODEL_GEMINI", "google/gemini-2.5-flash-lite"),
}

TEST_MODELS: dict[str, ModelConfig] = {
    key: ModelConfig(
        provider=ModelProvider.OPENAI,
        name=name,
        base_url=TEST_LLM_BASE_URL,
        api_key=TEST_LLM_API_KEY,
    )
    for key, name in _MODEL_NAMES.items()
}

LIVE_API = os.environ.get("OPENLRC_TEST_LIVE_API", "").lower() in ("1", "true", "yes")
STRESS_TEST = os.environ.get("OPENLRC_TEST_STRESS", "").lower() in ("1", "true", "yes")
