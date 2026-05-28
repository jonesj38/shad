"""Tests for provider routing in Shad's LLM abstraction."""

from unittest.mock import AsyncMock

import pytest

from shad.engine.llm import LLMProvider, ModelTier
from shad.utils.config import get_settings


@pytest.mark.asyncio
async def test_openai_provider_uses_selected_tier_model(monkeypatch):
    monkeypatch.setenv("SHAD_LLM_PROVIDER", "openai")
    get_settings.cache_clear()
    provider = LLMProvider(use_claude_code=True)
    provider._complete_openai = AsyncMock(return_value=("ok", 12))

    provider._worker_override = "gpt-5.5"
    response = await provider.complete("hello", tier=ModelTier.WORKER)

    assert response == ("ok", 12)
    provider._complete_openai.assert_awaited_once()
    kwargs = provider._complete_openai.await_args.kwargs
    assert kwargs["model"] == "gpt-5.5"


@pytest.mark.asyncio
async def test_openai_compatible_provider_uses_base_url_and_model(monkeypatch):
    monkeypatch.setenv("SHAD_LLM_PROVIDER", "openai-compatible")
    monkeypatch.setenv("SHAD_OPENAI_BASE_URL", "http://127.0.0.1:18789/v1")
    get_settings.cache_clear()
    provider = LLMProvider(use_claude_code=True)
    provider._complete_openai = AsyncMock(return_value=("ok", 12))

    provider._orchestrator_override = "openai-codex/gpt-5.5"
    await provider.complete("hello", tier=ModelTier.ORCHESTRATOR)

    kwargs = provider._complete_openai.await_args.kwargs
    assert kwargs["model"] == "openai-codex/gpt-5.5"
    assert kwargs["base_url"] == "http://127.0.0.1:18789/v1"


@pytest.mark.asyncio
async def test_auto_provider_prefers_gemini_cli_for_gemini_model(monkeypatch):
    monkeypatch.setenv("SHAD_LLM_PROVIDER", "auto")
    get_settings.cache_clear()
    provider = LLMProvider(use_claude_code=True, use_gemini_cli=True)
    provider._complete_gemini_cli = AsyncMock(return_value=("ok", 12))
    provider._worker_override = "gemini-3-pro-preview"

    await provider.complete("hello", tier=ModelTier.WORKER)

    provider._complete_gemini_cli.assert_awaited_once()
