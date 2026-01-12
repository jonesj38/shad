"""LLM Provider abstraction for multi-model support."""

from __future__ import annotations

import json
import logging
from enum import Enum

import anthropic
import openai

from shad.utils.config import get_settings

logger = logging.getLogger(__name__)


class ModelTier(str, Enum):
    """Model capability tiers."""

    ORCHESTRATOR = "orchestrator"  # Best reasoning/planning
    WORKER = "worker"  # Balanced mid-depth
    LEAF = "leaf"  # Fast/cheap parallel
    JUDGE = "judge"  # Evals, verification
    EMBEDDER = "embedder"  # Routing, similarity


class LLMProvider:
    """Abstraction over LLM providers with tiered model support."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._anthropic_client: anthropic.Anthropic | None = None
        self._openai_client: openai.OpenAI | None = None

    @property
    def anthropic_client(self) -> anthropic.Anthropic:
        """Get or create Anthropic client."""
        if self._anthropic_client is None:
            self._anthropic_client = anthropic.Anthropic(
                api_key=self.settings.anthropic_api_key
            )
        return self._anthropic_client

    @property
    def openai_client(self) -> openai.OpenAI:
        """Get or create OpenAI client."""
        if self._openai_client is None:
            self._openai_client = openai.OpenAI(api_key=self.settings.openai_api_key)
        return self._openai_client

    def get_model_for_tier(self, tier: ModelTier) -> str:
        """Get the model name for a given tier."""
        if tier == ModelTier.ORCHESTRATOR:
            return self.settings.orchestrator_model
        elif tier == ModelTier.WORKER:
            return self.settings.worker_model
        elif tier in (ModelTier.LEAF, ModelTier.JUDGE):
            return self.settings.leaf_model
        else:
            return self.settings.worker_model

    async def complete(
        self,
        prompt: str,
        tier: ModelTier = ModelTier.WORKER,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> tuple[str, int]:
        """
        Generate a completion using the appropriate model tier.

        Returns:
            Tuple of (response_text, tokens_used)
        """
        model = self.get_model_for_tier(tier)
        logger.debug(f"Using model {model} for tier {tier}")

        # Use Anthropic as primary provider
        if self.settings.anthropic_api_key:
            return await self._complete_anthropic(
                prompt=prompt,
                model=model,
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        elif self.settings.openai_api_key:
            return await self._complete_openai(
                prompt=prompt,
                model="gpt-4o",  # Fallback model
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        else:
            raise ValueError("No LLM provider configured. Set ANTHROPIC_API_KEY or OPENAI_API_KEY.")

    async def _complete_anthropic(
        self,
        prompt: str,
        model: str,
        system: str | None,
        max_tokens: int,
        temperature: float,
    ) -> tuple[str, int]:
        """Complete using Anthropic API."""
        messages = [{"role": "user", "content": prompt}]

        response = self.anthropic_client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system or "You are a helpful AI assistant.",
            messages=messages,
        )

        text = response.content[0].text if response.content else ""
        tokens = response.usage.input_tokens + response.usage.output_tokens
        return text, tokens

    async def _complete_openai(
        self,
        prompt: str,
        model: str,
        system: str | None,
        max_tokens: int,
        temperature: float,
    ) -> tuple[str, int]:
        """Complete using OpenAI API."""
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self.openai_client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore
            max_tokens=max_tokens,
            temperature=temperature,
        )

        text = response.choices[0].message.content or ""
        tokens = response.usage.total_tokens if response.usage else 0
        return text, tokens

    async def decompose_task(
        self,
        task: str,
        context: str = "",
        max_subtasks: int = 5,
    ) -> list[str]:
        """
        Decompose a task into subtasks using the orchestrator model.

        Returns:
            List of subtask strings
        """
        system = """You are a task decomposition expert. Break down complex tasks into smaller,
actionable subtasks. Return ONLY a JSON array of subtask strings, no explanation.
Each subtask should be self-contained and specific.
Limit to the requested number of subtasks."""

        prompt = f"""Decompose this task into {max_subtasks} or fewer subtasks:

Task: {task}

{"Context: " + context if context else ""}

Return a JSON array of subtask strings. Example: ["subtask 1", "subtask 2", "subtask 3"]"""

        response, _ = await self.complete(
            prompt=prompt,
            tier=ModelTier.ORCHESTRATOR,
            system=system,
            temperature=0.3,
        )

        try:
            # Parse JSON response
            subtasks = json.loads(response.strip())
            if isinstance(subtasks, list):
                return [str(s) for s in subtasks[:max_subtasks]]
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse decomposition response: {response}")
            # Fallback: split by newlines if JSON parsing fails
            lines = [line.strip() for line in response.split("\n") if line.strip()]
            return lines[:max_subtasks]

        return [task]  # Return original task if decomposition fails

    async def synthesize_results(
        self,
        task: str,
        subtask_results: list[tuple[str, str]],
        context: str = "",
    ) -> str:
        """
        Synthesize results from subtasks into a coherent answer.

        Args:
            task: Original task
            subtask_results: List of (subtask, result) tuples
            context: Optional context

        Returns:
            Synthesized answer
        """
        system = """You are a synthesis expert. Combine subtask results into a coherent,
comprehensive answer. Cite sources using [N] notation where N is the subtask number.
Be concise but thorough."""

        results_text = "\n\n".join(
            f"[{i + 1}] Subtask: {subtask}\nResult: {result}"
            for i, (subtask, result) in enumerate(subtask_results)
        )

        prompt = f"""Synthesize these subtask results into a comprehensive answer for the original task.

Original Task: {task}

{"Context: " + context if context else ""}

Subtask Results:
{results_text}

Provide a coherent synthesis with citations [N] to the relevant subtask results."""

        response, _ = await self.complete(
            prompt=prompt,
            tier=ModelTier.ORCHESTRATOR,
            system=system,
            temperature=0.5,
        )

        return response

    async def answer_task(
        self,
        task: str,
        context: str = "",
        tier: ModelTier = ModelTier.WORKER,
    ) -> tuple[str, int]:
        """
        Directly answer a task without decomposition.

        Returns:
            Tuple of (answer, tokens_used)
        """
        system = """You are a knowledgeable assistant. Answer the question or complete the task
directly and concisely. If context is provided, use it to inform your answer.
Always cite relevant sources if available."""

        context_section = f"Context:\n{context}" if context else ""
        prompt = f"""Task: {task}

{context_section}

Provide a direct, informative answer."""

        return await self.complete(
            prompt=prompt,
            tier=tier,
            system=system,
            temperature=0.5,
        )
