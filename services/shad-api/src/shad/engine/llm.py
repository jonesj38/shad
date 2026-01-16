"""LLM Provider abstraction for multi-model support."""

from __future__ import annotations

import asyncio
import json
import logging
from enum import Enum
from typing import Any

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

    def __init__(self, use_claude_code: bool = True) -> None:
        self.settings = get_settings()
        self.use_claude_code = use_claude_code
        self._anthropic_client: Any = None
        self._openai_client: Any = None

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

        # Use Claude Code CLI as primary (uses subscription, not API costs)
        if self.use_claude_code:
            return await self._complete_claude_code(
                prompt=prompt,
                system=system,
            )

        # Fallback to API if configured
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
                model="gpt-4o",
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        else:
            raise ValueError(
                "No LLM provider configured. Use Claude Code CLI or set API keys."
            )

    async def _complete_claude_code(
        self,
        prompt: str,
        system: str | None = None,
    ) -> tuple[str, int]:
        """Complete using Claude Code CLI (uses subscription, not API costs)."""
        # Combine system prompt and user prompt with clear structure
        full_prompt = prompt
        if system:
            full_prompt = f"""<system>
{system}
</system>

<task>
{prompt}
</task>"""

        # Log prompt details
        logger.info("[CLAUDE_CODE] Preparing to call Claude Code CLI")
        logger.info(f"[CLAUDE_CODE] Prompt length: {len(full_prompt)} chars")
        if "Context:" in prompt:
            logger.info("[CLAUDE_CODE] Prompt contains context section")
        logger.debug(f"[CLAUDE_CODE] Full prompt preview:\n{full_prompt[:1000]}...")

        # Run claude CLI in non-interactive mode
        try:
            logger.info("[CLAUDE_CODE] Executing: claude -p <prompt> --output-format text")
            process = await asyncio.create_subprocess_exec(
                "claude",
                "-p", full_prompt,
                "--output-format", "text",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                logger.error(f"[CLAUDE_CODE] CLI error: {error_msg}")
                raise RuntimeError(f"Claude Code CLI failed: {error_msg}")

            response = stdout.decode().strip()
            # Claude Code doesn't report token usage, estimate based on length
            estimated_tokens = len(response.split()) * 2
            logger.info(f"[CLAUDE_CODE] Success - response length: {len(response)} chars, ~{estimated_tokens} tokens")
            logger.debug(f"[CLAUDE_CODE] Response preview: {response[:500]}...")
            return response, estimated_tokens

        except FileNotFoundError as e:
            logger.error("[CLAUDE_CODE] claude CLI not found in PATH")
            raise RuntimeError(
                "Claude Code CLI not found. Install it or set use_claude_code=False"
            ) from e

    async def _complete_anthropic(
        self,
        prompt: str,
        model: str,
        system: str | None,
        max_tokens: int,
        temperature: float,
    ) -> tuple[str, int]:
        """Complete using Anthropic API."""
        import anthropic

        if self._anthropic_client is None:
            self._anthropic_client = anthropic.Anthropic(
                api_key=self.settings.anthropic_api_key
            )

        messages = [{"role": "user", "content": prompt}]

        response = self._anthropic_client.messages.create(
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
        import openai

        if self._openai_client is None:
            self._openai_client = openai.OpenAI(api_key=self.settings.openai_api_key)

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self._openai_client.chat.completions.create(
            model=model,
            messages=messages,
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
        logger.info(f"[DECOMPOSE] Starting decomposition: {task[:100]}...")
        logger.info(f"[DECOMPOSE] Context provided: {len(context)} chars")
        logger.info(f"[DECOMPOSE] Max subtasks: {max_subtasks}")

        system = """You are a task decomposition expert. Break down complex tasks into smaller,
actionable subtasks. Return ONLY a JSON array of subtask strings, no explanation.
Each subtask should be self-contained and specific.
Limit to the requested number of subtasks."""

        context_section = ""
        if context:
            context_section = f"""<retrieved_context>
{context}
</retrieved_context>

"""
            logger.info("[DECOMPOSE] Including retrieved context")

        prompt = f"""{context_section}Decompose this task into {max_subtasks} or fewer subtasks:

Task: {task}

Return a JSON array of subtask strings. Example: ["subtask 1", "subtask 2", "subtask 3"]"""

        response, _ = await self.complete(
            prompt=prompt,
            tier=ModelTier.ORCHESTRATOR,
            system=system,
            temperature=0.3,
        )

        try:
            # Try to extract JSON from response
            # Handle case where response has markdown code blocks
            json_str = response.strip()
            if "```" in json_str:
                # Extract content between code blocks
                parts = json_str.split("```")
                for part in parts:
                    part = part.strip()
                    if part.startswith("json"):
                        part = part[4:].strip()
                    if part.startswith("["):
                        json_str = part
                        break

            subtasks = json.loads(json_str)
            if isinstance(subtasks, list):
                return [str(s) for s in subtasks[:max_subtasks]]
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse decomposition response: {response}")
            # Fallback: split by newlines if JSON parsing fails
            lines = [line.strip() for line in response.split("\n") if line.strip()]
            # Filter out lines that look like JSON artifacts
            lines = [ln for ln in lines if not ln.startswith(("[", "]", "{", "}"))]
            return lines[:max_subtasks] if lines else [task]

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
        logger.info(f"[SYNTHESIZE] Starting synthesis for: {task[:100]}...")
        logger.info(f"[SYNTHESIZE] Subtask results: {len(subtask_results)}")
        logger.info(f"[SYNTHESIZE] Context provided: {len(context)} chars")

        system = """You are a synthesis expert. Combine subtask results into a coherent,
comprehensive answer. Cite sources using [N] notation where N is the subtask number.
Be concise but thorough."""

        results_text = "\n\n".join(
            f"[{i + 1}] Subtask: {subtask}\nResult: {result}"
            for i, (subtask, result) in enumerate(subtask_results)
        )

        context_section = ""
        if context:
            context_section = f"""<retrieved_context>
{context}
</retrieved_context>

"""

        prompt = f"""{context_section}Synthesize these subtask results into a comprehensive answer for the original task.

Original Task: {task}

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
        logger.info(f"[ANSWER_TASK] Starting task: {task[:100]}...")
        logger.info(f"[ANSWER_TASK] Context provided: {len(context)} chars")

        system = """You are a knowledgeable assistant. Answer the question or complete the task
directly and concisely. If context is provided, use it to inform your answer.
Always cite relevant sources if available."""

        if context:
            context_section = f"""<retrieved_context>
The following information was retrieved from the user's knowledge base and is highly relevant to this task:

{context}
</retrieved_context>"""
            logger.info("[ANSWER_TASK] Including retrieved context in prompt")
        else:
            context_section = ""
            logger.info("[ANSWER_TASK] No context to include")

        prompt = f"""{context_section}

<task>
{task}
</task>

Provide a direct, informative answer. If the retrieved context is relevant, use it and cite the sources."""

        return await self.complete(
            prompt=prompt,
            tier=tier,
            system=system,
            temperature=0.5,
        )

    async def generate_retrieval_script(
        self,
        task: str,
        vault_info: str = "",
    ) -> tuple[str, int]:
        """
        Generate a Python script to retrieve relevant context from the vault.

        Per OBSIDIAN_PIVOT.md Section 3.1: The LLM writes Python scripts
        that import MCP tools to dynamically query the vault.

        Args:
            task: The task requiring context retrieval
            vault_info: Optional info about vault structure

        Returns:
            Tuple of (script, tokens_used)
        """
        logger.info(f"[CODE_MODE] Generating retrieval script for: {task[:100]}...")

        system = """You are a code generation expert specializing in knowledge retrieval.
Generate Python scripts that query an Obsidian vault to gather context for a task.

You have access to an 'obsidian' object with these methods:
- obsidian.search(query: str, limit: int = 10, path_filter: str | None = None) -> list[dict]
  Returns: [{"path": "...", "content": "...", "score": float, "matched_line": "..."}]
- obsidian.read_note(path: str) -> str | None
  Returns the full content of a note
- obsidian.list_notes(directory: str = "", recursive: bool = False) -> list[str]
  Returns list of note paths
- obsidian.get_frontmatter(path: str) -> dict | None
  Returns YAML frontmatter as dict

IMPORTANT: Store your final result in __result__ variable. This should be a string
containing the most relevant context for the task, formatted clearly.

CRITICAL: When using list comprehensions, access dict keys directly on the loop variable.
WRONG: [f"{path}" for r in results]  # 'path' is undefined!
RIGHT: [f"{r['path']}" for r in results]  # access via r

Example script:
```python
# Search for relevant notes
results = obsidian.search("machine learning", limit=5)

# Extract and combine relevant content (use r['key'] in comprehensions)
context_parts = [f"## {r['path']}\\n{r['content'][:2000]}" for r in results]

# Or with a for loop:
# context_parts = []
# for r in results:
#     context_parts.append(f"## {r['path']}\\n{r['content'][:2000]}")

# Store final result
__result__ = "\\n\\n---\\n\\n".join(context_parts)
```

Generate ONLY the Python code, no explanations."""

        prompt = f"""Generate a Python retrieval script for this task:

Task: {task}

{f"Vault structure hint: {vault_info}" if vault_info else ""}

The script should:
1. Search for relevant notes using keywords from the task
2. Read additional detail from the most relevant notes if needed
3. Extract and format the most useful content
4. Store the result in __result__ as a formatted string

Generate only the Python code."""

        response, tokens = await self.complete(
            prompt=prompt,
            tier=ModelTier.WORKER,
            system=system,
            temperature=0.3,
        )

        # Extract code from markdown code blocks if present
        script = response.strip()
        if "```" in script:
            # Extract content between code blocks
            parts = script.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("python"):
                    script = part[6:].strip()
                    break
                elif part.startswith("py"):
                    script = part[2:].strip()
                    break
                elif "obsidian." in part or "__result__" in part:
                    script = part.strip()
                    break

        logger.info(f"[CODE_MODE] Generated script ({len(script)} chars)")
        logger.debug(f"[CODE_MODE] Script:\n{script[:500]}...")

        return script, tokens
