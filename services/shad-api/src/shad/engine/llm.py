"""LLM Provider abstraction for multi-model support."""

from __future__ import annotations

import asyncio
import json
import logging
from enum import Enum
from typing import Any

from shad.models.run import ModelConfig
from shad.utils.config import get_settings
from shad.utils.models import get_ollama_env, is_ollama_model, normalize_model_name, is_gemini_model

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

    def __init__(
        self,
        use_claude_code: bool = True,
        use_gemini_cli: bool = False,
        model_config: ModelConfig | None = None,
    ) -> None:
        self.settings = get_settings()
        self.use_claude_code = use_claude_code
        self.use_gemini_cli = use_gemini_cli
        self._anthropic_client: Any = None
        self._openai_client: Any = None

        # Model overrides (normalized to full model IDs)
        self._orchestrator_override: str | None = None
        self._worker_override: str | None = None
        self._leaf_override: str | None = None

        if model_config:
            if model_config.orchestrator_model:
                self._orchestrator_override = normalize_model_name(
                    model_config.orchestrator_model
                )
            if model_config.worker_model:
                self._worker_override = normalize_model_name(model_config.worker_model)
            if model_config.leaf_model:
                self._leaf_override = normalize_model_name(model_config.leaf_model)

    def get_model_for_tier(self, tier: ModelTier) -> str:
        """Get the model name for a given tier.

        Checks for overrides first, then falls back to settings.
        """
        if tier == ModelTier.ORCHESTRATOR:
            if self._orchestrator_override:
                return self._orchestrator_override
            return self.settings.gemini_orchestrator_model if self.use_gemini_cli else self.settings.orchestrator_model
        elif tier == ModelTier.WORKER:
            if self._worker_override:
                return self._worker_override
            return self.settings.gemini_worker_model if self.use_gemini_cli else self.settings.worker_model
        elif tier in (ModelTier.LEAF, ModelTier.JUDGE):
            if self._leaf_override:
                return self._leaf_override
            return self.settings.gemini_leaf_model if self.use_gemini_cli else self.settings.leaf_model
        else:
            if self._worker_override:
                return self._worker_override
            return self.settings.gemini_worker_model if self.use_gemini_cli else self.settings.worker_model

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

        # Check for Gemini CLI
        if self.use_gemini_cli and is_gemini_model(model):
            return await self._complete_gemini_cli(
                prompt=prompt,
                system=system,
                model=model,
            )

        # Use Claude Code CLI as primary (uses subscription, not API costs)
        if self.use_claude_code:
            return await self._complete_claude_code(
                prompt=prompt,
                system=system,
                model=model,
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

    def _get_cli_model_name(self, model: str) -> str:
        """Convert API model name to Claude CLI model name.

        For Claude models:
            API names like 'claude-sonnet-4-20250514' -> CLI names like 'sonnet'
        For Ollama models:
            Pass through as-is (e.g., 'llama3', 'qwen3:latest', 'deepseek-coder:6.7b')
        """
        # Ollama models are passed through as-is
        if is_ollama_model(model):
            return model

        # Claude models get converted to shorthand
        model_lower = model.lower()
        if "opus" in model_lower:
            return "opus"
        elif "sonnet" in model_lower:
            return "sonnet"
        elif "haiku" in model_lower:
            return "haiku"
        else:
            # Default to sonnet if unknown Claude model
            return "sonnet"

    async def _complete_gemini_cli(
        self,
        prompt: str,
        system: str | None = None,
        model: str = "gemini-3-pro-preview",
    ) -> tuple[str, int]:
        """Complete using Gemini CLI."""
        import os

        # Combine system prompt and user prompt
        full_prompt = prompt
        if system:
            full_prompt = f"<system>\n{system}\n</system>\n\n<task>\n{prompt}\n</task>"

        logger.info(f"[GEMINI_CLI] Preparing to call Gemini CLI (model: {model})")

        # Prepare environment
        env = os.environ.copy()
        # No API key handling - gemini CLI manages its own auth

        try:
            logger.info(f"[GEMINI_CLI] Executing: gemini --model {model} (prompt via stdin)")
            process = await asyncio.create_subprocess_exec(
                "gemini",
                "--model", model,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            stdout, stderr = await process.communicate(input=full_prompt.encode())

            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                logger.error(f"[GEMINI_CLI] CLI error: {error_msg}")
                raise RuntimeError(f"Gemini CLI failed: {error_msg}")

            response = stdout.decode().strip()
            estimated_tokens = len(response.split()) * 2
            logger.info(f"[GEMINI_CLI] Success - response length: {len(response)} chars")
            return response, estimated_tokens

        except FileNotFoundError as e:
            logger.error(f"[GEMINI_CLI] gemini CLI not found in PATH")
            raise RuntimeError(
                "Gemini CLI not found. Install it or set use_gemini_cli=False"
            ) from e

    async def _complete_claude_code(
        self,
        prompt: str,
        system: str | None = None,
        model: str = "claude-sonnet-4-20250514",
    ) -> tuple[str, int]:
        """Complete using Claude Code CLI (uses subscription, not API costs).

        For Ollama models, this sets special environment variables per-call
        to route the request through Ollama instead of Anthropic's API.
        """
        import os

        # Convert to CLI model name
        cli_model = self._get_cli_model_name(model)

        # Check if this is an Ollama model
        using_ollama = is_ollama_model(model)

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
        provider_label = "OLLAMA" if using_ollama else "CLAUDE_CODE"
        logger.info(f"[{provider_label}] Preparing to call Claude Code CLI (model: {cli_model})")
        logger.info(f"[{provider_label}] Prompt length: {len(full_prompt)} chars")
        if "Context:" in prompt:
            logger.info(f"[{provider_label}] Prompt contains context section")
        logger.debug(f"[{provider_label}] Full prompt preview:\n{full_prompt[:1000]}...")

        # Prepare environment: inherit current env, optionally override for Ollama
        env = os.environ.copy()
        if using_ollama:
            ollama_env = get_ollama_env()
            env.update(ollama_env)
            logger.info(f"[OLLAMA] Using Ollama backend at {ollama_env['ANTHROPIC_BASE_URL']}")

        # Run claude CLI in non-interactive mode
        # Pass prompt via stdin to avoid "Argument list too long" errors
        try:
            logger.info(f"[{provider_label}] Executing: claude -p - --model {cli_model} --output-format text (prompt via stdin)")
            process = await asyncio.create_subprocess_exec(
                "claude",
                "-p", "-",
                "--model", cli_model,
                "--output-format", "text",
                "--allowedTools", "Read,Edit,Write,Bash",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,  # Pass custom environment
            )

            stdout, stderr = await process.communicate(input=full_prompt.encode())

            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                logger.error(f"[{provider_label}] CLI error: {error_msg}")
                raise RuntimeError(f"Claude Code CLI failed: {error_msg}")

            response = stdout.decode().strip()
            # Claude Code doesn't report token usage, estimate based on length
            estimated_tokens = len(response.split()) * 2
            logger.info(f"[{provider_label}] Success - response length: {len(response)} chars, ~{estimated_tokens} tokens")
            logger.debug(f"[{provider_label}] Response preview: {response[:500]}...")
            return response, estimated_tokens

        except FileNotFoundError as e:
            logger.error(f"[{provider_label}] claude CLI not found in PATH")
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

AVAILABLE (already imported - do NOT use import statements):
- obsidian: Object with search/read methods (see below)
- re, json, collections, itertools, functools, math, hashlib, pathlib, datetime

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

CRITICAL RULES - READ CAREFULLY:
1. ALWAYS initialize collection variables BEFORE any conditionals or loops:
   all_results = []  # Initialize at top, not inside if/try blocks

2. NEVER use variables that might not be defined. Check with: if 'var' in dir()

3. Keep scripts SIMPLE. One search, extract content, done. Example:
   results = obsidian.search("keyword", limit=10)
   __result__ = "\\n".join(r.get("content", "")[:2000] for r in results) if results else ""

4. In comprehensions, ONLY use the loop variable:
   WRONG: [note_path for r in results]  # note_path undefined!
   RIGHT: [r['path'] for r in results]   # use r from the loop

5. Handle empty results gracefully:
   results = obsidian.search("query") or []
   __result__ = "No results" if not results else format_results(results)

Example script (KEEP IT SIMPLE):
```python
# Simple search and format - this is all you need!
results = obsidian.search("relevant keywords", limit=10) or []

if results:
    parts = []
    for r in results:
        path = r.get('path', 'unknown')
        content = r.get('content', '')[:2000]
        parts.append(f"## {path}\\n{content}")
    __result__ = "\\n\\n---\\n\\n".join(parts)
else:
    __result__ = "No relevant context found."
```

CRITICAL: Do NOT use import statements - all modules are pre-imported!
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

    def _extract_code_from_markdown(self, text: str) -> str:
        """Extract Python code from markdown code blocks."""
        if "```" not in text:
            return text.strip()

        # Split by code block markers
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            # Remove language identifier from first line if present
            lines = part.split("\n", 1)
            first_line = lines[0].lower().strip()
            if first_line in ("python", "py", "python3"):
                # Return code after language identifier
                return lines[1].strip() if len(lines) > 1 else ""
            elif "__result__" in part or "DOCUMENTS" in part or "obsidian." in part:
                # Looks like code, return as-is
                return part
        return text.strip()

    async def generate_extraction_script(
        self,
        task: str,
        documents: str,
    ) -> tuple[str, int]:
        """
        Generate a Python script to extract/filter/synthesize from pre-fetched documents.

        Unlike generate_retrieval_script, this does NOT do search - the documents
        are already provided. The script focuses purely on extraction and synthesis.

        Args:
            task: The task requiring context extraction
            documents: Pre-fetched document content from qmd search

        Returns:
            Tuple of (script, tokens_used)
        """
        logger.info(f"[CODE_MODE] Generating extraction script for: {task[:80]}...")

        system = """You are an extraction expert. Generate Python scripts that extract relevant information from provided documents.

AVAILABLE (pre-imported, NO import statements allowed):
- DOCUMENTS: String with documents separated by "=== DOCUMENT N: path ===" headers
- re, json, collections, datetime

Store result in __result__ as a formatted string.

ABSOLUTE RULES - VIOLATIONS CAUSE ERRORS:
1. NO import statements - modules are pre-imported
2. NO list comprehensions or generator expressions - USE EXPLICIT FOR LOOPS ONLY
3. NO referencing variables from outer scopes in any expression
4. Initialize ALL variables before use
5. NO multi-line strings - use single-line strings with \\n only
6. Use simple f-strings like f"text {var}" not f"{complex.method()}"

BANNED:
  [x for x in items]           # NO comprehensions!
  (x for x in items)           # NO generators!

REQUIRED PATTERN - follow this exactly:
```python
# Parse documents
sections = DOCUMENTS.split("=== DOCUMENT")
relevant_parts = []

for section in sections:
    if not section.strip():
        continue
    lines = section.strip().split("\\n")
    header = lines[0] if lines else ""
    content = "\\n".join(lines[1:])

    # Check relevance (do ALL logic inside the loop)
    is_relevant = False
    if "keyword1" in content.lower():
        is_relevant = True
    if "keyword2" in content.lower():
        is_relevant = True

    if is_relevant:
        relevant_parts.append(f"## {header}\\n{content[:2000]}")

# Join results
if relevant_parts:
    __result__ = "\\n\\n---\\n\\n".join(relevant_parts)
else:
    __result__ = "No relevant content found."
```

Generate ONLY Python code following the pattern above. No comprehensions, no generators."""

        prompt = f"""Generate a Python extraction script for this task:

Task: {task}

The DOCUMENTS variable contains {len(documents)} characters of pre-fetched content.
Extract and synthesize the most relevant information for the task.

Generate only the Python code."""

        response, tokens = await self.complete(
            prompt=prompt,
            tier=ModelTier.WORKER,
            system=system,
            temperature=0.3,
        )

        script = self._extract_code_from_markdown(response)

        logger.info(f"[CODE_MODE] Generated extraction script ({len(script)} chars)")
        logger.debug(f"[CODE_MODE] Extraction script:\n{script[:500]}...")

        return script, tokens
