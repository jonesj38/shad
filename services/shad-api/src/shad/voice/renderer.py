"""Voice rendering - Output transformation based on persona.

Per SPEC.md Section 8:
- Inside the run engine: Neutral + precise (structured outputs)
- At the boundary: Rendered through a voice layer
- Voice affects PRESENTATION, not TRUTH CONDITIONS

Voice may change:
- Phrasing and tone
- Structure and brevity
- Examples and analogies
- How it surfaces caveats

Voice must NOT change:
- Claims / facts
- Citations / provenance
- Numeric results
- Safety policies / CORE constraints
- Decisions and stop reasons
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from shad.engine.llm import LLMProvider, ModelTier
from shad.utils.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class VoiceSpec:
    """Voice specification loaded from YAML."""

    name: str
    description: str = ""
    tone: str = "concise"  # concise | blunt | warm | playful | formal
    verbosity: int = 3  # 1-5
    formatting: str = "bullets"  # bullets | headings | tables | prose
    profanity: str = "disallow"  # allow | disallow
    citation_style: str = "inline"  # inline | numbered | footnote
    error_style: str = "transparent"  # transparent | soft | technical
    signature: str | None = None
    guidelines: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> VoiceSpec:
        """Load voice spec from YAML file."""
        with path.open() as f:
            data = yaml.safe_load(f)

        return cls(
            name=data.get("name", path.stem),
            description=data.get("description", ""),
            tone=data.get("tone", "concise"),
            verbosity=data.get("verbosity", 3),
            formatting=data.get("formatting", "bullets"),
            profanity=data.get("profanity", "disallow"),
            citation_style=data.get("citation_style", "inline"),
            error_style=data.get("error_style", "transparent"),
            signature=data.get("signature"),
            guidelines=data.get("guidelines", []),
        )

    def to_prompt_instructions(self) -> str:
        """Convert voice spec to prompt instructions."""
        instructions = [
            f"Output Style: {self.tone}",
            f"Verbosity Level: {self.verbosity}/5",
            f"Formatting: Use {self.formatting}",
            f"Citations: {self.citation_style} style",
        ]

        if self.guidelines:
            instructions.append("\nGuidelines:")
            for g in self.guidelines:
                instructions.append(f"- {g}")

        return "\n".join(instructions)


class VoiceRenderer:
    """
    Renders structured output through a voice/persona layer.

    The renderer transforms neutral, structured results into
    outputs that match the specified voice while preserving:
    - Factual accuracy
    - Citations and provenance
    - Numeric data
    - Safety constraints
    """

    def __init__(
        self,
        voices_path: Path | None = None,
        llm_provider: LLMProvider | None = None,
    ):
        settings = get_settings()
        self.voices_path = voices_path or (settings.core_path / "Voices")
        self.llm = llm_provider or LLMProvider()
        self.voices: dict[str, VoiceSpec] = {}
        self._load_voices()

    def _load_voices(self) -> None:
        """Load all voice specs from the voices directory."""
        if not self.voices_path.exists():
            logger.warning(f"Voices path does not exist: {self.voices_path}")
            return

        for voice_file in self.voices_path.glob("*.yaml"):
            try:
                voice = VoiceSpec.load(voice_file)
                self.voices[voice.name] = voice
                logger.debug(f"Loaded voice: {voice.name}")
            except Exception as e:
                logger.warning(f"Failed to load voice from {voice_file}: {e}")

        # Ensure default voice exists
        if "default" not in self.voices:
            self.voices["default"] = VoiceSpec(name="default")

    def get_voice(self, name: str) -> VoiceSpec:
        """Get a voice spec by name, falling back to default."""
        return self.voices.get(name, self.voices.get("default", VoiceSpec(name="default")))

    async def render(
        self,
        content: str,
        voice_name: str = "default",
        context: dict[str, Any] | None = None,
    ) -> str:
        """
        Render content through the specified voice.

        Args:
            content: Raw structured content to render
            voice_name: Name of voice to use
            context: Optional context (goal, citations, etc.)

        Returns:
            Rendered content with voice applied
        """
        voice = self.get_voice(voice_name)

        # For minimal verbosity, do simple transformations
        if voice.verbosity <= 2:
            return self._apply_simple_transforms(content, voice)

        # For higher verbosity, use LLM to rephrase
        try:
            return await self._llm_render(content, voice, context or {})
        except Exception as e:
            logger.warning(f"LLM rendering failed, using simple transforms: {e}")
            return self._apply_simple_transforms(content, voice)

    def _apply_simple_transforms(self, content: str, voice: VoiceSpec) -> str:
        """Apply simple rule-based transformations."""
        result = content

        # Apply formatting
        if voice.formatting == "bullets":
            # Ensure bullet formatting
            lines = result.split("\n")
            formatted = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith(("-", "*", "#", "•")):
                    if len(line) < 100 and ":" in line:
                        formatted.append(f"- {line}")
                    else:
                        formatted.append(line)
                else:
                    formatted.append(line)
            result = "\n".join(formatted)

        # Apply verbosity (truncate if needed)
        if voice.verbosity <= 2:
            # Keep first N paragraphs based on verbosity
            paragraphs = result.split("\n\n")
            max_paragraphs = voice.verbosity * 2
            result = "\n\n".join(paragraphs[:max_paragraphs])

        # Add signature if specified
        if voice.signature:
            result = f"{result}\n\n{voice.signature}"

        return result

    async def _llm_render(
        self,
        content: str,
        voice: VoiceSpec,
        context: dict[str, Any],
    ) -> str:
        """Use LLM to render content with voice personality."""
        system_prompt = f"""You are a text stylist. Transform the input text to match these style requirements:

{voice.to_prompt_instructions()}

CRITICAL CONSTRAINTS (DO NOT VIOLATE):
1. PRESERVE all factual claims exactly - do not add, remove, or modify facts
2. PRESERVE all citations exactly as written (e.g., [1], [Source A])
3. PRESERVE all numeric values exactly
4. PRESERVE any warnings, errors, or safety notes
5. PRESERVE technical terms and proper nouns

You may ONLY change:
- Word choice and phrasing (while preserving meaning)
- Sentence structure
- Paragraph organization
- Tone and register
- Examples or analogies (if they clarify, not replace, facts)"""

        user_prompt = f"""Transform this content to match the voice style:

---
{content}
---

Remember: Preserve all facts, citations, and numbers exactly."""

        response, _ = await self.llm.complete(
            prompt=user_prompt,
            tier=ModelTier.WORKER,
            system=system_prompt,
            temperature=0.7,
        )

        # Validate that citations are preserved
        original_citations = self._extract_citations(content)
        rendered_citations = self._extract_citations(response)

        if not self._citations_preserved(original_citations, rendered_citations):
            logger.warning("Voice rendering may have altered citations, using original")
            # Fall back to simple transforms if citations were changed
            return self._apply_simple_transforms(content, voice)

        return response

    def _extract_citations(self, text: str) -> set[str]:
        """Extract citation markers from text."""
        import re

        # Match [N], [Source X], etc.
        pattern = r"\[[\w\s]+\]"
        return set(re.findall(pattern, text))

    def _citations_preserved(
        self,
        original: set[str],
        rendered: set[str],
    ) -> bool:
        """Check if all original citations are preserved."""
        return original <= rendered  # All original citations should still be present

    def list_voices(self) -> list[dict[str, Any]]:
        """List all available voices."""
        return [
            {
                "name": v.name,
                "description": v.description,
                "tone": v.tone,
                "verbosity": v.verbosity,
                "formatting": v.formatting,
            }
            for v in self.voices.values()
        ]

    def render_error(
        self,
        error: str,
        voice_name: str = "default",
    ) -> str:
        """Render an error message according to voice style."""
        voice = self.get_voice(voice_name)

        if voice.error_style == "transparent":
            return f"Error: {error}"
        elif voice.error_style == "soft":
            return f"I encountered an issue: {error}"
        elif voice.error_style == "technical":
            return f"[ERROR] {error}"
        else:
            return f"Error: {error}"

    def render_partial_result(
        self,
        result: str,
        completed_tasks: int,
        total_tasks: int,
        stop_reason: str,
        voice_name: str = "default",
    ) -> str:
        """Render a partial result with status information."""
        voice = self.get_voice(voice_name)

        status_line = f"Completed {completed_tasks}/{total_tasks} tasks (stopped: {stop_reason})"

        if voice.tone == "formal":
            header = f"## Partial Result\n\n{status_line}\n\n"
        elif voice.tone == "blunt":
            header = f"PARTIAL ({status_line}):\n\n"
        else:
            header = f"*{status_line}*\n\n"

        return f"{header}{result}"
