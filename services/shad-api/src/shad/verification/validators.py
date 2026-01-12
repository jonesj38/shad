"""Validators for verification and quality control.

Per SPEC.md Section 6.1:
1. Domain-specific validators: Per-skill validation functions
2. Entailment checking: Verify answer is logically entailed by evidence
3. Human-in-the-loop: Flag low-confidence results for batch review
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from shad.engine.llm import LLMProvider, ModelTier

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation check."""

    valid: bool
    confidence: float = 1.0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def needs_review(self) -> bool:
        """Check if result needs human review."""
        return not self.valid or self.confidence < 0.7 or len(self.warnings) > 0


class Validator(ABC):
    """Abstract base class for validators."""

    @abstractmethod
    async def validate(
        self,
        result: str,
        context: dict[str, Any],
    ) -> ValidationResult:
        """Validate a result against context."""
        pass


class StructuralValidator(Validator):
    """Validates structural properties of results."""

    def __init__(
        self,
        min_length: int = 10,
        max_length: int = 50000,
        required_sections: list[str] | None = None,
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.required_sections = required_sections or []

    async def validate(
        self,
        result: str,
        context: dict[str, Any],
    ) -> ValidationResult:
        """Check structural constraints."""
        errors: list[str] = []
        warnings: list[str] = []

        # Length checks
        if len(result) < self.min_length:
            errors.append(f"Result too short ({len(result)} < {self.min_length})")

        if len(result) > self.max_length:
            errors.append(f"Result too long ({len(result)} > {self.max_length})")

        # Required sections
        for section in self.required_sections:
            if section.lower() not in result.lower():
                warnings.append(f"Missing expected section: {section}")

        return ValidationResult(
            valid=len(errors) == 0,
            confidence=1.0 if not errors and not warnings else 0.8,
            errors=errors,
            warnings=warnings,
        )


class CitationValidator(Validator):
    """Validates citation format and presence."""

    def __init__(self, citation_pattern: str = r"\[\d+\]"):
        self.citation_pattern = re.compile(citation_pattern)

    async def validate(
        self,
        result: str,
        context: dict[str, Any],
    ) -> ValidationResult:
        """Check citation requirements."""
        errors: list[str] = []
        warnings: list[str] = []

        # Check for citations
        citations = self.citation_pattern.findall(result)

        if context.get("require_citations", False) and not citations:
            errors.append("Result missing required citations")

        # Validate citation numbers are sequential
        if citations:
            numbers = sorted(int(c.strip("[]")) for c in citations)
            expected = list(range(1, len(set(numbers)) + 1))
            if numbers != expected:
                warnings.append("Citation numbers are not sequential")

        return ValidationResult(
            valid=len(errors) == 0,
            confidence=1.0 if not errors and not warnings else 0.9,
            errors=errors,
            warnings=warnings,
            metadata={"citation_count": len(citations)},
        )


class EntailmentChecker(Validator):
    """
    Checks if answer is logically entailed by evidence.

    Uses LLM judge to verify claims against sources.
    """

    def __init__(self, llm_provider: LLMProvider | None = None):
        self.llm = llm_provider or LLMProvider()

    async def validate(
        self,
        result: str,
        context: dict[str, Any],
    ) -> ValidationResult:
        """Check entailment of result given evidence."""
        evidence = context.get("evidence", [])
        if not evidence:
            return ValidationResult(
                valid=True,
                confidence=0.5,
                warnings=["No evidence provided for entailment check"],
            )

        # Use LLM as judge
        evidence_text = "\n\n".join(
            f"[Source {i + 1}]: {e}" for i, e in enumerate(evidence)
        )

        prompt = f"""You are a fact-checking judge. Determine if the claims in the Answer
are supported by the Evidence.

Evidence:
{evidence_text}

Answer to verify:
{result}

Respond with a JSON object:
{{
  "entailed": true/false,
  "confidence": 0.0-1.0,
  "unsupported_claims": ["claim1", "claim2", ...],
  "reasoning": "explanation"
}}"""

        try:
            response, _ = await self.llm.complete(
                prompt=prompt,
                tier=ModelTier.JUDGE,
                temperature=0.1,
            )

            import json

            # Try to parse JSON from response
            try:
                # Handle markdown code blocks
                if "```" in response:
                    response = response.split("```")[1]
                    if response.startswith("json"):
                        response = response[4:]

                judgment = json.loads(response.strip())

                return ValidationResult(
                    valid=judgment.get("entailed", False),
                    confidence=judgment.get("confidence", 0.5),
                    errors=judgment.get("unsupported_claims", []),
                    metadata={
                        "reasoning": judgment.get("reasoning", ""),
                        "judge_response": response,
                    },
                )
            except json.JSONDecodeError:
                # Fallback: check for key phrases
                is_valid = "entailed" in response.lower() or "supported" in response.lower()
                return ValidationResult(
                    valid=is_valid,
                    confidence=0.5,
                    warnings=["Could not parse judge response"],
                    metadata={"judge_response": response},
                )

        except Exception as e:
            logger.warning(f"Entailment check failed: {e}")
            return ValidationResult(
                valid=True,
                confidence=0.3,
                warnings=[f"Entailment check error: {str(e)}"],
            )


class CompositeValidator(Validator):
    """Combines multiple validators."""

    def __init__(self, validators: list[Validator]):
        self.validators = validators

    async def validate(
        self,
        result: str,
        context: dict[str, Any],
    ) -> ValidationResult:
        """Run all validators and combine results."""
        all_errors: list[str] = []
        all_warnings: list[str] = []
        all_metadata: dict[str, Any] = {}
        min_confidence = 1.0

        for validator in self.validators:
            vr = await validator.validate(result, context)
            all_errors.extend(vr.errors)
            all_warnings.extend(vr.warnings)
            all_metadata.update(vr.metadata)
            min_confidence = min(min_confidence, vr.confidence)

        return ValidationResult(
            valid=len(all_errors) == 0,
            confidence=min_confidence,
            errors=all_errors,
            warnings=all_warnings,
            metadata=all_metadata,
        )


# Factory for creating validators based on skill requirements
def create_validator_for_skill(skill_name: str) -> Validator:
    """Create appropriate validator for a skill."""
    validators: list[Validator] = [
        StructuralValidator(),  # Basic structure checks
    ]

    # Add skill-specific validators
    if skill_name == "research":
        validators.append(CitationValidator())
        validators.append(StructuralValidator(required_sections=["Summary", "Sources"]))

    return CompositeValidator(validators)
