---
name: research
version: 1.0.0
description: Deep research with citation tracking
use_when:
  - "research *"
  - "investigate *"
  - "find evidence for *"
  - "what do we know about *"
intents: [research, investigate, summarize]
entities: [academic, technical, general]
inputs_schema:
  goal: string
  notebook_id: string
outputs_schema:
  summary: string
  citations: array
  confidence: number
tools_allowed: [retrieve, embed, summarize]
priority: 10
cost_profile: expensive
composes_with: [citations, images]
exclusions: ["quick questions", "simple lookups", "yes/no questions"]
default_voice: researcher
entry_workflows: [default, quick, thorough]
---

# Research Skill

Deep research capability with citation tracking and source synthesis.

## When to Use

This skill is triggered for:
- Research queries requiring multiple sources
- Investigative tasks needing evidence gathering
- Synthesis of complex topics from various materials

## Workflow

1. **Goal Analysis**: Parse the research question and identify key concepts
2. **Source Retrieval**: Query relevant notebooks for sources and notes
3. **Evidence Gathering**: Extract relevant passages with citations
4. **Synthesis**: Combine findings into coherent analysis
5. **Citation**: Format all sources with proper attribution

## Constraints

- Always cite sources
- Distinguish findings from interpretations
- Note confidence levels
- Flag contradictory evidence
