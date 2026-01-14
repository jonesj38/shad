# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Shad (Shannon's Daemon) is a personal AI infrastructure for long-context reasoning. It recursively decomposes problems, retrieves knowledge from an Obsidian vault via MCP, caches subtrees in Redis, and recomposes results.

**Core premise**: Long-context reasoning is an inference problem, not a prompting problem.

**Storage backend**: Local-first Markdown via **Obsidian**, accessed via **Model Context Protocol (MCP)**.

> See `OBSIDIAN_PIVOT.md` for the complete Obsidian integration specification.

## Architecture

```
User / n8n
   |
   v
Shad API / CLI (FastAPI + Click)
   |
   +-- Skill Router (goal → skill selection with weighted scoring)
   |
   +-- RLM Engine (recursive DAG execution)
   |       |
   |       +-- OpenNotebookLM (graph-based knowledge retrieval)
   |       +-- Redis (subtree caching with hierarchical keys)
   |       +-- LLM Provider (Claude Code CLI by default, API fallback)
   |
   +-- Verification (validators, entailment, novelty detection)
   |
   +-- History/ (structured run artifacts)
   |
   +-- Voice Renderer (persona-based output transformation)
```

### Key Execution Flow

1. Goal comes in via CLI (`shad run`) or API (`POST /v1/run`)
2. SkillRouter scores and selects appropriate skill(s)
3. RLMEngine decomposes goal into subtasks via LLM
4. Each subtask: check cache → retrieve context → execute → cache result
5. Results synthesized bottom-up with citations
6. Output rendered through voice layer
7. Artifacts saved to History/

### Module Responsibilities

| Module | Location | Purpose |
|--------|----------|---------|
| `engine/rlm.py` | Core | Recursive decomposition, budget enforcement, DAG execution |
| `engine/llm.py` | Core | LLM abstraction (Claude Code CLI primary, API fallback) |
| `cache/redis_cache.py` | Caching | Hierarchical keys, main + staging caches |
| `skills/router.py` | Routing | Weighted scoring: intent, triggers, context, priority |
| `verification/` | Quality | Validators, entailment checking, novelty detection, HITL queues |
| `voice/renderer.py` | Output | Transform structured output through persona layer |
| `learnings/` | Learning | Extract facts/patterns from runs, promotion pipeline |
| `history/manager.py` | Persistence | Structured artifacts: manifest, DAG, metrics, reports |

## Development Commands

```bash
cd services/shad-api

# Activate virtual environment
source .venv/bin/activate

# Install in editable mode
pip install -e ".[dev]"

# Run API server locally
uvicorn shad.api.main:app --reload --port 8000

# Run linter
ruff check src/shad/

# Run linter with auto-fix
ruff check src/shad/ --fix

# Type checking
mypy src/shad/

# Run tests
pytest

# Run single test file
pytest tests/test_rlm.py

# Run with coverage
pytest --cov=shad
```

### Docker Commands

```bash
# Start full stack
docker compose up -d --build

# View logs
docker compose logs -f shad-api

# Rebuild single service
docker compose up -d --build shad-api
```

### CLI Commands

```bash
# Execute a reasoning task
shad run "What are three benefits of git?" --max-depth 2

# Check run status
shad status <run_id>

# View execution tree
shad trace tree <run_id>

# Resume partial run
shad resume <run_id>
```

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /v1/run` | Execute reasoning task |
| `GET /v1/run/:id` | Get run status/results |
| `POST /v1/run/:id/resume` | Resume partial run |
| `GET /v1/runs` | List recent runs |
| `GET /v1/skills` | List available skills |
| `POST /v1/skills/route` | Route goal to skills |
| `GET /v1/admin/cache/stats` | Cache statistics |
| `GET /v1/admin/hitl/queue` | HITL review queue |
| `GET /v1/admin/voices` | List voices |

## LLM Provider

The system uses Claude Code CLI by default (`use_claude_code=True` in LLMProvider), which uses your Claude Code subscription instead of API costs. Falls back to Anthropic/OpenAI APIs if CLI unavailable.

## Budget System

Every run enforces hard limits defined in `RunConfig.budget`:
- `max_depth`: Maximum recursion depth
- `max_nodes`: Maximum DAG nodes
- `max_wall_time`: Total execution time (seconds)
- `max_tokens`: Total token budget
- `max_branching_factor`: Maximum children per node

When budgets are exhausted, run returns partial results with `status: partial`.

## Skills System

Skills are in `Skills/<SkillName>/SKILL.md` with YAML frontmatter:
```yaml
name: research
use_when: ["research *", "investigate *"]
intents: [research, investigate]
priority: 10
composes_with: [citations]
```

SkillRouter uses weighted scoring: intent match (0.3), trigger match (0.25), context (0.1), priority (0.1), minus exclusion penalties.

## History Artifacts

Each run creates `History/Runs/<run_id>/`:
- `run.manifest.json` - Config, budgets, versions
- `dag.json` - Complete DAG with node statuses
- `metrics/summary.json` - Token counts, timing
- `final.report.md` - Human-readable output
- `final.summary.json` - Machine-readable with suggested next actions

## Hard Invariants (from CORE/invariants.md)

1. **Never Auto-Publish**: No irreversible side effects without explicit human approval
2. **Never Exfiltrate**: No sending data externally unless explicitly permitted
3. **Never Self-Modify**: Cannot change own Skills/CORE without human review
