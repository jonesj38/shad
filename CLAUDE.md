# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Shad (Shannon's Daemon) enables AI to utilize virtually unlimited context by treating an Obsidian vault as an explorable environment. It recursively decomposes complex tasks, retrieves targeted context for each subtask via **Code Mode** (LLM-generated Python scripts), and assembles coherent outputs.

Core premise: **Long-context reasoning is an inference problem, not a prompting problem.**

## Development Commands

All development is in `services/shad-api/`:

```bash
cd services/shad-api

# Setup
source .venv/bin/activate
pip install -e ".[dev]"

# Linting
ruff check src/shad/
ruff check src/shad/ --fix

# Type checking
mypy src/shad/

# Tests
pytest                              # All tests
pytest tests/test_rlm_engine.py     # Single file
pytest -k "test_decompose"          # By name pattern
pytest --cov=shad                   # With coverage

# Format check (ruff handles this via lint rules)
ruff format src/shad/ --check
```

## Architecture

```
shad run "Build app" --vault ~/MyVault
         │
         ▼
    RLM Engine
         │
         ├── Strategy Selection → Decomposition (DAG of subtasks)
         │
         ├── For each node:
         │     Code Mode → CodeExecutor → ObsidianTools → Vault
         │                  (sandboxed)
         │
         ├── Verification Layer (syntax, types, imports)
         │
         └── Synthesis → File Manifest
```

### Key Modules (in `services/shad-api/src/shad/`)

| Module | Purpose |
|--------|---------|
| `engine/rlm.py` | Core RLM engine - DAG execution, Code Mode orchestration |
| `engine/llm.py` | LLM abstraction, retrieval script generation |
| `engine/strategies.py` | Strategy skeletons (software, research, analysis) |
| `engine/decomposition.py` | Task decomposition into subtasks |
| `sandbox/executor.py` | Sandboxed Python execution for Code Mode scripts |
| `sandbox/tools.py` | `ObsidianTools` API available to retrieval scripts |
| `mcp/client.py` | Direct filesystem client for vault operations |
| `verification/layer.py` | Verification orchestration (syntax, types, imports) |
| `output/manifest.py` | File manifest generation and writing |
| `sources/manager.py` | Automated source ingestion scheduling |

### Code Mode

LLM generates Python scripts for targeted retrieval:

```python
results = obsidian.search("authentication", limit=10)
pattern = obsidian.read_note("Patterns/Auth.md")
__result__ = {"context": ..., "citations": [...], "confidence": 0.72}
```

Scripts run in a sandbox with restricted builtins and only vault access via `ObsidianTools`.

### Strategy Skeletons

Strategies define required stages and constraints. Example (software):
- Required: clarify_requirements → types_contracts → implementation → verification → synthesis
- Constraints: contracts-first (types before implementation), imports must resolve

## CLI Usage

```bash
# Basic run
shad run "Your task" --vault /path/to/vault

# Software generation with file output
shad run "Build REST API" --vault ~/v --strategy software --verify strict --write-files -o ./out

# Multi-vault (priority order)
shad run "Build app" --vault ~/Project --vault ~/Patterns

# Check status / inspect
shad status <run_id>
shad trace tree <run_id>
shad resume <run_id>
shad export <run_id> --output ./out

# Server management
shad server start|stop|status|logs

# Source scheduling
shad sources add github <url> --vault ~/v --schedule weekly
shad sources sync
```

## Key Concepts

- **Contracts-first**: Types node runs before implementation; provides single source of truth
- **Soft deps + context packets**: Parallel execution with context sharing between siblings
- **Two-pass import resolution**: Build export index first, then generate implementations
- **Verification levels**: `off` → `basic` (default) → `build` → `strict`
- **Sandbox profiles**: `strict` (vault-only), `local` (+allowlisted paths), `extended` (+network)
- **Budget system**: Hierarchical token allocation; parent reserves 25% for synthesis

## Hard Invariants

1. **Never Auto-Publish**: No irreversible side effects without human approval
2. **Never Exfiltrate**: No sending data externally unless explicitly permitted
3. **Never Self-Modify**: Cannot change own Skills/CORE without human review

## Configuration

Environment variables (or `.env` file):
- `OBSIDIAN_VAULT_PATH`: Default vault path (optional, CLI `--vault` overrides)
- `REDIS_URL`: Redis connection (default: `redis://localhost:6379/0`)

## Reference

- `SPEC.md`: Full technical specification with decision log
- `README.md`: User-facing documentation and quick start
