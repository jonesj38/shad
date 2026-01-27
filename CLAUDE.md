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
| `engine/llm.py` | LLM abstraction (Claude Code CLI, Anthropic API, OpenAI fallback) |
| `engine/strategies.py` | Strategy skeletons (software, research, analysis) |
| `engine/decomposition.py` | Task decomposition into subtasks |
| `engine/context_packets.py` | Cross-subtask context sharing |
| `sandbox/executor.py` | Sandboxed Python execution for Code Mode scripts |
| `sandbox/tools.py` | `ObsidianTools` API available to retrieval scripts |
| `retrieval/layer.py` | RetrievalLayer protocol and RetrievalResult dataclass |
| `retrieval/qmd.py` | QmdRetriever - hybrid BM25 + vector search via qmd CLI |
| `retrieval/filesystem.py` | FilesystemRetriever - fallback when qmd not installed |
| `verification/layer.py` | Verification orchestration (syntax, types, imports) |
| `output/manifest.py` | File manifest generation and writing |
| `output/import_resolution.py` | Two-pass import validation |
| `refinement/manager.py` | Run states, delta verification, HITL checkpoints |
| `vault/ingestion.py` | Repository ingestion with presets (mirror/docs/deep) |
| `sources/manager.py` | Automated source ingestion scheduling |
| `sources/scheduler.py` | Source sync scheduling (hourly/daily/weekly/monthly) |
| `utils/models.py` | Model registry, shorthand aliases (opus/sonnet/haiku), API cache |

### Code Mode

LLM generates Python scripts for targeted retrieval:

```python
results = obsidian.search("authentication", limit=10, mode="hybrid")
pattern = obsidian.read_note("Patterns/Auth.md")
__result__ = {"context": ..., "citations": [...], "confidence": 0.72}
```

Scripts run in a sandbox with restricted builtins and vault access via `ObsidianTools`.
Search modes: `hybrid` (default, BM25 + vectors), `bm25` (fast keyword), `vector` (semantic).

### Strategy Skeletons

Strategies define required stages and constraints. Example (software):
- Required: clarify_requirements → types_contracts → implementation → verification → synthesis
- Constraints: contracts-first (types before implementation), imports must resolve

## CLI Usage

```bash
# Project setup (configures Claude Code permissions)
shad init
shad check-permissions

# Basic run (uses OBSIDIAN_VAULT_PATH env if --vault not specified)
shad run "Your task" --vault /path/to/vault
shad run "Your task"  # Uses default vault from env

# Software generation with file output
shad run "Build REST API" --vault ~/v --strategy software --verify strict --write-files -o ./out

# Multi-vault (priority order)
shad run "Build app" --vault ~/Project --vault ~/Patterns

# Retriever selection (auto-detects qmd by default)
shad run "Task" --retriever qmd          # Force qmd
shad run "Task" --retriever filesystem   # Force filesystem fallback

# Model selection (per-tier override)
shad models                                    # List Claude models
shad models --refresh                          # Force refresh from API
shad models --ollama                           # Include Ollama models
shad run "Complex task" -O opus -W sonnet -L haiku
shad run "Simple task" -O haiku -W haiku -L haiku

# Ollama integration (free local models)
shad run "Task" -O qwen3-coder -W llama3 -L llama3   # All Ollama
shad run "Task" -O opus -W llama3 -L qwen3:latest   # Mixed

# Check status / inspect
shad status <run_id>
shad trace tree <run_id>
shad trace node <run_id> <node_id>
shad resume <run_id>
shad export <run_id> --output ./out
shad debug <run_id>

# Vault operations
shad vault                           # Check retriever status
shad search "query"                  # Hybrid search (default)
shad search "query" --mode bm25      # Fast keyword search
shad search "query" --mode vector    # Semantic search

# Ingest content
shad ingest github <url> --vault ~/v --preset docs

# Server management
shad server start|stop|status|logs

# Source scheduling (automated sync)
shad sources add github <url> --vault ~/v --schedule weekly
shad sources add url <url> --vault ~/v --schedule daily
shad sources add feed <url> --vault ~/v --schedule hourly
shad sources list
shad sources status
shad sources sync [--force]
shad sources remove <source_id>
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
- `PLAN.md`: Implementation phases and status
