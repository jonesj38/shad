# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Shad (Shannon's Daemon) is a personal AI infrastructure (PAI) for long-context reasoning over large knowledge environments. It is a self-orchestrating cognitive system that treats context as an environment, not a prompt.

Key premise: Long-context reasoning is an inference problem, not a prompting problem.

## Architecture

```
User / n8n
   |
   v
Shad API / CLI
   |
   +-- OpenNotebookLM (memory, retrieval, notes)
   |
   +-- RLM Engine (recursive reasoning loop)
   |
   +-- Redis (subtree caching)
   |
   +-- History + Hooks
```

### Core Components

- **RLM Engine**: Recursive Language Model engine that treats prompts as external environments, writes code to inspect/slice/query that environment, recursively calls itself on sub-problems, and caches/verifies/recomposes results
- **OpenNotebookLM**: Memory substrate providing notebooks, sources, notes, full-text and vector search. Read-only during reasoning, write-only during persistence
- **n8n**: Workflow orchestrator for cron/event triggers, fan-out, parallel execution, and maintenance tasks
- **Redis**: Subtree caching to turn recursive reasoning from a tree into a DAG

### Skills System

Skills are modular, composable units of domain expertise:
```
Skills/<SkillName>/
├── SKILL.md        # routing rules + domain knowledge
├── workflows/      # step-by-step procedures
├── tools/          # deterministic helpers
└── tests/          # evals and regressions
```

## Planned Repository Structure

```
shad/
├── docker-compose.yml
├── services/
│   └── shad-api/          # RLM engine + orchestration API
├── Skills/                # personalization modules
├── CORE/                  # constitution, policies, invariants
├── hooks/                 # lifecycle automation
├── History/               # generated at runtime (volume)
├── scripts/               # deploy / maintenance helpers
└── .env.example
```

## Development Commands

Once infrastructure is set up:

```bash
# Deploy to server
scripts/deploy.sh

# Run services locally
docker compose up -d

# Rebuild and run
docker compose up -d --build
```

## API Endpoints (Planned)

- `POST /v1/run` - Execute a reasoning task (accepts `{ goal, notebook_id?, budgets? }`)
- `GET /v1/health` - Health check

## Current Status

This project is in early scaffold phase. See PLAN.md for implementation roadmap.
