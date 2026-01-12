# Shad (Shannon's Daemon)

**Shad** is a personal AI infrastructure (PAI) designed to operate over *arbitrarily large knowledge environments* using **Recursive Language Models (RLMs)**, **OpenNotebookLM**, and **workflow orchestration via n8n**.

Shad is not a chatbot.
Shad is not a prompt collection.
Shad is not a single model.

Shad is a **self-orchestrating cognitive system** that treats context as an environment, not a prompt.

---

## Quick Start

### Prerequisites

- Docker and Docker Compose
- An Anthropic API key (or OpenAI API key)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/shad.git
cd shad

# Copy environment template
cp .env.example .env

# Edit .env with your API key
# ANTHROPIC_API_KEY=your_key_here

# Deploy
./scripts/deploy.sh
```

### CLI Usage

```bash
# Install CLI (from services/shad-api directory)
cd services/shad-api
pip install -e .

# Run a reasoning task
shad run "What are the key themes in quantum computing?" --max-depth 2

# Check run status
shad status <run_id>

# View execution tree
shad trace tree <run_id>

# Resume a partial run
shad resume <run_id>
```

### API Usage

```bash
# Health check
curl http://localhost:8000/v1/health

# Execute a run
curl -X POST http://localhost:8000/v1/run \
  -H "Content-Type: application/json" \
  -d '{"goal": "Explain the key concepts in machine learning"}'

# Get run results
curl http://localhost:8000/v1/run/<run_id>

# Resume a partial run
curl -X POST http://localhost:8000/v1/run/<run_id>/resume
```

---

## Why Shad Exists

Most AI systems break down when:

* context grows too large
* reasoning requires multiple passes
* answers need to be verifiable
* workflows need to run unattended
* knowledge must persist and compound over time

Shad is built around a different premise:

> **Long-context reasoning is an inference problem, not a prompting problem.**

Shad combines:

* **Recursive Language Models (RLMs)** for inference-time scaling
* **OpenNotebookLM** as a persistent memory and retrieval substrate
* **n8n** for scheduling, fan-out, and workflow orchestration
* **Caching + verification** to make recursion efficient and safe

---

## Core Concepts

### 1. Prompt-as-Environment (RLM)

Shad does not shove massive context into a single prompt.

Instead:

* the "prompt" is treated as an **external environment**
* the model writes code to inspect, slice, and query that environment
* the system recursively calls itself on sub-problems
* results are cached, verified, and recomposed

This allows Shad to reason over **millions of tokens** without exceeding model context windows.

---

### 2. OpenNotebookLM as Memory OS

OpenNotebookLM provides:

* notebooks, sources, and notes
* full-text and vector search
* stable identifiers for retrieved knowledge

Shad uses OpenNotebookLM for:

* long-term memory
* evidence storage
* citation and traceability
* knowledge reuse across runs

OpenNotebookLM is treated as **read-only input during reasoning**, and **write-only output during persistence**.

---

### 3. Budget Controls

Every run enforces hard limits:

| Budget | Description |
|--------|-------------|
| `max_wall_time` | Total execution time |
| `max_tokens` | Per tier + total token budget |
| `max_nodes` | Maximum DAG nodes |
| `max_depth` | Maximum recursion depth |
| `max_branching_factor` | Maximum children per node |

When budgets are exhausted, Shad returns partial results with:
- Completed subtrees
- Missing branches list
- Suggested next run plan
- Resume commands

---

### 4. Skills (Personalization Layer)

Shad is extended through **Skills** — modular, composable units of domain expertise.

Each skill contains:

```
Skills/<SkillName>/
├── SKILL.md        # routing rules + domain knowledge
├── workflows/      # step-by-step procedures
├── tools/          # deterministic helpers
└── tests/          # evals and regressions
```

Skills allow Shad to:

* behave consistently across runs
* encode your preferences once
* improve without rewriting prompts

---

### 5. History & Artifacts

Every run produces structured artifacts:

```
History/Runs/<run_id>/
├── run.manifest.json      # Inputs, versions, config
├── events.jsonl           # Node lifecycle events
├── dag.json               # DAG structure with statuses
├── metrics/
│   ├── nodes.jsonl        # Per-node metrics
│   └── summary.json       # Rollup metrics
├── replay/
│   └── manifest.json      # Deterministic replay bundle
├── final.report.md        # Human-readable output
└── final.summary.json     # Machine-readable output
```

Nothing important is lost. Shad compounds.

---

## High-Level Architecture

```
User / n8n
   |
   v
Shad API / CLI (:8000)
   |
   +-- Skill Router (GoalSpec → Skill selection)
   |
   +-- RLM Engine (recursive DAG execution)
   |       |
   |       +-- Open Notebook (:5055 API, :8502 Web UI)
   |       |       |
   |       |       +-- SurrealDB (:8001)
   |       |
   |       +-- Redis (:6379)
   |       +-- LLM Providers (Claude Code CLI)
   |
   +-- History/ (structured run artifacts)
   |
   +-- Voice Renderer (persona layer)
```

---

## Repository Structure

```
shad/
├── docker-compose.yml
├── services/
│   └── shad-api/              # RLM engine + orchestration API
│       ├── src/shad/
│       │   ├── api/           # FastAPI endpoints
│       │   ├── cache/         # Redis caching
│       │   ├── cli/           # Click CLI
│       │   ├── engine/        # RLM + LLM providers
│       │   ├── history/       # Artifact management
│       │   ├── integrations/  # n8n integration
│       │   ├── learnings/     # Learning extraction
│       │   ├── models/        # Pydantic models
│       │   ├── notebook/      # Open Notebook client
│       │   ├── skills/        # Skill router
│       │   ├── utils/         # Configuration
│       │   ├── verification/  # Validators, HITL
│       │   └── voice/         # Voice rendering
│       ├── Dockerfile
│       └── pyproject.toml
├── Skills/                    # Personalization modules
│   └── research/              # Example skill
├── CORE/                      # Constitution, policies, invariants
│   ├── invariants.md
│   ├── invariants.yaml
│   └── Voices/
├── History/                   # Generated at runtime (volume)
├── scripts/
│   └── deploy.sh
├── .env.example
├── CLAUDE.md
├── PLAN.md
├── SPEC.md
└── README.md
```

---

## API Endpoints

### Runs
| Endpoint | Description |
|----------|-------------|
| `POST /v1/run` | Execute a reasoning task |
| `GET /v1/run/:id` | Get run status and results |
| `POST /v1/run/:id/resume` | Resume a partial/failed run |
| `GET /v1/runs` | List recent runs |

### Notebooks (Open Notebook)
| Endpoint | Description |
|----------|-------------|
| `GET /v1/notebooks` | List all notebooks |
| `GET /v1/notebooks/:id` | Get notebook details |
| `POST /v1/notebooks` | Create a new notebook |
| `GET /v1/notebooks/:id/sources` | List sources in a notebook |
| `POST /v1/notebooks/:id/sources` | Add a source to a notebook |
| `POST /v1/notebooks/:id/sources/url` | Add source from URL |
| `POST /v1/notebooks/:id/search` | Search notebook content |
| `GET /v1/notebooks/:id/notes` | List notes in a notebook |
| `POST /v1/notebooks/:id/notes` | Create a note |

### Admin & Health
| Endpoint | Description |
|----------|-------------|
| `GET /v1/health` | Health check |
| `GET /v1/skills` | List available skills |
| `GET /v1/admin/cache/stats` | Cache statistics |
| `GET /v1/admin/hitl/queue` | HITL review queue |

---

## CLI Commands

```bash
# Start a run
shad run "goal text" --notebook <id> --max-depth 3

# Check status
shad status <run_id>

# Inspect the trace
shad trace tree <run_id>
shad trace node <run_id> <node_id>

# Resume / replay
shad resume <run_id>

# Debug mode
shad debug <run_id>
```

---

## Current Status

Shad is **fully implemented** with all planned features:

### MVP Features (Complete)
- [x] Infrastructure scaffold (Docker, services)
- [x] Shad API v0 (`/v1/run`, `/v1/health`)
- [x] CLI commands (run, status, trace, resume, debug)
- [x] RLM recursion loop with decomposition
- [x] Budget controls with partial results
- [x] History artifacts generation
- [x] Resume from checkpoint
- [x] OpenNotebookLM data model

### Post-MVP Features (Complete)
- [x] Redis caching with hierarchical keys
- [x] Skill router + skill composition
- [x] n8n webhook integration
- [x] Verification loops / validators / entailment checking
- [x] HITL review queues
- [x] Novelty detection for pruning
- [x] Voice rendering system
- [x] Learnings extraction and promotion pipeline

### Open Notebook Integration (Complete)
- [x] Open Notebook service integration (lfnovo/open-notebook)
- [x] Notebook API endpoints for CRUD operations
- [x] Search and retrieval from notebooks
- [x] Context injection into RLM reasoning

**Future work:**

- [ ] Inspection code sandbox
- [ ] Payment/entitlement system
- [ ] Multi-tenant support
- [ ] Offline mode

---

## Philosophy

> Solve a problem once.
> Encode it as infrastructure.
> Never solve it again.

Shad is built to **augment a human**, not replace one.

---

## License

MIT
