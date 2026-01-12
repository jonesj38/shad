# Shad Implementation Plan

## Phase 0 — Foundation Decisions ✅ COMPLETE

- [x] Repo strategy: `shad` as primary repo
- [x] Runtime form: `shad-api` service (HTTP) + `shad` CLI
- [x] Server deployment: Docker Compose only

## Phase 1 — Repo Skeleton ✅ COMPLETE

- [x] `docker-compose.yml` with services
- [x] `services/shad-api/` with full implementation
- [x] `Skills/research/SKILL.md` (example skill)
- [x] `CORE/invariants.md` and `CORE/invariants.yaml`
- [x] `CORE/Voices/` directory with voice specs
- [x] `.env.example`
- [x] `scripts/deploy.sh`

## Phase 2 — Server Foundation

- [ ] Install Docker + Docker Compose on server
- [ ] Configure firewall rules
- [ ] Create `/opt/shad/` runtime directory
- [ ] Set up deployment workflow

## Phase 3 — Empty Stack ✅ COMPLETE

- [x] `docker-compose.yml` with:
  - [x] `shad-api` service
  - [x] `redis` cache
- [ ] Verify all ports accessible
- [ ] Test basic connectivity

## Phase 4 — Shad API v0 ✅ COMPLETE

- [x] `POST /v1/run` - Execute reasoning task
- [x] `GET /v1/run/:id` - Get run status/results
- [x] `POST /v1/run/:id/resume` - Resume partial run
- [x] `GET /v1/runs` - List recent runs
- [x] `GET /v1/health` - Health check
- [x] History folder generation (`History/Runs/<run_id>/`)
- [x] Structured artifacts (manifest, dag, metrics, reports)

## Phase 5 — CLI ✅ COMPLETE

- [x] `shad run` - Execute reasoning task
- [x] `shad status` - Check run status
- [x] `shad trace tree` - View DAG tree
- [x] `shad trace node` - Inspect specific node
- [x] `shad resume` - Resume partial/failed run
- [x] `shad debug` - Debug mode with full info

## Phase 6 — RLM Engine ✅ COMPLETE

- [x] Goal normalization (GoalSpec)
- [x] Task decomposition (LLM-powered)
- [x] Recursive execution with depth control
- [x] Result synthesis
- [x] Budget enforcement:
  - [x] `max_depth`
  - [x] `max_nodes`
  - [x] `max_wall_time`
  - [x] `max_tokens`
  - [x] `max_branching_factor`
- [x] Partial results on budget exhaustion
- [x] Resume from checkpoint

## Phase 7 — OpenNotebookLM Data Model ✅ COMPLETE

- [x] Node types (Notebook, Source, Note)
- [x] Edge types (DERIVED_FROM, SUMMARIZES, etc.)
- [x] Graph-based retrieval model
- [x] RetrievalResult structure
- [ ] Full OpenNotebookLM service integration (post-MVP)

## Phase 8 — History & Artifacts ✅ COMPLETE

- [x] `run.manifest.json` - Run inputs and config
- [x] `dag.json` - DAG structure with statuses
- [x] `events.jsonl` - Node lifecycle events
- [x] `metrics/nodes.jsonl` - Per-node metrics
- [x] `metrics/summary.json` - Rollup metrics
- [x] `final.report.md` - Human-readable output
- [x] `final.summary.json` - Machine-readable output
- [x] `replay/manifest.json` - Replay bundle

## Post-MVP Phases

### Phase 9 — Caching (Redis)

- [ ] Implement cache key generation
- [ ] Redis integration in RLM engine
- [ ] Cache hit/miss tracking
- [ ] Hierarchical key scheme

### Phase 10 — Skill Router

- [ ] Skill metadata parsing (SKILL.md frontmatter)
- [ ] GoalSpec → Skill scoring
- [ ] Candidate generation and ranking
- [ ] Skill composition

### Phase 11 — n8n Integration

- [ ] Webhook triggers
- [ ] Run completion callbacks
- [ ] Fan-out/fan-in workflows
- [ ] Scheduling automation

### Phase 12 — Verification & Quality

- [ ] Domain-specific validators
- [ ] Entailment checking
- [ ] HITL review queues
- [ ] Novelty/diminishing returns pruning

### Phase 13 — Voice Rendering

- [ ] Voice spec loading
- [ ] Output transformation
- [ ] Multiple voice support
- [ ] Skill default voices

### Phase 14 — Learnings System

- [ ] Learning capture as notes
- [ ] Patch proposals
- [ ] Eval-based testing
- [ ] HITL promotion pipeline

---

## Current Status: MVP COMPLETE

All MVP requirements from SPEC.md Section 14.1 are implemented:

- ✅ CLI command: `shad run "Goal..." --notebook <id> --max-depth 2`
- ✅ Notebook graph retrieval (data model ready)
- ✅ 2-3 level decomposition + recomposition
- ✅ Run graph + History artifacts
- ✅ Hard budgets + partial results
- ✅ Resume from checkpoint

### Files Created

```
services/shad-api/
├── pyproject.toml
├── Dockerfile
└── src/shad/
    ├── __init__.py
    ├── api/
    │   ├── __init__.py
    │   └── main.py
    ├── cli/
    │   ├── __init__.py
    │   └── main.py
    ├── engine/
    │   ├── __init__.py
    │   ├── llm.py
    │   └── rlm.py
    ├── history/
    │   ├── __init__.py
    │   └── manager.py
    ├── models/
    │   ├── __init__.py
    │   ├── goal.py
    │   ├── notebook.py
    │   └── run.py
    └── utils/
        ├── __init__.py
        └── config.py

CORE/
├── invariants.md
├── invariants.yaml
└── Voices/
    ├── default.yaml
    └── researcher.yaml

Skills/
└── research/
    └── SKILL.md

scripts/
└── deploy.sh

docker-compose.yml
.env.example
```
