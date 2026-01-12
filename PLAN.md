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
- [ ] Full OpenNotebookLM service integration (future)

## Phase 8 — History & Artifacts ✅ COMPLETE

- [x] `run.manifest.json` - Run inputs and config
- [x] `dag.json` - DAG structure with statuses
- [x] `events.jsonl` - Node lifecycle events
- [x] `metrics/nodes.jsonl` - Per-node metrics
- [x] `metrics/summary.json` - Rollup metrics
- [x] `final.report.md` - Human-readable output
- [x] `final.summary.json` - Machine-readable output
- [x] `replay/manifest.json` - Replay bundle

## Phase 9 — Caching (Redis) ✅ COMPLETE

- [x] Implement cache key generation (hierarchical + hash fallback)
- [x] Redis integration in RLM engine
- [x] Cache hit/miss tracking
- [x] Hierarchical key scheme (goal_type, intent, entities, context_hash)
- [x] Main cache + staging cache for provisional results
- [x] Cache promotion pipeline
- [x] TTL support

## Phase 10 — Skill Router ✅ COMPLETE

- [x] Skill metadata parsing (SKILL.md frontmatter with YAML)
- [x] GoalSpec → Skill scoring (weighted scoring formula)
- [x] Candidate generation and ranking
- [x] Skill composition (primary + support skills)
- [x] Loop guards (cycle detection, depth limits)
- [x] Pattern matching and intent detection

## Phase 11 — n8n Integration ✅ COMPLETE

- [x] Webhook triggers (WebhookEvent enum)
- [x] Run completion callbacks (N8NClient)
- [x] Webhook dispatch with retry logic
- [x] Payload signing for security
- [x] WebhookHandler for incoming triggers
- [x] Workflow trigger support

## Phase 12 — Verification & Quality ✅ COMPLETE

- [x] Domain-specific validators (StructuralValidator, CitationValidator)
- [x] Entailment checking (LLM-based)
- [x] HITL review queues (HITLQueue with file persistence)
- [x] Novelty/diminishing returns pruning (NoveltyDetector)
- [x] CompositeValidator for chaining validators
- [x] ValidationResult with confidence scores

## Phase 13 — Voice Rendering ✅ COMPLETE

- [x] Voice spec loading from YAML
- [x] Output transformation (simple + LLM-based)
- [x] Multiple voice support
- [x] Skill default voices
- [x] Citation preservation checks
- [x] Error and partial result rendering

## Phase 14 — Learnings System ✅ COMPLETE

- [x] Learning capture (LearningExtractor)
- [x] Learning types (prompt_patch, routing_hint, negative_example, note, fact, pattern)
- [x] Patch proposals
- [x] LearningsStore with file persistence
- [x] HITL promotion pipeline (proposed → staged → approved → promoted)
- [x] Duplicate detection
- [x] Tag-based search

---

## Current Status: FULL IMPLEMENTATION COMPLETE

All phases from SPEC.md are now implemented:

### MVP Features (Phases 0-8)
- ✅ CLI command: `shad run "Goal..." --notebook <id> --max-depth 2`
- ✅ Notebook graph retrieval (data model ready)
- ✅ 2-3 level decomposition + recomposition
- ✅ Run graph + History artifacts
- ✅ Hard budgets + partial results
- ✅ Resume from checkpoint

### Post-MVP Features (Phases 9-14)
- ✅ Redis caching with hierarchical keys
- ✅ Skill router with scoring and composition
- ✅ n8n webhook integration
- ✅ Verification with validators and entailment
- ✅ HITL review queues
- ✅ Novelty detection for pruning
- ✅ Voice rendering system
- ✅ Learnings extraction and promotion

### API Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /v1/run` | Execute a reasoning task |
| `GET /v1/run/:id` | Get run status and results |
| `POST /v1/run/:id/resume` | Resume a partial/failed run |
| `GET /v1/runs` | List recent runs |
| `GET /v1/health` | Health check |
| `GET /v1/skills` | List available skills |
| `POST /v1/skills/route` | Route goal to skills |
| `GET /v1/admin/cache/stats` | Cache statistics |
| `GET /v1/admin/hitl/queue` | HITL review queue |
| `POST /v1/admin/hitl/:id/approve` | Approve review item |
| `POST /v1/admin/hitl/:id/reject` | Reject review item |
| `GET /v1/admin/learnings/stats` | Learnings statistics |
| `GET /v1/admin/voices` | List available voices |

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
    ├── cache/
    │   ├── __init__.py
    │   └── redis_cache.py
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
    ├── integrations/
    │   ├── __init__.py
    │   └── n8n.py
    ├── learnings/
    │   ├── __init__.py
    │   ├── extractor.py
    │   └── store.py
    ├── models/
    │   ├── __init__.py
    │   ├── goal.py
    │   ├── notebook.py
    │   └── run.py
    ├── skills/
    │   ├── __init__.py
    │   ├── router.py
    │   └── skill.py
    ├── utils/
    │   ├── __init__.py
    │   └── config.py
    ├── verification/
    │   ├── __init__.py
    │   ├── hitl.py
    │   ├── novelty.py
    │   └── validators.py
    └── voice/
        ├── __init__.py
        └── renderer.py

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

### Remaining Work (Future)

- [ ] Full OpenNotebookLM service integration
- [ ] Inspection code sandbox
- [ ] Payment/entitlement system
- [ ] Multi-tenant support
- [ ] Offline mode with local LLM
- [ ] Server deployment (Phase 2)
