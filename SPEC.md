# SPEC.md

Shad (Shannon's Daemon) — Technical Specification

## Overview

Shad is a personal AI infrastructure (PAI) for long-context reasoning over large knowledge environments. It is a self-orchestrating cognitive system that treats context as an environment, not a prompt.

**Core premise**: Long-context reasoning is an inference problem, not a prompting problem.

**Storage backend**: Local-first Markdown via **Obsidian**, accessed via **Model Context Protocol (MCP)**.

> **Note**: See `OBSIDIAN_PIVOT.md` for the detailed Obsidian integration specification.

---

## 1. Architecture

### 1.1 High-Level Flow

```
User / n8n
   |
   v
Shad CLI / API
   |
   +-- Skill Router (GoalSpec → Skill selection)
   |
   +-- RLM Engine (recursive DAG execution)
   |       |
   |       +-- OpenNotebookLM (graph-based knowledge retrieval)
   |       +-- Redis (subtree caching)
   |       +-- LLM Providers (tiered model calls)
   |
   +-- History/ (structured run artifacts)
   |
   +-- Voice Renderer (persona layer)
```

### 1.2 Core Components

| Component | Role |
|-----------|------|
| **RLM Engine** | Recursive Language Model engine. Treats prompts as external environments, generates code to inspect/query that environment, recursively decomposes problems, caches/verifies/recomposes results. |
| **Obsidian Vault** | Local-first Markdown knowledge substrate. Uses frontmatter for structured metadata, Bases plugin for database-like views. |
| **MCP Client** | Connects to Obsidian MCP Server. Translates RLM intents into tool execution requests. |
| **Code Sandbox** | Docker container with vault bind-mounted. Executes RLM-generated Python scripts in isolation. |
| **Skill Router** | Converts goals into GoalSpecs, scores candidate skills, selects primary + support skills, enforces loop guards. |
| **Redis** | Subtree caching with hierarchical keys and hash validation. Central ledger for budget enforcement. |
| **n8n** | Thin trigger + automation layer. Handles scheduling, event wiring, fan-out/fan-in of runs (not nodes), integrations. |
| **History/** | Structured, append-only run artifacts stored inside the Obsidian vault at `Vault/Shad/History/`. |

### 1.3 Code Mode (RLM Pattern)

Instead of chat-based tool calling, Shad implements **Code Execution with MCP**:

1. RLM writes a Python script that imports MCP tools (e.g., `obsidian.search`, `obsidian.read_note`)
2. Script executes in sandboxed container with vault access
3. Script filters, aggregates, and processes vault data *before* returning results
4. Only final distilled output enters the context window

This reduces context pollution and enables complex vault queries.

### 1.3 Tenancy Model

- **v1**: Single-user only
- **Architecture**: Multi-tenant capable (proper isolation hooks designed in)
- **Deployment**: Do not deploy multi-tenant initially

---

## 2. Interface

### 2.1 Primary Mode: CLI-First, Run-Oriented

The core UX is triggering runs from the terminal. Everything else (web UI, chat, n8n dashboards) is a view or trigger on top of the run model.

```bash
# Start a run
shad run "answer X using notebook Y" --notebook <id> --max-depth 3

# Check status
shad status <run_id>

# Inspect the trace
shad trace tree <run_id>
shad trace node <run_id> <node_id>

# Resume / replay
shad resume <run_id>
shad replay --from-node <node_id>

# Debug mode (CLI power users)
shad debug <run_id>
```

### 2.2 API Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /v1/run` | Execute a reasoning task. Accepts `{ goal, notebook_id?, budgets?, voice? }` |
| `GET /v1/run/:id` | Get run status and results |
| `POST /v1/run/:id/resume` | Resume a partial/failed run |
| `GET /v1/health` | Health check |

---

## 3. OpenNotebookLM Data Model

### 3.1 Graph Structure (Not Hierarchy)

Notebook, Source, and Note are all **nodes in a graph**. Relationships are **typed edges**. Hierarchy is a projection, not the ontology.

| Node Type | Definition |
|-----------|------------|
| **Notebook** | Contextual container / "lens" / "view over the graph". Groups references, defines context for retrieval, may impose policies. Does NOT own content. |
| **Source** | External or primary artifact (PDF, web page, transcript, code repo, etc.). Has immutable identity, metadata, extracted representations. |
| **Note** | Derived, human- or AI-authored artifact (summary, annotation, synthesis, hypothesis, etc.). References sources and/or other notes. |

### 3.2 Edge Types

```
NOTE  ─DERIVED_FROM─>  SOURCE
NOTE  ─DERIVED_FROM─>  NOTE
NOTE  ─SUMMARIZES──>   SOURCE
NOTE  ─CONTRADICTS─>   NOTE
NOTE  ─SUPPORTS────>   NOTE
NOTE  ─REFERENCES──>   SOURCE
NOTE  ─PART_OF─────>   NOTEBOOK
SOURCE ─INCLUDED_IN─>  NOTEBOOK
```

### 3.3 Retrieval Model

Shad doesn't "load a notebook." It queries the graph:
> "Give me notes + sources reachable from Notebook X within depth D, filtered by trust/payment/provisional state."

---

## 4. Skill System

### 4.1 Skill Structure

```
Skills/<SkillName>/
├── SKILL.md        # Routing rules + domain knowledge (YAML frontmatter)
├── workflows/      # Step-by-step procedures
├── tools/          # Deterministic helpers
└── tests/          # Evals and regressions
```

### 4.2 Skill Metadata (SKILL.md frontmatter)

```yaml
name: research
version: 1.2.0
description: Deep research with citation tracking
use_when:
  - "research *"
  - "investigate *"
  - "find evidence for *"
intents: [research, investigate, summarize]
entities: [academic, technical]
inputs_schema: { goal: string, notebook_id: string }
outputs_schema: { summary: string, citations: array }
tools_allowed: [retrieve, embed, summarize]
priority: 10
cost_profile: expensive
composes_with: [citations, images]
exclusions: ["quick questions", "simple lookups"]
default_voice: researcher
entry_workflows: [default, quick, thorough]
```

### 4.3 Skill Routing Flow

1. **Normalize goal → GoalSpec**
   - `raw_goal`, `normalized_goal`, `intent`, `entities`, `constraints`, `context`, `risk_level`

2. **Candidate generation** (fast recall, 5-12 skills)
   - Keyword/pattern match (high precision)
   - Intent match (high precision)
   - Embedding similarity (high recall)
   - Context gates (hard filters)

3. **Ranking** (deterministic scoring)
   ```
   Score(skill) =
     + W_intent * intent_match
     + W_triggers * trigger_match_strength
     + W_context * context_match
     + W_embed * embedding_similarity
     + W_history * past_success_rate
     - W_risk * risk_mismatch
     - W_exclusion * exclusion_hit
     + W_priority * priority
   ```

4. **Selection**
   - One primary skill (always)
   - Support skills only if clearly needed

5. **Composition**: Primary skill orchestrates support skills via direct invocation

### 4.4 Skill Loop Guards

- **Call stack tracking**: Reject invocations that would create a cycle
- **Depth limits per skill**: Each skill has max invocation depth
- **Static analysis**: Lint tool to detect potential cycles (not runtime defense)

---

## 5. RLM Engine

### 5.1 Recursive Execution Model

```
Goal
  ├── Decompose into sub-questions/tasks
  │     ├── Sub-task 1 → Retrieve → Reason → Cache
  │     ├── Sub-task 2 → Retrieve → Reason → Cache
  │     └── Sub-task 3 → (recursive decomposition if needed)
  ├── Verify results
  └── Recompose into final answer
```

### 5.2 Budget Controls (Non-Negotiable)

Every run gets hard limits:

| Budget | Description |
|--------|-------------|
| `max_wall_time` | Total execution time |
| `max_tokens` | Per tier + total token budget |
| `max_nodes` | Maximum DAG nodes |
| `max_depth` | Maximum recursion depth |
| `max_branching_factor` | Maximum children per node |

**Behavior at budget exhaustion**: Return partial results with completed subtrees, missing branches list, and suggested next run plan.

### 5.3 Pruning Strategy

**Primary signal**: Diminishing returns (novelty metric)

Detection layers:
1. **Cheap prefilter**: Embedding distance from existing results
2. **Backbone**: Fact extraction diff (new facts vs. rephrased existing)
3. **Tie-breaker**: LLM judge for marginal value scoring

### 5.4 Model Tiering

**Profiles** (capability-based, not provider-specific):

| Profile | Use Case |
|---------|----------|
| `ORCHESTRATOR` | Best reasoning/planning (expensive) |
| `WORKER` | Balanced mid-depth recursion |
| `LEAF` | Fast/cheap parallel calls |
| `JUDGE` | Evals, novelty, verification |
| `EMBEDDER` | Routing, cache keys, similarity |

**Tier routing**: Depth-based degradation + uncertainty-aware verification gates

**Provider priority**:
1. Anthropic (Claude Opus/Sonnet/Haiku) — primary
2. OpenAI (GPT-5.x / GPT-4.1 / embeddings) — fallback
3. Google Gemini (Pro/Flash) — alternative
4. Local (Ollama/vLLM) — offline/privacy tier

### 5.5 Caching

**Key scheme**: Hierarchical keys as primary
```
(goal_type, intent, entities, key_slots...) → stable key
```

- **Fallback**: Exact string hash
- **Optional**: LLM canonicalization for expensive subcalls only
- **Avoid**: Pure embedding-hash as main mechanism

**Cache namespaces**:
- **Main cache**: Reviewed, trusted results
- **Staging cache**: Provisional results awaiting review
- Provisional results do NOT enter main cache until reviewed

### 5.6 Contradiction Handling

**Policy**: Surface both contradictions; let reasoning handle uncertainty

**Per-skill handling** (context-dependent):
- Probabilistic reasoning (carry forward multiple hypotheses)
- Investigative escalation (spawn sub-task to resolve)
- Flag and continue (note contradiction, proceed with best interpretation)

---

## 6. Verification & Quality

### 6.1 Verification Signals

Layered approach for caching decisions:

1. **Domain-specific validators**: Per-skill validation functions checking structural/factual constraints
2. **Entailment checking**: Verify answer is logically entailed by retrieved evidence
3. **Human-in-the-loop**: Flag low-confidence results for batch review

### 6.2 HITL Model

**Latency tolerance**: Batch review (async)
- Queue flagged items for periodic human review
- Runs complete with provisional results
- Provisional results get taint propagation + expiring cache

### 6.3 Eval Strategy

| Layer | Purpose | Method |
|-------|---------|--------|
| **0: Invariants** | Safety floor | DAG validity, budget bounds, provenance, citation format, tool policy |
| **1: Golden dataset** | Regression guards | Curated (GoalSpec, notebook slice, expected properties) pairs |
| **2: Comparative runs** | Quality improvement | Same goal, different configs; human judges winner |
| **3: Live feedback** | Long-term signal | Structured thumbs up/down with tags |

---

## 7. History & Observability

### 7.1 Directory Structure

```
History/Runs/<run_id>/
├── run.manifest.json      # Inputs, versions, config hashes
├── events.jsonl           # Node lifecycle events (flight recorder)
├── dag.json               # DAG structure with statuses
├── decisions/
│   ├── routing.json       # Skill routing decision
│   └── decomposition/     # Per-node decomposition decisions
├── metrics/
│   ├── nodes.jsonl        # Per-node metrics
│   └── summary.json       # Rollup metrics
├── errors/                # Error records with context
├── artifacts/             # Large payloads (referenced by hash)
├── inspection/            # Derived analysis outputs
├── replay/
│   └── manifest.json      # Deterministic replay bundle
├── final.report.md        # Human-readable output
├── final.summary.json     # Machine-readable output
└── final.<voice>.md       # Voice-rendered output (optional)
```

### 7.2 Correlation IDs

Every event carries:
- `run_id` (entire execution)
- `trace_id` (distributed tracing)
- `node_id` (each recursive subproblem)
- `parent_node_id`
- `depth`
- `attempt` (retry count)
- `skill_id`
- `cache_key(s)`

### 7.3 Node Lifecycle Events

```
NODE_CREATED → NODE_READY → NODE_STARTED →
  NODE_TOOL_CALL → NODE_MODEL_CALL →
  NODE_CACHE_HIT / NODE_CACHE_MISS →
  NODE_RETRY_SCHEDULED →
  NODE_FAILED / NODE_SUCCEEDED →
  NODE_PROMOTED (staging→main)
```

### 7.4 Retention Policy

| Tier | Duration | Contents |
|------|----------|----------|
| **Hot** | 14 days | Full fidelity (all traces, artifacts, raw outputs) |
| **Warm** | 180 days | Compressed + pruned (gzipped logs, deduplicated content) |
| **Cold** | Forever | Summaries + provenance (manifest, final outputs, decisions, learnings) |

**Special handling**:
- `shad run pin <run_id>` — Never purge full trace
- Size caps per run (prevent pathological explosions)
- GC job: `shad history gc --hot-days 14 --warm-days 180`

---

## 8. Voice & Persona

### 8.1 Architecture

- **Inside the run engine**: Neutral + precise (structured outputs)
- **At the boundary**: Rendered through a voice layer

Voice affects **presentation**, not **truth conditions**.

### 8.2 VoiceSpec

```yaml
# CORE/Voices/kai.yaml
name: kai
tone: playful        # concise | blunt | warm | playful | formal
verbosity: 3         # 1-5
formatting: bullets  # bullets | headings | tables | prose
profanity: allow
citation_style: inline
error_style: transparent
signature: null
```

### 8.3 Skill Default Voices

Skills declare `default_voice` in their metadata. CLI can override:
```bash
shad run --voice kai "..."
```

### 8.4 Voice Constraints

Voice layer may change:
- Phrasing and tone
- Structure and brevity
- Examples and analogies
- How it surfaces caveats

Voice layer must NOT change:
- Claims / facts
- Citations / provenance
- Numeric results
- Safety policies / CORE constraints
- Decisions and stop reasons

---

## 9. Failure Handling

### 9.1 UX Contract

On any non-success outcome, Shad produces:
- `final.summary.json` (machine-readable)
- `final.report.md` (human-readable)

Both include:
- `status`: complete | partial | failed | aborted
- What succeeded (completed nodes/subtrees + key outputs)
- What failed (node IDs, phase, error class, message)
- Impact (what parts may be incomplete/tainted)
- Suggested fixes (retry, switch model, tighten depth, add source)
- Stop reasons (budget hit, novelty pruned, tool blocked)
- Next actions (resume commands + recommended params)

### 9.2 Resume Capability

Every partial/failure emits a resume command:
```bash
shad resume <run_id>
shad resume <run_id> --budget +25% --max-depth 2 --model-tier safer
```

Resume behavior:
- Continue from last checkpoint
- Reuse cached completed nodes
- Optionally re-run failed nodes with changed parameters

### 9.3 Recovery Strategy

| Failure Type | Response |
|--------------|----------|
| Transient (API error, timeout) | Retry with exponential backoff |
| Budget exhaustion | Return partial with missing branches |
| Bad decomposition | Checkpoint + partial |
| Security/policy violation | Full abort |
| Corrupted state | Full abort |

---

## 10. Security & Invariants

### 10.1 Hard Invariants (Constitutional Constraints)

These are non-negotiable. Shad must be architecturally incapable of violating them without explicit, auditable override.

#### 1. Never Auto-Publish

> Shad must not publish, deploy, send, post, or execute irreversible side effects without explicit human approval.

**Covered actions**: Pushing code, deploying infrastructure, posting to social media, sending emails, modifying production systems, issuing payments, changing public-facing pods.

**Enforcement**: Hard gate in CORE; publish tools require `--confirm` or approval token + run ID + user identity; all attempts logged.

#### 2. Never Exfiltrate

> Shad must not send notebook data, derived notes, or pod contents to external services unless explicitly permitted for that run.

**Enforcement**: Default network deny; allowlist per run; "external send" is privileged capability; provenance + consent recorded in History.

#### 3. Never Self-Modify

> Shad must not directly change its own Skills, routing logic, or CORE policies without explicit human review and approval.

**Allowed**: Proposing patches, generating diffs, writing candidates to staging, running evals.

**Enforcement**: Skills/CORE mounted read-only at runtime; promotion pipeline requires human approval + version bump + recorded rationale.

### 10.2 CORE Enforcement Model

- **Hard validation gates**: Anything that can cause harm, leakage, irreversible action, or economic impact (payments, access grants, publishing, deletes, network calls)
- **Soft guidance in prompts**: Tone, formatting, preferences, heuristics

> "Prompts are education, gates are law enforcement."

### 10.3 Inspection Code Sandbox (Future)

**Must support**:
- Read-only access to History artifacts
- Deterministic transformations and analysis
- Local compute and lightweight modeling
- Controlled tool invocation (safe, explicit)
- Structured output to `History/Runs/<run_id>/inspection/`
- Visualization outputs (Mermaid, Graphviz, CSV)

**Must NOT support**:
- Read arbitrary filesystem paths
- Access environment secrets
- Spawn unrestricted subprocesses
- Open arbitrary sockets
- Write outside `inspection/` subtree
- Mutate Skills/CORE directly

---

## 11. Payment Model (Future)

### 11.1 Billing Primitives

| Layer | Model | Description |
|-------|-------|-------------|
| **Primary** | Per-query metered | Pay per retrieval + reasoning operation over a notebook subgraph |
| **Secondary** | Per-artifact unlock | Pay once to permanently unlock a specific note/source |
| **Convenience** | Subscription tiers | Pre-paid query credits, higher limits, freshness windows |

### 11.2 Metering Dimensions

Record for future pricing flexibility:
- Nodes traversed
- Depth reached
- Tokens generated
- Cache hits vs misses
- Fresh vs cached derivations

### 11.3 Entitlement Model

Every traversal request checked against entitlement:
> "Does this run have permission to traverse this edge and read this node?"

BSV micropayment issues short-lived capability tokens scoped to (notebook, depth, node types).

---

## 12. Deployment

### 12.1 v1 Target: Single VPS / Homelab

Docker Compose stack:
- `shad-api` (Run API + CLI entrypoint)
- `shad-worker` (optional: async runs)
- `redis` (cache + run state + queues)
- `opennotebooklm` (knowledge graph)
- `n8n` (scheduling + integrations)
- `caddy`/`traefik` (optional: TLS + reverse proxy)

### 12.2 Volumes

```yaml
volumes:
  - ./History:/data/history
  - ./Notebooks:/data/notebooks
  - ./redis-data:/data/redis
```

### 12.3 Secrets Management

- `.env` file with strict permissions (0600)
- Never write secrets into History
- Log "provider/model used" but redact keys/tokens
- Future: swap in SSM/Secret Manager/K8s Secrets via clean config loader

### 12.4 Ops Story

- Systemd unit or Docker restart policies
- Backups: History/, ONLM data, Redis persistence
- Updates: `git pull && docker compose build && docker compose up -d`

---

## 13. Design Constraints

### 13.1 Offline-First Capability

Shad must be able to operate without internet:
- Local cache (Redis + disk)
- Local embeddings (planned)
- Local LLM provider (Ollama/vLLM)

**Offline mode**:
- `--offline` flag
- Refuse external calls
- Degrade gracefully (partial outputs, "needs online to fetch X")
- Record what would have been done online

### 13.2 Deterministic Replay

Given the same:
- GoalSpec
- Notebook graph snapshot
- Tool outputs (or hashes)
- Model versions
- Prompts
- Random seeds

Shad should reproduce:
- Same DAG structure
- Same decisions
- Same citations
- Functionally equivalent outputs

### 13.3 Audit Everything

Every action must be logged:
- Every model call (provider/model/params/inputs hash/outputs hash/cost)
- Every routing decision (candidates + scores)
- Every DAG node creation/prune/stop reason
- Every mutation (cache writes, notebook writes, pod writes)
- Every external call attempt (even blocked ones)
- Every HITL approval and what it authorized

**Constraints**:
- No secrets in logs (redaction mandatory)
- Logs must be structured (JSONL) with human summaries alongside

---

## 14. MVP Scope

### 14.1 What Must Work

**A) CLI command**
```bash
shad run "Goal text…" --notebook <id> --max-depth 2
```

**B) Notebook graph retrieval**
- Load notebook by ID
- Retrieve relevant sources + notes (top K)
- Provide citations back to node IDs

**C) 2-3 level decomposition + recomposition**
- Decompose goal into 3-7 sub-tasks
- For each: retrieve context, run sub-call
- Recompose into final answer with summary, key points, citations

**D) Run graph + History artifacts**
- `run.manifest.json`
- `routing.json`
- `dag.json`
- `final.report.md`
- `final.summary.json`

**E) Hard budgets + partial results**
- `max-depth`, `max-nodes`, `max-time`
- Return partial with clear report on exhaustion

**F) Resume from checkpoint**
```bash
shad resume <run_id>
```

### 14.2 What MVP Does NOT Need

- Redis semantic cache keys / canonicalization
- Diminishing returns novelty pruning
- Verification loops / judges
- Skill router + skill-to-skill composition
- Learnings extraction + patch promotion
- n8n orchestration (beyond simple trigger)
- SOLID pods + micropayments gating

---

## 15. Learnings System (Post-MVP)

### 15.1 Capture → Promote Pipeline

1. **Capture everything as notes** (default layer)
2. **Propose patches/hints/negatives** (automated suggestions)
3. **Test via evals** (comparative runs)
4. **Promote via HITL review** (human approval)

This gives compounding improvement without self-edit instability.

### 15.2 Learning Types

| Type | Form | Effect |
|------|------|--------|
| Prompt patches | Amendments to skill prompts | Better reasoning |
| Routing hints | "Goals containing X → skill Y" | Better skill selection |
| Negative examples | Failure cases to avoid | Prevent repeated mistakes |
| Notes | OpenNotebookLM entries | Retrieved when relevant |

---

## 16. Top Technical Risk

**Recursion depth explosion**: The failure mode that can silently eat time, money, and trust.

### Mitigation Plan

1. **Hard budgets with graceful degradation**
2. **Diminishing returns pruning** (novelty metric critical)
3. **DAG checkpointing + resume**
4. **First-class stop reasons** (budget, novelty, cycle, confidence, error)
5. **Depth-aware AND uncertainty-aware tiering**

---

## Appendix A: n8n Integration

**Role**: Thin trigger + automation layer (not orchestration peer)

**Responsibilities**:
- Scheduling (cron)
- Event wiring (webhooks)
- Fan-out/fan-in of **runs** (not nodes)
- Integrations (Slack, email, SOLID, BSV)

**n8n workflow pattern**:
1. Start run
2. Wait for completion webhook
3. Fetch `final.summary.json`
4. Route by status:
   - `complete` → publish/notify
   - `partial` → create HITL review task or auto-resume
   - `failed` → alert + log

---

## Appendix B: Repository Structure (Planned)

```
shad/
├── docker-compose.yml
├── services/
│   └── shad-api/              # RLM engine + orchestration API
├── Skills/                    # Personalization modules
│   └── <SkillName>/
│       ├── SKILL.md
│       ├── workflows/
│       ├── tools/
│       └── tests/
├── CORE/                      # Constitution, policies, invariants
│   ├── invariants.md
│   ├── invariants.yaml
│   └── Voices/
│       └── <voice>.yaml
├── hooks/                     # Lifecycle automation
├── History/                   # Generated at runtime (volume)
├── scripts/                   # Deploy / maintenance helpers
├── .env.example
├── CLAUDE.md
├── PLAN.md
└── SPEC.md
```
