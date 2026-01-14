# Shad (Shannon's Daemon) — Obsidian Edition

## Technical Specification v2.0

**Status:** APPROVED
**Last Updated:** 2026-01-13
**Minimum Obsidian Version:** 1.10+

---

## 1. Overview

Shad is a personal AI infrastructure (PAI) for long-context reasoning over large knowledge environments.

**Core premise:** Long-context reasoning is an inference problem, not a prompting problem.

**Storage backend:** Local-first Markdown via Obsidian, accessed via Model Context Protocol (MCP).

### 1.1 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User Interface                               │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────────────────┐  │
│  │   CLI    │    │  HTTP    │    │           n8n                │  │
│  │ (shad)   │    │   API    │    │   (Watch Folder Triggers)    │  │
│  └────┬─────┘    └────┬─────┘    └──────────────┬───────────────┘  │
└───────┼───────────────┼─────────────────────────┼───────────────────┘
        │               │                         │
        └───────────────┴─────────────┬───────────┘
                                      │
┌─────────────────────────────────────┴───────────────────────────────┐
│                        Shad Core Engine                              │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                      RLM Engine                               │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │  │
│  │  │ Decomposer  │──│  Executor   │──│    Synthesizer      │  │  │
│  │  │ (LLM-driven)│  │ (Code Mode) │  │ (Partial Results)   │  │  │
│  │  └─────────────┘  └──────┬──────┘  └─────────────────────┘  │  │
│  └──────────────────────────┼───────────────────────────────────┘  │
│                             │                                       │
│  ┌──────────────┐  ┌────────┴────────┐  ┌──────────────────────┐  │
│  │ Skill Router │  │ Budget Enforcer │  │   Verification       │  │
│  │ (Layered)    │  │ (Redis Ledger)  │  │ (Entailment Check)   │  │
│  └──────────────┘  └─────────────────┘  └──────────────────────┘  │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    MCP Client (Python)                        │  │
│  └──────────────────────────────┬───────────────────────────────┘  │
└─────────────────────────────────┼───────────────────────────────────┘
                                  │
┌─────────────────────────────────┴───────────────────────────────────┐
│                      Code Execution Sandbox                          │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │              Docker Container (Isolated)                      │  │
│  │   - Vault bind-mounted at /mnt/data                          │  │
│  │   - Network: host.docker.internal access only                │  │
│  │   - No system file access outside vault                      │  │
│  └──────────────────────────────┬───────────────────────────────┘  │
└─────────────────────────────────┼───────────────────────────────────┘
                                  │
┌─────────────────────────────────┴───────────────────────────────────┐
│                        External Services                             │
│  ┌──────────────┐  ┌───────────────────────┐  ┌─────────────────┐  │
│  │    Redis     │  │  Obsidian MCP Server  │  │  LLM Provider   │  │
│  │  (Central    │  │  (cyanheads/obsidian- │  │  (Claude/Local) │  │
│  │   Ledger)    │  │   mcp-server)         │  │                 │  │
│  └──────────────┘  └───────────┬───────────┘  └─────────────────┘  │
└────────────────────────────────┼────────────────────────────────────┘
                                 │
┌────────────────────────────────┴────────────────────────────────────┐
│                     Obsidian Instance                                │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │              Local REST API Plugin (HTTPS)                    │  │
│  │              Port 27124 (TLS) / API Key Auth                 │  │
│  └──────────────────────────────┬───────────────────────────────┘  │
│                                 │                                   │
│  ┌──────────────────────────────┴───────────────────────────────┐  │
│  │                    Obsidian Vault                             │  │
│  │   /Shad/History/     - Run artifacts (inside vault)          │  │
│  │   /Shad/Skills/      - User skill overrides                  │  │
│  │   /Shad/Staging/     - Learnings pending review              │  │
│  │   /.base files       - Bases view definitions                │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Core Components

| Component | Role |
|-----------|------|
| **RLM Engine** | Recursive Language Model engine. Treats the Obsidian vault as an external environment. Generates **code** (Python) to inspect, query, and modify the vault via MCP tools, reducing context pollution. |
| **Obsidian Vault** | The knowledge substrate. A local file system of Markdown files. Uses the **Bases** plugin for database-like views and **Frontmatter** for structured metadata. |
| **MCP Client (Shad)** | Connects to the Obsidian MCP Server. Translates RLM intents into tool execution requests. |
| **Obsidian MCP Server** | The bridge between Shad and the vault. Communicates with the **Obsidian Local REST API** plugin to execute read/write/search operations. |
| **Redis** | Central ledger for budget enforcement. Caches reasoning steps and expensive tool outputs. Turns recursion tree into a DAG. |
| **Code Sandbox** | Docker container with vault bind-mounted. Executes RLM-generated scripts in isolation. |

---

## 3. Interface: Code Execution (The RLM Pattern)

Instead of standard chat-based tool calling (one turn per action), Shad implements **Code Execution with MCP**.

### 3.1 Workflow

1. RLM writes a Python script that imports MCP tools (e.g., `obsidian.search`, `obsidian.read`)
2. Script executes in sandboxed container with vault access
3. Script filters, aggregates, and processes vault data *before* returning results
4. Only final distilled output enters the context window

### 3.2 Context Initialization

**Strategy:** Minimal + Expand

- Runs start with **empty context** (goal text only)
- Code Mode scripts fetch data dynamically based on LLM decisions
- Prevents context rot from upfront loading
- LLM generates `obsidian.read_note()` or `obsidian.search()` calls as needed

### 3.3 Decomposition Logic

**Strategy:** LLM Decides

- No static heuristics for when to decompose
- Model determines whether to split based on context inspection
- System prompt instructs: "If the task is too large, decompose into sub-goals"
- Budget controls (`max_depth`, `max_branching_factor`) act as hard constraints only

---

## 4. Obsidian Vault Topology

Shad imposes a **Topology** onto Markdown files using Frontmatter and Folders, managed via the **Bases** plugin.

### 4.1 Node Types (Mapped to Frontmatter)

| Shad Concept | Obsidian Implementation |
|--------------|------------------------|
| **Notebook** | A **Base** view (defined in `.base` files) or a Folder context |
| **Source** | Markdown file with `type: source`. Contains raw text, transcripts, PDF exports |
| **Note** | Markdown file with `type: note`. Contains derived insights, summaries, synthesis |
| **Task** | Markdown file with `type: task`. Triggers n8n workflows when `status: pending` |

### 4.2 Legacy Note Handling

**Strategy:** Dual-mode with Progressive Standardization ("Gardener" Pattern)

- **Typed Notes:** Query via Bases logic for high-precision retrieval
- **Untyped Notes:** Retrieve via `obsidian_global_search` for high-recall retrieval
- **On Access:** When Shad reads an untyped note, it infers type and writes frontmatter back
- **Result:** Vault becomes more organized over time without requiring bulk migration

### 4.3 Wikilink Citations

**Strategy:** Full Path Always

- Shad emits `[[Folder/Subfolder/Filename]]` format for all citations
- Never bare `[[Filename]]` to avoid ambiguity with duplicate names
- Strip `.md` extension for aesthetics
- LinkValidator verifies paths exist via `os.path.exists(vault_root + link_path)`

### 4.4 History Storage

**Location:** Inside Vault at `Vault/Shad/History/`

```
Vault/Shad/History/
└── Run_<id>/
    ├── Run_<id>.md          # final.report.md (permanent, searchable)
    ├── dag.json             # DAG structure (for resume)
    ├── metrics/
    │   └── summary.json     # Token counts, timing
    └── artifacts/           # Heavy files (excluded from Obsidian views)
```

- Run notes are first-class citizens, searchable via `obsidian_global_search`
- Linkable from other notes: `[[Shad/History/Run_123|Analysis of Q3 Earnings]]`
- Configure Obsidian "Excluded files" to hide `artifacts/` from file explorer

### 4.5 Learnings System

**Promotion Strategy:** Frontmatter Flag

| Stage | Frontmatter | Location |
|-------|-------------|----------|
| Staging | `status: staging`, `confidence: <score>` | Appears in "Review Queue" Base |
| Promoted | `status: verified` | Appears in "Knowledge Base" Base |

- No folder moves required
- Bases views automatically update when frontmatter changes

---

## 5. Retrieval Model (MCP Tools)

### 5.1 Available Tools

| Tool | Purpose |
|------|---------|
| `obsidian_global_search` | Vector or full-text search across vault |
| `obsidian_read_note` | Retrieve specific file content and frontmatter |
| `obsidian_list_notes` | Scan directories for file structure |
| `obsidian_manage_frontmatter` | Update YAML properties |
| `obsidian_create_note` | Write new atomic notes |
| `obsidian_update_note` | Append/overwrite existing notes |
| `obsidian_delete_note` | Delete note (**HITL gated**) |

### 5.2 Vault Cache Service

- MCP server maintains in-memory map of file content and modification times
- Sub-100ms lookups for metadata
- Used for hash validation in caching strategy

---

## 6. Caching Strategy

### 6.1 Architecture

| Layer | Purpose | Invalidation |
|-------|---------|--------------|
| **Redis (Reasoning)** | DAG structure, subtask results, budget counters | Hash validation |
| **MCP VaultCacheService (Content)** | File stats, metadata | File watcher |

### 6.2 Cache Coherence

**Strategy:** Hash Validation

- Cache keys include `context_hash` derived from file content/mtime
- Before cache lookup: query MCP server for current file hash
- If hash mismatch: cache miss, re-compute reasoning
- Example key: `Task:Summarize:NoteA:Hash_abc123`

### 6.3 Cache Tiers

| Tier | Purpose |
|------|---------|
| **Main Cache** | Verified, trusted results |
| **Staging Cache** | Provisional results pending verification |

- Results enter Main Cache only after passing entailment checks
- Failed runs leave partial state in Staging only

---

## 7. Budget System

### 7.1 Hard Limits (RunConfig.budget)

| Parameter | Description |
|-----------|-------------|
| `max_depth` | Maximum recursion depth |
| `max_nodes` | Maximum DAG nodes |
| `max_wall_time` | Total execution time (seconds) |
| `max_tokens` | Total token budget |
| `max_branching_factor` | Maximum children per node |

### 7.2 Enforcement

**Strategy:** Central Ledger (Redis)

```python
def call_llm(prompt):
    cost_estimate = estimate_tokens(prompt)
    # Atomic check-and-deduct
    remaining = redis.decrby(f"run:{run_id}:budget", cost_estimate)
    if remaining < 0:
        raise BudgetExhaustedError("Global token limit reached")
    return execute_request(prompt)
```

- Sub-agents decrement shared Redis counter atomically before each LLM call
- Real-time enforcement across distributed execution tree
- If counter goes negative: action denied, return partial state

### 7.3 Timeout Behavior

**Strategy:** Synthesis Attempt

When `max_wall_time` exceeded:
1. Set `STOP_SIGNAL` in RLM Engine
2. Mark pending nodes as `skipped` or `budget_exhausted`
3. Call `Synthesizer.compile_partial(dag, cache)` with completed nodes
4. Generate `missing_branches` list for resume hint
5. Return partial results with actionable resume command

---

## 8. Concurrency Model

### 8.1 Write Operations

**Strategy:** Single-Writer

- Only **Root Agent** (or parent node) applies updates to existing files
- Sub-agents operate in **Read-Only** mode
- Sub-agents return data, root applies changes
- Prevents race conditions in distributed execution

### 8.2 Atomic Notes Exception

- Sub-agents **may** create new files with unique identifiers (e.g., `Search_Result_8a7b.md`)
- OS filesystem handles create concurrency
- Treated as scratchpad artifacts until verified

---

## 9. Error Recovery

### 9.1 Strategy: Checkpoint Model

- DAG structure persisted to Redis and `dag.json`
- Successful node results cached with content hash
- Failed nodes marked for re-execution
- In-progress node state discarded (atomic execution)

### 9.2 Resume Mechanics

```bash
shad run <id> --resume
```

1. Load `History/Runs/<id>/dag.json` to reconstruct task tree
2. Check Redis for cache keys of each node
3. Skip `completed` nodes with valid cache hits
4. Queue `pending` or `failed` nodes for execution

### 9.3 Error Presentation

**Strategy:** Inline in Output

- Stack traces embedded in `final.report.md` under `## Failures` header
- Formatted as markdown code blocks
- Includes specific Node ID that failed
- Concludes with CLI command to resume

### 9.4 Sync Conflict Handling

**Strategy:** Abort Run

- Pre-flight check: verify file `mtime` hasn't changed since read
- Detect conflict artifacts (e.g., `Note (conflict).md`)
- Raise `StateCorruptionError` if detected
- User resolves conflict in Obsidian, then runs `--resume`

---

## 10. Verification & Quality

### 10.1 Citation Verification

**Strategy:** Entailment Check

1. Shad generates claim citing `[[Note]]`
2. System fetches note content via `obsidian_read_note`
3. JUDGE model profile validates claim is entailed by text
4. Failed checks: discard result or flag as low-confidence
5. Occurs during Recomposition phase to prevent hallucination propagation

### 10.2 Validation Gates

| Operation | Gate Type |
|-----------|-----------|
| Delete note | **HITL Queue** (async approval required) |
| Publish externally | **HITL Queue** |
| Payment/access grant | **HITL Queue** |
| Frontmatter update | Automatic (validator checks YAML syntax) |
| Link creation | Automatic (validator checks path exists) |

### 10.3 HITL Queue Flow

```
Detection → Queuing → Continuation → Approval → Execution

POST /v1/admin/hitl/:id/approve  # User approves action
```

- Shad continues with non-blocked work while awaiting approval
- Returns `Partial Result` indicating pending approvals if relevant

---

## 11. Skills System

### 11.1 Discovery

**Strategy:** Layered Override

| Priority | Path | Purpose |
|----------|------|---------|
| 1 (highest) | `Vault/Shad/Skills/` | User overrides |
| 2 | `services/shad-api/Skills/` | Codebase defaults |

- Scan both paths for `SKILL.md` files
- Build registry keyed by skill name (YAML frontmatter)
- Name collision: User path wins

### 11.2 Skill Composition

**Strategy:** Prompt Injection

When primary skill declares `composes_with: [skill_a, skill_b]`:
1. Router loads primary skill
2. Resolves support skills
3. Concatenates all instructions into System Prompt
4. LLM can invoke support skill logic as needed

### 11.3 Voice/Persona Layer

**Strategy:** Optional Post-Process

- Voice as a Skill (stored in `Vault/Shad/Skills/Voices/`)
- Intermediate notes remain **neutral** (no persona transformation)
- Only `final.report.md` optionally passes through Voice filter
- Command: `shad run ... --voice <name>`

---

## 12. Security Model

### 12.1 Hard Invariants

1. **Never Auto-Publish:** No irreversible side effects without explicit human approval
2. **Never Exfiltrate:** No sending data externally unless explicitly permitted
3. **Never Self-Modify:** Cannot change own Skills/CORE without human review

### 12.2 Sandboxing

**Level:** Container (Docker)

- Vault bind-mounted at `/mnt/data`
- No system file access outside vault
- Network restricted to `host.docker.internal` (MCP server access)
- Scoped to specific vault directories per principle of least privilege

### 12.3 API Security

**Strategy:** Require HTTPS

```env
OBSIDIAN_API_KEY=your-key
OBSIDIAN_BASE_URL=https://127.0.0.1:27124
OBSIDIAN_VERIFY_SSL=false  # Self-signed cert acceptable
```

- Local REST API plugin defaults to HTTPS with self-signed cert
- API key auth in headers
- Traffic encrypted even for localhost
- `VERIFY_SSL=false` allows self-signed certs

### 12.4 Delete Operations

- Routed through **HITL Queue**
- Never auto-approved
- User must explicitly approve via admin API
- Wraps `obsidian_delete_note` capability in safety layer

---

## 13. n8n Orchestration

### 13.1 Trigger Configuration

**Strategy:** Explicit Task Notes (State Machine)

| State | Trigger? | Description |
|-------|----------|-------------|
| `type: task`, `status: pending` | **YES** | n8n fires workflow |
| `type: task`, `status: processing` | No | Task picked up, in progress |
| `type: task`, `status: complete` | No | Task finished |
| `type: note` | No | Output artifact, ignored |
| `type: report` | No | Output artifact, ignored |

### 13.2 Loop Prevention

1. User creates task note with `status: pending`
2. n8n triggers (filter: `type == 'task' AND status == 'pending'`)
3. First action: update to `status: processing` (removes from trigger criteria)
4. Shad generates outputs with `type: note` or `type: report`
5. On completion: update to `status: complete`

### 13.3 Webhook Integration

- Local REST API webhooks supported
- n8n can trigger via HTTP or Watch Folder
- Callback webhooks for async completion notification

---

## 14. CLI & API

### 14.1 CLI Commands

```bash
# Execute reasoning task
shad run "Synthesize my thoughts on RAG architectures" --context "Tech Notes/"

# Resume partial run
shad run <id> --resume

# Check status
shad status <run_id>

# View DAG tree (real-time during execution)
shad trace tree <run_id>

# Apply voice to existing run
shad voice apply <run_id> --voice pirate
```

### 14.2 Progress UX

**Strategy:** Tree Visualization

- Real-time ASCII tree showing DAG progress
- Node status indicators: `[+]` completed, `[...]` running, `[C]` cached
- Shows depth and branching to verify decomposition logic

### 14.3 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/run` | POST | Execute reasoning task |
| `/v1/run/:id` | GET | Get run status/results |
| `/v1/run/:id/resume` | POST | Resume partial run |
| `/v1/runs` | GET | List recent runs |
| `/v1/skills` | GET | List available skills |
| `/v1/admin/hitl/queue` | GET | HITL review queue |
| `/v1/admin/hitl/:id/approve` | POST | Approve HITL action |

### 14.4 API Streaming

**Strategy:** Server-Sent Events (SSE)

- `POST /v1/run` accepts `Accept: text/event-stream`
- Streams node lifecycle events: `node_created`, `node_complete`, `cache_hit`
- CLI consumes stream to update ASCII tree
- `final.summary.json` emitted as closing event
- MCP-compliant transport layer

---

## 15. Configuration & Compatibility

### 15.1 Environment Variables

```env
# Required
OBSIDIAN_API_KEY=your-key
OBSIDIAN_BASE_URL=https://127.0.0.1:27124
OBSIDIAN_VERIFY_SSL=false

# Optional
MCP_SERVER_TYPE=obsidian
LLM_PROVIDER=anthropic           # or: openai, ollama
SHAD_OFFLINE_MODE=false          # Enable for local LLM only
REDIS_URL=redis://localhost:6379
```

### 15.2 Obsidian Requirements

| Requirement | Version | Status |
|-------------|---------|--------|
| Obsidian | **1.10+** | Required (Bases core plugin) |
| Local REST API | Latest | Required (community plugin) |
| Bases | Core | Required |

### 15.3 Bases Fallback

**Strategy:** Frontmatter-Only (Progressive Enhancement)

- Bases is a **visualization layer** on top of frontmatter
- Without Bases: Shad continues to function (reads/writes YAML)
- User loses spreadsheet-like Table View only
- Native search (`["type": "task"]`) still works

### 15.4 Multi-Vault Support

**Strategy:** Single Vault Only

- One vault per Shad instance
- Prevents MCP tool naming collisions
- Maintains security isolation
- For multiple vaults: run separate Shad instances with distinct configs

### 15.5 Offline Mode

**Strategy:** Degraded Local

- Support local LLMs (Ollama, vLLM) with `LLM_PROVIDER=ollama`
- Explicit warnings about reduced quality/capability
- Provider priority: Anthropic > OpenAI > Google > Local
- `--offline` flag refuses external calls, degrades gracefully

---

## 16. Implementation Plan

### Phase 1 — Infrastructure & MCP Setup
- [ ] Install **Local REST API** community plugin
- [ ] Configure Local REST API (generate API key, enable HTTPS)
- [ ] Enable **Bases** core plugin (v1.10+)
- [ ] Deploy `cyanheads/obsidian-mcp-server` via Docker
- [ ] Configure env vars: `OBSIDIAN_API_KEY`, `OBSIDIAN_BASE_URL`
- [ ] Enable `OBSIDIAN_ENABLE_CACHE` for performance

### Phase 2 — Shad MCP Client & Code Mode
- [ ] Implement `MCPClient` class in Python
- [ ] Create Code Mode sandbox (Docker container)
- [ ] Wrap MCP tools as importable functions
- [ ] Implement `obsidian.search()`, `obsidian.read()`, `obsidian.write()`
- [ ] Add budget enforcement wrapper around LLM calls

### Phase 3 — RLM Engine Refactor
- [ ] Context initialization: minimal + expand strategy
- [ ] Update prompts for LLM-driven decomposition
- [ ] Citation handling: full-path wikilinks
- [ ] Implement entailment check for citations
- [ ] Sub-agents: read-only mode with data return

### Phase 4 — Caching & Budget
- [ ] Redis central ledger for budget counters
- [ ] Hash validation for cache coherence
- [ ] Staging vs Main cache tiers
- [ ] Timeout handling: synthesis attempt

### Phase 5 — Skills System Migration
- [ ] Layered skill discovery (codebase + vault)
- [ ] Skill composition via prompt injection
- [ ] Voice as optional skill
- [ ] Progressive standardization for legacy notes

### Phase 6 — HITL & Verification
- [ ] HITL queue for delete operations
- [ ] Entailment check integration
- [ ] Frontmatter validator
- [ ] Link validator

### Phase 7 — CLI & API
- [ ] Tree visualization for progress
- [ ] SSE streaming for API
- [ ] Resume command implementation
- [ ] Voice apply command

### Phase 8 — n8n Orchestration
- [ ] Task note state machine
- [ ] Watch folder trigger configuration
- [ ] Loop prevention validation

### Current Status: APPROVED, READY FOR IMPLEMENTATION

---

## Appendix A: Frontmatter Schemas

### Task Note
```yaml
---
type: task
status: pending | processing | complete | failed
goal: "Natural language description of task"
shad_run_id: null | "<uuid>"
created: 2026-01-13
---
```

### Source Note
```yaml
---
type: source
status: raw | processed | verified
source_type: transcript | pdf | web | manual
shad_processed: false | true
---
```

### Learning Note
```yaml
---
type: learning
status: staging | verified
confidence: 0.0-1.0
cited_by: []
contradicted_by: []
---
```

### Run Report
```yaml
---
type: report
shad_run_id: "<uuid>"
status: complete | partial | failed
goal: "Original goal text"
node_count: <int>
max_depth: <int>
token_usage: <int>
wall_time_seconds: <float>
---
```

---

## Appendix B: Decision Log

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Error Recovery | Checkpoint Model | DAG-based, supports resume without losing completed work |
| Cache Coherence | Hash Validation | Content hash in keys auto-invalidates on vault changes |
| Legacy Notes | Progressive Standardization | "Gardener" pattern tags notes on access |
| Wikilinks | Full Path Always | Deterministic resolution, future-proof |
| Concurrency | Single-Writer | Prevents race conditions, root applies all writes |
| Sandbox | Container-Level | Docker with bind-mounted vault, strong isolation |
| Delete Confirm | HITL Queue | Async approval, non-blocking |
| Citation Verify | Entailment Check | JUDGE model validates claims |
| Trigger Loops | Explicit Task Notes | State machine with frontmatter flags |
| Decomposition | LLM Decides | Model determines split, budget enforces limits |
| Context Init | Minimal + Expand | Empty start, Code Mode fetches dynamically |
| Voice Layer | Optional Post-Process | Neutral data in vault, voice applied on render |
| Skill Discovery | Layered Override | Codebase defaults + vault overrides |
| Skill Compose | Prompt Injection | Support skill instructions in system prompt |
| Learning Promo | Frontmatter Flag | Status field transitions, no folder moves |
| History Location | Inside Vault | Runs searchable and linkable |
| Resume State | DAG + Cache Only | Restore structure, re-execute incomplete |
| Timeout Cleanup | Synthesis Attempt | Compile partial answer from completed nodes |
| API Streaming | SSE | MCP-compliant, real-time progress |
| Progress UX | Tree Visualization | ASCII DAG with node statuses |
| Multi-Vault | Single Vault Only | Security isolation, prevent tool collisions |
| Version Floor | 1.10+ (Bases) | Hard requirement for Bases plugin |
| Bases Fallback | Frontmatter-Only | Progressive enhancement |
| Offline Mode | Degraded Local | Support with explicit warnings |
| API Security | HTTPS Required | Self-signed cert acceptable |
| Sync Conflicts | Abort Run | Detect mtime changes, require user resolution |
