# Shad Technical Specification

> **Version**: 2.0
> **Status**: Living Document
> **Last Updated**: 2026-01-14

---

## 0. Executive Summary

### What is Shad?

**Shad (Shannon's Daemon) enables AI to utilize virtually unlimited context.** It treats an Obsidian vault as an explorable environment rather than a fixed input, recursively decomposing complex tasks, retrieving targeted context for each subtask, generating outputs informed by vault knowledge, and assembling coherent results.

### Core Premise

> **Long-context reasoning is an inference problem, not a prompting problem.**

Instead of cramming context into a single prompt, Shad:
1. **Decomposes** complex tasks into subtasks recursively
2. **Retrieves** targeted context for each subtask via Code Mode
3. **Generates** outputs informed by relevant examples
4. **Assembles** results into coherent output

### Hard Invariants

1. **Never Auto-Publish**: No irreversible side effects without explicit human approval
2. **Never Exfiltrate**: No sending data externally unless explicitly permitted
3. **Never Self-Modify**: Cannot change own Skills/CORE without human review

### Architecture Overview

```
User
   │
   ▼
Shad CLI / API
   │
   ├── RLM Engine (recursive decomposition + execution)
   │       │
   │       ├── Strategy Selection → Decomposition
   │       ├── Code Mode → CodeExecutor → ObsidianTools → Vault(s)
   │       ├── Verification Layer → Repair Loop
   │       └── Synthesis → File Output
   │
   ├── Redis (cache + budget ledger)
   ├── History/ (run artifacts)
   └── Shadow Index (vault metadata)
```

### Implementation Status

| Phase | Status | Description |
|-------|--------|-------------|
| 1. Foundation | Complete | CLI, API, RLM Engine, budgets |
| 2. Obsidian Integration | Complete | MCP client, Code Mode, citations |
| 3. Task-Aware Decomposition | In Progress | Strategy skeletons, domain-specific |
| 4. File Output Mode | Planned | Multi-file codebases, manifests |
| 5. Verification Layer | Planned | Syntax, types, tests, repair |
| 6. Iterative Refinement | Planned | Error feedback, HITL checkpoints |
| 7. Vault Curation Tools | Planned | Ingestion, analysis, gap detection |

---

## 1. User Journey

This section describes the mental model and flow of a Shad run from start to finish.

### 1.1 Vault Setup

Shad operates against one or more Obsidian vaults containing curated knowledge. A **vault** is a directory of markdown files with optional frontmatter, organized for retrieval.

**Vault Layering**: Runs can declare multiple vaults with priority order:
```bash
shad run "Build auth system" \
  --vault ~/Project \
  --vault ~/Patterns \
  --vault ~/Docs
```

Earlier vaults have higher priority in search result ranking. Each citation includes vault provenance.

### 1.2 Initiating a Run

```bash
shad run "Build a task management app with auth and offline sync" \
  --vault ~/MobileDevVault \
  --max-depth 4 \
  --output ./TaskApp
```

Shad:
1. Normalizes the goal (extracts intent, entities, constraints)
2. Selects a **strategy** (software, research, analysis, creative)
3. Creates a **run** with unique ID and initializes budgets

### 1.3 Strategy Selection & Decomposition

**Strategy selection** is hybrid:
- Fast heuristic classifier runs first (keyword matching)
- User can override with `--strategy software|research|analysis|planning`
- LLM can request a strategy switch mid-execution with evidence

Each strategy defines a **skeleton** with required stages, optional stages, and constraints. The LLM fills in task-specific details within these guardrails.

**Software skeleton example**:
- Required: Clarify requirements → Project layout → Types & contracts → Implementation → Verification → Synthesis
- Optional: DB schema, Auth, OpenAPI, Migrations, Docs
- Constraints: contracts-first, imports must resolve

### 1.4 Retrieval (Code Mode)

For each subtask, Shad retrieves targeted context via **Code Mode**: the LLM generates a Python script that queries the vault.

```python
# LLM-generated retrieval for "Implement OAuth"
results = obsidian.search("OAuth refresh token", limit=10)
patterns = obsidian.read_note("Patterns/Auth/OAuth.md")

relevant = [r["content"][:2000] for r in results if "JWT" in r["content"]]
__result__ = {"context": "\n".join(relevant), "citations": [...], "confidence": 0.72}
```

**Retrieval recovery policy** (when results are poor):
1. **Tier A**: Regenerate script with hints (1 retry)
2. **Tier B**: Broadened direct search fallback
3. **Tier C**: Human checkpoint (high-impact nodes only)

### 1.5 Parallel Execution & Budgeting

Subtasks execute in parallel where dependencies allow. Each node draws from a **hierarchical token budget**:
- Parent reserves 25% for synthesis
- Children share the remainder via dynamic allocation
- Per-child caps prevent runaway consumption
- Late-stage protection enables graceful degradation

**Concurrency** is tiered adaptive:
- Separate limits for LLM calls vs local execution
- Backoff on 429 rate limits
- Slow-start increase when stable
- User can override with `--max-parallel N`

### 1.6 Context Propagation

**Soft dependencies** enable cross-subtask context sharing:
- Decomposition emits both hard_deps (must complete first) and soft_deps (useful if available)
- When a node completes, it produces a **context packet** (summary, artifacts, keywords)
- Scheduler injects packets into pending nodes' retrieval

### 1.7 Verification & Repair

After generation, the **verification layer** checks outputs:

| Check | Default Mode | Strict Mode |
|-------|--------------|-------------|
| Import resolution | Blocking | Blocking |
| Syntax/parse | Blocking | Blocking |
| Manifest integrity | Blocking | Blocking |
| Type errors | Advisory | Blocking |
| Unit tests | Advisory | Blocking |
| Lint | Advisory | Configurable |

**Error recovery** follows classification:
- **Syntax/lint**: Local repair only
- **Type errors**: Local repair with sibling context (contracts/types)
- **Integration failures**: Escalate to parent coordinator
- **Contract mismatch**: Parent coordination + contract update node

Escalation threshold: max 2 local retries per node, max 10 escalations per run.

### 1.8 File Output

For code generation tasks, Shad produces a **file manifest** (not raw text):

```json
{
  "files": [
    {"path": "src/types.ts", "content": "...", "language": "ts", "hash": "..."},
    {"path": "src/api/users.ts", "content": "...", "source_nodes": ["impl_api"]}
  ]
}
```

**Import resolution** uses two-pass generation:
1. **Pass 1**: Build export index (symbol → file mapping)
2. **Pass 2**: Generate implementations using export index as ground truth
3. **Validation**: Check all imports resolve to existing files/symbols

Writing to filesystem is **always explicit** (`--write-files` flag).

### 1.9 Iterative Refinement

If verification fails, Shad enters a repair loop. On max iterations:

| Task Type | Final State | Behavior |
|-----------|-------------|----------|
| High-impact | `NEEDS_HUMAN` | Pause with full context, await hints |
| Low-risk | `PARTIAL` | Return best-effort with known issues |
| Impossible | `FAILED` | Return diagnostics, suggest changes |

Shad **always** returns best-effort artifacts plus a diagnostic report.

### 1.10 Human-in-the-Loop Checkpoints

Checkpoints trigger when:
- Node is **high-impact** (security, data model, architecture, contracts) AND has low confidence
- Retrieval confidence < 0.45 on a gating node
- Generation confidence < 0.55 with repeated repair failures
- Any **irreversible side effect** (file writes, network, ingestion)

Explicit markers (`[REVIEW]`, `[APPROVE]`) always trigger checkpoints. Users can disable with `--no-checkpoints`.

### 1.11 Resume & Replay

Partial runs can be resumed:
```bash
shad resume <run_id>
```

**Delta verification**: Only re-verify nodes whose vault context changed (subset fingerprint mismatch). Users can force replay of specific nodes:
```bash
shad resume <run_id> --replay node_id
shad resume <run_id> --replay stale
```

### 1.12 Synthesis & Conflict Resolution

Child results are synthesized bottom-up. When children produce **conflicting outputs**:
1. Detect conflicts via decision table (decision slots with different values)
2. Preserve both perspectives with attribution in output
3. LLM attempts reconciliation with explicit criteria
4. Checkpoint only for high-impact unresolved conflicts

---

## 2. Reference Specification

### 2.1 Execution Engine

#### 2.1.1 DAG Lifecycle

```
CREATED → STARTED → SUCCEEDED | CACHE_HIT | FAILED | PRUNED
                ↓
           (decompose)
                ↓
           child nodes
```

**Node states**:
- `CREATED`: Node exists but not started
- `STARTED`: Execution in progress
- `SUCCEEDED`: Completed successfully
- `CACHE_HIT`: Result retrieved from cache
- `FAILED`: Execution failed
- `PRUNED`: Skipped (novelty check or budget)

#### 2.1.2 Scheduling & Concurrency

**Semaphores**:
- `max_llm_in_flight`: Default 2, adaptive
- `max_local_jobs`: Default min(4, cpu_count)

**Adaptive behavior**:
- On 429: Reduce `max_llm_in_flight` by 1, exponential backoff
- On stable window (60-120s, no errors): Increase by 1 up to cap
- Local jobs continue while LLM is rate-limited

**User overrides**:
- `--max-parallel N`: Upper bound on both
- `--max-llm N`: LLM-specific cap
- `--concurrency-mode fixed|adaptive`

#### 2.1.3 Budget Enforcement

| Budget | Default | Enforcement |
|--------|---------|-------------|
| `max_depth` | 3 | Checked before decomposition |
| `max_nodes` | 50 | Checked before creating nodes |
| `max_wall_time` | 300s | Checked periodically |
| `max_tokens` | 100,000 | Atomic Redis deduction |

**Hierarchical token allocation**:
```
Node receives envelope B
├── reserve = max(800, 0.25 * B)  # Protected for synthesis
└── spendable = B - reserve        # Distributed to children
```

Children receive base grants (300 tokens) with dynamic top-ups. Per-child cap: 35% of parent's spendable pool.

### 2.2 Decomposition & Strategies

#### 2.2.1 Strategy Skeletons

Each strategy defines:
- **Required stages**: Must exist in DAG
- **Optional stages**: Added when relevant
- **Constraints**: Invariants the LLM must respect
- **Default dependencies**: Edges between stages

**Software strategy**:
```yaml
required:
  - clarify_requirements
  - project_layout
  - types_contracts
  - implementation
  - verification
  - synthesis
optional:
  - db_schema
  - auth
  - openapi
  - migrations
  - docs
constraints:
  - contracts_first: true
  - imports_must_resolve: true
  - no_implicit_writes: true
```

**Research strategy**:
```yaml
required:
  - clarify_scope
  - gather_sources
  - synthesize
  - cite
optional:
  - compare_perspectives
  - identify_gaps
constraints:
  - must_cite_vault: true
  - max_claims_per_source: 5
```

#### 2.2.2 LLM Refinement

The decomposition LLM receives:
- Strategy name and skeleton
- Strategy-specific "hint pack" (system prompt additions)
- Budget caps

It returns a DAG that:
- Can add/remove optional nodes
- Can split implementation into specific modules
- Can add soft dependencies
- Cannot violate required stages without explicit waiver

#### 2.2.3 Strategy Selection

1. **Heuristic classifier** (no LLM call):
   - Pattern-match task text against keyword sets
   - Output: `strategy_guess` + `guess_confidence`

2. **Decision rule**:
   - confidence ≥ 0.7 → proceed with guess
   - confidence < 0.7 → default to `analysis`, allow LLM confirmation

3. **User override**: `--strategy X` always wins

4. **Mid-run switch**: LLM can emit `strategy_switch_request` with evidence

### 2.3 Vaults

#### 2.3.1 Single vs Layered Vaults

**Single vault** (default):
```bash
shad run "..." --vault ~/MyVault
```

**Layered vaults** (priority order):
```bash
shad run "..." --vault ~/Project --vault ~/Patterns --vault ~/Docs
```

**Retrieval behavior**:
- Search executes against all vaults in parallel
- Results merged with weighted scoring: `relevance + priority_bonus(vault_index)`
- Per-vault caps (e.g., max 10 results each), then global top N
- Citations include `vault_id:path`

**Optional namespacing** for Code Mode:
```python
obsidian.search("jwt", vault="patterns")
obsidian.search("config", vaults=["project", "docs"])
```

#### 2.3.2 Ingestion & Snapshots

**Snapshot model**: Immutable, timestamped snapshots.

```
Sources/<domain>/<source_id>/<YYYY-MM-DD>/
├── _entry.md           # Summary note
├── README.md
├── docs/...
├── meta/
│   ├── ingest.yaml     # Provenance metadata
│   └── file_tree.md
└── index/              # Only for 'deep' preset
    ├── symbols.md
    └── modules.md
```

**Frontmatter** (required):
```yaml
source_url: "https://github.com/org/repo"
source_type: "github_repo"
ingested_at: "2026-01-14T19:02:11-07:00"
source_revision: "abc123"
content_hash: "sha256:..."
snapshot_id: "source_id@2026-01-14"
```

**Ingestion presets**:

| Preset | What's ingested | Best for |
|--------|-----------------|----------|
| `mirror` | All files, minimal processing | Archival, small repos |
| `docs` (default) | README, docs/, comments, metadata | Most repos |
| `deep` | docs + semantic index (symbols, imports) | Heavily-used repos |

**CLI**:
```bash
shad ingest github <url> --preset docs
shad ingest github <url> --preset deep --languages ts,python
shad enrich <snapshot_id> --deep  # Post-hoc semantic indexing
```

#### 2.3.3 Shadow Index

The shadow index maps `source_url → latest_snapshot`. It lives **outside the vault** in `~/.shad/index.sqlite`.

**Schema**:
```sql
sources(source_url PK, source_id, source_type, update_policy)
snapshots(snapshot_id PK, source_id FK, ingested_at, source_revision, entry_paths, content_hash)
latest(source_id PK, latest_snapshot_id FK)
```

**Update policies**:
- `manual`: Detect changes, ingest only on user trigger
- `notify`: Detect changes, queue suggestion
- `auto`: Ingest on schedule with quotas

**Export** (optional):
```bash
shad sources export --format yaml --out <vault>/.shad/sources.yaml
shad sources ls
shad sources pin <url> --snapshot <id>
```

### 2.4 Retrieval (Code Mode)

#### 2.4.1 Sandbox Security Model

**Profiles** (configurable per-run):

| Profile | File Access | Network | Use Case |
|---------|-------------|---------|----------|
| `strict` (default) | Vault only via ObsidianTools | None | Safe, deterministic |
| `local` | + Read-only to allowlisted roots | None | Reference local repos |
| `extended` | + Read-only allowlist | HTTP GET to allowlisted domains | Fetch live docs |

**Sandbox constraints**:
- Disabled builtins: `eval`, `exec`, `compile`, `__import__`, raw `open`
- Allowed imports: `json`, `re`, `datetime`, `collections`, `itertools`, `functools`, `math`, `hashlib`, `pathlib`, `typing`, `dataclasses`, `enum`, `yaml`
- Resource limits: 60s timeout, 512MB memory
- Output via `__result__` only, capped at 200KB

**CLI**:
```bash
shad run "..." --profile strict  # Default
shad run "..." --profile local --fs-read ./myrepo
shad run "..." --profile extended --net-allow docs.example.com
```

#### 2.4.2 Structured Retrieval Results

Retrieval scripts must return structured results:

```python
__result__ = {
    "context": "<distilled context text>",
    "citations": [{"vault": "...", "path": "...", "snip_start": 0, "snip_end": 2000}],
    "queries": ["oauth refresh token", "jwt validation"],
    "signals": {"num_notes": 7, "total_chars": 12450, "keyword_hits": 18},
    "confidence": 0.72,
    "why": "Found OAuth pattern note + 3 examples mentioning refresh tokens"
}
```

#### 2.4.3 Retrieval Scoring & Recovery

**System-computed relevance score** (independent of script's self-reported confidence):

```
retrieval_score = w1*keyword_overlap + w2*coverage + w3*diversity - w4*boilerplate_penalty
```

**Recovery policy**:

| Tier | Trigger | Action |
|------|---------|--------|
| A | `retrieval_score < 0.45` OR `confidence < 0.5` | Regenerate script with hints (1 retry) |
| B | Tier A failed | Broadened direct search fallback |
| C | Tier B failed AND high-impact node | Human checkpoint |

**Thresholds**:
- `min_context_chars`: 2000
- `min_citations`: 2
- `low_score_threshold`: 0.45
- `max_retrieval_regens_per_node`: 1
- `max_total_retrieval_regens_per_run`: 5

### 2.5 Budgets

#### 2.5.1 Token Hierarchy

```
Run budget (max_tokens)
└── Root node envelope
    ├── reserve (25%, min 800)
    └── spendable → distributed to children
        ├── Child 1 envelope
        │   ├── retrieval (20-30%)
        │   ├── generation (50-60%)
        │   └── repair (10-20%)
        └── Child N envelope
```

**Allocation rules**:
- Base grant per child: 300 tokens
- Max child share: 35% of parent's spendable
- Late-stage protection: Disable expensive strategies when pool depleted

#### 2.5.2 Wall Time & Retry Caps

| Limit | Default | Scope |
|-------|---------|-------|
| `max_wall_time` | 300s | Per-run |
| `max_local_repairs_per_leaf` | 2 | Per-node |
| `max_escalations_per_run` | 10 | Per-run |
| `max_retrieval_regens_per_run` | 5 | Per-run |

### 2.6 Verification

#### 2.6.1 Verification Checks

| Check | What it validates |
|-------|-------------------|
| Import resolution | All imports resolve to existing files/symbols |
| Syntax/parse | Code is syntactically valid |
| Manifest integrity | No path traversal, no duplicates, within output root |
| Type check | TypeScript/Flow type errors (via `tsc --noEmit`) |
| Unit tests | Tests pass |
| Integration tests | Integration tests pass |
| Lint | Style conformance |

#### 2.6.2 Blocking vs Advisory

**Verification levels**:
- `--verify=off`: No checks
- `--verify=basic` (default): Imports + syntax + manifest blocking
- `--verify=build`: Basic + typecheck blocking
- `--verify=strict`: Build + tests blocking

**Per-check override**:
```bash
--block-on typecheck,unit_tests
--warn-only lint,integration_tests
```

Or via config:
```yaml
verification:
  import_resolution: blocking
  syntax: blocking
  typecheck: advisory
  unit_tests: advisory
```

#### 2.6.3 Repair Escalation

**Error classification drives repair scope**:

| Error Class | First Action | Context | Escalate When |
|-------------|--------------|---------|---------------|
| Syntax/lint | Local repair | None | >2 retries |
| Type errors | Local repair | types/contracts | >2 retries OR multi-leaf |
| Unit tests | Local repair | sibling outputs | >2 retries |
| Integration tests | Parent coordination | relevant siblings | Immediate |
| Contract mismatch | Parent coordination + update | contracts + dependents | Immediate |

**Error signature hashing** prevents infinite loops: Hash `(error_class, primary_files, key_lines)` to detect repeated same-error.

### 2.7 File Output

#### 2.7.1 Manifest Structure

```json
{
  "run_id": "abc123",
  "files": [
    {
      "path": "src/types.ts",
      "content": "export interface User { ... }",
      "language": "ts",
      "mode": "create",
      "hash": "sha256:...",
      "source_nodes": ["types_contracts"]
    }
  ],
  "notes": [
    {"kind": "contract_change_request", "detail": "Add UserRole enum"}
  ]
}
```

#### 2.7.2 Import Resolution (Two-Pass)

**Pass 1: Build export index**
```json
{
  "exports": [
    {"symbol": "User", "from": "src/types.ts", "type": "type"},
    {"symbol": "createUser", "from": "src/api/users.ts", "type": "function"}
  ],
  "path_aliases": {"@/": "src/"}
}
```

**Pass 2: Generate implementations**
- Each node receives export index as ground truth
- Instruction: "Import from export index mappings. Do not invent paths."

**Post-generation validation**:
```json
{
  "missing_modules": [{"file": "src/api/users.ts", "import": "@/db", "reason": "not in manifest"}],
  "missing_symbols": [{"file": "src/api/users.ts", "symbol": "User", "reason": "not exported"}]
}
```

#### 2.7.3 Type Consistency

**Contracts-first rule**:
1. A `Types & Contracts` node produces canonical type artifacts
2. Implementation nodes import from canonical artifacts only
3. New type needs emit `contract_change_request` artifacts
4. Reconciliation step merges identical proposals, flags conflicts

For software tasks spanning DB/API/Auth, Shad enforces **schema-first**: data model → contracts → implementation.

#### 2.7.4 Write Semantics

**Artifact-first**: Runs always produce a manifest. Writing to disk is explicit.

```bash
# Produce manifest only
shad run "..." --vault ~/v

# Produce manifest + write files
shad run "..." --vault ~/v --write-files --output ./out

# Export after inspection
shad export <run_id> --output ./out
```

**Write safeguards**:
- Only write under `output_root` (no `../` traversal)
- `--overwrite` flag for conflicts (default: fail)
- Emit write report with paths, skipped conflicts, hashes

### 2.8 Iteration & Resume

#### 2.8.1 Run States

```
PENDING → RUNNING → SUCCESS | PARTIAL | FAILED | NEEDS_HUMAN
```

**Terminal states**:
- `SUCCESS`: Meets acceptance criteria
- `PARTIAL`: Produced artifacts but did not meet criteria
- `FAILED`: Could not produce meaningful artifacts or safety stop

**Pausable state**:
- `NEEDS_HUMAN`: Run stopped, context preserved, awaiting human input

#### 2.8.2 Delta Verification on Resume

For each completed node, store:
- `used_notes[]`: Paths actually read
- `used_note_hashes{path → hash}`: Content hashes
- `subset_fingerprint`: Hash over used note hashes

**On resume**:
1. Check current vault manifest against stored hashes
2. Node is **stale** if any `used_note_hash` differs
3. Stale nodes undergo re-verification (or re-execution for contracts nodes)
4. Unchanged nodes are trusted

**Selective replay**:
```bash
shad resume <run_id> --replay node_id
shad resume <run_id> --replay subtree:node_id
shad resume <run_id> --replay stale
```

#### 2.8.3 Max Iterations Policy

On max-iteration exhaustion:

| Condition | Final State |
|-----------|-------------|
| High-impact task OR substantial artifacts exist | `NEEDS_HUMAN` |
| Low-risk task, verification advisory | `PARTIAL` |
| Cannot proceed safely OR no meaningful artifacts | `FAILED` |
| Same run hits max iterations twice without human intervention | `FAILED_WITH_DIAGNOSTIC` |

**Always return**:
- Best current artifacts
- Failure report (what criteria failed, what was tried)
- Suggested next inputs from human

### 2.9 Human-in-the-Loop

#### 2.9.1 Checkpoint Triggers

**Explicit markers** (always honored):
- `[REVIEW]`: Run node, pause before propagating outputs
- `[APPROVE]`: Pause before execution, require approval
- `--checkpoint-on <categories>`: Pause on specific node types
- `--no-checkpoints`: Disable all except hard safety

**Auto-triggers** (default enabled):
- High-impact node (security/auth, data model, architecture, contracts) at depth ≤ 1
- High-impact node changing contracts
- Low retrieval confidence (< 0.45) + high-impact or high fan-out
- Low generation confidence (< 0.55) + repeated repair failures

**Hard safety** (non-negotiable, cannot disable):
- File write outside sandboxed output root
- Network access beyond allowlist
- Vault ingestion (if auto-ingest configured)
- Any potential data leak

#### 2.9.2 Checkpoint Interface

When paused, Shad presents:
- Node summary (goal + plan)
- Decision/output to approve (diff or contract snippet)
- Confidence signals (retrieval_score, generation_confidence, verification status)
- Options: Approve | Edit | Provide hint | Skip node | Stop run

**Caps**: `max_checkpoints_per_run = 5` (then degrade to batch review at end)

### 2.10 Caching

#### 2.10.1 Cache Keys

**Hierarchical key scheme**:
```
shad:cache:main:<goal_type>:<intent>:<entities>:<context_hash>[:<extra_slots>]
```

**Subset fingerprint**: Hash of `(vault_id, path, content_hash)` tuples for notes actually used.

#### 2.10.2 Invalidation Rules

**Hybrid (hash + TTL)**:

| Condition | Cache behavior |
|-----------|----------------|
| Subset fingerprint matches AND TTL not expired | Hit |
| Subset fingerprint differs | Miss (invalidate) |
| TTL expired | Miss (invalidate) |

**TTL defaults**:
- Main cache: 30 days
- Staging cache: 24 hours
- Volatile folders (`Inbox/`, `Daily/`): 24 hours

**TTL per vault** (with layered vaults):
- `patterns` vault: 30 days
- `docs` vault: 7 days
- `project` vault: 24 hours

#### 2.10.3 Manual Controls

```bash
shad run "..." --no-cache          # Bypass reads/writes
shad run "..." --refresh-cache     # Ignore reads, write new
shad cache clear --scope run|source|all
```

### 2.11 Test Generation

#### 2.11.1 Strategy-Specific Defaults

| Strategy | Test behavior |
|----------|---------------|
| `software` | Spec-first stubs + post-implementation pass |
| `research` | No tests (verification = citation checks) |
| `analysis` | No tests (verification = criteria coverage) |
| `creative` | No tests |

#### 2.11.2 Software Test Flow

1. **After contracts**: Generate `TEST_PLAN.md` + test stubs (describe blocks, fixtures)
2. **After implementation**: Generate full test suite with complete codebase context
3. **Verification**: Run tests as advisory (default) or blocking (`--verify=strict`)

**Test co-generation** (optional): Leaf nodes may emit local unit tests for pure functions only.

**Override**:
```bash
--tests off|stubs|post|tdd|co
```

### 2.12 Vault Analysis

#### 2.12.1 Gap Detection (Combined Scoring)

```
gap_score = 0.55 * history_pain + 0.25 * coverage_miss + 0.20 * llm_score
```

**History pain** (primary signal):
- Query frequency in past runs
- Median retrieval_score per query cluster
- "No results" / fallback rate
- Correlation with downstream failures

**Coverage miss** (secondary):
- Missing anchor notes for common topics (auth, API, testing, etc.)
- Missing templates (spec, ADR, test plan)

**LLM audit** (optional, `--llm-audit`):
- Send vault summary + sample notes
- Ask for top 10 gaps with suggested note titles

#### 2.12.2 Gap Report Output

```markdown
## Top Gaps (Ranked)

### 1. Auth refresh tokens
- **Evidence**: 14 queries, median retrieval_score 0.31, 62% fallback
- **Suggested additions**:
  - `Patterns/Auth/RefreshTokens.md`
  - Ingest: https://auth0.com/docs/tokens/refresh-tokens
- **Priority**: High (frequent + painful)
```

---

## 3. Existing Implementation Details

This section documents what is already implemented (Phases 1-2).

### 3.1 LLM Provider Support

**Supported providers**:
- **Claude Code CLI** (primary, uses subscription)
- **Anthropic API** (fallback with API key)
- **OpenAI API** (fallback with API key)

**Model tiers**:
- `ModelTier.ORCHESTRATOR`: Best reasoning/planning (claude-sonnet-4)
- `ModelTier.WORKER`: Balanced mid-depth work (claude-sonnet-4)
- `ModelTier.LEAF`: Fast/cheap parallel execution (claude-haiku-4)
- `ModelTier.JUDGE`: Evaluation/verification (uses leaf_model)
- `ModelTier.EMBEDDER`: Routing/similarity (uses worker_model)

### 3.2 MCP Client (Vault Operations)

**Read operations**:
- `read_note(path)` → VaultNote with content, metadata, hash, mtime
- `search(query, limit, path_filter, type_filter)` → Keyword-based search with scoring
- `list_notes(directory, recursive)` → Directory enumeration
- `get_file_hash(path)` → SHA256 hash for cache validation

**Write operations**:
- `write_note(path, content, metadata, resolve_wikilinks)`
- `update_note(path, content, append)`
- `update_frontmatter(path, updates)`

**Delete operations** (HITL gated):
- `delete_note(path)` → Queues for human review
- `execute_delete(path)` → Executes pre-approved deletion

### 3.3 Code Executor (Sandbox)

**Restricted environment**:
- Dangerous builtins disabled
- Import whitelist enforced
- File access restricted to vault path
- 60s timeout, 512MB memory limit

**ObsidianTools API** available in scripts:
- `obsidian.search(query, limit, path_filter)`
- `obsidian.read_note(path)`
- `obsidian.write_note(path, content, note_type, frontmatter)`
- `obsidian.list_notes(directory, recursive)`
- `obsidian.get_frontmatter(path)`
- `obsidian.get_hash(path)`
- `obsidian.create_wikilink(path)`

### 3.4 History & Observability

**Per-run artifact directory**:
```
History/Runs/<run_id>/
├── run.manifest.json      # Inputs, versions, config hashes
├── events.jsonl           # Node lifecycle events
├── dag.json               # DAG structure with statuses
├── decisions/             # Routing, decomposition decisions
├── metrics/               # Per-node and summary metrics
├── errors/                # Error records with context
├── artifacts/             # Large payloads (by hash)
├── replay/manifest.json   # Deterministic replay bundle
├── final.report.md        # Human-readable output
└── final.summary.json     # Machine-readable output
```

**Per-node metrics**:
- `node_id`, `task`, `depth`, `status`, `tokens_used`
- `start_time`, `end_time`, `duration_ms`
- `cache_key`, `cache_hit`, `error`

### 3.5 Redis Cache

**Two-stage caching**:
- **Main cache**: Verified, long-lived (30 days TTL)
- **Staging cache**: Provisional, short-lived (24 hours TTL)
- **Promotion**: Staging → Main via `promote()` after HITL review

**Hash validation**:
- Cache keys include `context_hash` from vault content
- Before lookup, verify hash still matches
- Mismatch → invalidate and recompute

**Budget ledger**:
- `init_budget(run_id, token_budget)`
- `deduct_budget(run_id, tokens)` → Atomic deduction
- `get_remaining_budget(run_id)`

---

## 4. Decision Log

This appendix documents key design decisions with rationale.

### D1: Retrieval Recovery Policy

**Decision**: Confidence-gated regeneration + three-tier fallback (regen → direct search → human checkpoint)

**Options considered**:
1. Re-generate with hints only
2. Multi-strategy parallel retrieval
3. Graceful degradation (accept weak context)
4. Confidence scoring + tiered recovery

**Why this won**: Balances retrieval quality with budget constraints. Multi-strategy parallel is expensive; graceful degradation hides problems. Tiered approach provides measurable recovery with caps.

**Revisit if**: Retrieval models become cheap enough to always run parallel strategies.

---

### D2: DAG Cross-Subtask Dependencies

**Decision**: Soft dependencies + dynamic context injection (context packets)

**Options considered**:
1. Static DAG only
2. Dynamic re-parenting
3. Two-pass execution
4. Explicit soft dependency hints

**Why this won**: Soft deps preserve parallelism while enabling context sharing. Dynamic re-parenting causes thrashing; two-pass doubles cost. Context packets inject value without restructuring.

**Revisit if**: Complex tasks consistently produce better results with two-pass.

---

### D3: Type Consistency

**Decision**: Contracts-first node + convention-based merging as safety net

**Options considered**:
1. Early type extraction (dedicated node)
2. Convention-based merging
3. Schema-first decomposition
4. Shared mutable type registry

**Why this won**: Contracts-first ensures single source of truth. Convention-based merging catches drift. Mutable registry breaks determinism and caching.

**Revisit if**: LLMs become reliable enough to maintain consistency without explicit contracts.

---

### D4: Sandbox Security

**Decision**: Configurable profiles with strict (vault-only) as default

**Options considered**:
1. Vault-only strict
2. Read-only filesystem
3. Allowlisted network
4. Configurable profiles

**Why this won**: Profiles balance security with power-user needs. Strict default matches "never exfiltrate" invariant. Opt-in relaxation requires explicit per-run consent.

**Revisit if**: Common use cases require network access frequently enough to justify changing default.

---

### D5: Vault Ingestion Versioning

**Decision**: Immutable snapshots + shadow index (outside vault)

**Options considered**:
1. Immutable snapshots only
2. Update-in-place
3. Git-backed versioning
4. Shadow index + snapshots

**Why this won**: Immutable snapshots preserve provenance. Shadow index provides "latest" convenience without vault churn. External DB avoids sync conflicts.

**Revisit if**: Multi-user collaboration requires vault-internal index for portability.

---

### D6: Token Budget Distribution

**Decision**: Hierarchical reserves (parent keeps synthesis reserve)

**Options considered**:
1. Fixed per-depth allocation
2. Complexity-weighted pre-allocation
3. Shared pool (first-come-first-served)
4. Hierarchical reserves

**Why this won**: Guarantees synthesis always completes (the most important output). Dynamic allocation within children prevents waste. Fixed-per-depth is too rigid; shared pool starves synthesis.

**Revisit if**: Token costs drop enough that over-allocation is acceptable.

---

### D7: Verification Strictness

**Decision**: Progressive strictness with configurable per-check

**Options considered**:
1. All advisory
2. Syntax blocking only
3. Configurable per-check
4. Progressive strictness levels

**Why this won**: Sensible defaults (basic: imports + syntax) enable fast iteration. Strict mode enforces quality when needed. Per-check config satisfies power users.

**Revisit if**: Verification becomes fast/cheap enough to always run strict.

---

### D8: Resume Semantics

**Decision**: Delta verification (re-verify only changed context)

**Options considered**:
1. Trust all completed nodes
2. Re-verify all completed nodes
3. Selective replay (user-specified)
4. Delta verification

**Why this won**: Delta verification balances correctness with speed. Trusting blindly misses vault changes; re-verifying all wastes compute. Selective replay available as override.

**Revisit if**: Vault content rarely changes (then trust-all becomes safe default).

---

### D9: Concurrency Model

**Decision**: Tiered adaptive with separate LLM/local limits

**Options considered**:
1. Fixed thread pool
2. Adaptive to provider rate limits
3. User-configurable only
4. Tiered adaptive

**Why this won**: Adaptive backoff handles real-world rate limits gracefully. Separate limits prevent LLM saturation from blocking local work. User override preserves control.

**Revisit if**: Providers expose reliable rate limit headers (then query-based adaptation becomes viable).

---

### D10: Decomposition Strategies

**Decision**: Hybrid (template skeletons + LLM refinement)

**Options considered**:
1. Hardcoded templates
2. LLM + strategy hints
3. Vault-learned patterns
4. Hybrid templates + LLM

**Why this won**: Skeletons enforce invariants (contracts-first, verification placement). LLM fills task-specific details. Vault-learned is future enhancement, not MVP requirement.

**Revisit if**: Sufficient vault examples accumulate to enable pattern learning.

---

### D11: Test Generation Timing

**Decision**: Spec-first stubs + post-implementation pass (for software strategy)

**Options considered**:
1. Co-generation (alongside implementation)
2. Post-implementation only
3. TDD-style (full tests first)
4. Configurable per-strategy

**Why this won**: Spec-first shapes architecture toward testability. Post-implementation pass has full context for coherent tests. Co-generation produces inconsistent fixtures.

**Revisit if**: LLMs improve at maintaining test consistency across co-generation.

---

### D12: Multi-Vault Support

**Decision**: Layered vaults with priority + optional namespacing

**Options considered**:
1. Single vault only
2. Vault layering (priority order)
3. Vault namespacing
4. Defer to later

**Why this won**: Layering unlocks reusable pattern vaults without complexity. Namespacing adds precision for Code Mode. Single-vault remains the simple default.

**Revisit if**: Vault collision issues become common (then stronger namespacing needed).

---

### D13: Conflict Resolution in Synthesis

**Decision**: Preserve both perspectives + LLM reconciliation + checkpoint for high-impact

**Options considered**:
1. LLM always picks winner
2. Flag all conflicts for human
3. Preserve both with attribution
4. Confidence-weighted winner

**Why this won**: Conflicts often represent valid tradeoffs, not errors. Preserving both maintains transparency. LLM reconciliation attempts resolution. Human checkpoint only when it matters.

**Revisit if**: Users consistently want Shad to make decisions rather than present options.

---

### D14: Gap Detection Method

**Decision**: Combined scoring (history + patterns + optional LLM audit)

**Options considered**:
1. Pattern matching only
2. Query history analysis
3. LLM assessment
4. Combined scoring

**Why this won**: History-based gaps reflect actual pain points. Pattern coverage catches obvious omissions. LLM audit adds depth but is expensive—opt-in only.

**Revisit if**: LLM audit costs drop significantly.

---

## Appendix A: CLI Reference

```bash
# Core commands
shad run "goal" [options]
shad status <run_id>
shad trace tree <run_id>
shad trace node <run_id> <node_id>
shad resume <run_id> [--replay <target>]
shad export <run_id> --output <dir>

# Run options
--vault <path>              # Vault path (repeatable for layering)
--strategy <name>           # Force strategy (software|research|analysis|planning)
--max-depth <n>             # Max recursion depth (default: 3)
--max-nodes <n>             # Max DAG nodes (default: 50)
--max-time <seconds>        # Max wall time (default: 300)
--max-tokens <n>            # Max token budget (default: 100000)
--max-parallel <n>          # Concurrency cap
--profile <name>            # Sandbox profile (strict|local|extended)
--verify <level>            # Verification level (off|basic|build|strict)
--write-files               # Write output files to disk
--output <dir>              # Output directory (requires --write-files)
--no-code-mode              # Disable Code Mode (direct search only)
--no-cache                  # Bypass cache
--no-checkpoints            # Disable non-safety checkpoints
--tests <mode>              # Test generation (off|stubs|post|tdd|co)

# Vault commands
shad ingest github <url> [--preset docs|mirror|deep]
shad ingest <path> [--type file|folder]
shad enrich <snapshot_id> --deep
shad vault analyze [--llm-audit]
shad sources ls
shad sources export --format yaml --out <path>
shad sources pin <url> --snapshot <id>

# Cache commands
shad cache clear [--scope run|source|all]
```

---

## Appendix B: API Reference

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/run` | Execute reasoning task |
| `GET` | `/v1/run/:id` | Get run status/results |
| `POST` | `/v1/run/:id/resume` | Resume partial run |
| `GET` | `/v1/runs` | List recent runs |
| `GET` | `/v1/vault/status` | Check vault connection |
| `GET` | `/v1/vault/search` | Search vault |
| `GET` | `/v1/health` | Health check |

### POST /v1/run

```json
{
  "goal": "Build a REST API for user management",
  "vaults": [
    {"id": "project", "root": "/vaults/proj", "priority": 0},
    {"id": "patterns", "root": "/vaults/patterns", "priority": 1}
  ],
  "strategy": "software",
  "config": {
    "max_depth": 4,
    "max_tokens": 100000,
    "verify": "basic"
  },
  "write_files": false,
  "profile": "strict"
}
```

### Response

```json
{
  "run_id": "abc123",
  "status": "SUCCESS",
  "result": "...",
  "manifest": {
    "files": [...],
    "notes": [...]
  },
  "metrics": {
    "total_tokens": 45000,
    "duration_ms": 120000,
    "nodes_executed": 23
  }
}
```

---

## Appendix C: Glossary

| Term | Definition |
|------|------------|
| **Code Mode** | LLM generates Python scripts to query vault instead of keyword search |
| **Context packet** | Compact artifact from completed node (summary, artifacts, keywords) |
| **Contracts-first** | Types & contracts node runs before implementation |
| **DAG** | Directed Acyclic Graph of execution nodes |
| **Delta verification** | Re-verify only nodes whose context changed |
| **Export index** | Symbol → file mapping built in first pass of code generation |
| **Hard deps** | Dependencies that must complete before node starts |
| **HITL** | Human-in-the-loop checkpoint |
| **Manifest** | Structured file output (paths, content, metadata) |
| **RLM** | Recursive Language Model engine |
| **Shadow index** | External DB mapping source URLs to latest snapshots |
| **Skeleton** | Strategy template with required/optional stages |
| **Soft deps** | Dependencies that are useful but not blocking |
| **Subset fingerprint** | Hash of notes actually used by a node |
| **Vault layering** | Multiple vaults with priority order for retrieval |
