# Shad Implementation Plan

## The Vision

**Shad enables AI to utilize virtually unlimited context.**

The goal: Load an Obsidian vault with curated, up-to-date knowledge (documentation, code examples, architecture patterns, best practices), then ask Shad to accomplish complex tasks that would be impossible with a single context window.

**Example**: Build a production-quality mobile app in one shot by:
1. Loading a vault with React Native docs, great app examples, UI/UX patterns, API design guides
2. Running: `shad run "Build a task management app with auth, offline sync, and push notifications" --vault ~/MobileDevVault --strategy software --write-files`
3. Shad recursively decomposes, retrieves targeted context for each subtask, generates code with contracts-first consistency, verifies outputs, and assembles a complete codebase

This is not prompt engineering. This is **inference-time scaling** — treating context as an explorable environment, not a fixed input.

---

## Implementation Phases

### Phase 1 — Foundation (COMPLETE)

- [x] Repository structure and Docker setup
- [x] Shad API (`POST /v1/run`, `GET /v1/run/:id`, resume)
- [x] CLI (`shad run`, `shad status`, `shad trace`, `shad resume`)
- [x] RLM Engine with recursive decomposition
- [x] Budget enforcement (depth, nodes, time, tokens)
- [x] History artifacts (DAG, metrics, reports)
- [x] Redis caching with hierarchical keys

### Phase 2 — Obsidian Integration (COMPLETE)

- [x] MCP client for vault operations
- [x] Per-subtask context retrieval
- [x] Code Mode: LLM generates Python scripts for custom retrieval
- [x] Sandbox execution with `obsidian.search()`, `obsidian.read_note()`, etc.
- [x] Fallback to direct search when scripts fail
- [x] Full-path wikilink citations

### Phase 3 — Task-Aware Decomposition (IN PROGRESS)

**Decision**: Hybrid template skeletons + LLM refinement (see SPEC.md D10)

- [ ] **Strategy Skeletons**
  - Define required stages, optional stages, constraints per strategy
  - `software`: contracts-first, imports-must-resolve, schema-first when DB involved
  - `research`: must-cite-vault, max-claims-per-source
  - `analysis`: criteria-coverage, explicit-tradeoffs
  - `planning`: milestones, dependencies

- [ ] **Heuristic Strategy Selection** (no LLM call for obvious cases)
  - Pattern-match task text against keyword sets
  - Output: `strategy_guess` + `guess_confidence`
  - confidence ≥ 0.7 → proceed; < 0.7 → default to `analysis`
  - User can override with `--strategy X`

- [ ] **LLM Refinement**
  - LLM fills in task-specific nodes within skeleton constraints
  - Can add/remove optional stages, split implementation into modules
  - Cannot violate required stages without explicit waiver
  - May request strategy switch mid-execution with evidence

- [ ] **Soft Dependencies & Context Packets**
  - Decomposition emits `hard_deps` (must complete) and `soft_deps` (useful if available)
  - Completed nodes produce context packets (summary, artifacts, keywords)
  - Scheduler injects packets into pending nodes' retrieval
  - Limited rerun gate when soft dep completes and node would benefit

### Phase 4 — Code Generation Output

**Decision**: Two-pass with manifest + convention-based validation (see SPEC.md D3)

- [ ] **File Manifest Output**
  - All code-producing runs emit structured manifest: `{path, content, language, hash, source_nodes}`
  - Writing to filesystem is always explicit (`--write-files`)
  - CLI and API share same manifest contract
  - `shad export <run_id> --output ./out` to materialize later

- [ ] **Two-Pass Import Resolution**
  - **Pass 1**: Build export index (symbol → file mapping) via early contracts node
  - **Pass 2**: Generate implementations using export index as ground truth
  - Post-generation validation: check all imports resolve to existing files/symbols
  - Output structured report: `{missing_modules, missing_symbols}`

- [ ] **Contracts-First Type Consistency**
  - `Types & Contracts` node produces canonical type artifacts
  - Implementation nodes import from canonical artifacts only
  - New type needs emit `contract_change_request` artifacts
  - Reconciliation step merges identical proposals, flags conflicts
  - For DB/API/Auth tasks: enforce schema-first (data model → contracts → impl)

- [ ] **Write Semantics**
  - Only write under `output_root` (no `../` traversal)
  - `--overwrite` flag for conflicts (default: fail)
  - Emit write report with paths, skipped conflicts, hashes

### Phase 5 — Verification Layer

**Decision**: Progressive strictness with configurable per-check (see SPEC.md D7)

- [ ] **Verification Checks**
  - Import resolution: all imports resolve to existing files/symbols
  - Syntax/parse: code is syntactically valid
  - Manifest integrity: no path traversal, no duplicates, within output root
  - Type check: TypeScript/Flow errors via `tsc --noEmit`
  - Unit tests: tests pass
  - Lint: style conformance

- [ ] **Verification Levels**
  - `--verify=off`: No checks
  - `--verify=basic` (default): Imports + syntax + manifest blocking
  - `--verify=build`: Basic + typecheck blocking
  - `--verify=strict`: Build + tests blocking
  - Per-check override: `--block-on typecheck` / `--warn-only lint`

- [ ] **Error Classification & Repair**
  - Syntax/lint → local repair only
  - Type errors → local repair with sibling context (contracts/types)
  - Unit tests → local repair with sibling outputs
  - Integration tests → escalate to parent coordinator immediately
  - Contract mismatch → parent coordination + contract update node
  - Error signature hashing prevents infinite loops
  - Max 2 local retries per node, max 10 escalations per run

- [ ] **Test Generation** (software strategy)
  - Spec-first stubs: after contracts, generate `TEST_PLAN.md` + test stubs
  - Post-implementation pass: generate full test suite with complete codebase context
  - Tests advisory by default, blocking in `--verify=strict`
  - Override: `--tests off|stubs|post|tdd|co`

### Phase 6 — Iterative Refinement

**Decision**: Tiered fallback with NEEDS_HUMAN as default for high-impact (see SPEC.md D8)

- [ ] **Run States**
  - `SUCCESS`: Meets acceptance criteria
  - `PARTIAL`: Produced artifacts but did not meet criteria
  - `FAILED`: Could not produce meaningful artifacts or safety stop
  - `NEEDS_HUMAN`: Paused, context preserved, awaiting human input

- [ ] **Max Iterations Policy (Tiered Fallback)**
  - High-impact task OR substantial artifacts → `NEEDS_HUMAN`
  - Low-risk task, verification advisory → `PARTIAL`
  - Cannot proceed safely OR no artifacts → `FAILED`
  - Same run hits max iterations twice without human input → `FAILED_WITH_DIAGNOSTIC`
  - Always return: best artifacts + failure report + suggested next inputs

- [ ] **Delta Verification on Resume**
  - Store per completed node: `used_notes[]`, `used_note_hashes{}`, `subset_fingerprint`
  - On resume: check vault manifest against stored hashes
  - Node is stale if any `used_note_hash` differs
  - Stale nodes undergo re-verification (or re-execution for contracts nodes)
  - Unchanged nodes are trusted
  - Selective replay: `--replay node_id`, `--replay subtree:X`, `--replay stale`

- [ ] **Human-in-the-Loop Checkpoints**
  - Explicit markers: `[REVIEW]`, `[APPROVE]` in prompts
  - Auto-triggers: high-impact node at depth ≤ 1, low confidence + high fan-out
  - Hard safety (non-negotiable): file writes outside sandbox, network beyond allowlist
  - `--no-checkpoints` disables all except hard safety
  - Max 5 checkpoints per run, then degrade to batch review at end
  - Checkpoint presents: node summary, decision to approve, confidence signals, options

### Phase 7 — Vault Curation Tools

**Decision**: Combined scoring for gap detection, configurable ingestion presets (see SPEC.md D14)

- [ ] **Ingestion Pipeline**
  - `shad ingest github <url>` — Clone and process repository
  - Presets: `mirror` (raw files), `docs` (default, README/docs/metadata), `deep` (semantic index)
  - Immutable timestamped snapshots: `Sources/<domain>/<source_id>/<YYYY-MM-DD>/`
  - Required frontmatter: `source_url`, `source_type`, `ingested_at`, `source_revision`, `content_hash`
  - Post-hoc enrichment: `shad enrich <snapshot_id> --deep`

- [ ] **Shadow Index** (outside vault, in `~/.shad/index.sqlite`)
  - Maps `source_url → latest_snapshot`
  - Schema: `sources`, `snapshots`, `latest` tables
  - Update policies: `manual`, `notify`, `auto`
  - Export: `shad sources export --format yaml --out <path>`
  - Pin snapshots: `shad sources pin <url> --snapshot <id>`

- [ ] **Vault Analysis & Gap Detection**
  - `shad vault analyze [--llm-audit]`
  - Combined scoring: `0.55 * history_pain + 0.25 * coverage_miss + 0.20 * llm_score`
  - History pain: query frequency, median retrieval_score, fallback rate, downstream failures
  - Coverage miss: missing anchor notes for common topics, missing templates
  - LLM audit (optional): send vault summary, ask for top 10 gaps
  - Output: ranked gaps with evidence, suggested additions, priority

- [ ] **Note Standardization**
  - Enforce consistent frontmatter
  - Extract and tag code examples
  - Link related notes automatically
  - Progressive standardization ("Gardener Pattern")

---

## Current Status

### Complete
- RLM Engine with recursive decomposition
- Obsidian MCP integration
- Code Mode (LLM-generated retrieval scripts)
- Per-subtask context retrieval
- Budget enforcement and partial results
- History artifacts and resume
- Redis caching with hash validation

### In Progress
- Task-aware decomposition (Phase 3)
  - Strategy skeletons
  - Heuristic selection + LLM refinement
  - Soft dependencies and context packets

### Next Up
- File output mode (Phase 4)
- Verification layer (Phase 5)

---

## Success Metrics

1. **Context Utilization**: A 100MB vault should contribute meaningfully to output
2. **Task Complexity**: Successfully complete tasks requiring 10+ subtasks
3. **Code Quality**: Generated code passes linting, type-checking, basic tests
4. **One-Shot Success**: Complex tasks complete without human intervention
5. **Iteration Efficiency**: When iteration needed, converge in ≤3 passes
6. **Type Consistency**: Generated multi-file codebases have zero import resolution errors
7. **Resume Efficiency**: Delta verification re-executes only changed nodes

---

## Example: Building a Mobile App

```bash
# 1. Prepare vault with mobile dev knowledge
shad ingest github https://github.com/facebook/react-native --preset docs --vault ~/MobileVault
shad ingest github https://github.com/excellent-app/example --preset deep --vault ~/MobileVault
shad ingest ~/notes/mobile-patterns.md --vault ~/MobileVault

# 2. Run the build task
shad run "Build a task management mobile app with:
  - User authentication (email + OAuth)
  - Task CRUD with categories and due dates
  - Offline-first with background sync
  - Push notifications for reminders
  - Clean, modern UI following Material Design" \
  --vault ~/MobileVault \
  --strategy software \
  --verify strict \
  --write-files --output ./TaskApp \
  --max-depth 5

# 3. Result: Complete React Native project in ./TaskApp/
#    - Types & contracts generated first
#    - All imports resolve correctly
#    - Passes type checking and tests
```

The vault contains the "how" (patterns, examples, docs). Shad provides the "engine" (decomposition, retrieval, generation, verification, assembly). Together: complex tasks in one shot.

---

## Key Design Decisions

See [SPEC.md](SPEC.md) Section 4 (Decision Log) for full rationale on each decision:

| ID | Decision | Choice |
|----|----------|--------|
| D1 | Retrieval recovery | Confidence-gated 3-tier fallback |
| D2 | Cross-subtask deps | Soft deps + context packets |
| D3 | Type consistency | Contracts-first + convention merge |
| D4 | Sandbox security | Configurable profiles (strict default) |
| D5 | Vault versioning | Immutable snapshots + shadow index |
| D6 | Token budget | Hierarchical reserves (25% parent) |
| D7 | Verification | Progressive strictness levels |
| D8 | Resume | Delta verification |
| D9 | Concurrency | Tiered adaptive (separate LLM/local) |
| D10 | Decomposition | Template skeletons + LLM refinement |
| D11 | Test generation | Spec-first stubs + post-impl pass |
| D12 | Multi-vault | Layered with priority + namespacing |
| D13 | Conflict resolution | Preserve both + reconcile + checkpoint |
| D14 | Gap detection | Combined scoring (history + patterns) |
