# Shad Implementation Plan

## The Vision

**Shad enables AI to utilize virtually unlimited context.**

The goal: Load an Obsidian vault with curated, up-to-date knowledge (documentation, code examples, architecture patterns, best practices), then ask Shad to accomplish complex tasks that would be impossible with a single context window.

**Example**: Build a production-quality mobile app in one shot by:
1. Loading a vault with React Native docs, great app examples, UI/UX patterns, API design guides
2. Running: `shad run "Build a task management app with auth, offline sync, and push notifications" --vault ~/MobileDevVault`
3. Shad recursively decomposes, retrieves targeted context for each subtask, generates code, and assembles a complete codebase

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

The current decomposition is naive (length-based heuristic). For complex tasks like "build an app", we need **domain-aware decomposition**:

- [ ] **Software Architecture Decomposition**
  - Recognize software tasks and decompose by component/layer
  - Example: "Build mobile app" → Project structure, Navigation, State management, UI components, Data layer, Features...
  - Each feature → Screens, Components, Hooks, API calls, Tests

- [ ] **Decomposition Strategies**
  - `software`: By architecture layer and feature
  - `research`: By question and sub-question
  - `analysis`: By data source and dimension
  - `creative`: By section and element

- [ ] **Smart Depth Control**
  - Decompose until subtasks are "atomic" (directly answerable)
  - LLM decides decomposition, not hardcoded heuristics
  - Budget-aware: stop decomposing when approaching limits

### Phase 4 — Code Generation Output

Currently Shad outputs markdown. For software tasks, it should output **actual files**:

- [ ] **File Output Mode**
  - Detect when task requires code generation
  - Output structured file manifest: `{path, content, language}`
  - Write files to output directory or return as artifact

- [ ] **Multi-File Assembly**
  - Collect code from all subtasks
  - Resolve imports and dependencies
  - Generate project structure (package.json, requirements.txt, etc.)
  - Handle file conflicts (same path from different subtasks)

- [ ] **Code Context Awareness**
  - When generating file B, include relevant parts of file A as context
  - Track dependencies between generated files
  - Ensure consistent naming, imports, types across files

### Phase 5 — Verification Layer

Generated code should be verified before returning:

- [ ] **Syntax Validation**
  - Parse generated code (AST for Python/JS/TS)
  - Catch syntax errors before returning

- [ ] **Type Checking** (for TypeScript)
  - Run `tsc --noEmit` on generated code
  - Feed errors back for correction

- [ ] **Test Generation**
  - Generate tests alongside implementation
  - Run tests as verification step

- [ ] **Example Conformance**
  - Compare generated code structure to vault examples
  - Flag deviations from patterns in the vault

### Phase 6 — Iterative Refinement

Single-pass generation won't always succeed. Enable iteration:

- [ ] **Error Feedback Loop**
  - If verification fails, create correction subtask
  - Include error message and relevant context
  - Re-generate with learned constraints

- [ ] **Human-in-the-Loop Checkpoints**
  - Pause at configurable points for human review
  - Accept corrections and continue
  - Learn from corrections for future runs

- [ ] **Progressive Output**
  - Stream partial results as subtasks complete
  - Allow early termination with partial output
  - Resume from any checkpoint

### Phase 7 — Vault Curation Tools

The vault quality determines output quality. Provide tools for vault curation:

- [ ] **Ingestion Pipeline**
  - `shad ingest <url>` — Fetch and convert docs/repos to vault notes
  - `shad ingest <github-repo>` — Clone and index a repository
  - Auto-extract: README, API docs, code examples, patterns

- [ ] **Vault Analysis**
  - `shad vault analyze` — Report on vault coverage
  - Identify gaps (e.g., "no examples for authentication")
  - Suggest additions based on common queries

- [ ] **Note Standardization**
  - Enforce consistent frontmatter
  - Extract and tag code examples
  - Link related notes automatically

---

## Current Status

### Complete
- RLM Engine with recursive decomposition
- Obsidian MCP integration
- Code Mode (LLM-generated retrieval scripts)
- Per-subtask context retrieval
- Budget enforcement and partial results
- History artifacts and resume

### In Progress
- Task-aware decomposition (Phase 3)

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

---

## Example: Building a Mobile App

```bash
# 1. Prepare vault with mobile dev knowledge
shad ingest https://reactnative.dev/docs --vault ~/MobileVault
shad ingest https://github.com/excellent-app/example --vault ~/MobileVault
shad ingest ~/notes/mobile-patterns.md --vault ~/MobileVault

# 2. Run the build task
shad run "Build a task management mobile app with:
  - User authentication (email + OAuth)
  - Task CRUD with categories and due dates
  - Offline-first with background sync
  - Push notifications for reminders
  - Clean, modern UI following Material Design" \
  --vault ~/MobileVault \
  --output ./TaskApp \
  --max-depth 5

# 3. Result: Complete React Native project in ./TaskApp/
```

The vault contains the "how" (patterns, examples, docs). Shad provides the "engine" (decomposition, retrieval, generation, assembly). Together: complex tasks in one shot.
