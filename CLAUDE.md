# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Vision

**Shad enables AI to utilize virtually unlimited context.**

The goal: Load an Obsidian vault with curated knowledge (documentation, code examples, architecture patterns), then accomplish complex tasks that would be impossible with a single context window.

**Example use case**: Build a production-quality mobile app by loading a vault with React Native docs, great app examples, and UI patterns, then running:
```bash
shad run "Build a task management app with auth and offline sync" \
  --vault ~/MobileDevVault \
  --strategy software \
  --write-files --output ./TaskApp
```

Shad recursively decomposes the task, retrieves targeted context for each subtask from the vault, generates code, and assembles a complete codebase.

## Core Premise

> **Long-context reasoning is an inference problem, not a prompting problem.**

Instead of cramming context into a single prompt, Shad:
1. Treats the vault as an **explorable environment**
2. **Selects a strategy** (software, research, analysis) with domain-specific decomposition
3. **Decomposes** complex tasks into subtasks using strategy skeletons
4. **Retrieves** targeted context for each subtask via Code Mode
5. **Generates** outputs with contracts-first type consistency
6. **Verifies** outputs (syntax, types, tests) with configurable strictness
7. **Assembles** results into coherent output (file manifests for code)

## Architecture

```
User
   │
   ▼
Shad CLI / API
   │
   ├── RLM Engine (recursive decomposition + execution)
   │       │
   │       ├── Strategy Selection (heuristic + LLM refinement)
   │       │
   │       ├── Code Mode ─────────────────────────┐
   │       │   (LLM generates Python scripts)    │
   │       │                                      ▼
   │       ├── CodeExecutor ──────────> ObsidianTools
   │       │   (sandboxed, profile-based)    │
   │       │                                 ▼
   │       │                           Vault(s)
   │       │                           (layered, priority-ordered)
   │       │
   │       ├── Verification Layer (syntax, types, tests)
   │       │
   │       ├── Redis (cache + budget ledger)
   │       └── LLM Provider (Claude Code CLI)
   │
   ├── History/ (run artifacts)
   └── Shadow Index (~/.shad/index.sqlite)
```

### Key Architectural Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Strategy selection | Heuristic + LLM override | Fast default, flexible when needed |
| Decomposition | Template skeletons + LLM refinement | Enforces invariants (contracts-first) |
| Cross-subtask deps | Soft deps + context packets | Parallelism with context sharing |
| Type consistency | Contracts-first node | Single source of truth |
| Sandbox security | Configurable profiles (strict default) | Balance safety and power |
| Token budget | Hierarchical reserves (25% parent) | Guarantees synthesis completes |
| Verification | Progressive strictness | Fast iteration, strict when needed |
| Cache invalidation | Hash + TTL hybrid | Correctness + operational sanity |
| Resume | Delta verification | Only re-verify changed context |

### Code Mode (Key Innovation)

Instead of simple keyword search, **Code Mode** lets the LLM write custom retrieval logic:

```python
# LLM generates this script for: "How do I implement auth?"
results = obsidian.search("authentication React Native", limit=10)
auth_patterns = obsidian.read_note("Patterns/Authentication.md")

context_parts = []
for r in results:
    if "OAuth" in r["content"] or "JWT" in r["content"]:
        context_parts.append(f"## {r['path']}\n{r['content'][:2000]}")

__result__ = {
    "context": "\n\n".join(context_parts),
    "citations": [...],
    "confidence": 0.72,
    "why": "Found OAuth pattern note + 3 examples"
}
```

**Retrieval recovery** (when results are poor):
1. **Tier A**: Regenerate script with hints (1 retry)
2. **Tier B**: Broadened direct search fallback
3. **Tier C**: Human checkpoint (high-impact nodes only)

### Module Responsibilities

| Module | Location | Purpose |
|--------|----------|---------|
| `engine/rlm.py` | Core | Recursive decomposition, DAG execution, Code Mode orchestration |
| `engine/llm.py` | Core | LLM abstraction, retrieval script generation, strategy selection |
| `sandbox/executor.py` | Execution | Sandboxed Python execution with security profiles |
| `sandbox/tools.py` | Execution | `ObsidianTools` class with `search()`, `read_note()`, etc. |
| `mcp/client.py` | Integration | Direct MCP client for vault operations |
| `cache/redis_cache.py` | Caching | Hierarchical keys, hash validation, TTL |
| `history/manager.py` | Persistence | Run artifacts (DAG, metrics, reports) |

## Development Commands

```bash
cd services/shad-api

# Activate virtual environment
source .venv/bin/activate

# Install in editable mode
pip install -e ".[dev]"

# Run linter
ruff check src/shad/

# Run linter with auto-fix
ruff check src/shad/ --fix

# Type checking
mypy src/shad/

# Run tests
pytest

# Run with coverage
pytest --cov=shad
```

## CLI Commands

```bash
# Run with vault context (Code Mode enabled by default)
shad run "Your task" --vault /path/to/vault

# Use multiple vaults (priority order)
shad run "Build API" --vault ~/Project --vault ~/Patterns --vault ~/Docs

# Force a specific strategy
shad run "Build app" --vault ~/v --strategy software

# Control verification strictness
shad run "Build app" --vault ~/v --verify strict

# Generate files to disk
shad run "Build app" --vault ~/v --write-files --output ./out

# Use extended sandbox (allow network)
shad run "..." --vault ~/v --profile extended --net-allow docs.example.com

# Check run status
shad status <run_id>

# View execution DAG
shad trace tree <run_id>

# Resume with delta verification
shad resume <run_id>

# Replay specific nodes
shad resume <run_id> --replay node_id
shad resume <run_id> --replay stale

# Export files from a run
shad export <run_id> --output ./out
```

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /v1/run` | Execute reasoning task |
| `GET /v1/run/:id` | Get run status/results |
| `POST /v1/run/:id/resume` | Resume partial run |
| `GET /v1/runs` | List recent runs |
| `GET /v1/vault/status` | Check vault connection |
| `GET /v1/vault/search` | Search vault |

## Key Concepts

### Strategy Skeletons

Each strategy defines required stages, optional stages, and constraints:

**Software strategy**:
```yaml
required:
  - clarify_requirements
  - project_layout
  - types_contracts      # Contracts-first!
  - implementation
  - verification
  - synthesis
optional:
  - db_schema
  - auth
  - openapi
constraints:
  - contracts_first: true
  - imports_must_resolve: true
```

### Soft Dependencies & Context Packets

- **Hard deps**: Must complete before node starts
- **Soft deps**: Useful if available, but not blocking
- When a node completes, it produces a **context packet** (summary, artifacts, keywords)
- Scheduler injects packets into pending nodes' retrieval

### Two-Pass Import Resolution (File Output)

1. **Pass 1**: Build export index (symbol → file mapping)
2. **Pass 2**: Generate implementations using export index as ground truth
3. **Validation**: Check all imports resolve to existing files/symbols

### Verification Levels

- `--verify=off`: No checks
- `--verify=basic` (default): Imports + syntax + manifest blocking
- `--verify=build`: Basic + typecheck blocking
- `--verify=strict`: Build + tests blocking

### Sandbox Security Profiles

| Profile | File Access | Network | Use Case |
|---------|-------------|---------|----------|
| `strict` (default) | Vault only | None | Safe, deterministic |
| `local` | + Allowlisted roots | None | Reference local repos |
| `extended` | + Allowlist | HTTP GET to allowlist | Fetch live docs |

## Budget System

**Hierarchical token allocation**:
```
Run budget (max_tokens)
└── Root node envelope
    ├── reserve (25%, min 800)  # Protected for synthesis
    └── spendable → distributed to children
```

Every run enforces hard limits:
- `max_depth`: Maximum recursion depth (default: 3)
- `max_nodes`: Maximum DAG nodes (default: 50)
- `max_wall_time`: Total execution time in seconds (default: 300)
- `max_tokens`: Total token budget (default: 100000)

When budgets are exhausted, run returns partial results with `status: partial`.

## How Vault Context Flows

```
1. User: "Build a login screen"
         ↓
2. Strategy selected: software
         ↓
3. RLM decomposes using software skeleton:
   - Types & contracts (hard dep for all below)
   - "Design login UI layout"
   - "Implement form validation"
   - "Add OAuth integration"
         ↓
4. For each subtask, Code Mode:
   a. LLM generates retrieval script
   b. Script searches vault(s) for relevant examples
   c. Script returns structured result with confidence
   d. If confidence low → recovery (regen/fallback/checkpoint)
         ↓
5. LLM generates output using retrieved context
         ↓
6. Verification layer checks (based on --verify level)
         ↓
7. If errors → repair loop (local → escalate → human)
         ↓
8. Results synthesized bottom-up → file manifest
```

## Extending the System

### Adding Vault Content
The vault quality determines output quality. Good vault content includes:
- Official documentation converted to markdown
- Code examples with explanations
- Architecture patterns and best practices
- Common pitfalls and solutions

### Adding a New Strategy
1. Define skeleton in `engine/strategies/` with required/optional stages
2. Add hint pack (system prompt additions)
3. Update heuristic classifier keywords
4. Add acceptance tests

### Adding a Verification Check
1. Implement check in `verification/`
2. Add to verification level definitions
3. Define blocking vs advisory default
4. Implement repair action for failures

## Hard Invariants (from CORE/invariants.md)

1. **Never Auto-Publish**: No irreversible side effects without explicit human approval
2. **Never Exfiltrate**: No sending data externally unless explicitly permitted
3. **Never Self-Modify**: Cannot change own Skills/CORE without human review

## Key Files

| File | Purpose |
|------|---------|
| `engine/rlm.py` | Core RLM engine - decomposition, Code Mode retrieval, synthesis |
| `engine/llm.py` | `generate_retrieval_script()`, `decompose_task()`, strategy selection |
| `sandbox/executor.py` | `CodeExecutor` - runs LLM-generated scripts with security profiles |
| `sandbox/tools.py` | `ObsidianTools` - vault operations available to scripts |
| `verification/` | Syntax, type, test, import verification checks |
| `SPEC.md` | Full technical specification with decision log |
