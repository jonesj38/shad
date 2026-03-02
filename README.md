# Shad (Shannon's Daemon)

**Shad enables AI to utilize virtually unlimited context.**

Load any directory of markdown, code, or docs — then accomplish complex tasks that would be impossible with a single context window. Shad recursively decomposes tasks, retrieves targeted context for each subtask, generates outputs with type consistency, verifies them, and assembles coherent results.

```bash
# Build a full app using your team's patterns and docs
shad run "Build a task management app with auth, offline sync, and push notifications" \
  --vault ~/TeamDocs \
  --strategy software \
  --write-files --output ./TaskApp
```

---

## The Problem

AI systems break down when:
- Context grows beyond the model's window
- Tasks require reasoning over many documents
- Output quality depends on following specific patterns
- Generated code needs consistent types across files
- You need reproducible, verifiable results

Current solutions (RAG, long-context models) help but don't scale. You can't fit a 100MB documentation vault into any context window.

## The Solution

> **Long-context reasoning is an inference problem, not a prompting problem.**

Shad treats your vault as an **explorable environment**, not a fixed input:

1. **Decompose** — Break complex tasks into subtasks using domain-specific strategy skeletons
2. **Retrieve** — For each subtask, generate custom retrieval code that searches your vault(s)
3. **Generate** — Produce output with contracts-first type consistency
4. **Verify** — Check syntax, types, and tests with configurable strictness
5. **Assemble** — Synthesize subtask results into coherent output (file manifests for code)

This allows Shad to effectively utilize **gigabytes** of context — not by loading it all at once, but by intelligently retrieving what's needed for each subtask.

---

## Quick Start

### Prerequisites

- Python 3.11+
- At least one of:
  - [Claude CLI](https://claude.ai/code) — uses your Claude subscription (default)
  - [Gemini CLI](https://geminicli.com) — uses your Google subscription
  - [Ollama](https://ollama.com) — free, local open-source models
- A vault (any directory of markdown files, code, or docs)
- (Optional) Docker for Redis (enables cross-run caching)
- (Optional) [qmd](https://github.com/jonesj38/qmd) for hybrid semantic search

### Installation

```bash
# One-liner install
curl -fsSL https://raw.githubusercontent.com/jonesj38/shad/main/install.sh | bash

# Or clone and run manually
git clone https://github.com/jonesj38/shad.git
cd shad
./install.sh
```

The installer will:
- Clone the repo to `~/.shad`
- Create a Python virtual environment
- Install dependencies
- Install qmd for semantic search (if bun/npm available)
- Add `shad` to your PATH

After installation, restart your terminal or run:
```bash
source ~/.zshrc  # or ~/.bashrc
```

### Start the Server

```bash
shad server start     # Start Redis + API server
shad server status    # Check status
shad server logs -f   # Follow logs
```

### Basic Usage

```bash
# Run a task with vault context
shad run "Summarize the key concepts in my notes" --vault ~/MyVault

# Use multiple vaults (searched in priority order)
shad run "Build auth system" --vault ~/Project --vault ~/Patterns --vault ~/Docs

# Generate code with verification
shad run "Build a REST API for user management" \
  --vault ~/TeamDocs \
  --strategy software \
  --verify strict \
  --write-files --output ./api

# Quick context retrieval (no DAG, faster than run)
shad context "BSV authentication decisions" -v ~/Notes

# Search your vault directly
shad search "oauth refresh token" --mode hybrid

# Check environment health
shad doctor
shad doctor --fix   # Install qmd + register vault + embed
```

### Stop the Server

```bash
shad server stop
```

---

## LLM Providers

Shad supports three model backends. No API keys need to be configured in Shad — each CLI handles its own authentication.

### Claude CLI (default)

```bash
# Use model tier aliases
shad run "Complex task" -O opus -W sonnet -L haiku

# Use haiku for everything (faster, cheaper)
shad run "Simple task" -O haiku -W haiku -L haiku
```

### Gemini CLI

```bash
# Use Gemini for everything
shad run "Task" --gemini

# Specify Gemini models per tier
shad run "Task" --gemini -O gemini-3-pro-preview -W gemini-3-flash-preview
```

Requires [Gemini CLI](https://geminicli.com) installed and authenticated (`gemini auth login`).

### Ollama (local models)

```bash
# Use local models (free, runs on your hardware)
shad run "Task" -O qwen3-coder -W llama3 -L llama3

# Mix Ollama with Claude
shad run "Task" -O opus -W llama3 -L qwen3:latest
```

Requires [Ollama](https://ollama.com) installed with models pulled (`ollama pull llama3`). Any model name not matching Claude or Gemini patterns routes to Ollama automatically.

### Model Tiers

| Tier | Flag | Purpose | Claude Default | Gemini Default |
|------|------|---------|----------------|----------------|
| Orchestrator | `-O` | Planning and synthesis | `sonnet` | `gemini-3-pro-preview` |
| Worker | `-W` | Mid-depth execution | `sonnet` | `gemini-3-pro-preview` |
| Leaf | `-L` | Fast parallel execution | `haiku` | `gemini-3-flash-preview` |

---

## How It Works

### Code Mode: Intelligent Retrieval

Instead of simple keyword search, Shad uses **Code Mode** — the LLM writes Python scripts to retrieve exactly what it needs:

```python
# For task: "How should I implement OAuth?"
# LLM generates:

results = obsidian.search("OAuth implementation", limit=10)
patterns = obsidian.read_note("Patterns/Authentication/OAuth.md")

relevant = []
for r in results:
    if "refresh token" in r["content"].lower():
        relevant.append(r["content"][:2000])

__result__ = {
    "context": f"## OAuth Patterns\n{patterns[:3000]}\n\n## Examples\n{'---'.join(relevant)}",
    "citations": [...],
    "confidence": 0.72
}
```

This enables:
- **Multi-step retrieval** — search → read specific files → filter → aggregate
- **Query-specific logic** — different retrieval strategies per subtask
- **Context efficiency** — return only what's needed, not entire documents
- **Confidence scoring** — recovery when retrieval quality is low

Use `--no-code-mode` to disable Code Mode and use direct search instead.

### Strategy-Based Decomposition

Complex tasks are broken into manageable subtasks using **strategy skeletons**:

```
"Build a mobile app with auth" (software strategy)
         ↓
├── Types & Contracts (hard dependency for all below)
├── "Set up project structure"
├── "Implement navigation"
├── "Build authentication flow"
│   ├── "Create login screen"
│   ├── "Implement OAuth integration"
│   └── "Add session management"
├── "Create main features"
│   ├── "Task list view"
│   ├── "Task detail screen"
│   └── "Create/edit task form"
├── "Add offline sync"
└── Verification (syntax, types, tests)
```

Strategies: `software`, `research`, `analysis`, `planning`. Auto-selected by default, or override with `--strategy`.

### File Output with Type Consistency

For code generation, Shad uses **two-pass import resolution**:
1. Generate an export index (which symbols live where)
2. Generate implementations using the export index as ground truth
3. Validate all imports resolve correctly

Output is a structured **file manifest** — writing to disk requires explicit `--write-files`.

---

## Semantic Search (qmd)

For best retrieval quality, install [qmd](https://github.com/tobi/qmd) for hybrid BM25 + vector search with LLM reranking.

```bash
# Install (recommended fork with OpenAI embeddings)
bun install -g https://github.com/jonesj38/qmd#feat/openai-embeddings

# Register your vault as a collection
qmd collection add ~/MyVault --name myvault

# Generate embeddings
QMD_OPENAI=1 qmd embed
```

| Search Mode | Command | Use Case |
|-------------|---------|----------|
| `hybrid` | `qmd query` | Best quality (default) — BM25 + vector + RRF + reranking |
| `bm25` | `qmd search` | Fast keyword matching |
| `vector` | `qmd vsearch` | Pure semantic similarity |

Without qmd, Shad falls back to filesystem search (basic keyword matching). Use `shad doctor --fix` to install qmd and set up your vault automatically.

---

## CLI Reference

### Core Commands

```bash
# Execute a task with vault context
shad run "Your task" [options]

# Quick context retrieval (faster than run, richer than search)
shad context "query" -v ~/vault

# Search your vault
shad search "query" [--mode hybrid|bm25|vector]

# Check run status
shad status <run_id>

# View execution tree
shad trace tree <run_id>

# Inspect specific node
shad trace node <run_id> <node_id>

# Resume partial run
shad resume <run_id> [--profile deep] [--auto-profile] [--replay stale]

# Export files from completed run
shad export <run_id> --output ./out

# List available models
shad models [--refresh] [--ollama]
```

### Run Options

```
--vault, -v            Vault path(s) for context (repeatable)
--retriever, -r        Backend: auto|qmd|filesystem (default: auto)
--strategy, -s         Force strategy: software|research|analysis|planning
--profile              Budget preset: fast|balanced|deep
--auto-profile         Auto-select profile based on machine specs
--dry-run              Show budgets/models and exit (no execution)
--max-depth, -d        Maximum recursion depth (default: 3)
--max-nodes            Maximum DAG nodes (default: 50)
--max-time, -t         Maximum wall time in seconds (default: 1200)
--verify               Verification level: off|basic|build|strict
--write-files          Write output files to disk
--output-dir           Output directory (requires --write-files)
--no-code-mode         Disable Code Mode (use direct search)
--qmd-hybrid/--no-qmd-hybrid  Toggle hybrid search with reranking (default: on)
--quiet, -q            Suppress verbose output
-O                     Orchestrator model (opus, sonnet, haiku, or any model ID)
-W                     Worker model
-L                     Leaf model
--gemini               Use Gemini CLI instead of Claude CLI
```

### Server Management

```bash
shad server start      # Start Redis + API server
shad server stop       # Stop all services
shad server status     # Check service status
shad server logs [-f]  # View/follow logs
```

### Environment & Setup

```bash
shad doctor            # Check environment health (Python, qmd, Redis, vault)
shad doctor --fix      # Auto-fix: install qmd, register vault, generate embeddings
shad init              # Initialize project permissions for Claude Code
shad vault             # Check retriever status
```

### Sources Scheduler

Automatically sync content from external sources on a schedule.

```bash
# Add sources
shad sources add github https://github.com/org/repo --schedule weekly --vault ~/Vault
shad sources add url https://docs.example.com/api --schedule daily --vault ~/Vault
shad sources add feed https://blog.example.com/rss --schedule hourly --vault ~/Vault
shad sources add folder ~/LocalDocs --schedule daily --vault ~/Vault

# Manage
shad sources list              # List all sources
shad sources status            # Detailed status (schedule, last/next sync)
shad sources sync              # Sync due sources
shad sources sync --force      # Force sync all
shad sources remove <id>       # Remove a source
```

Schedules: `manual`, `hourly`, `daily`, `weekly`, `monthly`

### Vault Ingestion

```bash
# Ingest a GitHub repo into your vault
shad ingest github <url> --vault ~/Vault --preset docs

# Presets: mirror (all files), docs (documentation only), deep (with code)
```

---

## Performance Profiles

### Quick Patterns

```bash
# Cold-start (good default)
shad run "task" --vault ~/V -O sonnet -W sonnet -L haiku

# Fast + cheap
shad run "task" --vault ~/V --profile fast -O haiku -W haiku -L haiku

# Auto profile (adapts to your machine)
shad run "task" --vault ~/V --auto-profile

# Deep reasoning (large tasks)
shad run "task" --vault ~/V --profile deep -O opus -W sonnet -L haiku

# Preview before running
shad run "task" --vault ~/V --auto-profile --dry-run
```

### Budget Defaults by Machine

**Low-end laptop / small VM:**
```bash
DEFAULT_MAX_DEPTH=2
DEFAULT_MAX_NODES=30
DEFAULT_MAX_WALL_TIME=600
DEFAULT_MAX_TOKENS=800000
```

**Mid-range dev machine (recommended):**
```bash
DEFAULT_MAX_DEPTH=3
DEFAULT_MAX_NODES=50
DEFAULT_MAX_WALL_TIME=1200
DEFAULT_MAX_TOKENS=2000000
```

**High-end workstation:**
```bash
DEFAULT_MAX_DEPTH=4
DEFAULT_MAX_NODES=80
DEFAULT_MAX_WALL_TIME=1800
DEFAULT_MAX_TOKENS=3000000
```

---

## Architecture

```
User
   │
   ▼
Shad CLI / API
   │
   ├── RLM Engine
   │       │
   │       ├── Strategy Selection (heuristic + LLM)
   │       │
   │       ├── Decomposition (skeleton + LLM refinement)
   │       │
   │       ├── Code Mode (LLM generates retrieval scripts)
   │       │       │
   │       │       ▼
   │       ├── CodeExecutor ──> RetrievalLayer ──> Your Vault(s)
   │       │                         │
   │       │                    ┌────┴────┐
   │       │                    │         │
   │       │                   qmd    Filesystem
   │       │               (semantic)  (fallback)
   │       │
   │       ├── Verification (syntax, types, tests)
   │       │
   │       └── Synthesis (combine subtask results)
   │
   ├── Redis (cache + budget ledger)
   └── History (run artifacts)
```

### Key Components

| Component | Purpose |
|-----------|---------|
| **RLM Engine** | Recursive decomposition and execution |
| **Strategy Skeletons** | Domain-specific decomposition templates (`software`, `research`, `analysis`, `planning`) |
| **Code Mode** | LLM-generated retrieval scripts |
| **CodeExecutor** | Sandboxed Python execution (configurable profiles) |
| **RetrievalLayer** | Vault search abstraction (qmd or filesystem fallback) |
| **qmd** | Hybrid BM25 + vector search with LLM reranking |
| **Verification Layer** | Syntax, type, import, test checking (progressive strictness) |
| **Redis Cache** | Cache subtask results with hash validation |
| **LLM Provider** | Multi-backend: Claude CLI, Gemini CLI, Ollama |

---

## Configuration

Shad works with minimal configuration. Set optional environment variables in `~/.shad/.env` or your shell profile.

```bash
# Default vault (so you don't need --vault every time)
OBSIDIAN_VAULT_PATH=/path/to/your/vault

# Redis for cross-run caching (defaults to localhost:6379)
REDIS_URL=redis://localhost:6379/0

# Budget defaults
DEFAULT_MAX_DEPTH=3
DEFAULT_MAX_NODES=50
DEFAULT_MAX_WALL_TIME=1200
DEFAULT_MAX_TOKENS=2000000
```

### Data Directories

| Directory | Purpose |
|-----------|---------|
| `~/.shad/history/` | Run artifacts and history |
| `~/.shad/skills/` | Skill definitions |
| `~/.shad/CORE/` | Core system files |
| `~/.shad/repo/` | Installed Shad source |
| `~/.shad/venv/` | Python virtual environment |

---

## Vault Strategy

### One vs Many Vaults

| | One Vault | Many Vaults |
|---|---|---|
| **Pros** | Single source of truth, cross-topic connections, simpler management | Faster indexing, focused retrieval, easier sharing/permissions |
| **Cons** | Slower as it grows, noise in retrieval, harder to share subsets | Context fragmentation, can't find cross-vault connections, more overhead |

**Use one vault** for personal/work knowledge — memory, tasks, notes, projects all interconnected.

**Use separate vaults** for codebases, client deliverables needing isolation, or read-only reference material.

Multi-vault queries search in priority order:
```bash
shad run "Build auth system" --vault ~/Project --vault ~/Patterns --vault ~/Docs
```

### Vault Preparation Tips

- Use consistent frontmatter for better filtering
- Include code examples with context, not just snippets
- Link related notes for better discovery
- Keep notes focused (one concept per note)
- Authoritative sources and worked examples improve output quality

---

## Project Status

All core phases complete:

- [x] Foundation — CLI, API, RLM engine, Redis caching
- [x] qmd migration — hybrid search, multi-vault, no Obsidian dependency
- [x] Task-aware decomposition — strategy skeletons, soft dependencies
- [x] File output mode — two-pass imports, contracts-first
- [x] Verification layer — progressive strictness, repair loops
- [x] Iterative refinement — HITL checkpoints, delta resume
- [x] Vault curation — ingestion, gap detection
- [x] Sources scheduler — automated sync from GitHub, URLs, feeds, folders
- [x] Multi-provider — Claude CLI, Gemini CLI, Ollama support
- [x] Context command — fast retrieval + synthesis without DAG overhead
- [x] Doctor command — environment health checks with auto-fix
- [x] Performance profiles — fast/balanced/deep presets, auto-profile

See [SPEC.md](SPEC.md) for the technical specification, [QMD_PIVOT.md](QMD_PIVOT.md) for the qmd migration rationale.

---

## Philosophy

> Solve a problem once. Encode it as knowledge. Never solve it again.

Shad compounds your knowledge. Every document you add makes it more capable. The vault is the *how* — patterns, examples, documentation. Shad is the *engine* — decomposition, retrieval, generation, verification, assembly.

Together: complex tasks that learn from your accumulated knowledge.

---

## Contributing

Contributions welcome. See [SPEC.md](SPEC.md) for architecture details before submitting PRs.

## License

MIT
