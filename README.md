# Shad (Shannon's Daemon)

**Shad enables AI to utilize virtually unlimited context.**

Load an Obsidian vault with curated knowledge — documentation, code examples, architecture patterns, best practices — then accomplish complex tasks that would be impossible with a single context window.

```bash
# Load a vault with mobile dev knowledge, then build an app
shad run "Build a task management app with auth, offline sync, and push notifications" \
  --vault ~/MobileDevVault \
  --strategy software \
  --write-files --output ./TaskApp
```

Shad recursively decomposes the task, retrieves targeted context for each subtask, generates code with type consistency, verifies outputs, and assembles coherent results.

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

1. **Decompose**: Break complex tasks into subtasks using domain-specific strategy skeletons
2. **Retrieve**: For each subtask, generate custom retrieval code that searches your vault(s)
3. **Generate**: Produce output with contracts-first type consistency
4. **Verify**: Check syntax, types, and tests with configurable strictness
5. **Assemble**: Synthesize subtask results into coherent output (file manifests for code)

This allows Shad to effectively utilize **gigabytes** of context — not by loading it all at once, but by intelligently retrieving what's needed for each subtask.

---

## Quick Start

### Prerequisites

- Python 3.11+
- Docker (for Redis)
- [Claude CLI](https://claude.ai/code) installed and authenticated
- An Obsidian vault (or any directory of markdown files)
- (Optional) [qmd](https://github.com/tobi/qmd) for hybrid semantic search

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
# Start Redis and the Shad API
shad server start

# Check status
shad server status

# View logs
shad server logs -f
```

### Basic Usage

```bash
# Run a task with vault context
shad run "Summarize the key concepts in my notes" --vault ~/MyVault

# More complex task with deeper recursion
shad run "Compare all authentication approaches documented in my vault" \
  --vault ~/DevDocs \
  --max-depth 3

# Generate code with verification
shad run "Build a REST API for user management" \
  --vault ~/TeamDocs \
  --strategy software \
  --verify strict \
  --write-files --output ./api

# Use multiple vaults (priority order)
shad run "Build auth system" \
  --vault ~/Project \
  --vault ~/Patterns \
  --vault ~/Docs

# Check results
shad status <run_id>
shad trace tree <run_id>

# Resume a partial run
shad resume <run_id>
```

### Stop the Server

```bash
shad server stop
```

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
- **Multi-step retrieval**: Search → read specific notes → filter → aggregate
- **Query-specific logic**: Different retrieval strategies per subtask
- **Context efficiency**: Return only what's needed, not entire documents
- **Confidence scoring**: Recovery when retrieval quality is low

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

Each leaf node retrieves its own context from the vault, ensuring targeted, relevant information.

### File Output with Type Consistency

For code generation, Shad uses **two-pass import resolution**:
1. Generate an export index (which symbols live where)
2. Generate implementations using the export index as ground truth
3. Validate all imports resolve correctly

Output is a structured **file manifest** — writing to disk requires explicit `--write-files`.

---

## Example Use Cases

### 1. Build Software with Your Patterns

```bash
# Vault contains: Your team's code standards, architecture docs, example projects
shad run "Build a REST API for user management following our patterns" \
  --vault ~/TeamDocs \
  --strategy software \
  --verify strict \
  --write-files --output ./api
```

### 2. Research with Your Knowledge Base

```bash
# Vault contains: Research papers, notes, bookmarks
shad run "What are the key arguments for and against microservices in my notes?" \
  --vault ~/Research \
  --strategy research
```

### 3. Generate Documentation

```bash
# Vault contains: Codebase documentation, API specs
shad run "Write a getting started guide based on our API documentation" \
  --vault ~/ProjectDocs
```

### 4. Analysis with Domain Knowledge

```bash
# Vault contains: Industry reports, competitor analysis, market data
shad run "Analyze market trends based on my collected research" \
  --vault ~/MarketResearch \
  --strategy analysis
```

### 5. Keep Your Vault Current

```bash
# Set up automated ingestion from multiple sources
shad sources add github https://github.com/facebook/react --schedule weekly --vault ~/DevDocs
shad sources add feed https://blog.rust-lang.org/feed.xml --schedule daily --vault ~/DevDocs
shad sources add url https://docs.python.org/3/whatsnew/3.12.html --schedule monthly --vault ~/DevDocs

# Check what's due for sync
shad sources status

# Sync all due sources
shad sources sync
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
| **Strategy Skeletons** | Domain-specific decomposition templates |
| **Code Mode** | LLM-generated retrieval scripts |
| **CodeExecutor** | Sandboxed Python execution (configurable profiles) |
| **RetrievalLayer** | Vault search abstraction (qmd or filesystem) |
| **qmd** | Hybrid BM25 + vector search with LLM reranking |
| **Verification Layer** | Syntax, type, import, test checking |
| **Redis Cache** | Cache subtask results with hash validation |

---

## CLI Reference

### Server Management

```bash
shad server start     # Start Redis + API server
shad server stop      # Stop all services
shad server status    # Check service status
shad server logs      # View API logs
shad server logs -f   # Follow logs
```

### Project Setup

```bash
# Initialize project permissions for Claude Code integration
shad init

# Verify permissions are configured correctly
shad check-permissions
```

### Task Execution

```bash
# Execute a task
shad run "Your task" [options]

Options:
  --vault, -v       Path to vault (repeatable; falls back to OBSIDIAN_VAULT_PATH)
  --retriever, -r   Retrieval backend: auto|qmd|filesystem (default: auto)
  --strategy        Force strategy (software|research|analysis|planning)
  --max-depth, -d   Maximum recursion depth (default: 3)
  --max-nodes       Maximum DAG nodes (default: 50)
  --max-time, -t    Maximum wall time in seconds (default: 300)
  --verify          Verification level (off|basic|build|strict)
  --write-files     Write output files to disk
  --output, -o      Output directory (requires --write-files)
  --no-code-mode    Disable Code Mode (use direct search)
  --quiet, -q       Suppress verbose output (logging enabled by default)
  -O, --orchestrator-model   Model for planning/synthesis (opus, sonnet, haiku)
  -W, --worker-model         Model for mid-depth execution
  -L, --leaf-model           Model for fast parallel execution

# Check status
shad status <run_id>

# View execution tree
shad trace tree <run_id>

# Inspect specific node
shad trace node <run_id> <node_id>

# Resume partial run (with delta verification)
shad resume <run_id>
shad resume <run_id> --replay stale

# Export files from completed run
shad export <run_id> --output ./out
```

### Model Selection

Shad uses different models for different execution tiers. You can override the defaults with Claude models or Ollama open-source models:

```bash
# List available models
shad models
shad models --refresh    # Force refresh from API
shad models --ollama     # Include locally installed Ollama models

# Use specific Claude models for each tier
shad run "Complex task" -O opus -W sonnet -L haiku

# Use haiku for everything (cost-effective)
shad run "Simple task" -O haiku -W haiku -L haiku

# Full API names also work
shad run "Task" --orchestrator-model claude-opus-4-20250514

# Mix Claude and Ollama models
shad run "Task" -O opus -W llama3 -L qwen3:latest

# Use all Ollama models (free, runs locally)
shad run "Task" -O qwen3-coder -W llama3 -L llama3
```

Model tiers:
- **Orchestrator (-O)**: Planning and synthesis (default: sonnet)
- **Worker (-W)**: Mid-depth execution (default: sonnet)
- **Leaf (-L)**: Fast parallel execution (default: haiku)

**Ollama Integration**: Any model name that isn't a Claude model (opus, sonnet, haiku, or claude-*) is treated as an Ollama model. Requires [Ollama](https://ollama.com) installed locally with models pulled (e.g., `ollama pull llama3`).

### Vault Management

```bash
# Check retriever status
shad vault

# Search your vault
shad search "query"                    # Hybrid search (default)
shad search "query" --mode bm25        # Fast keyword search
shad search "query" --mode vector      # Semantic search

# Ingest external content into vault
shad ingest github <url> --preset docs --vault ~/MyVault

# Presets: mirror (all files), docs (documentation only), deep (with code)
```

### Semantic Search Setup (Optional)

For hybrid BM25 + vector search with LLM reranking, install [qmd](https://github.com/tobi/qmd):

```bash
# Install qmd (installer does this automatically if bun/npm available)
bun install -g https://github.com/tobi/qmd

# Register your vault as a collection
qmd collection add ~/MyVault --name myvault

# Generate embeddings (required for semantic search)
qmd embed
```

Without qmd, shad falls back to filesystem search (basic keyword matching).

### Sources Scheduler

Automatically sync content from GitHub repos, URLs, RSS feeds, and local folders on a schedule.

```bash
# Add sources
shad sources add github https://github.com/org/repo --schedule weekly --vault ~/MyVault
shad sources add url https://docs.example.com/api --schedule daily --vault ~/MyVault
shad sources add feed https://blog.example.com/rss --schedule hourly --vault ~/MyVault
shad sources add folder ~/LocalDocs --schedule daily --vault ~/MyVault

# List all sources
shad sources list

# View detailed status (schedule, last sync, next sync)
shad sources status

# Manually sync due sources
shad sources sync

# Force sync all sources
shad sources sync --force

# Remove a source
shad sources remove <source_id>
```

Schedules: `manual`, `hourly`, `daily`, `weekly`, `monthly`

---

## Configuration

Shad works with minimal configuration. Set environment variables directly or in a `.env` file:

```bash
# API Keys (set in shell or .env)
ANTHROPIC_API_KEY=your_key_here     # Required for Claude models
OPENAI_API_KEY=your_key_here        # Optional fallback

# Optional: Default vault path (so you don't need --vault every time)
OBSIDIAN_VAULT_PATH=/path/to/your/vault

# Optional: Redis (defaults to localhost:6379)
REDIS_URL=redis://localhost:6379/0

# Optional: Budget defaults
DEFAULT_MAX_DEPTH=3
DEFAULT_MAX_NODES=50
DEFAULT_MAX_WALL_TIME=300
DEFAULT_MAX_TOKENS=100000
```

### Default Vault

Set `OBSIDIAN_VAULT_PATH` to skip `--vault` on every command:

```bash
# In your shell profile (~/.zshrc or ~/.bashrc)
export OBSIDIAN_VAULT_PATH=/home/user/MyVault

# Now you can run without --vault
shad run "Summarize my notes"

# CLI --vault flag overrides the default
shad run "Query other vault" --vault ~/OtherVault
```

---

## Vault Preparation

The quality of Shad's output depends on your vault's content. Good vaults include:

### For Software Development
- Framework documentation (converted to markdown)
- Code examples with explanations
- Architecture decision records
- Common patterns and anti-patterns
- Your team's coding standards

### For Research
- Paper summaries and notes
- Key quotes and citations
- Concept explanations
- Related work connections

### For Any Domain
- Authoritative sources
- Worked examples
- Best practices
- Common pitfalls and solutions

### Tips
- Use consistent frontmatter for better filtering
- Include code examples with context, not just snippets
- Link related notes for better discovery
- Keep notes focused (one concept per note)

---

## Project Status

All phases complete:

- [x] **Phase 1**: Foundation (CLI, API, RLM engine, Redis caching)
- [x] **Phase 2**: Obsidian integration (Code Mode, per-subtask retrieval)
- [x] **Phase 3**: Task-aware decomposition (strategy skeletons, soft dependencies)
- [x] **Phase 4**: File output mode (two-pass imports, contracts-first)
- [x] **Phase 5**: Verification layer (progressive strictness, repair loops)
- [x] **Phase 6**: Iterative refinement (HITL checkpoints, delta resume)
- [x] **Phase 7**: Vault curation tools (ingestion, gap detection)
- [x] **Phase 8**: Sources scheduler (automated sync from GitHub, URLs, feeds, folders)

See [SPEC.md](SPEC.md) for detailed technical specification and [PLAN.md](PLAN.md) for implementation details.

---

## Philosophy

> Solve a problem once.
> Encode it as knowledge.
> Never solve it again.

Shad compounds your knowledge. Every document you add to your vault makes Shad more capable. The vault is the "how" — patterns, examples, documentation. Shad is the "engine" — decomposition, retrieval, generation, verification, assembly.

Together: complex tasks that learn from your accumulated knowledge.

---

## Contributing

Contributions welcome! Please read the [SPEC.md](SPEC.md) to understand the architecture before submitting PRs.

## License

MIT
