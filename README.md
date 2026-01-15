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
- An Obsidian vault with relevant content
- (Optional) [Obsidian Local REST API](https://github.com/coddingtonbear/obsidian-local-rest-api) plugin

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
   │       ├── CodeExecutor ──> ObsidianTools ──> Your Vault(s)
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
| **ObsidianTools** | Vault operations (`search`, `read_note`, `list_notes`) |
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

### Task Execution

```bash
# Execute a task
shad run "Your task" --vault /path/to/vault [options]

Options:
  --vault, -v       Path to Obsidian vault (repeatable for layering)
  --strategy        Force strategy (software|research|analysis|planning)
  --max-depth, -d   Maximum recursion depth (default: 3)
  --max-nodes       Maximum DAG nodes (default: 50)
  --max-time, -t    Maximum wall time in seconds (default: 300)
  --verify          Verification level (off|basic|build|strict)
  --write-files     Write output files to disk
  --output, -o      Output directory (requires --write-files)
  --no-code-mode    Disable Code Mode (use direct search)

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

### Vault Management

```bash
# Ingest external content into vault
shad ingest github <url> --preset docs --vault ~/MyVault

# Presets: mirror (all files), docs (documentation only), deep (with code)
```

---

## Configuration

Copy `.env.example` to `.env` and configure:

```bash
# Required
REDIS_URL=redis://localhost:6379/0

# Optional: Obsidian Local REST API
OBSIDIAN_API_KEY=your_key_here
OBSIDIAN_BASE_URL=https://127.0.0.1:27124
OBSIDIAN_VAULT_PATH=/path/to/your/vault

# Optional: Budget defaults
DEFAULT_MAX_DEPTH=3
DEFAULT_MAX_NODES=50
DEFAULT_MAX_WALL_TIME=300
DEFAULT_MAX_TOKENS=100000
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
