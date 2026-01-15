# Shad (Shannon's Daemon)

**Shad enables AI to utilize virtually unlimited context.**

Load an Obsidian vault with curated knowledge — documentation, code examples, architecture patterns, best practices — then accomplish complex tasks that would be impossible with a single context window.

```bash
# Load a vault with mobile dev knowledge, then build an app
shad run "Build a task management app with auth, offline sync, and push notifications" \
  --vault ~/MobileDevVault \
  --max-depth 4
```

Shad recursively decomposes the task, retrieves targeted context for each subtask, generates output informed by your vault's examples, and assembles coherent results.

---

## The Problem

AI systems break down when:
- Context grows beyond the model's window
- Tasks require reasoning over many documents
- Output quality depends on following specific patterns
- You need consistent, reproducible results

Current solutions (RAG, long-context models) help but don't scale. You can't fit a 100MB documentation vault into any context window.

## The Solution

> **Long-context reasoning is an inference problem, not a prompting problem.**

Shad treats your vault as an **explorable environment**, not a fixed input:

1. **Decompose**: Break complex tasks into subtasks recursively
2. **Retrieve**: For each subtask, generate custom retrieval code that searches your vault
3. **Generate**: Produce output informed by relevant examples from your vault
4. **Assemble**: Synthesize subtask results into coherent output

This allows Shad to effectively utilize **gigabytes** of context — not by loading it all at once, but by intelligently retrieving what's needed for each subtask.

---

## Quick Start

### Prerequisites
- Python 3.11+
- An Obsidian vault with relevant content

### Installation

```bash
# Clone
git clone https://github.com/yourusername/shad.git
cd shad/services/shad-api

# Install
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Basic Usage

```bash
# Run a task with vault context
shad run "Summarize the key concepts in my notes" --vault ~/MyVault

# More complex task with deeper recursion
shad run "Compare all authentication approaches documented in my vault" \
  --vault ~/DevDocs \
  --max-depth 3

# Check results
shad status <run_id>
shad trace tree <run_id>
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

__result__ = f"""
## OAuth Patterns
{patterns[:3000]}

## Relevant Examples
{"---".join(relevant)}
"""
```

This enables:
- **Multi-step retrieval**: Search → read specific notes → filter → aggregate
- **Query-specific logic**: Different retrieval strategies per subtask
- **Context efficiency**: Return only what's needed, not entire documents

### Recursive Decomposition

Complex tasks are broken into manageable subtasks:

```
"Build a mobile app with auth"
         ↓
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
└── "Add offline sync"
```

Each leaf node retrieves its own context from the vault, ensuring targeted, relevant information.

---

## Example Use Cases

### 1. Build Software with Your Patterns

```bash
# Vault contains: Your team's code standards, architecture docs, example projects
shad run "Build a REST API for user management following our patterns" \
  --vault ~/TeamDocs \
  --max-depth 4
```

### 2. Research with Your Knowledge Base

```bash
# Vault contains: Research papers, notes, bookmarks
shad run "What are the key arguments for and against microservices in my notes?" \
  --vault ~/Research
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
  --max-depth 3
```

---

## Architecture

```
User
   |
   v
Shad CLI / API
   |
   +-- RLM Engine
   |       |
   |       +-- Decomposition (break task into subtasks)
   |       |
   |       +-- Code Mode (LLM generates retrieval scripts)
   |       |       |
   |       |       v
   |       +-- CodeExecutor ──> ObsidianTools ──> Your Vault
   |       |
   |       +-- Generation (produce output with context)
   |       |
   |       +-- Synthesis (combine subtask results)
   |
   +-- Redis (cache subtrees)
   +-- History (run artifacts)
```

### Key Components

| Component | Purpose |
|-----------|---------|
| **RLM Engine** | Recursive decomposition and execution |
| **Code Mode** | LLM-generated retrieval scripts |
| **CodeExecutor** | Sandboxed Python execution |
| **ObsidianTools** | Vault operations (`search`, `read_note`, `list_notes`) |
| **Redis Cache** | Cache subtask results for reuse |

---

## CLI Reference

```bash
# Execute a task
shad run "Your task" --vault /path/to/vault [options]

Options:
  --vault, -v       Path to Obsidian vault
  --max-depth, -d   Maximum recursion depth (default: 3)
  --max-nodes       Maximum DAG nodes (default: 50)
  --max-time, -t    Maximum wall time in seconds (default: 300)
  --no-code-mode    Disable Code Mode (use direct search)
  --output, -o      Write result to file

# Check status
shad status <run_id>

# View execution tree
shad trace tree <run_id>

# Inspect specific node
shad trace node <run_id> <node_id>

# Resume partial run
shad resume <run_id>
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

## Roadmap

See [PLAN.md](PLAN.md) for the full roadmap. Current focus:

- [x] **Phase 1**: Foundation (CLI, API, RLM engine)
- [x] **Phase 2**: Obsidian integration (Code Mode, per-subtask retrieval)
- [ ] **Phase 3**: Task-aware decomposition (software architecture, research, etc.)
- [ ] **Phase 4**: File output mode (generate actual codebases)
- [ ] **Phase 5**: Verification layer (syntax check, type check, tests)
- [ ] **Phase 6**: Iterative refinement (error feedback, HITL checkpoints)

---

## Philosophy

> Solve a problem once.
> Encode it as knowledge.
> Never solve it again.

Shad compounds your knowledge. Every document you add to your vault makes Shad more capable. The vault is the "how" — patterns, examples, documentation. Shad is the "engine" — decomposition, retrieval, generation, assembly.

Together: complex tasks that learn from your accumulated knowledge.

---

## License

MIT
