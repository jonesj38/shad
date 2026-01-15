# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Vision

**Shad enables AI to utilize virtually unlimited context.**

The goal: Load an Obsidian vault with curated knowledge (documentation, code examples, architecture patterns), then accomplish complex tasks that would be impossible with a single context window.

**Example use case**: Build a production-quality mobile app by loading a vault with React Native docs, great app examples, and UI patterns, then running:
```bash
shad run "Build a task management app with auth and offline sync" --vault ~/MobileDevVault
```

Shad recursively decomposes the task, retrieves targeted context for each subtask from the vault, generates code, and assembles a complete codebase.

## Core Premise

> **Long-context reasoning is an inference problem, not a prompting problem.**

Instead of cramming context into a single prompt, Shad:
1. Treats the vault as an **explorable environment**
2. **Decomposes** complex tasks into subtasks recursively
3. **Retrieves** targeted context for each subtask via Code Mode
4. **Generates** outputs informed by relevant examples
5. **Assembles** results into coherent output

## Architecture

```
User
   |
   v
Shad CLI / API
   |
   +-- RLM Engine (recursive decomposition + execution)
   |       |
   |       +-- Code Mode ─────────────────────────┐
   |       |   (LLM generates Python scripts)    |
   |       |                                      v
   |       +-- CodeExecutor ──────────> ObsidianTools
   |       |   (sandboxed execution)        |
   |       |                                v
   |       |                          Obsidian Vault
   |       |                          (your knowledge)
   |       |
   |       +-- Redis (subtree caching)
   |       +-- LLM Provider (Claude Code CLI)
   |
   +-- History/ (run artifacts)
```

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

__result__ = "\n\n".join(context_parts)
```

This enables:
- Multi-step retrieval (search → read → filter)
- Query-specific logic (different strategies per task)
- Aggregation before returning (reduce context size)

### Module Responsibilities

| Module | Location | Purpose |
|--------|----------|---------|
| `engine/rlm.py` | Core | Recursive decomposition, DAG execution, Code Mode orchestration |
| `engine/llm.py` | Core | LLM abstraction, retrieval script generation |
| `sandbox/executor.py` | Execution | Sandboxed Python execution for Code Mode |
| `sandbox/tools.py` | Execution | `ObsidianTools` class with `search()`, `read_note()`, etc. |
| `mcp/client.py` | Integration | Direct MCP client for vault operations |
| `cache/redis_cache.py` | Caching | Hierarchical keys, subtree caching |
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

# Run without Code Mode (direct search only)
shad run "Your task" --vault /path/to/vault --no-code-mode

# Control recursion depth
shad run "Complex task" --vault ~/vault --max-depth 4

# Check run status
shad status <run_id>

# View execution DAG
shad trace tree <run_id>

# Resume partial run
shad resume <run_id>
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

## Key Files for the Vision

| File | Purpose |
|------|---------|
| `engine/rlm.py` | Core RLM engine - decomposition, Code Mode retrieval, synthesis |
| `engine/llm.py` | `generate_retrieval_script()` - LLM writes custom retrieval code |
| `sandbox/executor.py` | `CodeExecutor` - runs LLM-generated scripts safely |
| `sandbox/tools.py` | `ObsidianTools` - vault operations available to scripts |

## Budget System

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
2. RLM decomposes into subtasks:
   - "Design login UI layout"
   - "Implement form validation"
   - "Add OAuth integration"
         ↓
3. For each subtask, Code Mode:
   a. LLM generates retrieval script
   b. Script searches vault for relevant examples
   c. Script reads specific notes for detail
   d. Script returns distilled context
         ↓
4. LLM generates output using retrieved context
         ↓
5. Results synthesized bottom-up
```

## Extending the System

### Adding Vault Content
The vault quality determines output quality. Good vault content includes:
- Official documentation converted to markdown
- Code examples with explanations
- Architecture patterns and best practices
- Common pitfalls and solutions

### Improving Decomposition
Current decomposition is in `engine/llm.py:decompose_task()`. To improve:
- Add domain-specific decomposition strategies
- Teach the LLM to recognize task types
- Adjust branching based on task complexity

### Adding Verification
Post-generation verification goes in `verification/`. Could add:
- Syntax checking for generated code
- Type checking via `tsc --noEmit`
- Test generation and execution

## Hard Invariants (from CORE/invariants.md)

1. **Never Auto-Publish**: No irreversible side effects without explicit human approval
2. **Never Exfiltrate**: No sending data externally unless explicitly permitted
3. **Never Self-Modify**: Cannot change own Skills/CORE without human review
