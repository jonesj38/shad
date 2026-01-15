# Run Report: 6d45cb8b-fd94-49b9-a257-16976b700029

**Goal:** Explain the key components of the Shad architecture

**Status:** complete

## Metrics

- Total Nodes: 1
- Completed: 1
- Failed: 0
- Total Tokens: 714

## Result

# Shad Architecture Overview

Shad is a system that enables AI to utilize virtually unlimited context by treating knowledge vaults as explorable environments rather than cramming everything into a single prompt.

## Core Components

### 1. **RLM Engine** (`engine/rlm.py`)
The recursive decomposition and execution engine. It:
- Decomposes complex tasks into subtasks using strategy skeletons
- Manages DAG (directed acyclic graph) execution
- Orchestrates Code Mode retrieval

### 2. **Strategy System** (`engine/strategies/`)
Domain-specific decomposition strategies (software, research, analysis) that define:
- Required and optional stages
- Constraints (e.g., contracts-first for software)
- Selection via heuristic + LLM refinement

### 3. **Code Mode** (Key Innovation)
Instead of simple keyword search, the LLM generates custom Python retrieval scripts:
```python
results = obsidian.search("authentication React Native", limit=10)
auth_patterns = obsidian.read_note("Patterns/Authentication.md")
```
This enables intelligent, context-aware retrieval with confidence scoring and recovery tiers.

### 4. **CodeExecutor** (`sandbox/executor.py`)
Sandboxed Python execution with configurable security profiles:
- **strict** (default): Vault access only, no network
- **local**: Adds allowlisted file roots
- **extended**: Adds HTTP GET to allowlisted domains

### 5. **ObsidianTools** (`sandbox/tools.py`)
The API available to Code Mode scripts: `search()`, `read_note()`, and other vault operations.

### 6. **Verification Layer** (`verification/`)
Progressive strictness checking:
- `off` → `basic` (imports/syntax) → `build` (typecheck) → `strict` (tests)
- Includes repair loops for failures

### 7. **Supporting Infrastructure**
- **Redis** (`cache/redis_cache.py`): Caching with hash validation + TTL
- **History Manager** (`history/manager.py`): Run artifacts, DAG state, metrics
- **Shadow Index** (`~/.shad/index.sqlite`): Local indexing

## Data Flow

```
User Task → Strategy Selection → RLM Decomposition → Code Mode Retrieval
    → LLM Generation → Verification → Repair (if needed) → Synthesis
```

## Key Design Decisions

| Area | Choice | Why |
|------|--------|-----|
| Decomposition | Template skeletons + LLM refinement | Enforces invariants while staying flexible |
| Type consistency | Contracts-first node | Single source of truth for types |
| Token budget | Hierarchical reserves (25% parent) | Guarantees synthesis completes |
| Cross-subtask deps | Soft deps + context packets | Enables parallelism with context sharing |

The architecture prioritizes **correctness** (contracts-first, verification layers), **safety** (sandboxed execution, hard invariants), and **practicality** (configurable strictness, resume/replay support).
