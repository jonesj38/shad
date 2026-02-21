# QMD Pivot: From Obsidian API to Local Hybrid Search

> **Date**: 2026-01-27
> **Status**: Complete
> **Decision**: Replace Obsidian REST API + MCP with qmd as primary retrieval backend

---

## Summary

Shad originally used the Obsidian Local REST API for vault operations. This required Obsidian to be running and added complexity for multi-vault support. We pivoted to [qmd](https://github.com/tobi/qmd) as the primary retrieval backend, which provides better search quality with no runtime dependencies. For OpenAI embeddings support, use the fork: https://github.com/jonesj38/qmd/tree/feat/openai-embeddings.

---

## The Problem

### Obsidian REST API Limitations

1. **Runtime Dependency**: Obsidian must be running for shad to access vaults
2. **Single Vault**: API bound to one vault at a time
3. **Multi-Vault Complexity**: Would require multiple Obsidian instances or complex routing
4. **Search Quality**: Basic keyword search only, no semantic understanding
5. **Deployment**: Doesn't work on headless servers or CI/CD

### Multi-Vault Requirements

Users needed to search across multiple knowledge bases:
```bash
shad run "Build auth system" \
  --vault ~/Project \
  --vault ~/Patterns \
  --vault ~/Docs
```

With Obsidian API, this would require:
- Running multiple Obsidian instances (one per vault)
- Complex routing logic to dispatch queries
- Result merging with priority weighting
- Significant infrastructure overhead

---

## The Solution: qmd

[qmd](https://github.com/tobi/qmd) is a local CLI tool that provides:

| Feature | Obsidian API | qmd |
|---------|--------------|-----|
| Search type | Keyword only | Hybrid (BM25 + vectors + LLM reranking) |
| Runtime dependency | Obsidian running | None (standalone CLI) |
| Multi-vault | Complex routing | Native collections |
| Semantic search | No | Yes (local embeddings) |
| Headless/CI | No | Yes |
| Setup | Plugin install + API key | `bun install -g` + `qmd collection add` |

### qmd Architecture

```
shad run "task" --vault ~/V1 --vault ~/V2
         │
         ▼
    RetrievalLayer
         │
         ├─ [Primary] QmdRetriever
         │     │
         │     ├─ qmd search "..."     (BM25)
         │     ├─ qmd vsearch "..."    (vector)
         │     └─ qmd query "..."      (hybrid + rerank)
         │
         └─ [Fallback] FilesystemRetriever
               └─ Direct file scan (no qmd required)
```

### Search Modes

| Mode | Command | Use Case |
|------|---------|----------|
| `hybrid` | `qmd query` | Best quality (default) |
| `bm25` | `qmd search` | Fast keyword matching |
| `vector` | `qmd vsearch` | Pure semantic similarity |

---

## Implementation

### New Modules

```
retrieval/
├── __init__.py      # get_retriever() factory
├── layer.py         # RetrievalLayer protocol, RetrievalResult
├── qmd.py           # QmdRetriever (wraps qmd CLI)
└── filesystem.py    # FilesystemRetriever (fallback)
```

### RetrievalLayer Protocol

```python
@runtime_checkable
class RetrievalLayer(Protocol):
    async def search(
        self,
        query: str,
        mode: str = "hybrid",      # bm25 | vector | hybrid
        collections: list[str] | None = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[RetrievalResult]: ...

    async def get(self, path: str, collection: str | None = None) -> str | None: ...

    async def status(self) -> dict[str, Any]: ...

    @property
    def available(self) -> bool: ...
```

### Automatic Backend Selection

```python
def get_retriever(paths, collection_names, prefer="auto"):
    if prefer == "filesystem":
        return FilesystemRetriever(...)

    qmd = QmdRetriever()
    if qmd.available and prefer != "filesystem":
        return qmd  # Primary

    return FilesystemRetriever(...)  # Fallback
```

---

## Migration

### For Users

**Before (Obsidian API)**:
```bash
# Required Obsidian running with REST API plugin
export OBSIDIAN_API_KEY=your_key
export OBSIDIAN_BASE_URL=https://127.0.0.1:27124
shad run "task" --vault ~/Vault
```

**After (qmd)**:
```bash
# One-time setup
bun install -g https://github.com/jonesj38/qmd#feat/openai-embeddings
qmd collection add ~/Vault --name vault
QMD_OPENAI=1 qmd embed  # Generate embeddings

# Usage (no daemon required)
shad run "task" --vault ~/Vault
```

**Fallback (no qmd)**:
```bash
# Works without any setup, uses filesystem search
shad run "task" --vault ~/Vault
```

### Breaking Changes

1. **Removed settings**: `OBSIDIAN_API_KEY`, `OBSIDIAN_BASE_URL`, `OBSIDIAN_VERIFY_SSL`
2. **Removed modules**: `mcp/client.py`, `mcp/models.py`
3. **New CLI flag**: `--retriever auto|qmd|filesystem`
4. **New search modes**: `--mode hybrid|bm25|vector`

---

## Deleted Code

| File | Reason |
|------|--------|
| `mcp/client.py` | Replaced by retrieval layer |
| `mcp/models.py` | Replaced by RetrievalResult |
| `mcp/__init__.py` | Package removed |
| `tests/test_mcp_client.py` | Tests for deleted code |

---

## Benefits Realized

1. **No Runtime Dependency**: Shad works without Obsidian running
2. **Better Search**: Hybrid BM25 + vector search with LLM reranking
3. **Native Multi-Vault**: `--vault` flag accepts multiple paths
4. **Simpler Config**: No API keys or URLs to configure
5. **CI/CD Ready**: Works on headless servers
6. **Graceful Fallback**: Filesystem search when qmd not installed

---

## References

- [tobi/qmd](https://github.com/tobi/qmd) - Primary retrieval backend
- [sqlite-vec](https://github.com/asg017/sqlite-vec) - Vector search in SQLite
- [Hybrid search with SQLite](https://alexgarcia.xyz/blog/2024/sqlite-vec-hybrid-search/index.html)
