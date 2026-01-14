# Run Report: afb3b5d6-f710-4dbe-b301-b5859a7a3b5f

**Goal:** notebooks

**Status:** complete

## Metrics

- Total Nodes: 1
- Completed: 1
- Failed: 0
- Total Tokens: 428

## Result

## Notebooks in Shad

The **notebooks** system in Shad is based on **OpenNotebookLM**, a graph-based knowledge substrate used for context retrieval during reasoning tasks.

### Architecture

There are two implementations:

| Component | Location | Purpose |
|-----------|----------|---------|
| `OpenNotebookClient` | `src/shad/notebook/client.py` | REST client for external Open Notebook service (lfnovo/open-notebook) at `http://localhost:5055` |
| `NotebookStore` | `src/shad/notebook/store.py` | Local file-based implementation using JSON storage |

### Data Model (`src/shad/models/notebook.py`)

The knowledge graph consists of:

**Node Types:**
- `Notebook` - A contextual container/lens over the graph (has description, policies)
- `Source` - External artifacts like PDFs, web pages, transcripts, code
- `Note` - Derived artifacts authored by humans or AI (has confidence, provisional flag)

**Edge Types:**
- `DERIVED_FROM`, `SUMMARIZES`, `CONTRADICTS`, `SUPPORTS`, `REFERENCES`, `PART_OF`, `INCLUDED_IN`

### Key Features

1. **Context Retrieval** - The `retrieve()` method is used by `RLMEngine` to fetch relevant knowledge during recursive decomposition

2. **Text Search** - Simple word-overlap relevance scoring (can be upgraded to vector search)

3. **Source Ingestion** - Import text files, directories, or add sources from URLs

4. **Storage** - JSON files in `settings.notebooks_path`:
   - `notebooks.json`, `sources.json`, `notes.json`, `edges.json`

### Usage in RLM Engine

Per the CLAUDE.md, each subtask in the recursive reasoning flow:
1. Checks cache
2. **Retrieves context from OpenNotebookLM**
3. Executes via LLM
4. Caches result
