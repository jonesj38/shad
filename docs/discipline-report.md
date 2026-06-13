# Discipline Report Strategy

`discipline-report` is Shad's source-grounded strategy for building durable agent disciplines from a codebase or document corpus. It is designed for discipline artifacts, not implementation work.

## Why it exists

Generic `software` decomposition treats "build a discipline" like a software project and can create serial contract/implementation/verification loops. Discipline generation is a research-and-synthesis workflow with a known artifact shape. The `discipline-report` strategy makes the run faster and cleaner by using a wide, shallow section DAG.

## Run shape

The strategy decomposes once into these stages:

1. `source_map`
2. `product_model`
3. `repo_architecture`
4. `core_concepts`
5. `protocols_security`
6. `formal_methods`
7. `developer_workflows`
8. `operational_pitfalls`
9. `routing_hints`
10. `final_synthesis`
11. `quality_gate`

After `source_map`, the section stages are independent and can run in parallel. `final_synthesis` depends on the sections, and `quality_gate` verifies the result.

## Source map prepass

Generate a deterministic source map before the RLM run:

```bash
shad discipline source-map /path/to/source -o source-map.md
```

The source map includes:

- source roots and git revisions
- file/byte counts
- extension counts
- top directories
- important manifest/docs files
- formal/spec assets (`.tla`, `.cfg`, `.lean`, etc.)
- test/fuzz files
- deploy/runtime files

When `shad run --strategy discipline-report` receives filesystem `--sources`, Shad also injects a source map into run context automatically.

## Recommended command

```bash
shad run "Build the Example discipline from this source corpus" \
  --strategy discipline-report \
  --sources /path/to/source \
  --collection example-discipline \
  --profile deep \
  --orchestrator-model <strong-model> \
  --worker-model <fast-good-model> \
  --leaf-model <fast-good-model>
```

For QMD-only named collections, source-map injection requires a filesystem source path; otherwise the run still uses the wide discipline strategy and QMD retrieval packs.

## Model-tier guidance

- Planner/decomposition: deterministic for this strategy; no LLM planning call is needed.
- Section tasks: leaf tier, parallel, source-grounded.
- Final synthesis and quality gate: orchestrator tier.

Use faster worker/leaf models for section extraction and reserve the strongest model for final synthesis/verification.

## Quality rules

The strategy prompt forbids implementation drift:

- Do not design new packages, schemas, APIs, or code unless they already exist in the source corpus.
- Ground substantive claims in source paths, quotes, commands, specs, or symbols.
- Mark uncertainty explicitly.
- Include actionable `useWhen` / `avoidWhen` routing hints.

## Layered artifacts

A finished discipline should normally publish:

- `source-map.md`
- `architecture.md`
- `formal-methods.md`
- `runtime-concepts.md`
- `developer-workflows.md`
- `pitfalls-checklists.md`
- `routing-hints.md`
- `discipline-report.md` or `<id>-discipline-report.md`
- `quality-gate.md`

These can be indexed into an artifact collection such as `<id>-discipline-artifacts`, with raw source snapshots kept in `<id>-discipline`.

## QMD concurrency rule

Do not reintroduce Shad-side QMD search throttles to hide locking bugs. If QMD locking regresses, fix QMD read/write boundaries, SQLite transactions, busy timeouts, WAL/read-only behavior, or startup schema behavior.
