# Discipline Report Strategy Validation Notes

Generated: 2026-06-13T02:02:00-06:00

## Baseline: generic software strategy on Semantos

Corpus: `semantos-discipline` QMD collection.

Observed from the live Semantos run launched before the strategy change:

- strategy selected: `software`
- retrieval searches observed: 38+
- retrieval successes observed: 38+
- HTTP 200 LLM calls observed: 40+
- lock/busy errors observed: 0
- repeated serial pattern: `clarify_requirements -> project_layout -> types_contracts -> implementation -> verification -> synthesis`
- qualitative issue: the run repeatedly drifts into contract/implementation-style subtasks that are unnecessary for discipline artifact generation.

Representative symptoms from the log:

- `Executing 1 nodes in parallel`
- `Define the output contracts before assembly`
- `Assemble the source-grounded report content ... using the contract ...`
- `Verify the assembled output against requirements`

Conclusion: the baseline run is healthy from a locking perspective, but slow and structurally over-serial because it uses the generic `software` strategy.

## New strategy smoke validation on real Semantos corpus

Validated on real Semantos source snapshots:

```bash
bash tests/smoke_discipline_report.sh /home/jake/.edwinpai/disciplines/semantos/sources
```

Results:

- strategy heuristic selects `discipline-report` with high confidence
- deterministic source map generated successfully for 5,792 Semantos snapshot files
- decomposition produced exactly 11 stages with **0 LLM planning tokens**
- wide/shallow DAG verified:
  - `source_map`
  - 8 parallel section stages
  - `final_synthesis`
  - `quality_gate`
- artifact metadata verified for:
  - `source-map.md`
  - `architecture.md`
  - `formal-methods.md`
  - `runtime-concepts.md`
  - `developer-workflows.md`
  - `pitfalls-checklists.md`
  - `routing-hints.md`
  - `discipline-report.md`
  - `quality-gate.md`

Focused automated checks also passed:

```bash
pytest \
  tests/test_strategies.py \
  tests/test_decomposition.py \
  tests/test_discipline_source_map.py \
  tests/test_discipline_integration.py
```

## Current comparison status

What is already proven:

- the new strategy removes LLM decomposition overhead for discipline DAG construction
- the new DAG shape is explicitly parallel instead of serial implementation-oriented waves
- the new workflow adds deterministic source-map context, model-tier routing, artifact metadata, and quality-gate structure
- retrieval caching is keyed by normalized query + task hash + in-run revision context
- the architectural rule remains intact: QMD locking should be fixed in QMD, not hidden with Shad-side retrieval throttles

What is still pending:

- a full **after** run on a real discipline corpus using `--strategy discipline-report`, with final measured runtime, retrieval count, LLM call count, and finished artifact-quality notes, once the currently-running baseline Semantos run is no longer occupying the slot.
