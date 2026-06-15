# Discipline Multi-Run Scaffold

`shad discipline scaffold` creates a deterministic staged package for building a durable discipline without starting any expensive model calls.

It exists because large discipline builds perform better when they are split into focused, source-authority-aware stages instead of one giant recursive run. The scaffold creates:

- `00-Final-Reports/`
- `01-Methodology/discipline-plan.md`
- `02-Source-Map/source-map.md`
- `03-Analysis/`
- `04-Data/discipline-plan.json`
- `05-Scripts/stage-prompts/*.md`
- `05-Scripts/run-discipline-stage.sh`
- `06-Visualizations/`
- `07-Out-Reports/`
- `Sources/`

## Example

```bash
shad discipline scaffold EdwinPAI ~/.edwinpai/disciplines/edwinpai-next \
  --source ~/Desktop/edwin \
  --source ~/Desktop/edwin-desktop \
  --exclude ~/Desktop/edwin-docs-private
```

The command records explicit source roots and excludes, writes a deterministic source map, and generates prompts for these stages:

1. architecture, runtime, storage-data
2. protocols-security, formal-methods, developer-workflows
3. pitfalls, routing-hints
4. verification
5. final-synthesis

The first three waves are intentionally parallelizable. Verification and final synthesis depend on the stage outputs.

## Why this is faster and better

- deterministic source map before model work
- focused stage prompts reduce implementation drift
- stage outputs can be resumed or rerun independently
- parallel waves reduce wall-clock time while preserving recursive depth
- verification gets its own pass instead of being hidden inside synthesis
- excludes prevent accidental ingestion of unrelated/private repositories

## Notes

The generated `run-discipline-stage.sh` is a starter runner. For production workflows, schedule stages with the host workflow runner or another supervisor so logs, retries, approvals, and artifacts are captured consistently.
