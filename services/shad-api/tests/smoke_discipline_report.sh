#!/usr/bin/env bash
# Smoke test for discipline-report strategy.
#
# Validates:
#   1. Strategy selection heuristic picks discipline-report
#   2. Source map generation succeeds on a real or synthetic corpus
#   3. Decomposition produces the expected wide/shallow DAG (no LLM)
#   4. Artifact metadata is correctly attached
#
# Usage:
#   bash tests/smoke_discipline_report.sh [corpus_path]
#
# If corpus_path is omitted, a temporary synthetic corpus is created.

set -euo pipefail

SHAD_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PYTHON="${SHAD_ROOT}/.venv/bin/python"
if [[ ! -x "$VENV_PYTHON" ]]; then
  VENV_PYTHON="${HOME}/.shad/venv/bin/python"
fi

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "ERROR: no usable python found (.venv or ~/.shad/venv)"
  exit 1
fi

export PYTHONPATH="${SHAD_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

CORPUS="${1:-}"

cleanup() {
  if [[ -n "${TMPDIR_CREATED:-}" ]]; then
    rm -rf "$CORPUS"
  fi
}
trap cleanup EXIT

# Create synthetic corpus if none provided
if [[ -z "$CORPUS" ]]; then
  CORPUS="$(mktemp -d)"
  TMPDIR_CREATED=1
  echo "Creating synthetic corpus at $CORPUS"

  mkdir -p "$CORPUS/src" "$CORPUS/tests" "$CORPUS/spec" "$CORPUS/deploy" "$CORPUS/docs"
  echo '# Example Project' > "$CORPUS/README.md"
  echo '[package]\nname = "example"' > "$CORPUS/Cargo.toml"
  echo 'fn main() {}' > "$CORPUS/src/main.rs"
  echo 'mod utils {}' > "$CORPUS/src/utils.rs"
  echo '#[test] fn it_works() {}' > "$CORPUS/tests/basic_test.rs"
  echo '---- MODULE Protocol ----' > "$CORPUS/spec/Protocol.tla"
  echo '[Service]\nExecStart=/usr/bin/example' > "$CORPUS/deploy/example.service"
  echo '# Architecture\nThis project...' > "$CORPUS/docs/architecture.md"
fi

echo "=== Shad Discipline Report Smoke Test ==="
echo "Corpus: $CORPUS"
echo ""

# --- Test 1: Strategy Selection ---
echo "--- Test 1: Strategy selection heuristic ---"
$VENV_PYTHON -c "
from shad.engine.strategies import StrategySelector, StrategyType

selector = StrategySelector()
tests = [
    'Build a source-grounded discipline for Semantos',
    'Build discipline report with useWhen avoidWhen',
    'Create edwinpai discipline from corpus',
]
for task in tests:
    result = selector.select(task)
    status = 'PASS' if result.strategy_type == StrategyType.DISCIPLINE_REPORT else 'FAIL'
    print(f'  [{status}] \"{task[:60]}...\" -> {result.strategy_type.value} (conf={result.confidence:.2f})')
    assert result.strategy_type == StrategyType.DISCIPLINE_REPORT, f'Expected discipline-report, got {result.strategy_type}'

# Override test
result = selector.select('Build something', override=StrategyType.DISCIPLINE_REPORT)
assert result.is_override and result.confidence == 1.0
print('  [PASS] Override forces discipline-report with confidence=1.0')
print()
"

# --- Test 2: Source map generation ---
echo "--- Test 2: Source map generation ---"
$VENV_PYTHON -c "
from pathlib import Path
from shad.discipline.source_map import SourceMapGenerator

corpus = Path('$CORPUS')
gen = SourceMapGenerator(max_files_per_section=80)
source_map = gen.generate([corpus])
md = source_map.to_markdown()

print(f'  Files:      {source_map.total_files}')
print(f'  Bytes:      {source_map.total_bytes}')
print(f'  Extensions: {dict(list(source_map.extension_counts.items())[:8])}')
print(f'  Important:  {source_map.important_files[:5]}')
print(f'  Formal:     {source_map.formal_files[:5]}')
print(f'  Tests:      {source_map.test_files[:5]}')
print(f'  Deploy:     {source_map.deploy_files[:5]}')
print(f'  Markdown:   {len(md)} chars')

assert source_map.total_files > 0, 'Source map should have files'
assert '## Roots' in md, 'Markdown should have Roots section'
assert '## Extension Counts' in md, 'Markdown should have Extension Counts'
print('  [PASS] Source map generated successfully')
print()
"

# --- Test 3: Deterministic DAG decomposition ---
echo "--- Test 3: Deterministic DAG decomposition ---"
$VENV_PYTHON -c "
import asyncio
from unittest.mock import MagicMock
from shad.engine.decomposition import StrategyDecomposer
from shad.engine.strategies import StrategyType, get_strategy

decomposer = StrategyDecomposer(llm_provider=MagicMock())
strategy = get_strategy(StrategyType.DISCIPLINE_REPORT)

result = asyncio.run(decomposer.decompose(
    task='Build the Example discipline from source snapshots',
    strategy=strategy,
    max_nodes=50,
))

stages = [n.stage_name for n in result.nodes]
print(f'  Stages ({len(stages)}): {stages}')
print(f'  Valid:  {result.is_valid}')
print(f'  LLM tokens: {result.tokens_used}')

assert result.is_valid, f'DAG invalid: {result.validation_errors}'
assert result.tokens_used == 0, 'Discipline DAG should be deterministic (0 tokens)'
assert len(stages) == 11, f'Expected 11 stages, got {len(stages)}'

# Check parallelism: 8 sections depend only on source_map
sections = [n for n in result.nodes if n.stage_name not in ('source_map', 'final_synthesis', 'quality_gate')]
for node in sections:
    assert node.hard_deps == ['source_map'], f'{node.stage_name} should depend only on source_map'

# Check final_synthesis depends on all 8 sections
final = next(n for n in result.nodes if n.stage_name == 'final_synthesis')
assert len(final.hard_deps) == 8, f'final_synthesis should depend on 8 sections, got {len(final.hard_deps)}'

# Check quality_gate depends on final_synthesis
gate = next(n for n in result.nodes if n.stage_name == 'quality_gate')
assert gate.hard_deps == ['final_synthesis']

print('  [PASS] DAG is wide/shallow with correct dependencies')
print()
"

# --- Test 4: Artifact metadata ---
echo "--- Test 4: Artifact metadata ---"
$VENV_PYTHON -c "
import asyncio
from unittest.mock import MagicMock
from shad.engine.decomposition import StrategyDecomposer
from shad.engine.strategies import StrategyType, get_strategy

decomposer = StrategyDecomposer(llm_provider=MagicMock())
strategy = get_strategy(StrategyType.DISCIPLINE_REPORT)

result = asyncio.run(decomposer.decompose(
    task='Build discipline',
    strategy=strategy,
    max_nodes=50,
))

for node in result.nodes:
    assert 'artifact' in node.metadata, f'{node.stage_name} missing artifact'
    assert node.metadata['artifact'].endswith('.md'), f'{node.stage_name} artifact not .md'
    assert node.metadata.get('discipline_report_section') is True
    print(f'  {node.stage_name:25s} -> {node.metadata[\"artifact\"]}')

print()
print('  [PASS] All nodes have correct artifact metadata')
"

echo ""
echo "=== All smoke tests passed ==="
