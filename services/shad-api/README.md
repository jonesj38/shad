# Shad API

The core API and CLI for Shannon's Daemon - Personal AI Infrastructure for long-context reasoning.

## Installation

```bash
pip install -e .
```

## CLI Usage

```bash
shad run "Your goal here" --max-depth 2
shad run "Complex task" -O opus -W sonnet -L haiku  # Claude models
shad run "Task" -O opus -W llama3 -L qwen3:latest   # Mix Claude + Ollama
shad models                          # List Claude models
shad models --ollama                 # Include Ollama models
shad status <run_id>
shad trace tree <run_id>
shad resume <run_id>
```

## API Usage

```bash
uvicorn shad.api.main:app --reload
```

See the main project README for full documentation.
