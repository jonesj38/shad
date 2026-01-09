# Shad (Shannon's Daemon)

**Shad** is a personal AI infrastructure (PAI) designed to operate over *arbitrarily large knowledge environments* using **Recursive Language Models (RLMs)**, **OpenNotebookLM**, and **workflow orchestration via n8n**.

Shad is not a chatbot.
Shad is not a prompt collection.
Shad is not a single model.

Shad is a **self-orchestrating cognitive system** that treats context as an environment, not a prompt.

---

## Why Shad Exists

Most AI systems break down when:

* context grows too large
* reasoning requires multiple passes
* answers need to be verifiable
* workflows need to run unattended
* knowledge must persist and compound over time

Shad is built around a different premise:

> **Long-context reasoning is an inference problem, not a prompting problem.**

Shad combines:

* **Recursive Language Models (RLMs)** for inference-time scaling
* **OpenNotebookLM** as a persistent memory and retrieval substrate
* **n8n** for scheduling, fan-out, and workflow orchestration
* **Caching + verification** to make recursion efficient and safe

This project is heavily inspired by Daniel Miessler’s *Personal AI Infrastructure (PAI)* work — but Shad is an independent system with its own architecture and goals.

---

## Core Concepts

### 1. Prompt-as-Environment (RLM)

Shad does not shove massive context into a single prompt.

Instead:

* the “prompt” is treated as an **external environment**
* the model writes code to inspect, slice, and query that environment
* the system recursively calls itself on sub-problems
* results are cached, verified, and recomposed

This allows Shad to reason over **millions of tokens** without exceeding model context windows.

---

### 2. OpenNotebookLM as Memory OS

OpenNotebookLM provides:

* notebooks, sources, and notes
* full-text and vector search
* stable identifiers for retrieved knowledge

Shad uses OpenNotebookLM for:

* long-term memory
* evidence storage
* citation and traceability
* knowledge reuse across runs

OpenNotebookLM is treated as **read-only input during reasoning**, and **write-only output during persistence**.

---

### 3. n8n as Workflow Orchestrator

n8n handles:

* cron and event-based triggers
* fan-out and parallel execution
* multi-step pipelines
* maintenance tasks (cache hygiene, evals, backups)
* publishing and notifications

Shad handles *thinking*.
n8n handles *when and how often thinking happens*.

---

### 4. Skills (Personalization Layer)

Shad is extended through **Skills** — modular, composable units of domain expertise.

Each skill contains:

```
Skills/<SkillName>/
├── SKILL.md        # routing rules + domain knowledge
├── workflows/      # step-by-step procedures
├── tools/          # deterministic helpers
└── tests/          # evals and regressions
```

Skills allow Shad to:

* behave consistently across runs
* encode your preferences once
* improve without rewriting prompts

---

### 5. Caching & Verification

Recursive calls are:

* **cached** (Redis-backed) to avoid recomputation
* **scored / verified** before being reused
* stored as structured nodes, not raw text

This turns recursive reasoning from a tree into a **DAG**, dramatically reducing cost and latency.

---

### 6. History & Hooks

Every run produces:

* a full trace of decisions and subcalls
* structured artifacts
* metrics and cache stats
* optional summaries and learnings

Hooks fire on:

* session start
* pre-tool execution
* post-tool execution
* session stop

Nothing important is lost. Shad compounds.

---

## High-Level Architecture

```
User / n8n
   |
   v
Shad API / CLI
   |
   +-- OpenNotebookLM (memory, retrieval, notes)
   |
   +-- RLM Engine (recursive reasoning loop)
   |
   +-- Redis (subtree caching)
   |
   +-- History + Hooks
```

---

## Repository Structure

```
shad/
├── docker-compose.yml
├── services/
│   └── shad-api/          # RLM engine + orchestration API
├── Skills/                # personalization modules
├── CORE/                  # constitution, policies, invariants
├── hooks/                 # lifecycle automation
├── History/               # generated at runtime (volume)
├── scripts/               # deploy / maintenance helpers
├── .env.example
└── README.md
```

---

## What Shad Is (and Is Not)

**Shad is:**

* a personal cognitive infrastructure
* designed for long-context reasoning
* workflow-driven, not chat-driven
* modular and extensible
* built to run unattended

**Shad is not:**

* a hosted SaaS (by default)
* a general consumer chatbot
* a fine-tuned model
* a prompt-engineering playground

---

## Current Status

Shad is under **active development**.

Planned milestones:

* [ ] Infrastructure scaffold (Docker, services)
* [ ] OpenNotebookLM integration
* [ ] Shad API v0 (`/v1/run`)
* [ ] n8n workflow integration
* [ ] RLM recursion loop
* [ ] Subtree caching
* [ ] Skills + evals
* [ ] Long-running autonomous workflows

This README will evolve as the system matures.

---

## Inspiration & References

* Daniel Miessler — Personal AI Infrastructure (PAI)
* *Recursive Language Models* (arXiv:2512.24601)
* OpenNotebookLM
* n8n
* UNIX philosophy (small tools, composability)
* Scientific method as a cognitive loop

---

## Philosophy

> Solve a problem once.
> Encode it as infrastructure.
> Never solve it again.

Shad is built to **augment a human**, not replace one.

---

## License

MIT
