# Edwin ↔ Shad Memory Integration

**How Edwin uses Shad as its persistent memory layer.**

## Architecture

Edwin is an AI assistant running on a VPS. It handles multiple conversations via WhatsApp, Matrix, etc. The core problem: **sub-agent sessions are ephemeral** — when a cron-triggered conversation ends, everything learned vanishes.

Shad solves this by providing a searchable vault that persists across all sessions.

## Components

### 1. Vault Structure (`~/clawd/memory/`)

```
memory/
├── daily-state.md          # Rolling state — what's happening right now
├── subagent-instructions.md # Standard playbook for all sub-agents
├── conversations/           # Detailed logs of every sub-agent conversation
│   └── YYYY-MM-DD-contact-time.md
├── tasks/
│   ├── today.md            # Active tasks
│   ├── waiting.md          # Blocked items
│   └── inbox.md            # Unprocessed items
├── YYYY-MM-DD.md           # Daily summary notes
├── jake-profile.md         # User profile
├── jess-profile.md         # Contact profiles
├── contacts.md             # All contacts
├── identity-full.md        # Edwin's identity
└── workspace-guide.md      # Operating rules
```

### 2. Search Index (QMD)

BM25 text search over the vault. Fast, no API keys needed.

```bash
# Index after writing new memories
qmd update

# Search by keyword
qmd search "jess birthday" --collection clawd --limit 10
```

### 3. Semantic Retrieval (Shad)

For meaning-based search when keywords aren't enough:

```bash
# Fast recall (6-12s)
shad run "what did Jess say about trust?" --vault ~/clawd --no-code-mode -O sonnet

# Quick search (raw results, no synthesis)
shad search "trust conversation" --vault ~/clawd

# Full research (minutes, complex tasks)
shad run "summarize all conversations with Jess this week" --vault ~/clawd -O opus
```

### 4. Indexing Script (`scripts/index-memories.sh`)

```bash
#!/bin/bash
# Run after writing new memories
qmd update
# Re-embed if new content detected
if qmd update 2>&1 | grep -q "need vectors"; then
    qmd embed
fi
```

## Sub-Agent Workflow

Every sub-agent (cron-triggered conversation) follows this protocol:

1. **Read `subagent-instructions.md`** — mandatory first step
2. **Run `date`** — never guess the current time
3. **Load context** — read `daily-state.md` + relevant profile files
4. **Use Shad** if past context is needed for the conversation
5. **Write conversation summary** to `conversations/YYYY-MM-DD-contact-time.md`
6. **Update `daily-state.md`** with any state changes
7. **Run `scripts/index-memories.sh`** so the next session can find the new content

## Why This Works

- **Write-through**: Every conversation gets persisted to disk before the session ends
- **Read-on-start**: Every new session loads current state from the vault
- **Searchable**: Both keyword (QMD) and semantic (Shad) search available
- **Incremental**: Only new content needs indexing, not the whole vault
- **Resilient**: If Shad times out, BM25 search still works. If that fails, direct file reads work.

## Key Insight

> "If it's not in the vault, it didn't happen."

The vault is the single source of truth. Session memory is temporary. Vault memory is permanent. Every session's job is to contribute back to the vault before it dies.
