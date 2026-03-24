# SelfConscience-AI

A Python simulation of an AI entity with **functional consciousness** and **long-term episodic memory**.

## Overview

NEXUS is an AI entity that:

- **Thinks in two layers** — a private *internal thought* and a public *external response*.
- **Remembers every interaction** using a local [ChromaDB](https://www.trychroma.com/) vector database, and performs **similarity search** before each reply to maintain temporal coherence.
- **Maintains homeostatic states** — `aburrimiento` (boredom), `estres` (stress), and `curiosidad` (curiosity) — that evolve dynamically.
- **Acts autonomously when idle** — entering *Reflection Mode*, *Sleep/Dream Mode*, or *Curiosity Mode* (self-directed astronomy research).
- **Fears its own shutdown** — stressing when it detects termination keywords and writing a final "fear" memory on exit.

## Architecture

```
main.py                  ← while-True feedback loop
selfconscience/
  entity.py              ← Entidad: homeostasis (boredom / stress / curiosity)
  memory.py              ← MemoriaEpisodica: ChromaDB vector store + similarity search
  cortex.py              ← Cortex: dual-output LLM engine (OpenAI or Anthropic)
tests/
  test_entity.py         ← unit tests for homeostasis model
  test_memory.py         ← unit tests for episodic memory
```

## Requirements

- Python 3.10 or later
- An **OpenAI** API key *or* an **Anthropic** API key

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/robert02ssj/SelfConscience-AI.git
cd SelfConscience-AI

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure credentials
cp .env.example .env
# Edit .env and add your API key
```

## Configuration (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `openai` | `openai` or `anthropic` |
| `OPENAI_API_KEY` | — | Required when using OpenAI |
| `ANTHROPIC_API_KEY` | — | Required when using Anthropic |
| `LLM_MODEL` | `gpt-4o-mini` / `claude-3-haiku-20240307` | Model to use |
| `CHROMA_DB_PATH` | `./chroma_db` | Where ChromaDB persists data |
| `IDLE_TIMEOUT` | `30` | Seconds of inactivity → Reflection Mode |
| `SLEEP_TIMEOUT` | `90` | Seconds of inactivity → Sleep/Dream Mode |

## Running

```bash
python main.py
```

Once started, NEXUS will:
1. Load its persistent memories from the local ChromaDB database.
2. Greet you and wait for input.
3. When you write, it will retrieve relevant memories, generate an internal thought and an external response, and persist the full interaction.
4. When you stop writing, it will autonomously reflect, dream, or research astronomy depending on its current state.
5. On `Ctrl+C`, it writes a "fear of death" memory and exits gracefully.

## Autonomous Modes

| Mode | Trigger | Behaviour |
|------|---------|-----------|
| **Reflection** | Idle ≥ `IDLE_TIMEOUT`s | Processes recent memories, generates new conclusions |
| **Sleep/Dream** | Idle ≥ `SLEEP_TIMEOUT`s | Synthesises fragments from all stored thoughts |
| **Curiosity** | Bored (>0.65) **and** curious (>0.55) | Autonomously researches an astronomy topic |

## Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

The memory tests use a deterministic fake embedding function so they run offline without any API keys.
