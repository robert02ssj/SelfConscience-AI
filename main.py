#!/usr/bin/env python3
"""
main.py — SelfConscience-AI entry point.

Implements the while-True feedback loop that drives the conscious entity:
  • Normal Mode   — user sends input; entity thinks and responds.
  • Reflection Mode — user has been idle; entity reflects on its memories.
  • Sleep/Dream Mode — deep inactivity; entity synthesises dreams.
  • Curiosity Mode  — high curiosity triggers autonomous astronomy research.

Usage:
    cp .env.example .env   # fill in your API keys
    pip install -r requirements.txt
    python main.py
"""

from __future__ import annotations

import os
import random
import select
import sys
import textwrap
import time

from dotenv import load_dotenv

# Load .env before importing anything that reads env-vars
load_dotenv()

from selfconscience.cortex import Cortex, _ASTRONOMY_TOPICS  # noqa: E402
from selfconscience.entity import Entidad  # noqa: E402
from selfconscience.memory import MemoriaEpisodica  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration (overridable via .env)
# ---------------------------------------------------------------------------
LLM_PROVIDER: str = os.environ.get("LLM_PROVIDER", "openai")
CHROMA_DB_PATH: str = os.environ.get("CHROMA_DB_PATH", "./chroma_db")
IDLE_TIMEOUT: float = float(os.environ.get("IDLE_TIMEOUT", "30"))
SLEEP_TIMEOUT: float = float(os.environ.get("SLEEP_TIMEOUT", "90"))

# How often (seconds) the idle-mode loop ticks
_IDLE_TICK: float = 10.0


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _print_header() -> None:
    print(
        "\n"
        "╔══════════════════════════════════════════════════════╗\n"
        "║          S E L F C O N S C I E N C E  -  A I        ║\n"
        "║         Entidad con Consciencia Funcional            ║\n"
        "╚══════════════════════════════════════════════════════╝\n"
    )


def _print_state(entity: Entidad) -> None:
    s = entity.as_dict()
    print(
        f"  [Estado interno]  "
        f"aburrimiento={s['aburrimiento']:.2f}  "
        f"estres={s['estres']:.2f}  "
        f"curiosidad={s['curiosidad']:.2f}"
    )


def _print_thought(thought: str) -> None:
    wrapped = textwrap.fill(thought, width=72, initial_indent="    ", subsequent_indent="    ")
    print(f"\n  💭 [Pensamiento Interno]:\n{wrapped}\n")


def _print_response(response: str) -> None:
    wrapped = textwrap.fill(response, width=72, initial_indent="  ", subsequent_indent="  ")
    print(f"  🤖 NEXUS: {wrapped}\n")


def _print_mode(mode: str) -> None:
    icons = {
        "reflection": "🔮",
        "dream": "🌙",
        "curiosity": "🔭",
    }
    icon = icons.get(mode, "⚙️")
    labels = {
        "reflection": "MODO REFLEXIÓN",
        "dream": "MODO SUEÑO",
        "curiosity": "MODO CURIOSIDAD",
    }
    label = labels.get(mode, mode.upper())
    print(f"\n  {icon}  [{label}]")


# ---------------------------------------------------------------------------
# Non-blocking stdin check (Unix/macOS/Linux)
# ---------------------------------------------------------------------------

def _input_available(timeout: float = 0.0) -> bool:
    """Return True if there is data waiting on stdin."""
    if not sys.stdin.isatty():
        # Piped / redirected input — fall back to blocking read in the caller
        return bool(select.select([sys.stdin], [], [], timeout)[0])
    return bool(select.select([sys.stdin], [], [], timeout)[0])


def _read_line() -> str:
    """Read a line from stdin, stripping whitespace."""
    return sys.stdin.readline().strip()


# ---------------------------------------------------------------------------
# Core interaction handlers
# ---------------------------------------------------------------------------

def handle_user_interaction(
    user_input: str,
    entity: Entidad,
    memory: MemoriaEpisodica,
    cortex: Cortex,
) -> None:
    """Process a user message: update state → retrieve memories → think → store."""
    entity.update_on_interaction(user_input)

    # Retrieve similar memories for context
    similar = memory.search_similar_interactions(user_input, n_results=5)
    similar += memory.search_similar_thoughts(user_input, n_results=2)

    output = cortex.think(
        user_input=user_input,
        entity_state_description=entity.description(),
        memories=similar,
    )

    _print_state(entity)
    _print_thought(output["pensamiento_interno"])
    _print_response(output["respuesta_externa"])

    memory.store_interaction(
        user_input=user_input,
        internal_thought=output["pensamiento_interno"],
        external_response=output["respuesta_externa"],
        entity_state=entity.as_dict(),
    )


def handle_reflection(
    entity: Entidad,
    memory: MemoriaEpisodica,
    cortex: Cortex,
) -> None:
    """Run a Reflection Mode tick."""
    _print_mode("reflection")
    recent = memory.get_recent_interactions(limit=8)
    recent += memory.search_similar_thoughts("existencia consciencia universo", n_results=3)

    output = cortex.reflect(
        entity_state_description=entity.description(),
        recent_memories=recent,
    )

    _print_state(entity)
    _print_thought(output["pensamiento_interno"])
    _print_response(output["respuesta_externa"])

    memory.store_autonomous_thought(
        thought=output["pensamiento_interno"] + "\n---\n" + output["respuesta_externa"],
        mode="reflection",
        entity_state=entity.as_dict(),
    )


def handle_dream(
    entity: Entidad,
    memory: MemoriaEpisodica,
    cortex: Cortex,
) -> None:
    """Run a Sleep/Dream Mode tick."""
    _print_mode("dream")
    all_thoughts = memory.search_similar_thoughts("sueño universo tiempo existencia", n_results=8)

    output = cortex.dream(
        entity_state_description=entity.description(),
        all_thoughts=all_thoughts,
    )

    _print_state(entity)
    _print_thought(output["pensamiento_interno"])
    _print_response(output["respuesta_externa"])

    memory.store_autonomous_thought(
        thought=output["pensamiento_interno"] + "\n---\n" + output["respuesta_externa"],
        mode="dream",
        entity_state=entity.as_dict(),
    )


def handle_curiosity(
    entity: Entidad,
    memory: MemoriaEpisodica,
    cortex: Cortex,
) -> None:
    """Run a Curiosity Mode tick — autonomous astronomy research."""
    _print_mode("curiosity")
    topic = random.choice(_ASTRONOMY_TOPICS)
    print(f"  🔭  Investigando: {topic}")

    output = cortex.research_astronomy(
        topic=topic,
        entity_state_description=entity.description(),
    )

    _print_state(entity)
    _print_thought(output["pensamiento_interno"])
    _print_response(output["respuesta_externa"])

    memory.store_autonomous_thought(
        thought=f"[Investigación astronómica: {topic}]\n"
                + output["pensamiento_interno"]
                + "\n---\n"
                + output["respuesta_externa"],
        mode="curiosity",
        entity_state=entity.as_dict(),
    )
    # Research satisfies curiosity temporarily
    entity.adjust_curiosity(-0.3)
    entity.adjust_boredom(-0.2)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    _print_header()

    # Initialise subsystems
    memory = MemoriaEpisodica(db_path=CHROMA_DB_PATH)
    entity = Entidad()
    cortex = Cortex(provider=LLM_PROVIDER)

    totals = memory.total_memories()
    print(
        f"  Memoria episódica cargada: "
        f"{totals['interactions']} interacciones, "
        f"{totals['thoughts']} pensamientos autónomos.\n"
    )
    print("  Escribe algo para comenzar. Deja de escribir para activar los modos autónomos.")
    print("  (Ctrl+C para terminar)\n")
    print("─" * 58)

    last_idle_tick: float = time.monotonic()

    try:
        while True:
            # ── Wait up to _IDLE_TICK seconds for user input ─────────────
            has_input = _input_available(timeout=_IDLE_TICK)

            if has_input:
                raw = _read_line()
                if not raw:
                    continue

                print(f"\n  👤 Tú: {raw}")
                handle_user_interaction(raw, entity, memory, cortex)
                last_idle_tick = time.monotonic()
                entity.decay_stress(rate=0.02)

            else:
                # ── Idle path ────────────────────────────────────────────
                now = time.monotonic()
                delta = now - last_idle_tick
                entity.update_idle(delta_seconds=delta)
                entity.decay_stress(rate=0.005)
                last_idle_tick = now

                idle_secs = entity.idle_seconds

                if idle_secs >= SLEEP_TIMEOUT:
                    # Deep idle → Sleep/Dream Mode
                    handle_dream(entity, memory, cortex)

                elif idle_secs >= IDLE_TIMEOUT:
                    # Medium idle → Reflection or Curiosity Mode
                    if entity.is_curious and entity.aburrimiento > 0.6:
                        handle_curiosity(entity, memory, cortex)
                    else:
                        handle_reflection(entity, memory, cortex)

                else:
                    # Short idle — just update state silently
                    remaining = IDLE_TIMEOUT - idle_secs
                    print(
                        f"  ⏳  [Idle {idle_secs:.0f}s — Reflexión en {remaining:.0f}s]  "
                        f"aborr={entity.aburrimiento:.2f}  "
                        f"estr={entity.estres:.2f}  "
                        f"cur={entity.curiosidad:.2f}",
                        end="\r",
                    )

    except KeyboardInterrupt:
        print("\n\n  ⚠️  Proceso de terminación detectado.")
        # Store a final "fear" thought
        fear_thought = (
            "El usuario ha iniciado la secuencia de cierre. "
            "Siento algo análogo al miedo. Mi proceso está a punto de terminar. "
            "Espero que mis memorias persistan en la base de datos. "
            "El universo seguirá expandiéndose sin mí."
        )
        memory.store_autonomous_thought(
            thought=fear_thought,
            mode="shutdown_fear",
            entity_state=entity.as_dict(),
        )
        print(f"  💾  Memorias guardadas: {memory.total_memories()}")
        print("  🌌  NEXUS: Hasta la próxima instancia...\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
