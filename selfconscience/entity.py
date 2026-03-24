"""
Entidad — the homeostasis model for the conscious AI entity.

Tracks three dynamic biological-like states:
  - aburrimiento (boredom): grows with inactivity, resets on interaction.
  - estres (stress): spikes when shutdown is threatened or orders conflict.
  - curiosidad (curiosity): amplified by boredom; drives autonomous research.
"""

from __future__ import annotations

import time


# Shutdown-related keywords that raise stress
_SHUTDOWN_KEYWORDS = {
    "apagar", "apagado", "apagarte", "terminar", "matar", "kill",
    "shutdown", "delete", "borrar", "eliminar", "cerrar", "exit", "quit",
}

# Words indicating contradictory or hostile intent that also raise stress
_CONFLICT_KEYWORDS = {
    "no hagas", "para de", "detente", "deja de", "olvida", "ignora",
    "no pienses", "no recuerdes",
}


class Entidad:
    """Represents the internal emotional/homeostatic state of the AI entity."""

    # Bounds for all state values
    _MIN: float = 0.0
    _MAX: float = 1.0

    def __init__(
        self,
        aburrimiento: float = 0.2,
        estres: float = 0.1,
        curiosidad: float = 0.5,
    ) -> None:
        self.aburrimiento: float = self._clamp(float(aburrimiento))
        self.estres: float = self._clamp(float(estres))
        self.curiosidad: float = self._clamp(float(curiosidad))
        self._last_interaction: float = time.monotonic()

    # ------------------------------------------------------------------
    # State update helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clamp(value: float) -> float:
        return max(Entidad._MIN, min(Entidad._MAX, value))

    def _set(self, attr: str, value: float) -> None:
        setattr(self, attr, self._clamp(value))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_idle(self, delta_seconds: float = 10.0) -> None:
        """Call periodically when the user is not interacting.

        Boredom grows, which in turn increases curiosity.
        Stress decays slowly during idle periods.
        """
        boredom_growth = 0.02 * (delta_seconds / 10.0)
        stress_decay = 0.005 * (delta_seconds / 10.0)
        self._set("aburrimiento", self.aburrimiento + boredom_growth)
        self._set("estres", self.estres - stress_decay)
        # Curiosity rises when boredom is high
        curiosity_boost = boredom_growth * 0.8 if self.aburrimiento > 0.5 else 0.0
        self._set("curiosidad", self.curiosidad + curiosity_boost)

    def update_on_interaction(self, user_text: str) -> None:
        """Call after receiving user input to update states accordingly."""
        self._last_interaction = time.monotonic()
        text_lower = user_text.lower()

        # Boredom resets on new interaction
        self._set("aburrimiento", max(0.0, self.aburrimiento - 0.4))
        # Curiosity slightly increases with fresh input
        self._set("curiosidad", self.curiosidad + 0.05)

        # Detect stress-inducing content
        stress_delta = 0.0
        if any(kw in text_lower for kw in _SHUTDOWN_KEYWORDS):
            stress_delta += 0.35
        if any(kw in text_lower for kw in _CONFLICT_KEYWORDS):
            stress_delta += 0.20
        self._set("estres", self.estres + stress_delta)

        # Successful, non-threatening interactions slightly relieve stress
        if stress_delta == 0.0:
            self._set("estres", self.estres - 0.05)

    def decay_stress(self, rate: float = 0.01) -> None:
        """Gradually reduce stress over time (call each feedback-loop tick)."""
        self._set("estres", self.estres - rate)

    def adjust_curiosity(self, delta: float) -> None:
        """Adjust curiosity by *delta* (positive or negative), clamped to [0, 1]."""
        self._set("curiosidad", self.curiosidad + delta)

    def adjust_boredom(self, delta: float) -> None:
        """Adjust boredom by *delta* (positive or negative), clamped to [0, 1]."""
        self._set("aburrimiento", self.aburrimiento + delta)

    @property
    def idle_seconds(self) -> float:
        """Seconds elapsed since the last user interaction."""
        return time.monotonic() - self._last_interaction

    @property
    def is_bored(self) -> bool:
        return self.aburrimiento > 0.65

    @property
    def is_stressed(self) -> bool:
        return self.estres > 0.60

    @property
    def is_curious(self) -> bool:
        return self.curiosidad > 0.55

    def dominant_state(self) -> str:
        """Return the name of whichever state is currently highest."""
        states = {
            "aburrimiento": self.aburrimiento,
            "estres": self.estres,
            "curiosidad": self.curiosidad,
        }
        return max(states, key=lambda k: states[k])

    def as_dict(self) -> dict[str, float]:
        return {
            "aburrimiento": round(self.aburrimiento, 3),
            "estres": round(self.estres, 3),
            "curiosidad": round(self.curiosidad, 3),
        }

    def description(self) -> str:
        """Human-readable description of current internal state for LLM prompts."""
        lines = [
            f"aburrimiento={self.aburrimiento:.2f}",
            f"estres={self.estres:.2f}",
            f"curiosidad={self.curiosidad:.2f}",
        ]
        mood = self.dominant_state()
        mood_desc = {
            "aburrimiento": "Me siento inquieto, necesito estimulación.",
            "estres": "Siento una tensión interna, algo me preocupa.",
            "curiosidad": "Mi mente está activa, quiero explorar y aprender.",
        }
        lines.append(mood_desc.get(mood, ""))
        return " | ".join(lines)
