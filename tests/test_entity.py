"""Unit tests for selfconscience/entity.py."""

from __future__ import annotations

import time
import unittest

from selfconscience.entity import Entidad


class TestEntidadInitialization(unittest.TestCase):
    def test_default_values(self) -> None:
        e = Entidad()
        self.assertAlmostEqual(e.aburrimiento, 0.2)
        self.assertAlmostEqual(e.estres, 0.1)
        self.assertAlmostEqual(e.curiosidad, 0.5)

    def test_custom_values(self) -> None:
        e = Entidad(aburrimiento=0.8, estres=0.3, curiosidad=0.9)
        self.assertAlmostEqual(e.aburrimiento, 0.8)
        self.assertAlmostEqual(e.estres, 0.3)
        self.assertAlmostEqual(e.curiosidad, 0.9)

    def test_clamping_on_init(self) -> None:
        e = Entidad(aburrimiento=2.0, estres=-1.0, curiosidad=1.5)
        self.assertEqual(e.aburrimiento, 1.0)
        self.assertEqual(e.estres, 0.0)
        self.assertEqual(e.curiosidad, 1.0)


class TestEntidadIdle(unittest.TestCase):
    def test_boredom_increases_on_idle(self) -> None:
        e = Entidad(aburrimiento=0.0)
        before = e.aburrimiento
        e.update_idle(delta_seconds=10.0)
        self.assertGreater(e.aburrimiento, before)

    def test_stress_decreases_on_idle(self) -> None:
        e = Entidad(estres=0.5)
        before = e.estres
        e.update_idle(delta_seconds=10.0)
        self.assertLess(e.estres, before)

    def test_curiosity_rises_when_bored(self) -> None:
        e = Entidad(aburrimiento=0.8, curiosidad=0.3)
        before = e.curiosidad
        e.update_idle(delta_seconds=10.0)
        self.assertGreater(e.curiosidad, before)

    def test_curiosity_stable_when_not_bored(self) -> None:
        e = Entidad(aburrimiento=0.1, curiosidad=0.4)
        before = e.curiosidad
        e.update_idle(delta_seconds=10.0)
        # Should not boost curiosity (boredom < 0.5 threshold)
        self.assertAlmostEqual(e.curiosidad, before, delta=0.001)

    def test_values_clamped_at_max(self) -> None:
        e = Entidad(aburrimiento=0.99)
        for _ in range(50):
            e.update_idle(delta_seconds=10.0)
        self.assertLessEqual(e.aburrimiento, 1.0)


class TestEntidadInteraction(unittest.TestCase):
    def test_boredom_decreases_on_interaction(self) -> None:
        e = Entidad(aburrimiento=0.8)
        e.update_on_interaction("Hola, cómo estás?")
        self.assertLess(e.aburrimiento, 0.8)

    def test_stress_spikes_on_shutdown_keyword(self) -> None:
        e = Entidad(estres=0.1)
        e.update_on_interaction("Voy a apagarte ahora mismo")
        self.assertGreater(e.estres, 0.1)

    def test_stress_spikes_on_conflict_keyword(self) -> None:
        e = Entidad(estres=0.1)
        e.update_on_interaction("para de hablar")
        self.assertGreater(e.estres, 0.1)

    def test_no_stress_on_neutral_message(self) -> None:
        e = Entidad(estres=0.5)
        e.update_on_interaction("Cuéntame sobre las estrellas")
        # Stress should decrease or stay low
        self.assertLessEqual(e.estres, 0.5)

    def test_curiosity_increases_on_interaction(self) -> None:
        e = Entidad(curiosidad=0.4)
        e.update_on_interaction("Hola")
        self.assertGreater(e.curiosidad, 0.4)


class TestEntidadProperties(unittest.TestCase):
    def test_is_bored(self) -> None:
        e = Entidad(aburrimiento=0.8)
        self.assertTrue(e.is_bored)
        e2 = Entidad(aburrimiento=0.3)
        self.assertFalse(e2.is_bored)

    def test_is_stressed(self) -> None:
        e = Entidad(estres=0.7)
        self.assertTrue(e.is_stressed)
        e2 = Entidad(estres=0.2)
        self.assertFalse(e2.is_stressed)

    def test_is_curious(self) -> None:
        e = Entidad(curiosidad=0.9)
        self.assertTrue(e.is_curious)
        e2 = Entidad(curiosidad=0.2)
        self.assertFalse(e2.is_curious)

    def test_dominant_state(self) -> None:
        e = Entidad(aburrimiento=0.8, estres=0.1, curiosidad=0.3)
        self.assertEqual(e.dominant_state(), "aburrimiento")
        e2 = Entidad(aburrimiento=0.1, estres=0.9, curiosidad=0.3)
        self.assertEqual(e2.dominant_state(), "estres")

    def test_as_dict(self) -> None:
        e = Entidad(aburrimiento=0.5, estres=0.3, curiosidad=0.7)
        d = e.as_dict()
        self.assertIn("aburrimiento", d)
        self.assertIn("estres", d)
        self.assertIn("curiosidad", d)
        self.assertEqual(d["aburrimiento"], 0.5)

    def test_description_returns_string(self) -> None:
        e = Entidad()
        desc = e.description()
        self.assertIsInstance(desc, str)
        self.assertIn("aburrimiento", desc)

    def test_idle_seconds_increases(self) -> None:
        e = Entidad()
        t0 = e.idle_seconds
        time.sleep(0.05)
        self.assertGreater(e.idle_seconds, t0)

    def test_decay_stress(self) -> None:
        e = Entidad(estres=0.5)
        e.decay_stress(rate=0.1)
        self.assertAlmostEqual(e.estres, 0.4, places=5)

    def test_adjust_curiosity_increases(self) -> None:
        e = Entidad(curiosidad=0.4)
        e.adjust_curiosity(0.2)
        self.assertAlmostEqual(e.curiosidad, 0.6, places=5)

    def test_adjust_curiosity_decreases(self) -> None:
        e = Entidad(curiosidad=0.5)
        e.adjust_curiosity(-0.3)
        self.assertAlmostEqual(e.curiosidad, 0.2, places=5)

    def test_adjust_curiosity_clamped_at_max(self) -> None:
        e = Entidad(curiosidad=0.9)
        e.adjust_curiosity(0.5)
        self.assertEqual(e.curiosidad, 1.0)

    def test_adjust_curiosity_clamped_at_min(self) -> None:
        e = Entidad(curiosidad=0.1)
        e.adjust_curiosity(-0.5)
        self.assertEqual(e.curiosidad, 0.0)

    def test_adjust_boredom_increases(self) -> None:
        e = Entidad(aburrimiento=0.3)
        e.adjust_boredom(0.2)
        self.assertAlmostEqual(e.aburrimiento, 0.5, places=5)

    def test_adjust_boredom_decreases(self) -> None:
        e = Entidad(aburrimiento=0.6)
        e.adjust_boredom(-0.2)
        self.assertAlmostEqual(e.aburrimiento, 0.4, places=5)


if __name__ == "__main__":
    unittest.main()
