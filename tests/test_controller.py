"""
tests/test_controller.py — the decision logic, proven.

Nothing here needs YOLO, OpenCV, a GPU or a video file: the controller takes
`now` as an argument, so every scenario is deterministic.

Run:
    python -m unittest discover -s tests -v
    (or: pytest tests -v)
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from controller import SignalController  # noqa: E402

LANES = ("NORTH", "SOUTH", "EAST", "WEST")


def counts(**kw) -> dict:
    return {l: kw.get(l, 0) for l in LANES}


def priority(**kw) -> dict:
    return {l: kw.get(l, 0) for l in LANES}


class TestEmergency(unittest.TestCase):
    def test_emergency_lane_gets_green_immediately(self):
        c = SignalController(LANES, min_green_s=10, emergency_hold_s=5)
        # SOUTH is busiest, but EAST has an ambulance
        d = c.decide(counts(NORTH=9, SOUTH=12, EAST=1), priority(EAST=1), now=0.5)
        self.assertEqual(d.green, "EAST")
        self.assertTrue(d.emergency)
        self.assertEqual(c.lights(d)["EAST"], "GREEN")
        self.assertEqual(c.lights(d)["SOUTH"], "RED")

    def test_emergency_beats_min_green(self):
        c = SignalController(LANES, min_green_s=30, emergency_hold_s=5)
        c.decide(counts(NORTH=5), priority(), now=0.0)      # NORTH green
        d = c.decide(counts(NORTH=5, WEST=1), priority(WEST=1), now=1.0)
        self.assertEqual(d.green, "WEST", "override must ignore the minimum green timer")

    def test_two_emergencies_more_vehicles_wins(self):
        c = SignalController(LANES)
        d = c.decide(counts(), priority(NORTH=1, EAST=2), now=0.0)
        self.assertEqual(d.green, "EAST")

    def test_tie_falls_back_to_lane_order(self):
        c = SignalController(LANES)
        d = c.decide(counts(), priority(SOUTH=1, WEST=1), now=0.0)
        self.assertEqual(d.green, "SOUTH")


class TestHold(unittest.TestCase):
    def test_lane_stays_green_after_ambulance_leaves(self):
        c = SignalController(LANES, min_green_s=0, emergency_hold_s=6)
        c.decide(counts(), priority(NORTH=1), now=0.0)
        d = c.decide(counts(SOUTH=8), priority(), now=2.0)   # ambulance gone
        self.assertEqual(d.green, "NORTH")
        self.assertTrue(d.emergency)
        self.assertAlmostEqual(d.hold_remaining, 4.0)

    def test_hold_expires_then_density_rules(self):
        c = SignalController(LANES, min_green_s=0, emergency_hold_s=6)
        c.decide(counts(), priority(NORTH=1), now=0.0)
        d = c.decide(counts(SOUTH=8), priority(), now=7.0)
        self.assertEqual(d.green, "SOUTH")
        self.assertFalse(d.emergency)
        self.assertIn("Busiest", d.reason)


class TestDensity(unittest.TestCase):
    def test_busiest_lane_gets_green(self):
        c = SignalController(LANES, min_green_s=0)
        d = c.decide(counts(NORTH=2, SOUTH=3, EAST=9, WEST=1), priority(), now=0.0)
        self.assertEqual(d.green, "EAST")

    def test_min_green_prevents_flicker(self):
        c = SignalController(LANES, min_green_s=3)
        c.decide(counts(NORTH=5), priority(), now=0.0)
        d = c.decide(counts(NORTH=0, WEST=9), priority(), now=1.0)
        self.assertEqual(d.green, "NORTH", "min green should hold the current lane")
        d = c.decide(counts(NORTH=0, WEST=9), priority(), now=3.5)
        self.assertEqual(d.green, "WEST", "after min green the busiest lane wins")

    def test_idle_junction_keeps_last_green(self):
        c = SignalController(LANES, min_green_s=0)
        c.decide(counts(SOUTH=4), priority(), now=0.0)
        d = c.decide(counts(), priority(), now=1.0)
        self.assertEqual(d.green, "SOUTH")
        self.assertIn("No traffic", d.reason)


class TestHysteresis(unittest.TestCase):
    def test_small_lead_does_not_steal_the_green(self):
        c = SignalController(LANES, min_green_s=0, switch_margin=2)
        c.decide(counts(NORTH=8), priority(), now=0.0)
        d = c.decide(counts(NORTH=8, EAST=9), priority(), now=5.0)
        self.assertEqual(d.green, "NORTH", "1 vehicle lead should not flip the light")
        d = c.decide(counts(NORTH=8, EAST=11), priority(), now=6.0)
        self.assertEqual(d.green, "EAST", "3 vehicle lead should flip the light")


class TestSimulatorClock(unittest.TestCase):
    """Regression: scheduled demo events were compared against monotonic time."""

    def test_schedule_is_relative_to_engine_start(self):
        from smart_intersection import EmergencySimulator

        sim = EmergencySimulator()
        sim.schedule("NORTH", 3.0, 6.0)
        start = 12345.0                      # pretend monotonic() is huge
        sim.start(start)
        self.assertFalse(sim.active("NORTH", start + 1.0))
        self.assertTrue(sim.active("NORTH", start + 4.0))
        self.assertFalse(sim.active("NORTH", start + 10.0))
        self.assertFalse(sim.active("SOUTH", start + 4.0))


class TestLights(unittest.TestCase):
    def test_exactly_one_green(self):
        c = SignalController(LANES, min_green_s=0)
        d = c.decide(counts(NORTH=1), priority(), now=0.0)
        lights = c.lights(d)
        self.assertEqual(sum(1 for v in lights.values() if v == "GREEN"), 1)
        self.assertEqual(set(lights), set(LANES))

    def test_empty_lane_names_rejected(self):
        with self.assertRaises(ValueError):
            SignalController([])


if __name__ == "__main__":
    unittest.main(verbosity=2)
