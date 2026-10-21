"""
controller.py — the "brain".

Pure decision logic, deliberately free of OpenCV / YOLO so it can be
unit-tested and reasoned about on paper.

Rules, in priority order
------------------------
1. EMERGENCY  : if any lane has an ambulance/police/fire vehicle,
                that lane gets GREEN and everyone else gets RED.
                If two lanes have one, the lane with more priority
                vehicles wins (ties -> lane order).
2. HOLD       : after the emergency vehicle disappears, its lane stays
                GREEN for `emergency_hold_s` seconds so the convoy can
                clear the junction. (A real ambulance is followed by
                other vehicles.)
3. DENSITY    : otherwise the lane with the most vehicles gets GREEN, but only
                if it beats the current lane by `switch_margin` vehicles.
4. FAIRNESS   : a lane keeps GREEN at least `min_green_s` seconds to stop
                flickering, and an idle junction (0 vehicles everywhere)
                keeps whatever was green.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence


@dataclass
class Decision:
    green: str                    # lane name that gets GREEN
    reason: str                   # human readable, goes straight into the HUD
    emergency: bool = False       # emergency override currently active
    hold_remaining: float = 0.0   # seconds left of the post-emergency hold


class SignalController:
    """Decides which lane is green. No side effects, no I/O, no cv2."""

    def __init__(
        self,
        lane_names: Sequence[str] = ("NORTH", "SOUTH", "EAST", "WEST"),
        min_green_s: float = 3.0,
        emergency_hold_s: float = 6.0,
        switch_margin: int = 2,
    ) -> None:
        if not lane_names:
            raise ValueError("lane_names cannot be empty")
        self.lane_names: List[str] = list(lane_names)
        self.min_green_s = float(min_green_s)
        self.emergency_hold_s = float(emergency_hold_s)
        self.switch_margin = max(0, int(switch_margin))

        self.current: str = self.lane_names[0]
        self._changed_at: Optional[float] = None     # None = "just started"
        self._hold_lane: Optional[str] = None
        self._hold_until: float = 0.0
        self.history: List[Decision] = []

    # ------------------------------------------------------------------ #
    def _switch(self, lane: str, now: float) -> None:
        if lane != self.current:
            self.current = lane
            self._changed_at = now

    def _elapsed(self, now: float) -> float:
        if self._changed_at is None:
            self._changed_at = now
        return now - self._changed_at

    # ------------------------------------------------------------------ #
    def decide(
        self,
        counts: Mapping[str, int],
        priority: Mapping[str, int],
        now: float,
    ) -> Decision:
        """Return the lane that should be green at time `now` (seconds)."""
        elapsed = self._elapsed(now)

        # ---- 1. live emergency -------------------------------------- #
        with_priority = [l for l in self.lane_names if priority.get(l, 0) > 0]
        if with_priority:
            best = max(
                with_priority,
                key=lambda l: (priority.get(l, 0), -self.lane_names.index(l)),
            )
            self._switch(best, now)
            self._hold_lane = best
            self._hold_until = now + self.emergency_hold_s
            return self._record(
                Decision(best, f"EMERGENCY vehicle on {best}", True, self.emergency_hold_s)
            )

        # ---- 2. post-emergency hold --------------------------------- #
        if self._hold_lane and now < self._hold_until:
            self._switch(self._hold_lane, now)
            return self._record(
                Decision(
                    self._hold_lane,
                    "Emergency hold (letting the convoy clear)",
                    True,
                    self._hold_until - now,
                )
            )
        self._hold_lane = None

        # ---- 3. density --------------------------------------------- #
        busiest = max(self.lane_names, key=lambda l: counts.get(l, 0))

        # ---- 4. fairness -------------------------------------------- #
        if counts.get(busiest, 0) == 0:
            return self._record(Decision(self.current, "No traffic — holding last green"))

        if elapsed < self.min_green_s:
            return self._record(
                Decision(
                    self.current,
                    f"Minimum green ({self.min_green_s:.0f}s) not over yet",
                )
            )

        # hysteresis: only take over if clearly busier, otherwise the light
        # would flip between two lanes on every single detection wobble
        if busiest != self.current and (
            counts.get(busiest, 0) >= counts.get(self.current, 0) + self.switch_margin
        ):
            self._switch(busiest, now)
            return self._record(
                Decision(busiest, f"Busiest lane ({counts.get(busiest, 0)} vehicles)")
            )

        if counts.get(self.current, 0) >= counts.get(busiest, 0):
            return self._record(
                Decision(self.current, f"Still busiest ({counts.get(self.current, 0)} vehicles)")
            )
        return self._record(
            Decision(
                self.current,
                f"Holding green (lead too small: {counts.get(self.current, 0)} vs "
                f"{counts.get(busiest, 0)})",
            )
        )

    # ------------------------------------------------------------------ #
    def _record(self, decision: Decision) -> Decision:
        if not self.history or self.history[-1] != decision:
            self.history.append(decision)
        return decision

    def lights(self, decision: Optional[Decision] = None) -> Dict[str, str]:
        """Map lane -> 'GREEN' / 'RED'."""
        green = (decision or self.history[-1]).green if decision or self.history else self.current
        return {lane: ("GREEN" if lane == green else "RED") for lane in self.lane_names}
