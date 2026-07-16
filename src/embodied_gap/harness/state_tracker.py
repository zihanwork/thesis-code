from __future__ import annotations

from dataclasses import dataclass

from embodied_gap.core.state_schema import WorldState


@dataclass
class StateTracker:
    current: WorldState

    def update(self, state: WorldState) -> None:
        self.current = state
