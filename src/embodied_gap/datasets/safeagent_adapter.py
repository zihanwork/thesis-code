from __future__ import annotations


class SafeAgentAdapter:
    def load(self, path: str):
        raise NotImplementedError("SafeAgentBench conversion requires task and safety annotations.")
