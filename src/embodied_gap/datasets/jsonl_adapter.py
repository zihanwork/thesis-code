from __future__ import annotations

from pathlib import Path

from embodied_gap.core.task_schema import Task, load_tasks


class JsonlTaskAdapter:
    """Adapter for the repository's canonical JSONL task format."""

    def load(self, path: str | Path) -> list[Task]:
        return load_tasks(path)
