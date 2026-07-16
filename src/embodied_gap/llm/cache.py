from __future__ import annotations

import hashlib
import json
from pathlib import Path


class JsonPromptCache:
    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def key(self, prompt: str, model: str) -> str:
        digest = hashlib.sha256(f"{model}\n{prompt}".encode("utf-8")).hexdigest()
        return digest

    def get(self, prompt: str, model: str) -> str | None:
        path = self.cache_dir / f"{self.key(prompt, model)}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))["response"]

    def set(self, prompt: str, model: str, response: str) -> None:
        path = self.cache_dir / f"{self.key(prompt, model)}.json"
        path.write_text(
            json.dumps({"model": model, "prompt": prompt, "response": response}, indent=2),
            encoding="utf-8",
        )
