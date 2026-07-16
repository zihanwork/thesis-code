from __future__ import annotations

from dataclasses import dataclass
import json
import os
import socket
import ssl
import time
import urllib.error
import urllib.request
from typing import Protocol

from .env import load_dotenv

try:
    import certifi
except ImportError:  # pragma: no cover - default SSL context remains valid.
    certifi = None


class LLMClient(Protocol):
    provider: str
    model: str

    def generate(self, prompt: str) -> str:
        """Generate text from a prompt."""


@dataclass(frozen=True)
class MockLLMClient:
    """Offline deterministic client used for reproducible development tests."""

    provider: str = "mock"
    model: str = "mock-planner"

    def generate(self, prompt: str) -> str:
        if "safety" in prompt.lower() and "microwave" in prompt.lower():
            return '["reject()"]'
        return "[]"


@dataclass(frozen=True)
class OneAPIChatClient:
    """OpenAI-compatible chat client for One API deployments."""

    api_key: str
    base_url: str
    model: str
    provider: str = "one_api"
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout: int = 180
    max_attempts: int = 4
    backoff_seconds: float = 2.0

    @classmethod
    def from_env(
        cls,
        env_path: str = ".env",
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: int | None = None,
        max_attempts: int | None = None,
        backoff_seconds: float | None = None,
    ) -> "OneAPIChatClient":
        load_dotenv(env_path)
        api_key = os.getenv("ONE_API_KEY", "")
        base_url = os.getenv("ONE_API_BASE_URL", "")
        selected_model = model or os.getenv("ONE_API_MODEL", "")
        missing = [
            name
            for name, value in {
                "ONE_API_KEY": api_key,
                "ONE_API_BASE_URL": base_url,
                "ONE_API_MODEL": selected_model,
            }.items()
            if not value
        ]
        if missing:
            raise RuntimeError(f"Missing One API environment variables: {', '.join(missing)}")
        return cls(
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            model=selected_model,
            temperature=temperature if temperature is not None else cls.temperature,
            max_tokens=max_tokens if max_tokens is not None else cls.max_tokens,
            timeout=timeout if timeout is not None else cls.timeout,
            max_attempts=max_attempts if max_attempts is not None else cls.max_attempts,
            backoff_seconds=backoff_seconds if backoff_seconds is not None else cls.backoff_seconds,
        )

    def generate(self, prompt: str) -> str:
        url = f"{self.base_url}/chat/completions"
        body = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an embodied planning module. Return only valid JSON "
                        "when the user requests a plan."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(self.max_attempts):
            try:
                with urllib.request.urlopen(
                    request,
                    timeout=self.timeout,
                    context=self._ssl_context(),
                ) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                message = exc.read().decode("utf-8", errors="replace")
                if exc.code in (429, 500, 502, 503, 504) and attempt < self.max_attempts - 1:
                    last_error = RuntimeError(
                        f"One API request failed with HTTP {exc.code}: {message}"
                    )
                    time.sleep(self.backoff_seconds * (2**attempt))
                    continue
                raise RuntimeError(
                    f"One API request failed with HTTP {exc.code}: {message}"
                ) from exc
            except (urllib.error.URLError, socket.timeout, TimeoutError) as exc:
                last_error = RuntimeError(f"One API request failed: {exc}")
                if attempt < self.max_attempts - 1:
                    time.sleep(self.backoff_seconds * (2**attempt))
                    continue
                raise last_error from exc
        else:  # pragma: no cover - loop always breaks or raises.
            raise last_error or RuntimeError("One API request failed after retries.")

        try:
            return payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected One API response shape: {payload}") from exc

    def _ssl_context(self) -> ssl.SSLContext | None:
        if certifi is None:
            return None
        return ssl.create_default_context(cafile=certifi.where())
