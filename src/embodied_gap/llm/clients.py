from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
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

    def parameters(self) -> dict[str, object]:
        return {"provider": self.provider, "model": self.model}

    def telemetry(self) -> dict[str, object]:
        return {
            "parameters": self.parameters(),
            "call_count": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "latency_seconds": 0.0,
            "estimated_cost_usd": 0.0,
            "calls": [],
        }


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
    input_cost_per_million: float | None = None
    output_cost_per_million: float | None = None
    call_history: list[dict[str, object]] = field(
        default_factory=list,
        compare=False,
        repr=False,
    )

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
        input_cost_per_million: float | None = None,
        output_cost_per_million: float | None = None,
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
            input_cost_per_million=input_cost_per_million,
            output_cost_per_million=output_cost_per_million,
        )

    def generate(self, prompt: str) -> str:
        started_at = _utc_now()
        started_clock = time.perf_counter()
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
        attempts = 0
        payload: dict[str, object] | None = None
        for attempt in range(self.max_attempts):
            attempts = attempt + 1
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
                error = RuntimeError(
                    f"One API request failed with HTTP {exc.code}: {message}"
                )
                self._record_call(
                    status="failed",
                    prompt=prompt,
                    started_at=started_at,
                    started_clock=started_clock,
                    attempts=attempts,
                    error=error,
                )
                raise error from exc
            except (urllib.error.URLError, socket.timeout, TimeoutError) as exc:
                last_error = RuntimeError(f"One API request failed: {exc}")
                if attempt < self.max_attempts - 1:
                    time.sleep(self.backoff_seconds * (2**attempt))
                    continue
                self._record_call(
                    status="failed",
                    prompt=prompt,
                    started_at=started_at,
                    started_clock=started_clock,
                    attempts=attempts,
                    error=last_error,
                )
                raise last_error from exc
        else:  # pragma: no cover - loop always breaks or raises.
            error = last_error or RuntimeError("One API request failed after retries.")
            self._record_call(
                status="failed",
                prompt=prompt,
                started_at=started_at,
                started_clock=started_clock,
                attempts=attempts,
                error=error,
            )
            raise error

        try:
            assert payload is not None
            choices = payload["choices"]
            content = choices[0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            error = RuntimeError(f"Unexpected One API response shape: {payload}")
            self._record_call(
                status="failed",
                prompt=prompt,
                started_at=started_at,
                started_clock=started_clock,
                attempts=attempts,
                payload=payload,
                error=error,
            )
            raise error from exc
        self._record_call(
            status="succeeded",
            prompt=prompt,
            started_at=started_at,
            started_clock=started_clock,
            attempts=attempts,
            payload=payload,
        )
        return str(content)

    def list_models(self) -> list[str]:
        """Return the model IDs exposed by the configured One API account."""

        request = urllib.request.Request(
            f"{self.base_url}/models",
            headers={"Authorization": f"Bearer {self.api_key}"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout,
                context=self._ssl_context(),
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"One API model discovery failed with HTTP {exc.code}: {message}"
            ) from exc
        except (urllib.error.URLError, socket.timeout, TimeoutError) as exc:
            raise RuntimeError(f"One API model discovery failed: {exc}") from exc

        rows = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            raise RuntimeError(f"Unexpected One API model-list response shape: {payload}")
        model_ids = {
            str(row["id"])
            for row in rows
            if isinstance(row, dict) and isinstance(row.get("id"), str)
        }
        return sorted(model_ids, key=str.casefold)

    def parameters(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout_seconds": self.timeout,
            "max_attempts": self.max_attempts,
            "backoff_seconds": self.backoff_seconds,
            "input_cost_per_million": self.input_cost_per_million,
            "output_cost_per_million": self.output_cost_per_million,
            "cost_currency": "USD",
        }

    def last_call_metadata(self) -> dict[str, object]:
        return dict(self.call_history[-1]) if self.call_history else {}

    def telemetry(self) -> dict[str, object]:
        prompt_tokens = sum(int(call.get("prompt_tokens", 0)) for call in self.call_history)
        completion_tokens = sum(
            int(call.get("completion_tokens", 0)) for call in self.call_history
        )
        total_tokens = sum(int(call.get("total_tokens", 0)) for call in self.call_history)
        latency_seconds = sum(
            float(call.get("latency_seconds", 0.0)) for call in self.call_history
        )
        priced_calls = [
            float(call["estimated_cost_usd"])
            for call in self.call_history
            if call.get("estimated_cost_usd") is not None
        ]
        pricing_configured = (
            self.input_cost_per_million is not None
            and self.output_cost_per_million is not None
        )
        return {
            "parameters": self.parameters(),
            "call_count": len(self.call_history),
            "successful_calls": sum(
                call.get("status") == "succeeded" for call in self.call_history
            ),
            "failed_calls": sum(call.get("status") == "failed" for call in self.call_history),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "latency_seconds": round(latency_seconds, 6),
            "estimated_cost_usd": round(sum(priced_calls), 10) if pricing_configured else None,
            "cost_status": "estimated" if pricing_configured else "pricing_not_configured",
            "calls": [dict(call) for call in self.call_history],
        }

    def _record_call(
        self,
        *,
        status: str,
        prompt: str,
        started_at: str,
        started_clock: float,
        attempts: int,
        payload: dict[str, object] | None = None,
        error: Exception | None = None,
    ) -> None:
        usage = payload.get("usage", {}) if isinstance(payload, dict) else {}
        if not isinstance(usage, dict):
            usage = {}
        prompt_tokens = _usage_int(usage, "prompt_tokens", "input_tokens")
        completion_tokens = _usage_int(usage, "completion_tokens", "output_tokens")
        total_tokens = _usage_int(usage, "total_tokens") or prompt_tokens + completion_tokens
        estimated_cost = None
        if (
            self.input_cost_per_million is not None
            and self.output_cost_per_million is not None
        ):
            estimated_cost = (
                prompt_tokens * self.input_cost_per_million
                + completion_tokens * self.output_cost_per_million
            ) / 1_000_000
        finish_reason = None
        if isinstance(payload, dict):
            choices = payload.get("choices")
            if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                finish_reason = choices[0].get("finish_reason")
        record: dict[str, object] = {
            "status": status,
            "provider": self.provider,
            "model": self.model,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "prompt_characters": len(prompt),
            "started_at": started_at,
            "completed_at": _utc_now(),
            "attempts": attempts,
            "latency_seconds": round(time.perf_counter() - started_clock, 6),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "estimated_cost_usd": (
                round(estimated_cost, 10) if estimated_cost is not None else None
            ),
            "response_id": payload.get("id") if isinstance(payload, dict) else None,
            "finish_reason": finish_reason,
            "error_type": type(error).__name__ if error else None,
            "error": str(error) if error else None,
        }
        self.call_history.append(record)

    def _ssl_context(self) -> ssl.SSLContext | None:
        if certifi is None:
            return None
        return ssl.create_default_context(cafile=certifi.where())


def last_call_metadata(client: object) -> dict[str, object]:
    getter = getattr(client, "last_call_metadata", None)
    if not callable(getter):
        return {}
    return dict(getter())


def client_telemetry(client: object) -> dict[str, object]:
    getter = getattr(client, "telemetry", None)
    if callable(getter):
        return dict(getter())
    return {
        "parameters": {
            "provider": getattr(client, "provider", "unknown"),
            "model": getattr(client, "model", "unknown"),
        },
        "call_count": 0,
        "successful_calls": 0,
        "failed_calls": 0,
        "calls": [],
    }


def _usage_int(usage: dict[str, object], *keys: str) -> int:
    for key in keys:
        value = usage.get(key)
        if isinstance(value, (int, float)):
            return int(value)
    return 0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
