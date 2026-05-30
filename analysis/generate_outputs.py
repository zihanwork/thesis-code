#!/usr/bin/env python3
"""Unified multi-vendor generator for EAI/VirtualHome experiments.

Reads an EAI ``helm_prompt.json`` file and writes outputs into the
benchmark layout:

    <out-dir>/<model>_<variant>_outputs.json

The script supports several providers behind a single CLI:

* ``openai``            - OpenAI Chat Completions
* ``anthropic``         - Anthropic Messages
* ``gemini``            - Google Generative AI
* ``openai_compatible`` - Any OpenAI-compatible endpoint (vLLM, Ollama,
                          DeepInfra, Together, ...). Requires ``--base-url``.
* ``dry_run``           - No network call. Returns a deterministic stub
                          for smoke-testing the rest of the pipeline.

The variant logic lives in :mod:`analysis.prompt_variants`.
"""
from __future__ import annotations

import os

# Set HF offline mode before any huggingface imports (when KB_BACKEND=persistent,
# the BGE model is already cached locally and needs no network access).
if os.environ.get("KB_BACKEND", "").lower() == "persistent":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import json
import os.path as osp
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

THIS_DIR = osp.dirname(osp.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from prompt_variants import PromptVariant, get_variant, list_variants


def _strip_code_fences(text: str) -> str:
    if not text:
        return ""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


@dataclass
class CallResult:
    text: str
    raw: Optional[Dict] = None


ProviderFn = Callable[[str, str, str, Dict], CallResult]


def _extract_openai_text(resp) -> str:
    """Read content; fall back to reasoning_content for reasoning models (e.g. Kimi-K2.5).
    Also strips inline <think>...</think> tags emitted by some MiniMax/MoE models."""
    import re
    msg = resp.choices[0].message
    text = (getattr(msg, "content", None) or "")
    if not text.strip():
        # Reasoning models (Kimi K2.5, DeepSeek-R1, etc.) put output in reasoning_content
        rc = getattr(msg, "reasoning_content", None)
        if rc is None and hasattr(msg, "model_dump"):
            rc = msg.model_dump().get("reasoning_content")
        text = rc or ""
    # Strip <think>...</think> blocks (MiniMax M2 family) — keep only content after </think>
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # If unclosed <think> swallows everything, drop it
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
    return text.strip()


def _provider_openai(
    api_model: str,
    system_prompt: str,
    user_prompt: str,
    options: Dict,
) -> CallResult:
    from openai import OpenAI

    client = options.get("_client") or OpenAI(
        api_key=options.get("api_key") or os.environ.get("OPENAI_API_KEY"),
        timeout=options.get("request_timeout", 60.0),
        max_retries=options.get("max_retries", 2),
    )
    options["_client"] = client
    resp = client.chat.completions.create(
        model=api_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt[: options.get("max_user_chars", 120000)]},
        ],
        temperature=options.get("temperature", 0),
        max_tokens=options.get("max_tokens", 2048),
    )
    text = _extract_openai_text(resp)
    return CallResult(text=_strip_code_fences(text))


def _provider_openai_compatible(
    api_model: str,
    system_prompt: str,
    user_prompt: str,
    options: Dict,
) -> CallResult:
    try:
        from openai import OpenAI

        client = options.get("_client") or OpenAI(
            api_key=options.get("api_key") or os.environ.get("OPENAI_API_KEY", "EMPTY"),
            base_url=options.get("base_url"),
            timeout=options.get("request_timeout", 60.0),
            max_retries=options.get("max_retries", 2),
        )
        options["_client"] = client
        resp = client.chat.completions.create(
            model=api_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt[: options.get("max_user_chars", 120000)]},
            ],
            temperature=options.get("temperature", 0),
            max_tokens=options.get("max_tokens", 2048),
        )
        text = _extract_openai_text(resp)
        return CallResult(text=_strip_code_fences(text))
    except ModuleNotFoundError:
        return _provider_openai_compatible_urllib(api_model, system_prompt, user_prompt, options)


def _provider_openai_compatible_urllib(
    api_model: str,
    system_prompt: str,
    user_prompt: str,
    options: Dict,
) -> CallResult:
    base_url = (options.get("base_url") or "").rstrip("/")
    if not base_url:
        raise ValueError("--base-url is required for openai_compatible without openai package")
    if base_url.endswith("/v1"):
        url = base_url + "/chat/completions"
    else:
        url = base_url + "/v1/chat/completions"
    api_key = options.get("api_key") or os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN") or "EMPTY"
    payload = {
        "model": api_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt[: options.get("max_user_chars", 120000)]},
        ],
        "temperature": options.get("temperature", 0),
        "max_tokens": options.get("max_tokens", 2048),
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=options.get("request_timeout", 120.0)) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI-compatible HTTP {exc.code}: {detail[:500]}") from exc
    blob = json.loads(body)
    choice = (blob.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    text = message.get("content") or message.get("reasoning_content") or ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
    return CallResult(text=_strip_code_fences(text.strip()), raw=blob)


def _provider_anthropic(
    api_model: str,
    system_prompt: str,
    user_prompt: str,
    options: Dict,
) -> CallResult:
    import anthropic

    client = options.get("_client") or anthropic.Anthropic(
        api_key=options.get("api_key") or os.environ.get("ANTHROPIC_API_KEY")
    )
    options["_client"] = client
    resp = client.messages.create(
        model=api_model,
        system=system_prompt,
        max_tokens=options.get("max_tokens", 2048),
        temperature=options.get("temperature", 0),
        messages=[{"role": "user", "content": user_prompt[: options.get("max_user_chars", 120000)]}],
    )
    text = "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")
    return CallResult(text=_strip_code_fences(text.strip()))


def _provider_gemini(
    api_model: str,
    system_prompt: str,
    user_prompt: str,
    options: Dict,
) -> CallResult:
    import google.generativeai as genai

    if not options.get("_configured"):
        genai.configure(api_key=options.get("api_key") or os.environ.get("GOOGLE_API_KEY"))
        options["_configured"] = True
    model = options.get("_client") or genai.GenerativeModel(
        model_name=api_model, system_instruction=system_prompt
    )
    options["_client"] = model
    resp = model.generate_content(
        user_prompt[: options.get("max_user_chars", 120000)],
        generation_config={
            "temperature": options.get("temperature", 0),
            "max_output_tokens": options.get("max_tokens", 2048),
        },
    )
    text = getattr(resp, "text", "") or ""
    return CallResult(text=_strip_code_fences(text.strip()))


def _provider_dry_run(
    api_model: str,
    system_prompt: str,
    user_prompt: str,
    options: Dict,
) -> CallResult:
    eval_type = options.get("eval_type")
    if eval_type == "action_sequencing":
        text = (
            '{"WALK":["floor_lamp","1000"]}{"SWITCHON":["floor_lamp","1000"]}'
        )
    else:
        text = json.dumps(
            {
                "node goals": [{"name": "floor_lamp", "state": "ON"}],
                "edge goals": [],
                "action goals": [
                    {"action": "SWITCHON", "description": "switch on the floor lamp"}
                ],
            },
            ensure_ascii=False,
        )
    return CallResult(text=text)


PROVIDERS: Dict[str, ProviderFn] = {
    "openai": _provider_openai,
    "openai_compatible": _provider_openai_compatible,
    "anthropic": _provider_anthropic,
    "gemini": _provider_gemini,
    "dry_run": _provider_dry_run,
}


def _generate_one(
    provider: str,
    api_model: str,
    variant: PromptVariant,
    user_prompt: str,
    options: Dict,
    eval_type: str,
    identifier: Optional[str] = None,
) -> str:
    options["eval_type"] = eval_type
    fn = PROVIDERS[provider]
    wrapped = variant.user_wrapper(user_prompt, identifier=identifier)
    first = fn(api_model, variant.system_prompt, wrapped, options)
    if not variant.requires_second_pass:
        return first.text

    critique_system = variant.critique_system_prompt or variant.system_prompt
    critique_user_fn = variant.critique_user_wrapper or (
        lambda prompt, draft, **_: f"Original prompt:\n{prompt}\n\nDraft:\n{draft}\n\nReturn ONLY the corrected output."
    )
    critique_prompt = critique_user_fn(user_prompt, first.text, identifier=identifier)
    second = fn(api_model, critique_system, critique_prompt, options)
    return second.text or first.text


def _outputs_filename(model: str, variant_label: str) -> str:
    if variant_label == "baseline":
        return f"{model}_outputs.json"
    return f"{model}_{variant_label}_outputs.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", required=True, choices=sorted(PROVIDERS.keys()))
    parser.add_argument("--api-model", required=True, help="Provider-specific model id, e.g. gpt-4o-mini")
    parser.add_argument("--model-name", required=True, help="Friendly model id used in output filenames")
    parser.add_argument("--variant", required=True, help="Prompt variant name (see prompt_variants.py)")
    parser.add_argument("--eval-type", choices=("action_sequencing", "goal_interpretation"), required=True)
    parser.add_argument("--helm-prompt", required=True, help="Path to EAI helm_prompt.json")
    parser.add_argument("--out-dir", required=True, help="Directory for <model>_<variant>_outputs.json")
    parser.add_argument("--max-prompts", type=int, default=10, help="0 means run the full file")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--sleep", type=float, default=0.4)
    parser.add_argument("--base-url", default=None, help="Required for openai_compatible")
    parser.add_argument("--api-key-env", default=None, help="Override the env var used for the API key")
    parser.add_argument("--list-variants", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip identifiers already present in the output file")
    args = parser.parse_args()

    if args.list_variants:
        for name in list_variants(args.eval_type):
            print(name)
        return

    variant = get_variant(args.eval_type, args.variant)

    api_key = None
    if args.api_key_env:
        api_key = os.environ.get(args.api_key_env)
    options: Dict = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "api_key": api_key,
        "base_url": args.base_url,
    }

    with open(args.helm_prompt, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = osp.join(args.out_dir, _outputs_filename(args.model_name, variant.label))

    existing: Dict[str, Dict] = {}
    if args.resume and osp.exists(out_path):
        try:
            existing_rows = json.load(open(out_path, "r", encoding="utf-8"))
            existing = {
                row["identifier"]: row
                for row in existing_rows
                if str(row.get("llm_output", "")).strip()
            }
        except Exception:
            existing = {}

    out_rows: List[Dict] = list(existing.values())
    seen = set(existing.keys())
    written = 0

    limit = args.max_prompts if args.max_prompts and args.max_prompts > 0 else len(prompts)
    for i, row in enumerate(prompts):
        if i >= limit:
            break
        identifier = str(row.get("identifier"))
        if identifier in seen:
            continue
        user_prompt = row.get("llm_prompt", "")
        try:
            text = _generate_one(
                args.provider,
                args.api_model,
                variant,
                user_prompt,
                options,
                args.eval_type,
                identifier=identifier,
            )
        except Exception as exc:
            print(f"[WARN] {identifier} failed: {exc}", file=sys.stderr)
            text = ""
        out_rows.append({"identifier": identifier, "llm_output": text})
        seen.add(identifier)
        written += 1
        if written % 20 == 0:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(out_rows, f, indent=2, ensure_ascii=False)
            print(f"[PROGRESS] {written} done ({len(out_rows)}/{limit}) -> {out_path}", flush=True)
        if args.sleep:
            time.sleep(args.sleep)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_rows, f, indent=2, ensure_ascii=False)
    print(f"[DONE] {written} new rows ({len(out_rows)} total) -> {out_path}")


if __name__ == "__main__":
    main()
