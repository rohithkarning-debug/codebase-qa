"""Minimal multi-provider chat client (no vendor SDKs needed).

Providers
---------
* OpenAI-compatible HTTP API: gemini, groq, openai, openrouter, ollama
* Anthropic Messages API: anthropic
* offline: a deterministic stand-in used when no API key is configured, so the
  UI, the agent loop and the tests still run end-to-end.

Responses are cached on disk (data/cache/llm) keyed by the full request, so
re-running an evaluation or a demo question costs nothing.
"""
from __future__ import annotations

import hashlib
import json
import random
import time
from typing import Callable

import httpx

from . import config

Message = dict  # {"role": "system"|"user"|"assistant", "content": str}


class LLMError(RuntimeError):
    pass


class LLM:
    def __init__(self, settings: config.LLMSettings | None = None,
                 offline_policy: Callable[[list[Message]], str] | None = None):
        self.s = settings or config.llm_settings()
        self.offline_policy = offline_policy
        self.calls = 0
        self.cache_hits = 0
        if self.s.provider not in ("offline", "ollama") and not self.s.api_key:
            raise LLMError(f"No API key for provider {self.s.provider!r}. "
                           f"Set {config.LLM_PRESETS[self.s.provider][2]} in .env")

    @property
    def name(self) -> str:
        return f"{self.s.provider}:{self.s.model}"

    @property
    def is_offline(self) -> bool:
        return self.s.provider == "offline"

    # ------------------------------------------------------------------ API
    def chat(self, messages: list[Message], stop: list[str] | None = None,
             max_tokens: int | None = None, temperature: float | None = None,
             json_mode: bool = False) -> str:
        if self.is_offline:
            if not self.offline_policy:
                raise LLMError("Offline mode has no policy for this task; configure an LLM API key.")
            return self.offline_policy(messages)

        payload = {
            "messages": messages, "stop": stop or [], "model": self.s.model,
            "max_tokens": max_tokens or self.s.max_tokens,
            "temperature": self.s.temperature if temperature is None else temperature,
            "json": json_mode, "provider": self.s.provider,
        }
        key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        cache_f = config.CACHE_DIR / "llm" / f"{key}.json"
        if config.LLM_CACHE and cache_f.exists():
            self.cache_hits += 1
            return json.loads(cache_f.read_text())["text"]

        text = self._with_retries(lambda: self._call(payload))
        if stop:  # some providers ignore stop sequences: enforce them client-side
            for s in stop:
                if s in text:
                    text = text.split(s)[0]
        self.calls += 1
        if config.LLM_CACHE:
            cache_f.parent.mkdir(parents=True, exist_ok=True)
            cache_f.write_text(json.dumps({"text": text, "model": self.name}))
        return text

    # ------------------------------------------------------------ internals
    def _with_retries(self, fn, attempts: int = 6) -> str:
        delay = 2.0
        for i in range(attempts):
            try:
                return fn()
            except httpx.HTTPStatusError as e:
                code = e.response.status_code
                if code in (429, 500, 502, 503, 504, 529) and i < attempts - 1:
                    wait = float(e.response.headers.get("retry-after", delay) or delay)
                    time.sleep(min(wait, 60) + random.random())
                    delay *= 2
                    continue
                raise LLMError(f"{self.name} HTTP {code}: {e.response.text[:300]}") from e
            except (httpx.TransportError, httpx.TimeoutException) as e:
                if i < attempts - 1:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise LLMError(f"{self.name} connection failed: {e}") from e
        raise LLMError("unreachable")

    def _call(self, p: dict) -> str:
        if self.s.provider == "anthropic":
            return self._anthropic(p)
        return self._openai_compatible(p)

    def _openai_compatible(self, p: dict) -> str:
        body = {"model": p["model"], "messages": p["messages"], "temperature": p["temperature"],
                "max_tokens": p["max_tokens"]}
        if p["stop"]:
            body["stop"] = p["stop"][:4]
        if p["json"]:
            body["response_format"] = {"type": "json_object"}
        if self.s.reasoning_effort:
            body["reasoning_effort"] = self.s.reasoning_effort
        headers = {"Content-Type": "application/json"}
        if self.s.api_key:
            headers["Authorization"] = f"Bearer {self.s.api_key}"
        r = httpx.post(f"{self.s.base_url.rstrip('/')}/chat/completions", json=body,
                       headers=headers, timeout=120)
        r.raise_for_status()
        data = r.json()
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError) as e:
            raise LLMError(f"Unexpected response: {str(data)[:300]}") from e

    def _anthropic(self, p: dict) -> str:
        system = "\n\n".join(m["content"] for m in p["messages"] if m["role"] == "system")
        msgs = [m for m in p["messages"] if m["role"] != "system"]
        body = {"model": p["model"], "max_tokens": p["max_tokens"], "messages": msgs,
                "temperature": p["temperature"]}
        if system:
            body["system"] = system
        if p["stop"]:
            body["stop_sequences"] = p["stop"]
        r = httpx.post(f"{self.s.base_url.rstrip('/')}/messages", json=body, timeout=120, headers={
            "x-api-key": self.s.api_key, "anthropic-version": "2023-06-01",
            "content-type": "application/json"})
        r.raise_for_status()
        return "".join(b.get("text", "") for b in r.json().get("content", []))


def make_llm(provider: str | None = None, model: str | None = None,
             offline_policy=None) -> LLM:
    return LLM(config.llm_settings(provider, model), offline_policy=offline_policy)
