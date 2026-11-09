"""Multi-provider LLM client: keys, provider detection, request formats, retries, cache."""
import pytest

from cqa import config
from cqa.llm import LLM, LLMError


def test_llm_requires_key_for_hosted_provider(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(LLMError):
        LLM(config.llm_settings("groq"))


def test_offline_llm_without_policy_raises():
    with pytest.raises(LLMError):
        LLM(config.LLMSettings(provider="offline", model="x")).chat([{"role": "user", "content": "hi"}])


def test_stop_sequences_enforced_and_cached(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(config, "LLM_CACHE", True)
    llm = LLM(config.LLMSettings(provider="openai", model="m", api_key="k", base_url="http://x"))
    calls = []
    monkeypatch.setattr(llm, "_call", lambda p: calls.append(p) or "Action: grep\nObservation: fake")
    msgs = [{"role": "user", "content": "q"}]
    assert llm.chat(msgs, stop=["\nObservation:"]) == "Action: grep"
    assert llm.chat(msgs, stop=["\nObservation:"]) == "Action: grep"
    assert len(calls) == 1 and llm.cache_hits == 1


def test_provider_autodetect(monkeypatch):
    monkeypatch.delenv("CQA_LLM_PROVIDER", raising=False)
    for k in ("GEMINI_API_KEY", "GROQ_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    assert config.llm_settings().provider == "offline"
    monkeypatch.setenv("GROQ_API_KEY", "x")
    s = config.llm_settings()
    assert s.provider == "groq" and s.api_key == "x" and "groq.com" in s.base_url


class _FakeResponse:
    def __init__(self, data, status=200):
        self._data, self.status_code, self.headers, self.text = data, status, {}, str(data)

    def raise_for_status(self):
        if self.status_code >= 400:
            import httpx
            raise httpx.HTTPStatusError("err", request=None, response=self)

    def json(self):
        return self._data


def test_openai_compatible_request_shape(monkeypatch):
    import httpx
    sent = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        sent.update(url=url, body=json, headers=headers)
        return _FakeResponse({"choices": [{"message": {"content": "pong"}}]})

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(config, "LLM_CACHE", False)
    s = config.LLMSettings(provider="groq", model="openai/gpt-oss-120b", api_key="k",
                           base_url="https://api.groq.com/openai/v1", reasoning_effort="low")
    assert LLM(s).chat([{"role": "user", "content": "ping"}], stop=["\nObservation:"]) == "pong"
    assert sent["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert sent["headers"]["Authorization"] == "Bearer k"
    assert sent["body"]["stop"] == ["\nObservation:"] and sent["body"]["reasoning_effort"] == "low"


def test_anthropic_request_shape(monkeypatch):
    import httpx
    sent = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        sent.update(url=url, body=json, headers=headers)
        return _FakeResponse({"content": [{"type": "text", "text": "hello"}]})

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(config, "LLM_CACHE", False)
    s = config.LLMSettings(provider="anthropic", model="claude-haiku-5-5", api_key="k",
                           base_url="https://api.anthropic.com/v1")
    out = LLM(s).chat([{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}], stop=["X"])
    assert out == "hello"
    assert sent["body"]["system"] == "sys" and sent["body"]["messages"] == [{"role": "user", "content": "hi"}]
    assert sent["body"]["stop_sequences"] == ["X"] and sent["headers"]["x-api-key"] == "k"


def test_retries_on_rate_limit(monkeypatch):
    import httpx
    import cqa.llm as llm_mod
    calls = []

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append(1)
        if len(calls) < 3:
            r = _FakeResponse({}, 429)
            r.headers = {"retry-after": "0"}
            return r
        return _FakeResponse({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(llm_mod.time, "sleep", lambda s: None)
    monkeypatch.setattr(config, "LLM_CACHE", False)
    s = config.LLMSettings(provider="openai", model="m", api_key="k", base_url="http://x")
    assert LLM(s).chat([{"role": "user", "content": "q"}]) == "ok" and len(calls) == 3
