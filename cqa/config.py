"""Central configuration for the Codebase QA assistant.

Everything can be overridden with environment variables or a `.env` file in the
project root (see `.env.example`).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Tiny .env loader (avoids an extra dependency). Existing env vars win."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = Path(os.environ.get("CQA_DATA_DIR", PROJECT_ROOT / "data"))
REPOS_DIR = DATA_DIR / "repos"
INDEX_DIR = DATA_DIR / "indexes"
CACHE_DIR = DATA_DIR / "cache"
RESULTS_DIR = Path(os.environ.get("CQA_RESULTS_DIR", PROJECT_ROOT / "results"))
REGISTRY_FILE = DATA_DIR / "registry.json"
EVAL_REPOS_FILE = PROJECT_ROOT / "eval" / "repos.json"

# File types we index. Python uses the stdlib `ast`; the others use tree-sitter.
LANGUAGE_BY_EXT = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".php": "php",
    ".kt": "kotlin",
}

# Directories skipped while walking a repository.
DEFAULT_EXCLUDE_DIRS = {
    ".git", ".github", "node_modules", "venv", ".venv", "env", "__pycache__",
    "build", "dist", "site-packages", ".tox", ".mypy_cache", ".pytest_cache",
    "docs", "doc", "tests", "test", "testing", "examples", "example", "benchmarks",
    "vendor", "third_party",
}
MAX_FILE_BYTES = 400_000

# --- chunking -------------------------------------------------------------
AST_MAX_LINES = int(os.environ.get("CQA_AST_MAX_LINES", 80))     # split classes bigger than this
AST_SPLIT_LONG = int(os.environ.get("CQA_AST_SPLIT_LONG", 160))  # window very long functions
FIXED_CHUNK_LINES = int(os.environ.get("CQA_FIXED_CHUNK_LINES", 40))
FIXED_OVERLAP_LINES = int(os.environ.get("CQA_FIXED_OVERLAP_LINES", 10))


# --- embeddings -----------------------------------------------------------
@dataclass(frozen=True)
class EmbedderSpec:
    key: str
    model: str
    description: str
    query_prefix: str = ""
    doc_prefix: str = ""
    max_chars: int = 2000   # ~600 tokens; keeps CPU memory low for long-context models


EMBEDDERS: dict[str, EmbedderSpec] = {
    "code": EmbedderSpec(
        key="code",
        model=os.environ.get("CQA_CODE_EMBED_MODEL", "jinaai/jina-embeddings-v2-base-code"),
        description="Code-specific model (Jina v2 base code, trained on code/text pairs)",
    ),
    "general": EmbedderSpec(
        key="general",
        model=os.environ.get("CQA_GENERAL_EMBED_MODEL", "BAAI/bge-base-en-v1.5"),
        description="General-purpose English text model (BGE base v1.5)",
        query_prefix="Represent this sentence for searching relevant passages: ",
    ),
}
# Retrievers = dense embedders + lexical / hybrid baselines.
RETRIEVERS = ["code", "general", "bm25", "hybrid"]
CHUNKERS = ["ast", "fixed"]
DEFAULT_CHUNKER = os.environ.get("CQA_DEFAULT_CHUNKER", "ast")
DEFAULT_RETRIEVER = os.environ.get("CQA_DEFAULT_RETRIEVER", "code")

# "fastembed" (real models, default) or "hash" (offline hashing embedder used by
# the unit tests and by machines without internet access).
EMBED_BACKEND = os.environ.get("CQA_EMBED_BACKEND", "fastembed").lower()


# --- LLM ------------------------------------------------------------------
@dataclass
class LLMSettings:
    provider: str = ""
    model: str = ""
    api_key: str = ""
    base_url: str = ""
    temperature: float = 0.0
    max_tokens: int = 4096
    reasoning_effort: str = ""     # for "thinking" models (Gemini 3 Flash, gpt-oss): keep it low
    extra: dict = field(default_factory=dict)


LLM_PRESETS = {
    # provider: (base_url, default model, env var holding the key)
# Model ids change often; override with CQA_LLM_MODEL. Checked Nov 2026: Gemini 2.5 Flash and
# Groq llama-3.3-70b were retired, so the defaults are the "latest Flash" alias and gpt-oss-120b.
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "gemini-flash-latest", "GEMINI_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "openai/gpt-oss-120b", "GROQ_API_KEY"),
    "openai": ("https://api.openai.com/v1", "gpt-4o-mini", "OPENAI_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "qwen/qwen-2.5-coder-32b-instruct", "OPENROUTER_API_KEY"),
    "ollama": ("http://localhost:11434/v1", "qwen2.5-coder:7b", ""),
    "anthropic": ("https://api.anthropic.com/v1", "claude-haiku-5-5", "ANTHROPIC_API_KEY"),
    "offline": ("", "heuristic", ""),
}


def llm_settings(provider: str | None = None, model: str | None = None) -> LLMSettings:
    """Resolve which LLM to use. Order: explicit arg > CQA_LLM_PROVIDER > first key found."""
    provider = (provider or os.environ.get("CQA_LLM_PROVIDER", "")).lower().strip()
    if not provider:
        for name in ("gemini", "groq", "openai", "anthropic", "openrouter"):
            if os.environ.get(LLM_PRESETS[name][2]):
                provider = name
                break
        else:
            provider = "offline"
    if provider not in LLM_PRESETS:
        raise ValueError(f"Unknown LLM provider {provider!r}. Choose from {sorted(LLM_PRESETS)}")
    base_url, default_model, key_env = LLM_PRESETS[provider]
    return LLMSettings(
        provider=provider,
        model=model or os.environ.get("CQA_LLM_MODEL") or default_model,
        api_key=os.environ.get(key_env, "") if key_env else "",
        base_url=os.environ.get("CQA_LLM_BASE_URL", base_url),
        temperature=float(os.environ.get("CQA_LLM_TEMPERATURE", 0.0)),
        max_tokens=int(os.environ.get("CQA_LLM_MAX_TOKENS", 4096)),
        reasoning_effort=os.environ.get("CQA_REASONING_EFFORT", "low") if provider in ("gemini", "groq") else "",
    )


def judge_settings() -> LLMSettings:
    """The judge can use a different (ideally stronger) model than the agent."""
    return llm_settings(
        os.environ.get("CQA_JUDGE_PROVIDER") or None,
        os.environ.get("CQA_JUDGE_MODEL") or None,
    )


AGENT_MAX_STEPS = int(os.environ.get("CQA_AGENT_MAX_STEPS", 8))
LLM_CACHE = os.environ.get("CQA_LLM_CACHE", "1") != "0"

for _d in (DATA_DIR, REPOS_DIR, INDEX_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)
