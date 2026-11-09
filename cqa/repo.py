"""Repository ingestion: cloning GitHub repos, walking source files, building links."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import config

GITHUB_RE = re.compile(r"(?:https?://)?(?:www\.)?github\.com/([\w.-]+)/([\w.-]+?)(?:\.git)?/?$")


@dataclass
class RepoSpec:
    name: str                      # short id used everywhere (e.g. "requests")
    url: str                       # https://github.com/owner/repo
    ref: str = ""                  # branch / tag / commit ("" = default branch)
    commit: str = ""               # resolved commit SHA after cloning
    exclude_dirs: list[str] = field(default_factory=list)
    include_dirs: list[str] = field(default_factory=list)  # empty = whole repo
    description: str = ""

    @property
    def path(self) -> Path:
        return config.REPOS_DIR / self.name

    @property
    def owner_repo(self) -> tuple[str, str] | None:
        m = GITHUB_RE.match(self.url.strip())
        return (m.group(1), m.group(2)) if m else None

    def github_url(self, rel_path: str, start: int | None = None, end: int | None = None) -> str:
        """Permalink to a file (and line range) pinned to the indexed commit."""
        orr = self.owner_repo
        if not orr:
            return ""
        ref = self.commit or self.ref or "HEAD"
        url = f"https://github.com/{orr[0]}/{orr[1]}/blob/{ref}/{rel_path}"
        if start:
            url += f"#L{start}" + (f"-L{end}" if end and end != start else "")
        return url

    def to_dict(self) -> dict:
        return asdict(self)


def parse_github_url(url: str) -> tuple[str, str]:
    m = GITHUB_RE.match(url.strip())
    if not m:
        raise ValueError(f"Not a GitHub repository URL: {url!r}")
    return m.group(1), m.group(2)


# --------------------------------------------------------------------------
# Registry of known repos (the 3 evaluation repos + any added from the UI/CLI)
# --------------------------------------------------------------------------
def _load_json(path: Path) -> list[dict]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return []


def load_registry() -> dict[str, RepoSpec]:
    specs: dict[str, RepoSpec] = {}
    for item in _load_json(config.EVAL_REPOS_FILE) + _load_json(config.REGISTRY_FILE):
        spec = RepoSpec(**{k: v for k, v in item.items() if k in RepoSpec.__dataclass_fields__})
        specs[spec.name] = spec
    return specs


def save_to_registry(spec: RepoSpec) -> None:
    items = [i for i in _load_json(config.REGISTRY_FILE) if i.get("name") != spec.name]
    items.append(spec.to_dict())
    config.REGISTRY_FILE.write_text(json.dumps(items, indent=2), encoding="utf-8")


def get_repo(name: str) -> RepoSpec:
    reg = load_registry()
    if name not in reg:
        raise KeyError(f"Unknown repo {name!r}. Known: {sorted(reg)}")
    spec = reg[name]
    if not spec.commit and (spec.path / ".git").exists():
        spec.commit = current_commit(spec.path)
    return spec


def spec_from_url(url: str, ref: str = "", name: str = "") -> RepoSpec:
    owner, repo = parse_github_url(url)
    safe = re.sub(r"[^\w.-]", "-", name or repo).lower()
    return RepoSpec(name=safe, url=f"https://github.com/{owner}/{repo}", ref=ref)


# --------------------------------------------------------------------------
# Cloning
# --------------------------------------------------------------------------
def _git(*args: str, cwd: Path | None = None) -> str:
    out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=600)
    if out.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {out.stderr.strip()}")
    return out.stdout.strip()


def current_commit(path: Path) -> str:
    try:
        return _git("rev-parse", "HEAD", cwd=path)
    except Exception:
        return ""


def ensure_cloned(spec: RepoSpec, force: bool = False) -> RepoSpec:
    """Shallow-clone the repo at `spec.ref` (tag/branch or commit SHA)."""
    dest = spec.path
    if force and dest.exists():
        shutil.rmtree(dest)
    if not (dest / ".git").exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            shutil.rmtree(dest)
        is_sha = bool(re.fullmatch(r"[0-9a-f]{7,40}", spec.ref or ""))
        if spec.ref and not is_sha:
            _git("clone", "--depth", "1", "--branch", spec.ref, spec.url, str(dest))
        else:
            _git("clone", "--depth", "1" if not is_sha else "50", spec.url, str(dest))
            if is_sha:
                try:
                    _git("checkout", spec.ref, cwd=dest)
                except RuntimeError:
                    _git("fetch", "--depth", "1", "origin", spec.ref, cwd=dest)
                    _git("checkout", spec.ref, cwd=dest)
    spec.commit = current_commit(dest)
    return spec


# --------------------------------------------------------------------------
# Walking source files
# --------------------------------------------------------------------------
def language_for(path: str | Path) -> str | None:
    return config.LANGUAGE_BY_EXT.get(Path(path).suffix.lower())


def iter_source_files(spec: RepoSpec) -> list[Path]:
    """Return repo-relative paths of indexable source files (sorted, deterministic)."""
    root = spec.path
    excluded = set(config.DEFAULT_EXCLUDE_DIRS) | set(spec.exclude_dirs)
    roots = [root / d for d in spec.include_dirs] if spec.include_dirs else [root]
    files: list[Path] = []
    for base in roots:
        if base.is_file():
            files.append(base.relative_to(root))
            continue
        for p in base.rglob("*"):
            rel = p.relative_to(root)
            if any(part in excluded or part.startswith(".") for part in rel.parts[:-1]):
                continue
            if not p.is_file() or language_for(p) is None:
                continue
            if p.stat().st_size > config.MAX_FILE_BYTES:
                continue
            files.append(rel)
    return sorted(set(files), key=lambda x: x.as_posix())


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def safe_join(spec: RepoSpec, rel_path: str) -> Path:
    """Resolve a repo-relative path and refuse anything that escapes the repo."""
    root = spec.path.resolve()
    target = (root / rel_path.lstrip("/")).resolve()
    if root != target and root not in target.parents:
        raise PermissionError(f"Path escapes repository: {rel_path}")
    return target
