from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re
import subprocess

try:
    import tomllib as _toml
except ModuleNotFoundError:  # pragma: no cover - fallback for older Python
    try:
        import tomli as _toml  # type: ignore
    except ModuleNotFoundError:  # pragma: no cover - no TOML parser available
        _toml = None  # type: ignore


def _find_project_root(start: Path) -> Path:
    for base in [start, *start.parents]:
        if (base / ".git").exists() or (base / "pyproject.toml").exists():
            return base
    return start


def _parse_pyproject_version(pyproject_path: Path) -> str | None:
    try:
        raw = pyproject_path.read_bytes()
    except Exception:
        return None

    if _toml is not None:
        try:
            data = _toml.loads(raw.decode("utf-8"))
            project = data.get("project") or {}
            if isinstance(project, dict) and project.get("version"):
                return str(project["version"])
            poetry = (data.get("tool") or {}).get("poetry") or {}
            if isinstance(poetry, dict) and poetry.get("version"):
                return str(poetry["version"])
        except Exception:
            return None

    text = raw.decode("utf-8", errors="replace")
    return _parse_version_fallback(text)


def _parse_version_fallback(text: str) -> str | None:
    def _find_in_section(section: str) -> str | None:
        header = re.search(rf"^\[{re.escape(section)}\]\s*$", text, flags=re.M)
        if not header:
            return None
        start = header.end()
        next_header = re.search(r"^\[.*\]\s*$", text[start:], flags=re.M)
        end = start + (next_header.start() if next_header else len(text))
        section_text = text[start:end]
        match = re.search(r'^\s*version\s*=\s*["\']([^"\']+)["\']', section_text, flags=re.M)
        return match.group(1) if match else None

    return _find_in_section("project") or _find_in_section("tool.poetry")


def _read_git_sha(repo_root: Path) -> str | None:
    try:
        p = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=True,
        )
        value = (p.stdout or "").strip()
        return value or None
    except Exception:
        return None


@lru_cache(maxsize=1)
def get_build_info() -> str:
    start = Path(__file__).resolve()
    repo_root = _find_project_root(start)
    pyproject = repo_root / "pyproject.toml"
    if pyproject.exists():
        version = _parse_pyproject_version(pyproject)
        if version:
            return version

    git_sha = _read_git_sha(repo_root)
    return git_sha or "unknown"
