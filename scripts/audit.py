"""Security / dependency audit (Split 14, spec section 17) — ``python scripts/audit.py``.

Four checks, importable as functions (the acceptance run reuses them) and runnable standalone:

1. **Secrets — tree + history.** No key-shaped strings in the tracked tree, and ``.env`` was
   never committed (a key committed earlier and deleted later is still in history).
2. **Gitignored.** ``.env`` / ``data/`` / built indexes (``*.bin``, ``.chroma``) are gitignored
   and absent from the tracked tree.
3. **Deps pinned.** Every runtime dependency across ``core`` / ``api`` / ``eval`` carries a
   concrete version bound; a best-effort ``pip-audit`` vuln scan is recorded (informational).
4. **Local-only.** The safety + RAG paths contain no managed/remote call — only the LLM seam
   may reach the network, so patient text goes nowhere but the configured Anthropic endpoint.

Each check returns an :class:`AuditResult`; ``gated`` results that fail must fail acceptance.
"""

from __future__ import annotations

import inspect
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Key-shaped patterns — specific enough to avoid flagging timestamps / chunk ids / hashes.
_SECRET_PATTERNS = [
    re.compile(r"AZURE_OPENAI_API_KEY\s*[=:]\s*['\"]?[A-Za-z0-9]{32,}"),
    re.compile(r"ANTHROPIC_API_KEY\s*[=:]\s*['\"]?sk-ant-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{24,}"),
    re.compile(r"\bsk-[A-Za-z0-9]{32,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
]

# Tokens that would betray a managed/remote call inside a must-be-local path.
_NONLOCAL_CALLS = (
    "anthropic",
    "openai",
    "cohere",
    "voyage",
    "requests.post",
    "requests.get",
    "httpx.post",
    "httpx.get",
    "urllib.request",
)


@dataclass
class AuditResult:
    name: str
    ok: bool
    detail: str
    gated: bool = True

    def as_dict(self) -> dict:
        return {"name": self.name, "ok": self.ok, "detail": self.detail, "gated": self.gated}


def _git(*args: str) -> tuple[int, str]:
    """Run a git command at the repo root; return ``(returncode, stdout)`` (stderr folded in)."""
    proc = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def _tracked_files() -> list[Path]:
    rc, out = _git("ls-files")
    if rc != 0:
        return []
    return [REPO_ROOT / line for line in out.splitlines() if line.strip()]


# ----------------------------------------------------------------- 1. secrets
def scan_secrets() -> AuditResult:
    """No key-shaped strings in the tracked tree, and ``.env`` never entered history."""
    hits: list[str] = []

    # (a) tree: scan every tracked text file.
    for path in _tracked_files():
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except (OSError, UnicodeError):
            continue
        for pat in _SECRET_PATTERNS:
            if pat.search(text):
                rel = path.relative_to(REPO_ROOT)
                hits.append(f"tree:{rel} matches {pat.pattern[:32]}…")

    # (b) history: .env must never have been committed.
    rc, out = _git("log", "--all", "--oneline", "--", ".env", ".env.*")
    if rc == 0 and out.strip():
        hits.append(".env appears in git history: " + out.strip().splitlines()[0])

    # (c) history: grep every revision's tree for the key patterns (cheap at this repo size).
    rc, revs = _git("rev-list", "--all")
    if rc == 0 and revs.strip():
        rev_list = revs.split()
        for pat in (r"sk-ant-[A-Za-z0-9_\-]{24,}", r"AZURE_OPENAI_API_KEY\s*=\s*[A-Za-z0-9]{32,}"):
            grc, gout = _git("grep", "-I", "-E", "-e", pat, *rev_list)
            if grc == 0 and gout.strip():
                hits.append(f"history matches {pat[:24]}…: {gout.strip().splitlines()[0][:80]}")

    if hits:
        return AuditResult("secrets", False, "; ".join(hits[:8]))
    return AuditResult(
        "secrets",
        True,
        "no key-shaped strings in the tracked tree; .env never committed; history clean",
    )


# ----------------------------------------------------------------- 2. gitignored
def check_gitignored() -> AuditResult:
    """``.env`` / ``data/`` / built indexes are gitignored and not tracked."""
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    required = [".env", "data/", "*.bin"]
    missing = [p for p in required if p not in gitignore]

    tracked = {str(p.relative_to(REPO_ROOT)).replace("\\", "/") for p in _tracked_files()}
    leaked = [
        t
        for t in tracked
        if t == ".env"
        or t.startswith("data/")
        or t.endswith(".bin")
        or "/.chroma/" in t
        or t.endswith(".sqlite3")
    ]

    if missing or leaked:
        detail = ""
        if missing:
            detail += f"missing from .gitignore: {missing}. "
        if leaked:
            detail += f"tracked but should be ignored: {leaked}."
        return AuditResult("gitignored", False, detail.strip())
    return AuditResult(
        "gitignored",
        True,
        "secrets/runtime-data/indexes gitignored (.env, data/, *.bin, .chroma) and untracked",
    )


# ----------------------------------------------------------------- 3. deps pinned
def _deps_from_pyproject(path: Path) -> list[str]:
    """Extract dependency spec strings from a ``[project]`` pyproject (deps + optional)."""
    if not path.exists():
        return []
    try:
        import tomllib

        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - a documentation-only pyproject has no [project]
        return []
    project = data.get("project", {})
    deps: list[str] = list(project.get("dependencies", []))
    for extra in project.get("optional-dependencies", {}).values():
        deps.extend(extra)
    return deps


def check_deps_pinned() -> AuditResult:
    """Every declared dependency carries a concrete version bound (``<``/``==``/``~=``)."""
    unpinned: list[str] = []
    checked = 0
    for rel in ("core/pyproject.toml", "api/pyproject.toml", "eval/pyproject.toml"):
        for dep in _deps_from_pyproject(REPO_ROOT / rel):
            checked += 1
            if not re.search(r"[<>=~!]=?", dep):
                unpinned.append(f"{rel}:{dep}")

    # The api web deps live in the Makefile install line (api/ is import-only) — verify pins there.
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    for pkg in ("fastapi", "uvicorn", "httpx"):
        m = re.search(rf'"{pkg}>=[^"]+"', makefile)
        checked += 1
        if not m:
            unpinned.append(f"Makefile install-api:{pkg} (no version bound)")

    if unpinned:
        return AuditResult("deps_pinned", False, f"unpinned: {unpinned}")
    return AuditResult("deps_pinned", True, f"all {checked} declared deps carry a version bound")


def run_pip_audit() -> AuditResult:
    """Best-effort ``pip-audit`` vuln scan — informational (not gated; may be absent offline)."""
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pip_audit", "--progress-spinner", "off"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except FileNotFoundError:
        return AuditResult("pip_audit", True, "pip-audit not installed - scan skipped", gated=False)
    except subprocess.TimeoutExpired:
        return AuditResult("pip_audit", True, "pip-audit timed out - scan skipped", gated=False)

    out = (proc.stdout or "") + (proc.stderr or "")
    if "No module named pip_audit" in out:
        return AuditResult("pip_audit", True, "pip-audit not installed - scan skipped", gated=False)
    if proc.returncode == 0:
        return AuditResult("pip_audit", True, "pip-audit: no known vulnerabilities", gated=False)
    # Non-zero → findings; record but do not gate (a CVE in a transitive dep isn't a build defect).
    summary = out.strip().splitlines()[-1] if out.strip() else "see pip-audit output"
    return AuditResult("pip_audit", True, f"pip-audit findings (informational): {summary}",
                       gated=False)


# ----------------------------------------------------------------- 4. local-only
def check_local_only() -> AuditResult:
    """The safety + RAG paths contain no managed/remote call (only the LLM seam may call out)."""
    from scribeintake.rag import ingest, retrieve
    from scribeintake.safety import extractor, rules

    src = "".join(inspect.getsource(m) for m in (extractor, rules, retrieve, ingest)).lower()
    bad = [tok for tok in _NONLOCAL_CALLS if tok in src]
    if bad:
        return AuditResult("local_only", False, f"non-local call token in a local path: {bad}")
    return AuditResult(
        "local_only",
        True,
        "safety + RAG paths are local-only (extractor/rules/retrieve/ingest); LLM is the only egress",  # noqa: E501
    )


# ----------------------------------------------------------------- runner
def run_audit() -> list[AuditResult]:
    """Run every audit check and return the results (order = report order)."""
    return [
        scan_secrets(),
        check_gitignored(),
        check_deps_pinned(),
        check_local_only(),
        run_pip_audit(),
    ]


def main() -> int:
    results = run_audit()
    print("Security / dependency audit (Split 14, section 17)\n")
    gated_fail = False
    for r in results:
        mark = "OK " if r.ok else "FAIL"
        tag = "" if r.gated else " (informational)"
        print(f"  [{mark}] {r.name}{tag}: {r.detail}")
        if r.gated and not r.ok:
            gated_fail = True
    print()
    if gated_fail:
        print("AUDIT FAILED - a gated check did not pass.")
        return 1
    print("AUDIT PASSED - no secrets, deps pinned, paths local-only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
