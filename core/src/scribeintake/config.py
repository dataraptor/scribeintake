"""Settings, pinned model IDs, constants, and data paths.

Environment-driven via ``pydantic-settings`` (``.env`` supported, gitignored).
Importing :data:`settings` never requires an API key — the deterministic test
tier and the per-commit CI gate run with no secrets.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# --- Pinned model IDs (spec section 19; verified against the claude-api skill 2026-06-20) ---
MODEL_INTAKE = "claude-sonnet-4-6"
MODEL_SUMMARY = "claude-opus-4-8"
MODEL_JUDGE = "claude-opus-4-8"

# --- Loop / limits ---
MAX_AGENT_STEPS = 4  # tool calls per turn
MAX_INTAKE_TURNS = 20

# --- Prompt-cache floors (tokens, spec section 16) ---
CACHE_FLOOR_OPUS = 4096
CACHE_FLOOR_SONNET = 2048

# --- Locale / crisis routing ---
LOCALE = "en-US"
CRISIS_NUMBERS = {"lifeline": "988", "emergency": "911"}

# --- Reproducibility versions (recorded per turn, spec section 13) ---
RULES_VERSION = "v1"
PROMPT_VERSION = "v1"

# config.py lives at <repo>/core/src/scribeintake/config.py; the repo root is 3
# parents up. This resolves correctly for an editable install (the canonical dev
# setup), where __file__ still points at the source tree.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_PACKAGE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    """Env-driven settings singleton.

    Secrets and overridable paths come from the environment; everything else
    (model IDs, limits) lives as module-level constants above.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Secret — env ANTHROPIC_API_KEY only, never hard-coded. Optional so imports
    # and the deterministic tier work without it.
    anthropic_api_key: str | None = None

    # Overridable via env DATA_DIR; defaults to <repo>/data (gitignored).
    data_dir: Path = _REPO_ROOT / "data"

    @property
    def DATA_DIR(self) -> Path:
        """Data directory, created on first access if missing."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return self.data_dir

    @property
    def DB_PATH(self) -> Path:
        """SQLite database path under :pyattr:`DATA_DIR`."""
        return self.DATA_DIR / "scribeintake.db"

    @property
    def KB_DIR(self) -> Path:
        """Curated guideline corpus directory (ships with the package)."""
        return _PACKAGE_DIR / "kb"

    @property
    def CHROMA_DIR(self) -> Path:
        """Local Chroma store path (Split 05)."""
        return self.DATA_DIR / ".chroma"


settings = Settings()
