"""Settings, pinned model IDs, constants, and data paths.

Environment-driven via ``pydantic-settings`` (``.env`` supported, gitignored).
Importing :data:`settings` never requires an API key — the deterministic test
tier and the per-commit CI gate run with no secrets.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# --- Pinned model IDs (spec section 19; verified against the claude-api skill 2026-06-20) ---
# These are the spec's *intended* Claude pins, kept for documentation and forward
# compatibility. The Split-01/03 environment ships an **Azure OpenAI GPT-5.5** key
# instead of an Anthropic key (see the PROGRESS log), so the live agent loop runs on
# ``ACTIVE_INTAKE_MODEL`` below. The deviation is conceptually consistent with the
# spec's "no sampling knobs" API-conformance rule: GPT-5.5 is itself a reasoning model
# that rejects ``temperature``/``top_p``/``seed`` (verified 2026-06-20).
MODEL_INTAKE = "claude-sonnet-4-6"
MODEL_SUMMARY = "claude-opus-4-8"
MODEL_JUDGE = "claude-opus-4-8"

# Active provider model for the wired agent loop (Split 03). Overridable via env
# CHAT_LLM_MODEL; defaults to the deployment shipped in ``.env``.
DEFAULT_CHAT_MODEL = "gpt-5.5"

# Agent reasoning-effort routes (maps to OpenAI ``reasoning_effort`` / the spec's
# ``output_config.effort``). Routine intake turns ask "which slot next?" — not deep reasoning —
# so the intake route is env-tunable via INTAKE_EFFORT (default ``medium`` to keep eval/CI
# reproducible; the container ships ``low`` for latency, since each turn makes ~3 sequential
# model calls and effort is paid on every one). The terminal SOAP summary stays ``high``
# (quality-critical clinical reasoning); triage refinement stays ``medium``.
EFFORT_INTAKE = os.getenv("INTAKE_EFFORT", "medium")
EFFORT_SUMMARY = "high"
EFFORT_TRIAGE = "medium"

# --- Loop / limits ---
MAX_AGENT_STEPS = 4  # tool calls per turn
MAX_INTAKE_TURNS = 20
MAX_SUMMARY_TOKENS = 4096  # terminal SOAP call (bumped once on max_tokens, §3.4)
MAX_TRIAGE_TOKENS = 2048  # terminal triage call

# --- RAG retrieval (Split 05, spec §11/§12) ---
RETRIEVE_K = 5  # chunks returned to the caller after rerank
RAG_CANDIDATES = 20  # BM25 top-N and dense top-N recall depth (union → rerank)
# Citation binding (build_summary): an observation binds to a chunk only when their content
# terms overlap enough — otherwise it is flagged ``uncited`` (never a fabricated source). The
# model is shown the retrieved passages, so a grounded observation echoes the chunk's vocabulary
# (high overlap), while a generic screening note ("none triggered") shares little and stays
# uncited. Conservative by design: prefer a missed citation to a fabricated one.
CITATION_MIN_OVERLAP = 0.2  # fraction of the statement's content terms found in the chunk
CITATION_MIN_SHARED = 2  # and at least this many distinct content terms shared

# --- Prompt-cache floors (tokens, spec section 16) ---
CACHE_FLOOR_OPUS = 4096
CACHE_FLOOR_SONNET = 2048

# --- Latency targets (ms, spec section 18) — used by observability/latency.py to flag breaches.
LATENCY_TARGET_INTAKE_P50_MS = 3000
LATENCY_TARGET_INTAKE_P95_MS = 6000
LATENCY_TARGET_SUMMARY_MS = 8000  # the terminal SOAP/triage call has its own (looser) budget

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

    # Azure OpenAI (GPT-5.5) — the provider actually wired in Split 03. All optional
    # so importing :data:`settings` and the keyless deterministic tier never need them;
    # they are required only when the *live* agent loop is constructed.
    azure_openai_endpoint: str | None = None
    azure_openai_api_key: str | None = None
    openai_api_version: str = "2025-01-01-preview"
    chat_llm_model: str | None = None  # e.g. "gpt-5.5"

    # Prompt caching (spec section 7/16). With the spec's Anthropic pin this toggle would gate
    # placement of ``cache_control`` breakpoints. The wired provider (Azure OpenAI GPT-5.5) does
    # **automatic** prefix caching that cannot be disabled per-request, so for that backend this
    # flag is **reporting-only**: the no-cache baseline is computed by repricing the observed
    # cached tokens at full input price (an exact counterfactual — see
    # ``pricing.no_cache_cost_usd`` / the Split 09 session log), which is more rigorous than
    # re-running. Env: PROMPT_CACHE_ENABLED.
    prompt_cache_enabled: bool = True

    # Warm the RAG retriever (torch + the two BGE models) at server startup instead of on the
    # first patient message. Off by default so unit tests and the deterministic eval tier — which
    # build the app but never touch the live index — never pay the model load. The container sets
    # WARM_MODELS_ON_STARTUP=1 so the first real turn isn't penalised by the cold load (§18).
    warm_models_on_startup: bool = False

    @property
    def ACTIVE_INTAKE_MODEL(self) -> str:
        """The model id the live agent loop calls (env ``CHAT_LLM_MODEL`` or default)."""
        return self.chat_llm_model or DEFAULT_CHAT_MODEL

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
    def RAG_INDEX_DIR(self) -> Path:
        """Local RAG index dir (Split 05): the persisted vector store + BM25 store.

        A dependency-light JSON vector store (``vectors.json``) + ``bm25.json`` live here. (The
        spec pinned Chroma; it is replaced by this local store — see the Split 05 session log —
        because chromadb 0.4.24 is incompatible with the environment's NumPy 2.x. The
        ``DenseIndex`` seam keeps a Chroma/FAISS backend a drop-in for larger corpora.)
        """
        return self.DATA_DIR / ".rag_index"


settings = Settings()
