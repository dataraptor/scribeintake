# syntax=docker/dockerfile:1
# ScribeIntake — single-process image (FastAPI API + static UI + in-process orchestrator).
#
# The whole stack runs from one container: the API mounts app/ at / and serves the committed
# Proof artifacts at /proof/*. The RAG index (dense vectors + BM25) is built **at image-build
# time** and the local embedder + cross-encoder reranker are warmed into the HF cache, so the
# running container needs no network for retrieval — citations work offline, out of the box.
# (This is why Docker is the recommended test surface: torch/sentence-transformers install
# cleanly on Linux, where a Windows host may fail to load them.)
#
# Secrets are NEVER baked in: the LLM credentials are passed at runtime via compose `env_file`
# (.env) or `-e`. Only the LLM call leaves the box — embeddings/rerank/BM25/gate/storage are
# all local (the HIPAA-boundary posture, see DEPLOY.md §3).

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Tolerate a flaky network during build (PyPI/torch index read timeouts). The pip cache is
    # kept (NOT disabled) so the cache-mounts below let each retried build reuse already-fetched
    # wheels instead of re-downloading the big torch stack from scratch.
    PIP_DEFAULT_TIMEOUT=180 \
    PIP_RETRIES=10 \
    # Keep the model cache in-image at a stable path so runtime retrieval is offline.
    HF_HOME=/app/.hf_cache \
    # The wired provider; override at runtime via env if using the Anthropic/Claude path.
    CHAT_LLM_MODEL=gpt-5.5

WORKDIR /app

# CPU-only torch first (the default linux wheel pulls ~2GB of CUDA we don't need). Pinning the
# CPU index keeps the image small; sentence-transformers then sees torch already satisfied. The
# pip cache mount persists downloaded wheels across (re)builds — robust to a flaky network.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2,<3"

# Copy the project (the .dockerignore excludes secrets, data/, caches, tmp/).
COPY . /app

# Install the core engine + the RAG extra (embedder + reranker) + the API web deps.
# api/ and eval/ are import-only, resolved via the repo-root on sys.path at runtime.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -e "./core[dev,rag]" \
        "fastapi>=0.110,<1" "uvicorn>=0.27,<1" "httpx>=0.27,<1"

# Build the RAG index (downloads + caches bge-small-en-v1.5) and warm the cross-encoder
# reranker (bge-reranker-base) into HF_HOME so the running container retrieves fully offline.
RUN python -m scribeintake.rag.ingest \
 && python -c "from scribeintake.rag.rerank import load_default_reranker; load_default_reranker(); print('reranker warmed')"

EXPOSE 8000

# Liveness: hit /health with stdlib only (no curl in slim).
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health',timeout=4).status==200 else 1)"

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
