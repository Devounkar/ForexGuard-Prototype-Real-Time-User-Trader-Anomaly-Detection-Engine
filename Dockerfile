FROM python:3.11-slim

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python deps (cached layer) ────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Source code ───────────────────────────────────────────────────────────────
COPY . .

# ── Runtime config ────────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8501

# Health-check against Streamlit's built-in health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# ── Entrypoint: generate data + train model, THEN start dashboard ─────────────
# Running `main.py run` here (not at build time) means data is generated fresh
# each container start and can be overridden by mounting a data/ volume.
ENTRYPOINT ["/bin/sh", "-c", \
    "python main.py run && \
     streamlit run app/dashboard.py \
       --server.port=8501 \
       --server.address=0.0.0.0 \
       --server.headless=true \
       --server.fileWatcherType=none"]
