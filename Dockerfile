FROM python:3.10-slim

# ─── System dependencies ────────────────────────────────────────────
# curl: required for HEALTHCHECK
# libboost-all-dev + swig + build-essential + cmake: required for AutoDock Vina Python bindings
# libffi-dev: required for cffi / some Python C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libboost-all-dev \
    swig \
    build-essential \
    cmake \
    libffi-dev \
    openbabel \
    && rm -rf /var/lib/apt/lists/*

# ─── Create non-root user (HF Spaces requires uid 1000) ─────────────
RUN useradd -m -u 1000 appuser

# ─── App directory ──────────────────────────────────────────────────
WORKDIR /app

# ─── Install Python dependencies (cached layer) ────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip six && \
    sed -i '/^oddt/d' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --no-build-isolation oddt>=0.7

# ─── Copy application code ─────────────────────────────────────────
COPY . .

# ─── Override Streamlit config for Docker/HF environment ────────────
RUN mkdir -p /app/.streamlit && \
    printf '[server]\nport = 7860\naddress = "0.0.0.0"\nheadless = true\nenableCORS = false\nenableXsrfProtection = false\nmaxUploadSize = 200\n\n[browser]\ngatherUsageStats = false\n\n[theme]\nprimaryColor = "#6366f1"\nbackgroundColor = "#0a0e1a"\nsecondaryBackgroundColor = "#111827"\ntextColor = "#f1f5f9"\nfont = "sans serif"\n' > /app/.streamlit/config.toml

# ─── Set permissions ────────────────────────────────────────────────
RUN chown -R appuser:appuser /app
USER appuser

# ─── HF Spaces expects port 7860 ───────────────────────────────────
EXPOSE 7860

# ─── Health check ───────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl --fail http://localhost:7860/_stcore/health || exit 1

# ─── Run Streamlit ──────────────────────────────────────────────────
ENTRYPOINT ["streamlit", "run", "app.py", \
    "--server.port=7860", \
    "--server.address=0.0.0.0", \
    "--server.headless=true", \
    "--browser.gatherUsageStats=false"]
