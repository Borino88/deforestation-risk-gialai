# ==============================================================================
# Stage 1: Build & Dependency Packaging
# ==============================================================================
FROM python:3.11-slim-bookworm AS builder

WORKDIR /build

# Install compilation dependencies for numpy/scikit-learn
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Build wheels for data science dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip wheel --no-cache-dir --wheel-dir /build/wheels -r requirements.txt

# ==============================================================================
# Stage 2: Production Runtime
# ==============================================================================
FROM python:3.11-slim-bookworm AS runtime

# OCI Standard Metadata Labels
LABEL org.opencontainers.image.title="Deforestation Risk Vietnam Pipeline"
LABEL org.opencontainers.image.description="Interpretable machine learning pipeline for forest loss risk prediction in Vietnam at 1 km grid resolution."
LABEL org.opencontainers.image.source="https://github.com/Borino88/deforestation-risk-vietnam"
LABEL org.opencontainers.image.url="https://fattahi.xyz"
LABEL org.opencontainers.image.documentation="https://github.com/Borino88/deforestation-risk-vietnam#readme"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.authors="Mahdi Fattahi <a.borino88@gmail.com>"

# Create unprivileged non-root runtime user and group
RUN groupadd -g 1000 appgroup && \
    useradd -u 1000 -g appgroup -s /bin/bash -m appuser

WORKDIR /app

# Install precompiled wheels from builder stage
COPY --from=builder /build/wheels /tmp/wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --no-index --find-links=/tmp/wheels -r requirements.txt \
    && rm -rf /tmp/wheels

# Copy application source code and master data datasets
COPY src/ ./src/
COPY data/ ./data/

# Create output directory for predictions and figures
RUN mkdir -p /app/outputs && chown -R appuser:appgroup /app

# Switch to non-root runtime user
USER appuser

# Health check verification
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD python -c "import pandas, sklearn; print('OK')" || exit 1

# Execute spatial prediction pipeline by default
CMD ["python", "-c", "from src.pipeline import run_pipeline; res = run_pipeline('data/KBang_TRAIN_master_rain_elev_lossyear.csv', 'data/KBang_TEST_master_rain_elev_lossyear.csv', 'data/MangYang_TRAIN_master_rain_elev_lossyear.csv', 'data/MangYang_TEST_master_rain_elev_lossyear.csv', 'outputs'); print('Pipeline successfully executed. Output directory:', res['out_dir'])"]
