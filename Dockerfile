# ============================================================
# Dockerfile multi-stage para o recomendador de e-commerce.
#
# Estagio 1 (builder): instala as dependencias com o uv.
# Estagio 2 (final): copia apenas o ambiente pronto e o codigo,
#   resultando em uma imagem menor e mais segura.
# ============================================================

# ----- Estagio 1: builder -----
FROM python:3.12-slim AS builder

# Instala o uv via pip (abordagem robusta, sem dependencia externa)
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copia apenas os arquivos de dependencia primeiro (aproveita cache)
COPY pyproject.toml uv.lock ./

# Instala as dependencias de producao no .venv (sem o grupo dev)
RUN uv sync --frozen --no-dev --no-install-project

# Copia o codigo do projeto e o instala
COPY src ./src
COPY README.md ./
RUN uv sync --frozen --no-dev

# ----- Estagio 2: imagem final -----
FROM python:3.12-slim AS final

# Cria um usuario nao-root por seguranca (boa pratica do Tema 3)
RUN useradd --create-home --uid 1000 appuser

WORKDIR /app

# Copia o ambiente virtual pronto e o codigo do estagio builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY pyproject.toml README.md ./

# Cria pastas gravaveis para dados e experimentos, e da posse ao appuser
RUN mkdir -p /app/data /app/mlruns && chown -R appuser:appuser /app

# Garante que o .venv seja usado por padrao
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src"
ENV MLFLOW_TRACKING_URI="file:/app/mlruns"

# Roda como usuario nao-root
USER appuser

# Comando padrao: executa o treino
CMD ["python", "-m", "recomendador.training.train"]