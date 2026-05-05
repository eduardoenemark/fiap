FROM python:3.12.3-slim

WORKDIR /app

# ---------- ENV ----------
# Define o backend do matplotlib para 'Agg' para funcionar corretamente em ambientes Docker sem monitor (headless).
# Isso previne erros comuns ao usar 'plt.show()'.
ENV MPLBACKEND=Agg
ENV PYTHONUNBUFFERED=1
ENV REPO_URL="https://github.com/eduardoenemark/fiap/tree/main/challenge/fase1"

# ---------- LABEL ----------
LABEL org.opencontainers.image.ref.name="fiap-challenge-fase1-9iadt-rm370509" \
    org.opencontainers.image.version="1.0"                  \
    org.opencontainers.image.authors="Eduardo Vieira Barbosa"       \
    org.opencontainers.image.source="${REPO_URL}"           \
    org.opencontainers.image.title="FIAP Challenge Fase 1 - Pós Tech 9IADT" \
    org.opencontainers.image.description="H2 Database Engine container image" \
    org.opencontainers.image.created="${CREATED_DATETIME}"  \
    org.opencontainers.image.licenses="GPL-3.0"       \
    org.opencontainers.image.purpose="Exercício avaliativo da primeira fase" \
    org.opencontainers.image.vendor="t.me/eduardoenemark"   \
    org.opencontainers.image.url="${REPO_URL}"              \
    org.opencontainers.image.documentation="${REPO_URL}"  \
    maintainer="Eduardo Vieira <eduardoenemark@gmail.com>"

# ---------- COPY ----------
COPY requirements.txt .
COPY functions.py .
COPY challenge-b.py .
COPY kaggle/ ./kaggle/

# ---------- RUN ----------
# Instala bibliotecas do sistema necessárias para o scikit-learn (libgomp1 para n_jobs=-1)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -r requirements.txt

# ---------- ENTRYPOINT/CMD ----------
# Executa o script principal
CMD ["python", "challenge-b.py"]
