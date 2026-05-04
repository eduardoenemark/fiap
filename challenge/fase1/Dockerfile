FROM python:3.12.3-slim

WORKDIR /app

# Define o backend do matplotlib para 'Agg' para funcionar corretamente em ambientes Docker sem monitor (headless).
# Isso previne erros comuns ao usar 'plt.show()'.
ENV MPLBACKEND=Agg

# Instala bibliotecas do sistema necessárias para o scikit-learn (libgomp1 para n_jobs=-1)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copia as dependências e instala os pacotes Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia os arquivos do projeto
COPY functions.py .
COPY challenge-b.py .
COPY kaggle/ ./kaggle/

# Executa o script principal
CMD ["python", "challenge-b.py"]
