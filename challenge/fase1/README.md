### Instruções de Uso
1. **Construção da Imagem:**

docker build -t desafio-cancer .

1. **Execução:** Como o script lê um arquivo CSV de um caminho específico relativo ao diretório de trabalho (`kaggle/datasets/...`), você precisará montar esse diretório como um volume no contêiner para que ele encontre os dados. `challenge-b.py`

docker run -v /caminho/para/seus/dados/kaggle:/app/kaggle desafio-cancer

