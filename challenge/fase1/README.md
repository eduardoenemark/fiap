## FIAP Challenge Fase 1 — Pós Tech 9IADT

### Sobre

O câncer de mama é uma das maiores ameaças à saúde pública mundial e a principal causa de morte por câncer entre as mulheres em praticamente todos os países. De acordo com os dados do GLOBOCAN 2022, foram registrados **2.296.840 novos casos** e **666.103 óbitos** em apenas um ano — representando aproximadamente **1 em cada 4 novos diagnósticos de câncer no mundo**.

Este projeto treina um modelo de machine learning para **prever se uma paciente está viva ou morta** com base em características clínicas e patológicas do [SEER Breast Cancer Dataset](https://www.kaggle.com/datasets/reihanenamdari/breast-cancer), disponível no Kaggle.

O fluxo cobre: carregamento e validação dos dados, pré-processamento, detecção de outliers (LOF), balanceamento, análise exploratória (histogramas, boxplots, heatmap) e treinamento de um ensemble `VotingClassifier` com Logistic Regression, Random Forest, SVC, KNN, Decision Tree e Linear SVC.

---

### Pré-requisitos

- [Docker](https://docs.docker.com/get-docker/) **ou** [Podman](https://podman.io/getting-started/installation)
- Dataset disponível localmente em: `kaggle/datasets/reihanenamdari/breast-cancer/versions/1/Breast_Cancer.csv`

---

### 1. Construção da Imagem

**Linux / macOS:**
```bash
./build-image.sh
```

**Windows:**
```cmd
build-image.cmd
```

> Os scripts detectam automaticamente se `docker` ou `podman` está disponível e utilizam o primeiro encontrado.

---

### 2. Execução

**Via Docker Compose (recomendado):**
```bash
docker compose up
```

**Via Docker / Podman diretamente:**
```bash
docker run --rm fiap-challenge-fase1-9iadt-rm370509:1.0
```

> O dataset já é copiado para dentro da imagem durante o build (`COPY kaggle/ ./kaggle/`), portanto não é necessário montar volumes.
