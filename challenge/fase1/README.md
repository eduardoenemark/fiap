# FIAP Challenge Fase 1 — Pós Tech 9IADT

## Sobre o Projeto

Este projeto desenvolve um modelo de Machine Learning para prever o status vital (vivo ou morto) de pacientes com câncer de mama, utilizando características clínicas e patológicas do SEER Breast Cancer Dataset. O câncer de mama é uma das maiores ameaças à saúde pública mundial, correspondendo a aproximadamente 1 em cada 4 novos diagnósticos de câncer no ano.

## Dataset

O modelo foi treinado e validado com uma versão pré-processada do [SEER Breast Cancer Dataset](https://www.kaggle.com/datasets/reihanenamdari/breast-cancer). 
- **Variável Alvo:** `STATUS` (Alive / Dead).
- **Features:** Incluem Idade, Raça, Estadiamento TNM (T Stage, N Stage), Grau de Diferenciação, Tamanho do Tumor, Receptores de Estrogênio/Progesterona, entre outras.
- **Observações Técnicas:** O arquivo original contém um erro de digitação na coluna `Regiol Node Positive` e o tamanho do tumor pode variar entre milímetros e centímetros dependendo da extração. Pacientes com tempo de sobrevida inferior a 1 mês ou dados críticos faltantes foram removidos, o que pode introduzir viés para casos mais avançados.

## Pré-requisitos
- [Docker](https://docs.docker.com/get-docker/) ou [Podman](https://podman.io/getting-started/installation)
- Dataset disponível localmente em: `kaggle/datasets/reihanenamdari/breast-cancer/versions/1/Breast_Cancer.csv`

## Instalação & Construção

O projeto oferece scripts automatizados (`build-image.sh` para Linux/macOS e `build-image.cmd` para Windows) que detectam e utilizam automaticamente o Docker ou Podman instalado. Durante o build, a imagem é gerada com a tag `fiap-challenge-fase1-9iadt-rm370509:1.0`. O `Dockerfile` configura variáveis de ambiente essenciais, como `MPLBACKEND=Agg`, para garantir o funcionamento correto do matplotlib em ambientes headless.

> *Os scripts detectam automaticamente se `docker` ou `podman` está disponível e utilizam o primeiro encontrado.*

## Execução

A aplicação é executada em ambiente containerizado. Você pode rodá-la de duas formas:

**Via Docker Compose (Recomendado):**
```bash
docker compose up
```

**Via CLI direto (Docker ou Podman):**
```bash
mkdir ./output
docker run --rm -v ${PWD}/output:/app/output fiap-challenge-fase1-9iadt-rm370509:1.0
```

> **Nota:** O dataset já é copiado para o interior da imagem durante o processo de build.

> **Volume de Saída (`/app/output`):** Este diretório é mapeado para o host para facilitar a extração dos resultados. Ao final da execução, você encontrará neste volume o arquivo [`challenge-b.ipynb`](challenge-b.ipynb) (notebook formatado e executado) e o relatório [`challenge-b.html`](challenge-b.html), contendo toda a pipeline, visualizações e métricas avaliadas.

## Metodologia & Pipeline

O fluxo de trabalho segue as etapas padrão de ciência de dados, garantindo reprodutibilidade e robustez:
1. **Pré-processamento:** Separação estratificada dos dados em treino (80%) e teste (20%), utilizando `RANDOM_STATE = 42` para garantir reprodutibilidade. Colunas categóricas são convertidas para representações numéricas adequadas aos algoritmos.
2. **Tratamento de Dados:** Detecção de outliers utilizando o algoritmo Local Outlier Factor (LOF) e balanceamento da classe alvo para corrigir assimetrias na distribuição.
3. **Modelagem:** Implementação de um `VotingClassifier` com estratégia de votação `"soft"`, combinando seis modelos base: Logistic Regression, Random Forest, SVC, KNN, Decision Tree e Linear SVC (calibrado via `CalibratedClassifierCV`).
4. **Avaliação:** O desempenho do modelo é monitorado por métricas clássicas de classificação, assegurando uma análise equilibrada entre cobertura e precisão das previsões.

##  Dependências

As bibliotecas do ambiente Python são gerenciadas via `requirements.txt`, incluindo `scikit-learn`, `pandas`, `numpy`, `matplotlib` e `seaborn`.

## Licença & Autoria
- 
- **Licença:** GPL-3.0
- **Autor:** Eduardo Vieira Barbosa (rm370509) - Pós Tech 9IADT - FIAP
- **Telegram:** @eduardoenemark 
