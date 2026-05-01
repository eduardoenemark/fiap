"""
OBJETIVO: Através do dataset fornecido é possível saber quais são as variáveis que contribuem para a morte de uma paciente diagnosticada com câncer de mama?
"""
import sys

# PENDÊNCIAS:
# zip: entender melhor esta função.

# ANOTAÇÃO:
# >> Juntar o significado de cada coluna do dataset para compreensão da correlação entre elas, ou seja, A -> B, B + C -> D.

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

import functions as func

# -----------------------------------------------------------------------------------------------------------------------
# Colunas (Opus 4.6):
# Age (Idade) — Idade da paciente no momento do diagnóstico. Domínio: valores numéricos inteiros (ex.: 30–69).
# Race (Raça/Etnia) — Raça da paciente conforme classificação do SEER. Domínio: "White", "Black", "Other".
# Marital Status (Estado Civil) — Estado civil no momento do diagnóstico. Domínio: "Married", "Single", "Divorced", "Widowed", "Separated".
# T Stage (Estágio T — Tumor) — Tamanho/extensão do tumor primário na classificação TNM. Domínio: T1 (≤20mm), T2 (>20mm e ≤50mm), T3 (>50mm), T4 (qualquer tamanho com extensão à parede torácica ou pele).
# N Stage (Estágio N — Linfonodos) — Grau de acometimento dos linfonodos regionais na classificação TNM. Domínio: N1 (1–3 linfonodos positivos), N2 (4–9), N3 (≥10).
# 6th Stage (Estadiamento AJCC 6ª edição) — Estadiamento geral do câncer combinando T e N. Domínio: IIA, IIB, IIIA, IIIB, IIIC.
# differentiate (Diferenciação) — Grau de diferenciação celular do tumor (quão semelhantes as células tumorais são às células normais). Domínio: "Well differentiated" (bem diferenciado — menos agressivo), "Moderately differentiated" (moderadamente), "Poorly differentiated" (pouco diferenciado — mais agressivo), "Undifferentiated" (indiferenciado).
# Grade (Grau histológico) — Representação numérica da diferenciação. Domínio: 1 (bem diferenciado), 2 (moderadamente), 3 (pouco diferenciado), 4 (indiferenciado).
# A Stage (Estágio de extensão) — Extensão anatômica da doença. Domínio: "Regional" (doença restrita à região da mama/linfonodos próximos), "Distant" (metástase à distância).
# Tumor Size (Tamanho do Tumor) — Tamanho do tumor em milímetros. Domínio: valores numéricos inteiros (ex.: 1–140).
# Estrogen Status (Status de Receptor de Estrogênio) — Se o tumor possui receptores de estrogênio (importante para decisão terapêutica). Domínio: "Positive", "Negative".
# Progesterone Status (Status de Receptor de Progesterona) — Se o tumor possui receptores de progesterona. Domínio: "Positive", "Negative".
# Regional Node Examined (Linfonodos Regionais Examinados) — Quantidade de linfonodos regionais que foram examinados cirurgicamente. Domínio: valores numéricos inteiros (ex.: 1–61).
# Reginol Node Positive (Linfonodos Regionais Positivos) — Quantidade de linfonodos examinados que continham células cancerígenas. Domínio: valores numéricos inteiros (ex.: 1–46). Nota: o nome da coluna contém um erro de grafia ("Reginol" em vez de "Regional").
# Survival Months (Meses de Sobrevida) — Número de meses que a paciente sobreviveu após o diagnóstico. Domínio: valores numéricos inteiros (≥1).
# Status (Situação) — Se a paciente está viva ou falecida ao final do período de acompanhamento. Domínio: "Alive" (viva), "Dead" (falecida).
#
# -----------------------------------------------------------------------------------------------------------------------
# Colunas (qwen3.6:35b-a3b-q8_0):
# Abaixo está a explicação detalhada de cada coluna do dataset, com seu significado clínico/econômico e o domínio de valores esperado. A tabela foi organizada para facilitar a consulta, seguida de notas técnicas importantes.
#
# | Coluna no CSV | Significado (Português) | Domínio / Valores Possíveis | Observações Clínicas |
# |--------------|------------------------|----------------------------|----------------------|
# | `Age` | Idade da paciente no momento do diagnóstico | Numérico inteiro | Geralmente entre 20 e 90+ anos. No trecho: `40, 47, 50, 51, 58, 68` |
# | `Race` | Raça/etnia da paciente | Categórico | `White`, `Black`, `Asian`, `Hispanic`, `Other`, `Unknown` (varia conforme o ano do SEER) |
# | `Marital Status` | Estado civil | Categórico | `Married`, `Single`, `Divorced`, `Widowed`, `Separated`, `Unknown` |
# | `T Stage` | Estágio do tumor primário (sistema TNM) | Ordinal/Categórico | `TX`, `T0`, `Tis`, `T1`, `T2`, `T3`, `T4` (T1: ≤2 cm; T2: >2-5 cm; T3: >5 cm; T4: invasão de pele/pescoço) |
# | `N Stage` | Envolvimento de linfonodos regionais (sistema TNM) | Ordinal/Categórico | `NX`, `N0`, `N1`, `N2`, `N3` (N1: 1-3 nós; N2: 4-9; N3: ≥10) |
# | `6th Stage` | Estadiamento global da doença (6ª edição do AJCC) | Ordinal/Categórico | `I`, `IA`, `IB`, `II`, `IIA`, `IIB`, `III`, `IIIA`, `IIIB`, `IIIC`, `IV`, `IVA`, `IVB` |
# | `differentiate` | Grau de diferenciação histológica do tumor | Ordinal/Categórico | `Well differentiated` (1), `Moderately differentiated` (2), `Poorly differentiated` (3), `Undifferentiated` (4) |
# | `Grade` | Escore numérico de graduação histológica | Ordinal/Categórico (numérico) | `1`, `2`, `3`, `4` (corresponde diretamente à coluna acima) |
# | `A Stage` | Agrupamento anatômico / Estadiamento resumido (SEER) | Categórico | `Localized`, `Regional`, `Distant`, `Unknown` (reflete extensão espacial da doença) |
# | `Tumor Size` | Tamanho máximo do tumor | Numérico (contínuo) | Valores no trecho: `4, 8, 18, 20, 30, 35, 41, 63`. No SEER, **geralmente está em milímetros (mm)** ou centímetros. 63 mm (6,3 cm) é plausível; 63 cm seria impossível. Verifique a unidade original no dicionário de dados. |
# | `Estrogen Status` | Presença de receptores de estrogênio (ER) | Categórico | `Positive`, `Negative`, `Unknown` (positivo indica possível resposta a hormonioterapia) |
# | `Progesterone Status` | Presença de receptores de progesterona (PR) | Categórico | `Positive`, `Negative`, `Unknown` (geralmente correlacionado com ER) |
# | `Regional Node Examined` | Número de linfonodos regionais dissecados/examinados patologicamente | Inteiro | Geralmente `0` a `100+`. Quanto maior, melhor a precisão do estadiamento. |
# | `Regiol Node Positive` *(grifo: erro de digitação no CSV)* | Número de linfonodos regionais com metástase | Inteiro | `0` a `Regional Node Examined`. Valores do trecho: `1, 2, 5, 7` |
# | `Survival Months` | Meses de sobrevida desde o diagnóstico até o desfecho ou último contato | Inteiro | `1` a `100+`. Inclui sobrevida global e por causa específica. |
# | `Status` | Estado vital da paciente no encerramento do acompanhamento | Categórico | `Alive`, `Dead` |
#
# ### Notas Técnicas Importantes
# 1. **Sistema TNM e AJCC**: As colunas `T Stage`, `N Stage`, `6th Stage` e `A Stage` referem-se a classificações oncológicas padronizadas pelo *American Joint Committee on Cancer (AJCC)* e pelo programa SEER. O estadiamento combina T, N e (implicitamente) M para definir o prognóstico.
# 2. **`differentiate` vs `Grade`**: São informações redundantes mas coletadas de formas diferentes. `Grade` é um escore numérico (1-4) usado em escores como Nottingham; `differentiate` é a descrição textual equivalente.
# 3. **`Regiol Node Positive`**: O CSV original apresenta um erro de digitação (`Regiol` em vez de `Regional`). Não afeta a lógica, mas recomenda-se corrigir ao carregar os dados.
# 4. **`Tumor Size`**: Verifique o dicionário oficial do SEER ou o arquivo `codebook` do Kaggle. Em versões mais recentes, o tamanho é reportado em **cm**; nesta versão específica, os valores sugerem **mm** ou uma gravação inconsistente.
# 5. **Viés de exclusão**: Conforme a descrição, pacientes com tempo de vida < 1 mês, tamanho tumoral desconhecido, linfonodos não examinados ou positivos desconhecidos foram removidos. Isso pode enviesar a distribuição para estágios mais avançados ou melhor acompanhados.
#
# Se precisar de ajuda para tratar essas colunas (ex: codificação, normalização, divisão treino/teste, ou mapeamento para o TNM 7ª/8ª edição), é só avisar!
# -----------------------------------------------------------------------------------------------------------------------

# ----- DATASET LOG -----
dataset = pd.read_csv("../kaggle/datasets/reihanenamdari/breast-cancer/versions/1/Breast_Cancer.csv")

# ----- **** -----
RANDOM_STATE = 42

# ----- COLUNAS DO DATASET -----
COL_IDADE = 'Age'  # type int
COL_RACA = 'Race'  # type str
COL_ESTADO_CIVIL = 'Marital Status'  # type str
COL_ESTAGIO_T = 'T Stage '  # type str
COL_ESTAGIO_N = 'N Stage'  # type str
COL_ESTADIO_AJCC = '6th Stage'  # type str
COL_DIFERENCIACAO = 'differentiate'  # type str
COL_GRAU_HISTOLOGICO = 'Grade'  # type int
COL_ESTAGIO_DE_EXTENSAO = 'A Stage'  # type str
COL_TAMANHO_DO_TUMOR = 'Tumor Size'  # type int
COL_ESTROGENIO_STATUS = 'Estrogen Status'  # type str
COL_PROGESTERONA_STATUS = 'Progesterone Status'  # type str
COL_LINFONODOS_REGIONAIS_EXAMINADOS = 'Regional Node Examined'  # type int
COL_LINFONODOS_REGIONAIS_POSITIVOS = 'Reginol Node Positive'  # type int
COL_MESES_DE_SOBREVIDA = 'Survival Months'  # type int
COL_STATUS = 'Status'  # type str

COLUNAS = [COL_IDADE, COL_RACA, COL_ESTADO_CIVIL, COL_ESTAGIO_T, COL_ESTAGIO_N, COL_ESTADIO_AJCC, COL_DIFERENCIACAO,
           COL_GRAU_HISTOLOGICO, COL_ESTAGIO_DE_EXTENSAO, COL_TAMANHO_DO_TUMOR, COL_ESTROGENIO_STATUS,
           COL_PROGESTERONA_STATUS, COL_LINFONODOS_REGIONAIS_EXAMINADOS, COL_LINFONODOS_REGIONAIS_POSITIVOS,
           COL_MESES_DE_SOBREVIDA, COL_STATUS]

COLUNAS_NUMERICAS = [COL_IDADE, COL_GRAU_HISTOLOGICO, COL_TAMANHO_DO_TUMOR, COL_LINFONODOS_REGIONAIS_EXAMINADOS,
                     COL_LINFONODOS_REGIONAIS_POSITIVOS, COL_MESES_DE_SOBREVIDA]

COLUNAS_STRINGS = [COL_RACA, COL_ESTADO_CIVIL, COL_ESTAGIO_T, COL_ESTAGIO_N, COL_ESTADIO_AJCC, COL_DIFERENCIACAO,
                   COL_ESTAGIO_DE_EXTENSAO, COL_ESTROGENIO_STATUS, COL_PROGESTERONA_STATUS, COL_STATUS]

# ----- DOMINIO DE VALORES -----
# Arrays de domínio para referência e testes de validação
# Valores em lowercase para consistência com o processamento dos dados

# Colunas Categóricas (Strings)
RACA_DOMINIO = ['black', 'white', 'other']
ESTADO_CIVIL_DOMINIO = ['divorced', 'married', 'separated', 'single', 'widowed']
ESTAGIO_T_DOMINIO = ['tx', 't0', 'tis', 't1', 't2', 't3', 't4']
ESTAGIO_N_DOMINIO = ['nx', 'n0', 'n1', 'n2', 'n3']
ESTADIO_AJCC_DOMINIO = ['ia', 'ib', 'iia', 'iib', 'iiia', 'iiib', 'iiic']
DIFERENCIACAO_DOMINIO = ['well differentiated', 'moderately differentiated', 'poorly differentiated',
                         'undifferentiated']
GRAU_HISTOLOGICO_DOMINIO = ['1', '2', '3', '4']
ESTAGIO_DE_EXTENSAO_DOMINIO = ['regional', 'distant']
ESTROGENIO_STATUS_DOMINIO = ['positive', 'negative']
PROGESTERONA_STATUS_DOMINIO = ['positive', 'negative']
STATUS_DOMINIO = ['dead', 'alive']

# Colunas Numéricas (Intervalos: (min, max))
IDADE_RANGE = (0, 120)
TAMANHO_DO_TUMOR_RANGE = (1, 140)
LINFONODOS_REGIONAIS_EXAMINADOS_RANGE = (1, 61)
LINFONODOS_REGIONAIS_POSITIVOS_RANGE = (1, 46)
MESES_DE_SOBREVIDA_RANGE = (0, 12000)

# ----- VERIFICACAO DE VALORES NULOS OU VAZIOS -----
assert (dataset.isnull().sum() == 0).all()

# ----- VERIFICACAO DE VALORES NULOS OU VAZIOS -----
assert (dataset.isnull().sum() == 0).all()

# ----- TRANSFORMACAO BASICA DOS DADOS -----
for i, coluna in enumerate(COLUNAS):
    dataset[coluna] = dataset[coluna].astype(str).str.strip().str.lower()

# ----- VALIDACAO DO RANGE DE VALORES DAS COLUNAS -----
INT_REGEX = r'^[0-9]+$'
STR_REGEX = r'^[[:alpha:][:digit:][:space:]]+$'
for index, row in dataset.iterrows():
    try:
        # Validação numérica usando os intervalos de domínio definidos
        assert IDADE_RANGE[0] <= int(row[COL_IDADE]) <= IDADE_RANGE[1]
        assert TAMANHO_DO_TUMOR_RANGE[0] <= int(row[COL_TAMANHO_DO_TUMOR]) <= TAMANHO_DO_TUMOR_RANGE[1]
        assert LINFONODOS_REGIONAIS_EXAMINADOS_RANGE[0] <= int(row[COL_LINFONODOS_REGIONAIS_EXAMINADOS]) <= \
               LINFONODOS_REGIONAIS_EXAMINADOS_RANGE[1]
        assert LINFONODOS_REGIONAIS_POSITIVOS_RANGE[0] <= int(row[COL_LINFONODOS_REGIONAIS_POSITIVOS]) <= \
               LINFONODOS_REGIONAIS_POSITIVOS_RANGE[1]
        assert MESES_DE_SOBREVIDA_RANGE[0] <= int(row[COL_MESES_DE_SOBREVIDA]) <= MESES_DE_SOBREVIDA_RANGE[1]
        # Validação categórica usando os arrays de domínio definidos
        assert row[COL_RACA] in RACA_DOMINIO
        assert row[COL_ESTADO_CIVIL] in ESTADO_CIVIL_DOMINIO
        assert row[COL_ESTAGIO_T] in ESTAGIO_T_DOMINIO
        assert row[COL_ESTAGIO_N] in ESTAGIO_N_DOMINIO
        assert row[COL_ESTADIO_AJCC] in ESTADIO_AJCC_DOMINIO
        assert row[COL_DIFERENCIACAO] in DIFERENCIACAO_DOMINIO
        assert row[COL_GRAU_HISTOLOGICO] in GRAU_HISTOLOGICO_DOMINIO
        assert row[COL_ESTAGIO_DE_EXTENSAO] in ESTAGIO_DE_EXTENSAO_DOMINIO
        assert row[COL_ESTROGENIO_STATUS] in ESTROGENIO_STATUS_DOMINIO
        assert row[COL_PROGESTERONA_STATUS] in PROGESTERONA_STATUS_DOMINIO
        assert row[COL_STATUS] in STATUS_DOMINIO
    except AssertionError as error:
        # print(f">> Removida a linha {index} de valor inválido {error}")
        dataset.drop(index, inplace=True)
        # print(f'Dataset shape depois de drop: {dataset.shape}')

# --------------- LINHAS DUPLICADAS --------------------------------------------------
duplicated = dataset.duplicated().sum()
print(f'>> Linhas duplicadas: {duplicated}')

dataset.drop_duplicates(inplace=True)
print(f'Dataset shape depois de drop: {dataset.shape}')

# --------------- DETECÇÃO DE OUTLIERS --------------------------------------------------
# outliers_lof = func.detect_outliers_lof(dataset[COLUNAS_NUMERICAS])
# outliers_lof_labels = func.map_in_and_outlier_labels(outliers_lof)
#
# func.plot_distribution_grid(outliers_lof_labels,
#                             outliers_lof_labels.columns,
#                             plot_type='count',
#                             suptitle='Distribuição de Outliers (Local Outlier Factor - LOF)')

# ----- HISTOGRAMA DAS COLUNAS -----
# colunas numéricas hist:
func.plot_distribution_grid(dataset, COLUNAS_NUMERICAS, plot_type='hist')

# colunas string count:
# func.plot_distribution_grid(dataset, COLUNAS_STRINGS, plot_type='count')

sys.exit(0)
