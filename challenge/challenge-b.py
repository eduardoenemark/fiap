# --------------- IMPORTS ------------------------------------------------------------
import sys

import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import functions as func

"""
    NOTA TÉCNICA:
    - Devido a complexidade do assunto tratado transposto em código de programação então optei por dividir em arquivo dedicado somente
      a funções para o suporte do fluxo tratado, functions.py, e um segundo dedicado para o desenvolvimento de todo o fluxo de análise, 
      pré-processamento e treinamento do modelo, challenge-b.py.
    - Isto é apenas uma prática de um exercício (desafio) de machine learning. Logo não tem nenhuma pretensão de ser usada em ambiente
      de produção.
    - Sinta-se livre para copiar ou modificar.
"""

# --------------- ABOUT --------------------------------------------------------------
"""
    O câncer de mama é, atualmente, uma das maiores ameaças à saúde pública mundial e a principal causa de morte por câncer
    entre as mulheres em praticamente todos os países. De acordo com os dados do GLOBOCAN 2022, da Agência Internacional de
    Pesquisa em Câncer (IARC/OMS), foram registrados no mundo 2.296.840 novos casos da doença e 666.103 óbitos em apenas um
    ano. O câncer de mama representa aproximadamente 1 em cada 4 novos diagnósticos de câncer no mundo, correspondendo a cerca
    de 25% de todos os casos diagnosticados anualmente. É o tipo de câncer mais frequente entre as mulheres em 161 dos 185 
    países analisados.
    IARC / OMS — GLOBOCAN 2022. Global Cancer Observatory. Disponível em: <https://gco.iarc.who.int>


    Descrição sobre o dataset utilizado neste trabalho:
    - O dataset é uma versão pré-processada do SEER Breast Cancer Dataset, disponível no Kaggle: https://www.kaggle.com/datasets/reihanenamdari/breast-cancer

    - Colunas e seus significados:
    ----------------------------------------------------------------------------------------------------------------------
    | Coluna no CSV | Significado (Português) | Domínio / Valores Possíveis | Observações Clínicas |
    |--------------|------------------------|----------------------------|----------------------|
    | `Age` | Idade da paciente no momento do diagnóstico | Numérico inteiro | Geralmente entre 20 e 90+ anos. No trecho: `40, 47, 50, 51, 58, 68` |
    | `Race` | Raça/etnia da paciente | Categórico | `White`, `Black`, `Asian`, `Hispanic`, `Other`, `Unknown` (varia conforme o ano do SEER) |
    | `Marital Status` | Estado civil | Categórico | `Married`, `Single`, `Divorced`, `Widowed`, `Separated`, `Unknown` |
    | `T Stage` | Estágio do tumor primário (sistema TNM) | Ordinal/Categórico | `TX`, `T0`, `Tis`, `T1`, `T2`, `T3`, `T4` (T1: ≤2 cm; T2: >2-5 cm; T3: >5 cm; T4: invasão de pele/pescoço) |
    | `N Stage` | Envolvimento de linfonodos regionais (sistema TNM) | Ordinal/Categórico | `NX`, `N0`, `N1`, `N2`, `N3` (N1: 1-3 nós; N2: 4-9; N3: ≥10) |
    | `6th Stage` | Estadiamento global da doença (6ª edição do AJCC) | Ordinal/Categórico | `I`, `IA`, `IB`, `II`, `IIA`, `IIB`, `III`, `IIIA`, `IIIB`, `IIIC`, `IV`, `IVA`, `IVB` |
    | `differentiate` | Grau de diferenciação histológica do tumor | Ordinal/Categórico | `Well differentiated` (1), `Moderately differentiated` (2), `Poorly differentiated` (3), `Undifferentiated` (4) |
    | `Grade` | Escore numérico de graduação histológica | Ordinal/Categórico (numérico) | `1`, `2`, `3`, `4` (corresponde diretamente à coluna acima) |
    | `A Stage` | Agrupamento anatômico / Estadiamento resumido (SEER) | Categórico | `Localized`, `Regional`, `Distant`, `Unknown` (reflete extensão espacial da doença) |
    | `Tumor Size` | Tamanho máximo do tumor | Numérico (contínuo) | Valores no trecho: `4, 8, 18, 20, 30, 35, 41, 63`. No SEER, **geralmente está em milímetros (mm)** ou centímetros. 63 mm (6,3 cm) é plausível; 63 cm seria impossível. Verifique a unidade original no dicionário de dados. |
    | `Estrogen Status` | Presença de receptores de estrogênio (ER) | Categórico | `Positive`, `Negative`, `Unknown` (positivo indica possível resposta a hormonioterapia) |
    | `Progesterone Status` | Presença de receptores de progesterona (PR) | Categórico | `Positive`, `Negative`, `Unknown` (geralmente correlacionado com ER) |
    | `Regional Node Examined` | Número de linfonodos regionais dissecados/examinados patologicamente | Inteiro | Geralmente `0` a `100+`. Quanto maior, melhor a precisão do estadiamento. |
    | `Regiol Node Positive` *(grifo: erro de digitação no CSV)* | Número de linfonodos regionais com metástase | Inteiro | `0` a `Regional Node Examined`. Valores do trecho: `1, 2, 5, 7` |
    | `Survival Months` | Meses de sobrevida desde o diagnóstico até o desfecho ou último contato | Inteiro | `1` a `100+`. Inclui sobrevida global e por causa específica. |
    | `Status` | Estado vital da paciente no encerramento do acompanhamento | Categórico | `Alive`, `Dead` |
    
    ###Notas Técnicas Importantes:
    1. **Sistema TNM e AJCC**: As colunas `T Stage`, `N Stage`, `6th Stage` e `A Stage` referem-se a classificações oncológicas padronizadas pelo *American Joint Committee on Cancer (AJCC)* e pelo programa SEER. O estadiamento combina T, N e (implicitamente) M para definir o prognóstico.
    2. **`differentiate` vs `Grade`**: São informações redundantes mas coletadas de formas diferentes. `Grade` é um escore numérico (1-4) usado em escores como Nottingham; `differentiate` é a descrição textual equivalente.
    3. **`Regiol Node Positive`**: O CSV original apresenta um erro de digitação (`Regiol` em vez de `Regional`). Não afeta a lógica, mas recomenda-se corrigir ao carregar os dados.
    4. **`Tumor Size`**: Verifique o dicionário oficial do SEER ou o arquivo `codebook` do Kaggle. Em versões mais recentes, o tamanho é reportado em **cm**; nesta versão específica, os valores sugerem **mm** ou uma gravação inconsistente.
    5. **Viés de exclusão**: Conforme a descrição, pacientes com tempo de vida < 1 mês, tamanho tumoral desconhecido, linfonodos não examinados ou positivos desconhecidos foram removidos. Isso pode enviesar a distribuição para estágios mais avançados ou melhor acompanhados.
"""

# --------------- LOAD DATASET -------------------------------------------------------

dataset = pd.read_csv("../kaggle/datasets/reihanenamdari/breast-cancer/versions/1/Breast_Cancer.csv")
dataset_initial_rows = func.get_total_rows(dataset)
func.fprint(f"dataset inicial rows: {dataset_initial_rows}")

# --------------- VERIFICACAO DE VALORES NULOS OU VAZIOS -----------------------------
"""
    A verificação de valores nulos ou vazios é importante para garantir a qualidade dos dados antes de realizar análises
    ou o treinamento do modelo.
"""
assert (dataset.empty == False)
assert (dataset.isnull().sum() == 0).all()

# --------------- COLUNAS DO DATASET -------------------------------------------------
"""
    O mapeamento do nome das colunas do CSV para variáveis é feito para facilitar a leitura e manutenção do código, além de evitar erros de digitação.
    As colunas são categorizadas em numéricas e string para facilitar as etapas de pré-processamento e análise.
"""
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

"""
    A distinção entre colunas numéricas e string (str) é importante para as etapas de pré-processamento (transformação dos dados) 
    e conjunto de colunas que vão compor determinado gráfico, por exemplo.

    Neste processo de análise dos dados do dataset, além da inspeção visual em primeiro momento, também, temos a recuperação de
    outros dados que nos ajudam a compreender os tipos (info), grupo de dados (groupby) e quantidades (count).
"""
COLUNAS_NUMERICAS = [COL_IDADE, COL_GRAU_HISTOLOGICO, COL_TAMANHO_DO_TUMOR, COL_LINFONODOS_REGIONAIS_EXAMINADOS,
                     COL_LINFONODOS_REGIONAIS_POSITIVOS, COL_MESES_DE_SOBREVIDA]

COLUNAS_STRINGS = [COL_RACA, COL_ESTADO_CIVIL, COL_ESTAGIO_T, COL_ESTAGIO_N, COL_ESTADIO_AJCC, COL_DIFERENCIACAO,
                   COL_ESTAGIO_DE_EXTENSAO, COL_ESTROGENIO_STATUS, COL_PROGESTERONA_STATUS, COL_STATUS]

"""
   info: sumário das colunas, tipos de dados e contagem de valores não nulos
"""
func.fprint(f"info:\n{dataset.info(verbose=True, max_cols=1000, show_counts=True)}")

"""
    describe: estatísticas descritivas para colunas numéricas (contagem, média, desvio padrão, min, quartis e max).
    Usamos a coluna idade como exemplo:
"""
func.fprint(f"describe:\n{dataset[COL_IDADE].describe()}")

"""
    groupby e count: contagem de ocorrências para cada categoria na coluna de raça, por exemplo. Isso ajuda a entender
    a distribuição dos dados.
"""
func.fprint(dataset.groupby(COL_RACA).count())

# --------------- DOMINIO DE VALORES -------------------------------------------------
"""
    Realizado análise de todos os possíveis valores de domínio. Os valores string (literais) são colocados em lowercase
    para garantir consistência, enquanto os numéricos são validados dentro de intervalos possíveis.
"""
RACA_DOMINIO = ['black', 'white', 'other']
ESTADO_CIVIL_DOMINIO = ['divorced', 'married', 'separated', 'single', 'widowed']
ESTAGIO_T_DOMINIO = ['tx', 't0', 'tis', 't1', 't2', 't3', 't4']
ESTAGIO_N_DOMINIO = ['nx', 'n0', 'n1', 'n2', 'n3']
ESTADIO_AJCC_DOMINIO = ['ia', 'ib', 'iia', 'iib', 'iiia', 'iiib', 'iiic']
DIFERENCIACAO_DOMINIO = ['well differentiated', 'moderately differentiated', 'poorly differentiated',
                         'undifferentiated']
GRAU_HISTOLOGICO_DOMINIO = [1, 2, 3, 4]
ESTAGIO_DE_EXTENSAO_DOMINIO = ['regional', 'distant']
ESTROGENIO_STATUS_DOMINIO = ['positive', 'negative']
PROGESTERONA_STATUS_DOMINIO = ['positive', 'negative']
STATUS_DOMINIO = ['dead', 'alive']

IDADE_RANGE = (0, 80)
TAMANHO_DO_TUMOR_RANGE = (1, 140)
LINFONODOS_REGIONAIS_EXAMINADOS_RANGE = (1, 61)
LINFONODOS_REGIONAIS_POSITIVOS_RANGE = (1, 46)
MESES_DE_SOBREVIDA_RANGE = (0, 720)

# --------------- TRANSFORMACAO BASICA DOS DADOS E VALIDACAO -------------------------
"""
    A transformação básica dos dados inclui a padronização de strings (lowercase e strip) e a conversão de colunas
    numéricas para o tipo numérico, tratando erros como NaN. Caso uma linha apresente erro então é removida do dataset.
"""
error_counter = 0
for index, row in dataset.iterrows():
    for coluna in COLUNAS_STRINGS:
        try:
            dataset.loc[index, coluna] = str(row[coluna]).strip().lower()
        except ValueError as error:
            dataset.drop(index, inplace=True)
            error_counter += 1

func.fprint(f"Total de erros de conversão para string: {error_counter}")

"""
    Converte colunas numéricas para float64 (padrão pandas com suporte a NaN).
"""
dataset = func.convert_columns_to_numeric(dataset, COLUNAS_NUMERICAS)

"""
    A validação é realizada para garantir que os dados estejam dentro dos domínios e intervalos esperados. Caso uma linha
    contenha um valor inválido, ela é removida do dataset.
"""
error_counter = 0
INT_REGEX = r'^[0-9]+$'
STR_REGEX = r'^[[:alpha:][:digit:][:space:]]+$'
for index, row in dataset.iterrows():
    try:
        assert IDADE_RANGE[0] <= row[COL_IDADE] <= IDADE_RANGE[1]
        assert TAMANHO_DO_TUMOR_RANGE[0] <= row[COL_TAMANHO_DO_TUMOR] <= TAMANHO_DO_TUMOR_RANGE[1]
        assert LINFONODOS_REGIONAIS_EXAMINADOS_RANGE[0] <= row[COL_LINFONODOS_REGIONAIS_EXAMINADOS] <= \
               LINFONODOS_REGIONAIS_EXAMINADOS_RANGE[1]
        assert LINFONODOS_REGIONAIS_POSITIVOS_RANGE[0] <= row[COL_LINFONODOS_REGIONAIS_POSITIVOS] <= \
               LINFONODOS_REGIONAIS_POSITIVOS_RANGE[1]
        assert MESES_DE_SOBREVIDA_RANGE[0] <= row[COL_MESES_DE_SOBREVIDA] <= MESES_DE_SOBREVIDA_RANGE[1]
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
        error_counter += 1
        dataset.drop(index, inplace=True)

func.fprint(f"Total de erros de validacao: {error_counter}")

"""
    Verificar se há possíveis linhas duplicadas no conjunto de dados, também, constitui uma etapa importante.
"""
sum_duplicated_lines = dataset.duplicated().sum()
func.fprint(f"Total de linhas duplicadas: {sum_duplicated_lines}")

if sum_duplicated_lines > 0:
    dataset.drop_duplicates(inplace=True)

func.fprint(
    f"Total de linhas removidas apos as transformacoes e validacoes do dataset {dataset_initial_rows - func.get_total_rows(dataset)}")

# --------------- DETECÇÃO DE OUTLIERS -----------------------------------------------
"""
    A detecção de outliers pode ser realizada utilizando o método Local Outlier Factor (LOF), que é um algoritmo de
    detecção de anomalias baseado em densidade. Ele identifica pontos que estão em regiões de baixa densidade em
    comparação com seus vizinhos, o que pode indicar que são outliers.
    
    Quando de certa forma não somos especialistas nos dados tratados em um dataset procuramos estratégias como "find best"
    para parâmetros como n_neighbors_range, por exemplo. Esta estratégia tem a desvantagem de desprender maior custo
    computacional de processamento de dados.
"""

func.fprint(f"len(COLUNAS_NUMERICAS)={len(COLUNAS_NUMERICAS)}")
lof_n_neighbors, lof_contamination = func.find_best_lof_parameters(X=dataset[COLUNAS_NUMERICAS],
                                                                   n_neighbors_range=range(3, len(COLUNAS_NUMERICAS)))
func.fprint(f"LOF n neighbors: {lof_n_neighbors}, contamination: {lof_contamination}")

outliers_lof = func.detect_outliers_lof(X=dataset[COLUNAS_NUMERICAS],
                                        n_neighbors=lof_n_neighbors,
                                        contamination=lof_contamination)

func.fprint(f"LOF percentual de outliers: {func.get_outlier_percentage(outliers_lof)}%")

outliers_lof_labels = func.map_in_and_outlier_labels(outliers_lof)

func.plot_distribution_grid(outliers_lof_labels,
                            outliers_lof_labels.columns,
                            plot_type='count',
                            suptitle=f"Distribuição de Outliers (LOF): {func.get_outlier_percentage(outliers_lof)}%")


# --------------- HISTOGRAMA DAS COLUNAS ---------------------------------------------
"""
    O histograma com um gráfico de barras que representa a distribuição de frequência de um conjunto de dados, nos ajuda a 
    visualizar quantidades, como estão distribuídas e possíveis diferenças acentuadas.
    
    A função plot_distribution_grid em plot_type do tipo hist (histograma) traz sobre o gráfico de barras o KDE (Kernel Density
    Estimation) que é uma técnica estatística que cria uma curva que representa a distribuição de dados, mostra onde os dados 
    estão mais concentrados.

    Quando temos colunas não numéricas fazemos o gráfico de contagem (count) que é um tipo de gráfico de barras que mostra a
    frequência de cada categoria em uma coluna categórica.
"""
# colunas numéricas hist:
func.plot_distribution_grid(dataset, COLUNAS_NUMERICAS, plot_type='hist')

# colunas string count:
func.plot_distribution_grid(dataset, COLUNAS_STRINGS, plot_type='count')

# --------------- BOXPLOT DAS COLUNAS ------------------------------------------------
"""
    O boxplot é uma representação gráfica que mostra a distribuição de um conjunto de dados numéricos através de seus quartis, mediana e possíveis outliers. Ele é útil para identificar a presença de outliers, a simetria da distribuição e a dispersão dos dados.
"""
func.plot_boxplot_outliers(dataset, COLUNAS_NUMERICAS)

# --------------- HEATMAP DAS COLUNAS ------------------------------------------------
"""

"""
func.plot_correlation_heatmap(dataset, COLUNAS_NUMERICAS)

# --------------- COLUMN TRANSFORMER -------------------------------------------------
"""
"""
X = dataset.drop(COL_STATUS, axis=1)
y = dataset[COL_STATUS]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=func.RANDOM_STATE, stratify=y)

preprocessor = func.build_preprocessor(COLUNAS_STRINGS, COLUNAS_NUMERICAS, COL_STATUS)

x_train = preprocessor.fit_transform(x_train, y_train)
x_test = preprocessor.fit_transform(x_test, y_test)

y_train = func.encode_labels(y_train)
y_test = func.encode_labels(y_test)

# --------------- TRAIN E TESTS ------------------------------------------------------
"""
    Suponha que você faça uma pergunta complexa a milhares de pessoas aleatórias e, em seguida, agregue as respostas delas.
    Em muitos casos, você descobrirá que essa resposta agregada é melhor do que a resposta de um especialista.
    Isso é chamado de sabedoria popular.
    (tradução: Capitulo 7, Aurélien Géron. Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow. 3ed.O'Reilly)

    O VotingClassifier é um meta-classificador que combina as previsões de vários classificadores base para melhorar a
    precisão geral. Ele pode usar votação "hard" (a classe mais votada é a previsão final) ou "soft" (as probabilidades
    previstas são somadas e a classe com a maior probabilidade é a previsão final).
"""

voting_classifier = VotingClassifier(estimators=[
    ('logistic_regression', func.get_logistic_regression()),
    ('random_forest', func.get_random_forest()),
    ('svc', func.get_svc()),
    ('knn', func.get_best_knn_classifier(x_train, y_train, x_test, y_test)),
    ('decision_tree', func.get_decision_tree()),
    ('linear_svc', func.get_linear_svc_for_voting())],
    voting='soft')
voting_classifier.named_estimators['svc'].probability = True

voting_classifier.fit(x_train, y_train)

for name, classifier in voting_classifier.named_estimators_.items():
    print(f"train voting {name} score = {classifier.score(x_train, y_train):.4f}")

voting_classifier.fit(x_train, y_train)
print(f"train voting all score = {voting_classifier.score(x_train, y_train):.4f}")

y_pred = voting_classifier.predict(x_test)

voting_classifier_accuracy_score = accuracy_score(y_true=y_test, y_pred=y_pred)
print(f"voting_classifier_accuracy_score: {voting_classifier_accuracy_score:.4f}")

# --------------- END ----------------------------------------------------------------
sys.exit(0)
