import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from pandas.core.interchange.dataframe_protocol import DataFrame
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# --------------- CONSTANTES ---------------------------------------------------------
"""
    Semente padrão utilizada para garantir reprodutibilidade nos algoritmos de Machine Learning.
"""
RANDOM_STATE = 42


# --------------- FUNCTIONS ----------------------------------------------------------

def fprint(str):
    """
    Imprime uma mensagem formatada no console com linhas de separação visuais.

    Args:
        str (str): A mensagem a ser exibida.
    """
    print(f'{"-" * 80}\n{str}')


def get_total_elements(df):
    """
    Retorna o número total de elementos (linhas * colunas) em um DataFrame.

    Args:
        df (pd.DataFrame): O dataset de entrada.

    Returns:
        int: Número total de elementos no DataFrame.
    """
    return df.shape[0] * df.shape[1]


def get_total_rows(df):
    """
    Retorna o número total de linhas (amostras) em um DataFrame.

    Args:
        df (pd.DataFrame): O dataset de entrada.

    Returns:
        int: Número total de linhas no DataFrame.
    """
    return df.shape[0]


def convert_columns_to_string(df, columns):
    """
    Converte múltiplas colunas especificadas de um DataFrame para o tipo string.
    Utiliza métodos vetorizados do Pandas para melhor performance e consistência.

    Args:
        df (pd.DataFrame): O dataset de entrada.
        columns (list): Lista com os nomes das colunas a serem convertidas.

    Returns:
        pd.DataFrame: O DataFrame com as colunas convertidas para string.
    """
    df[columns] = df[columns].astype(str)
    return df

def convert_columns_to_numeric(df, columns):
    """
    Converte múltiplas colunas especificadas de um DataFrame para o tipo numérico.
    Utiliza pd.to_numeric com tratamento de erros para garantir consistência.

    Args:
        df (pd.DataFrame): O dataset de entrada.
        columns (list): Lista com os nomes das colunas a serem convertidas.

    Returns:
        pd.DataFrame: O DataFrame com as colunas convertidas para numérico.
    """
    # Converte cada coluna, transformando strings inválidas em NaN ao invés de disparar erro
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def plot_distribution_grid(df, columns, plot_type='hist', suptitle='Distribuição das Colunas'):
    """
    Gera um grid de subplots para visualização da distribuição das colunas.

    Args:
        df (pd.DataFrame): O dataset a ser plotado.
        columns (list): Lista de nomes das colunas a serem plotadas.
        plot_type (str): Tipo de gráfico ('hist' para numéricas, 'count' para categóricas).
        suptitle (str): Título principal do gráfico.
    """
    num_cols = len(columns)
    if num_cols == 0:
        return

    rows = math.ceil(num_cols / 2)
    fig, axes = plt.subplots(rows, 2, figsize=(8, 5 * rows))
    axes_flat = axes.flatten()

    for i, col in enumerate(columns):
        ax = axes_flat[i]
        if plot_type == 'hist':
            sb.histplot(df[col].dropna(),
                        kde=True,
                        bins=100,
                        ax=ax,
                        color='steelblue')
            ax.set_title(f'Histograma: {col}', fontsize=10)
            ax.set_xlabel('')
            ax.xaxis.set_major_locator(plt.MaxNLocator(10))
            ax.tick_params(axis='x', rotation=90)
        elif plot_type == 'count':
            sb.countplot(data=df, x=col, ax=ax, color='steelblue')
            ax.set_title(f'Distribuição: {col}', fontsize=10)
            ax.set_xlabel('')
            ax.tick_params(axis='x', rotation=45)
            for p in ax.patches:
                ax.annotate(f'{int(p.get_height())}',
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='bottom', fontsize=8)
    # Hide unused axes
    for ax in axes_flat[num_cols:]:
        ax.set_visible(False)

    plt.suptitle(f'{suptitle} ({plot_type})', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def plot_boxplot_outliers(df, columns, bins=30, y_group_size=None):
    """
    Gera um grid de boxplots para as colunas especificadas, marcando os outliers.
    Adiciona agrupamento de valores no eixo Y para facilitar a leitura.

    Args:
        df (pd.DataFrame): O dataset a ser plotado.
        columns (list): Lista de nomes das colunas a serem plotadas.
        bins (int): Número de bins para histograma (padrão: 30).
        y_group_size (int): Tamanho do intervalo para agrupamento de valores no eixo Y.
                           Se None, usa valor automático baseado nos dados.
    """
    # Filter to only numeric columns
    numeric_columns = []
    for col in columns:
        try:
            # Test if column can be converted to numeric
            pd.to_numeric(df[col].dropna())
            numeric_columns.append(col)
        except (ValueError, TypeError):
            continue  # Skip non-numeric columns

    num_cols = len(numeric_columns)
    if num_cols == 0:
        print("Nenhuma coluna numérica encontrada para plotar.")
        return

    # Adjust figure size based on number of columns to prevent label overlap
    fig_width = max(10, num_cols * 3)
    fig_height = max(5, (num_cols + 1) // 2 * 3)
    fig, axes = plt.subplots((num_cols + 1) // 2, 2, figsize=(fig_width, fig_height))
    axes_flat = axes.flatten()

    for i, col in enumerate(numeric_columns):
        ax = axes_flat[i]

        # Convert to numeric explicitly, handling errors
        col_data = pd.to_numeric(df[col], errors='coerce').dropna()

        if len(col_data) == 0:
            continue

        sb.boxplot(y=col_data, ax=ax,
                   flierprops={'markerfacecolor': 'red', 'marker': 'o', 'markersize': 8})
        ax.set_title(f'Boxplot: {col}', fontsize=10)
        ax.set_xlabel('')

        # Configure Y-axis grouping
        if y_group_size is None:
            # Auto-calculate group size based on data range
            if len(col_data) > 0:
                range_val = col_data.max() - col_data.min()
                # Aim for ~5-10 ticks on Y axis
                y_group_size_auto = max(1, int(range_val / 8))
            else:
                y_group_size_auto = 1
        else:
            y_group_size_auto = y_group_size

        # Set Y-axis ticks to group values
        if len(col_data) > 0:
            y_min = col_data.min()
            y_max = col_data.max()
            # Create evenly spaced ticks with group size
            y_ticks = np.arange(math.floor(y_min / y_group_size_auto) * y_group_size_auto,
                                math.ceil(y_max / y_group_size_auto) * y_group_size_auto + y_group_size_auto,
                                y_group_size_auto)
            ax.set_yticks(y_ticks)

    # Hide unused axes
    for ax in axes_flat[num_cols:]:
        ax.set_visible(False)

    plt.suptitle('Boxplots com Outliers', fontsize=12)
    plt.tight_layout()
    plt.show()


# ----- FUNÇÃO PARA ENCONTRAR MELHOR KNN -----
def get_best_knn_classifier(x_train, y_train, x_test, y_test, k_range=range(1, 20)):
    """
    Encontra e retorna o classificador KNN com a melhor acurácia no conjunto de teste.

    O método testa diferentes valores de 'k' e mantém o melhor modelo encontrado.

    Args:
        x_train (array-like): Dados de treinamento (features).
        y_train (array-like): Dados de treinamento (alvo).
        x_test (array-like): Dados de teste (features).
        y_test (array-like): Dados de teste (alvo).
        k_range (range): Intervalo de valores de vizinhos (k) para testar.

    Returns:
        KNeighborsClassifier: O melhor modelo treinado.
    """
    best_acc = 0
    best_instance = None
    for i in k_range:
        instance = KNeighborsClassifier(n_neighbors=i)
        instance.fit(x_train, y_train)
        y_pred = instance.predict(x_test)
        acc = accuracy_score(y_test, y_pred)
        if acc > best_acc:
            best_acc = acc
            best_instance = instance
    return best_instance


def get_linear_svc_classifier(random_state=42):
    """
    Retorna uma instância de LinearSVC configurada.

    Args:
        random_state (int): Semente para reprodutibilidade.

    Returns:
        LinearSVC: Uma instância de SVC linear.
    """
    return LinearSVC(random_state=random_state)


def get_linear_svc_for_voting(random_state=42):
    """
    Wrapper para LinearSVC que habilita predict_proba via CalibratedClassifierCV.
    Essencial para VotingClassifier com voting='soft', evitando o AttributeError.

    Args:
        random_state (int): Semente para reprodutibilidade.

    Returns:
        CalibratedClassifierCV: Um calibrador com SVC linear dentro.
    """
    return CalibratedClassifierCV(
        LinearSVC(C=1.0, random_state=random_state, max_iter=10000),
        cv=3
    )


def get_logistic_regression(random_state=42):
    """
    Retorna uma instância de LogisticRegression configurada.

    Args:
        random_state (int): Semente para reprodutibilidade.

    Returns:
        LogisticRegression: Uma instância de regressão logística.
    """
    return LogisticRegression(random_state=random_state)


def get_random_forest(random_state=42):
    """
    Retorna uma instância de RandomForestClassifier com hiperparâmetros específicos.

    Args:
        random_state (int): Semente para reprodutibilidade.

    Returns:
        RandomForestClassifier: Uma instância de Random Forest.
    """
    return RandomForestClassifier(n_estimators=100,
                                  max_leaf_nodes=10,
                                  n_jobs=-1,
                                  random_state=random_state)


def get_svc(random_state=42):
    """
    Retorna uma instância de SVC (Support Vector Classifier) habilitada para probabilidade.

    Args:
        random_state (int): Semente para reprodutibilidade.

    Returns:
        SVC: Uma instância de Support Vector Classifier.
    """
    return SVC(probability=True,
               random_state=random_state)


def get_decision_tree(random_state=42):
    """
    Retorna uma instância de Decision Tree Classifier.

    Args:
        random_state (int): Semente para reprodutibilidade.

    Returns:
        DecisionTreeClassifier: Uma instância de Árvore de Decisão.
    """
    return DecisionTreeClassifier(random_state=random_state)


def get_decision_tree(random_state=42):
    """
    Retorna uma instância de Decision Tree Classifier.
    (Nota: Esta definição é uma duplicata da função anterior no código original).

    Args:
        random_state (int): Semente para reprodutibilidade.

    Returns:
        DecisionTreeClassifier: Uma instância de Árvore de Decisão.
    """
    return DecisionTreeClassifier(random_state=random_state)


def detect_outliers_lof(X, n_neighbors=20, contamination='auto'):
    """
    Detecta outliers utilizando o algoritmo Local Outlier Factor (LOF).

    Args:
        X (array-like): Dados de entrada (features) para detecção.
        n_neighbors (int): Número de vizinhos para considerar.
        contamination (str): Proporção esperada de outliers.

    Returns:
        labels (array-like): Rótulos onde 1 é inlier e -1 é outlier.
    """
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    return lof.fit_predict(X)


def detect_outliers_isolation_forest(X_train, contamination=0.1, random_state=42):
    """
    Detecta outliers no conjunto de treinamento usando Isolation Forest.

    Args:
        X_train (array-like): Features numéricas de treinamento.
        contamination (float): Proporção esperada de outliers no dataset (float entre 0 e 0.5).
        random_state (int): Semente para reprodutibilidade.

    Returns:
        tuple: (labels, model) onde labels são -1 para outliers e 1 para inliers.
    """
    model = IsolationForest(contamination=contamination, random_state=random_state)
    labels = model.fit_predict(X_train)
    return labels, model


def map_in_and_outlier_labels(labels, column: str = 'in_and_outlier_labels') -> pd.DataFrame:
    """
    Mapeia rótulos numéricos de outliers (1/-1) para rótulos descritivos ('inlier'/'outlier')
    e adiciona o resultado como uma nova coluna no DataFrame.

    Args:
        labels (array-like): Array contendo 1 (inlier) e -1 (outlier).
        column (str): Nome da coluna a ser adicionada com os rótulos mapeados.

    Returns:
        pd.DataFrame: Cópia do DataFrame com a coluna de rótulos adicionada.
    """
    mapping = {1: 'inlier', -1: 'outlier'}
    return pd.DataFrame({column: pd.Series(labels).map(mapping)})


def get_outlier_percentage(outliers):
    """
    Calculate the percentage of outliers with 4 decimal places precision.

    Args:
        outliers: array-like, outlier labels where -1 indicates an outlier and 1 indicates normal data

    Returns:
        float: percentage of outliers with 4 decimal places
    """
    total_count = len(outliers)
    outlier_count = (outliers == -1).sum()
    percentage = (outlier_count / total_count) * 100
    return round(percentage, 4)


def plot_correlation_heatmap(df, columns=None, figsize=(10, 8)):
    """
    Gera um mapa de calor (heatmap) da correlação entre colunas numéricas.

    Args:
        df (pd.DataFrame): O dataset a ser plotado.
        columns (list): Lista de nomes das colunas a serem incluídas. Se None, usa todas as colunas numéricas.
        figsize (tuple): Tamanho da figura para o plot (largura, altura).
    """
    # # Select numeric columns if not specified
    if columns is None:
        numeric_df = df.select_dtypes(include='number')
    else:
        numeric_df = df[columns].select_dtypes(include='number')

    # Check if there are enough numeric columns
    if numeric_df.empty or numeric_df.shape[1] < 2:
        print("No numeric columns available for correlation heatmap. Need at least 2 numeric columns.")
        return

    # Calculate correlation matrix
    corr_matrix = numeric_df.corr()

    # Create heatmap
    plt.figure(figsize=figsize)
    sb.heatmap(corr_matrix,
               annot=True,
               cmap='coolwarm',
               center=0,
               square=True,
               fmt='.2f',
               cbar_kws={"shrink": .8})

    plt.title('Mapa de Calor de Correlação')
    plt.tight_layout()
    plt.show()


def build_preprocessor(colunas_strings, colunas_numericas, col_target):
    """
    Constrói um ColumnTransformer que aplica OneHotEncoder nas colunas categóricas
    e StandardScaler nas colunas numéricas, excluindo a coluna alvo.

    Args:
        colunas_strings (list): Lista de colunas categóricas.
        colunas_numericas (list): Lista de colunas numéricas.
        col_target (str): Nome da coluna alvo a ser excluída do pré-processamento.

    Returns:
        ColumnTransformer: Preprocessador configurado.
    """
    transformers = []
    cols_str = [c for c in colunas_strings if c != col_target]
    cols_num = [c for c in colunas_numericas if c != col_target]

    for c in cols_str:
        transformers.append(('ohe_' + c, OneHotEncoder(sparse_output=False), [c]))

    for c in cols_num:
        transformers.append((c, StandardScaler(), [c]))

    return ColumnTransformer(transformers=transformers, remainder='passthrough')


def encode_labels(y):
    """
    Codifica uma série de rótulos categóricos em valores numéricos usando LabelEncoder.

    Args:
        y (array-like): Série alvo com rótulos categóricos.

    Returns:
        tuple: (y_encoded (np.ndarray), encoder (LabelEncoder))
    """
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    return y_encoded

