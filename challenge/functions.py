import math

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# ----- FUNÇÃO AUXILIAR PARA GRÁFICOS -----
def plot_distribution_grid(df, columns, plot_type='hist', suptitle='Distribuição das Colunas'):
    """
    Gera um grid de subplots para visualização da distribuição das colunas.
    plot_type: 'hist' para numéricas, 'count' para categóricas.
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
                        cbar=True,
                        ax=ax,
                        color='steelblue')
            ax.set_title(f'Histograma: {col}', fontsize=10)
            ax.set_xlabel('')
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
    plt.tight_layout()
    plt.show()


# ----- FUNÇÃO PARA ENCONTRAR MELHOR KNN -----
def get_best_knn_classifier(x_train, y_train, x_test, y_test, k_range=range(1, 20)):
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
    """Retorna uma instância de LinearSVC configurada."""
    return LinearSVC(random_state=random_state)


def get_linear_svc_for_voting(random_state=42):
    """
    Wrapper para LinearSVC que habilita predict_proba via CalibratedClassifierCV.
    Essencial para VotingClassifier com voting='soft', evitando o AttributeError.
    """
    return CalibratedClassifierCV(
        LinearSVC(C=1.0, random_state=random_state, max_iter=10000),
        cv=3
    )


def get_logistic_regression(random_state=42):
    return LogisticRegression(random_state=random_state)


def get_random_forest(random_state=42):
    return RandomForestClassifier(n_estimators=100,
                                  max_leaf_nodes=10,
                                  n_jobs=-1,
                                  random_state=random_state)


def get_svc(random_state=42):
    return SVC(probability=True,
               random_state=random_state)


def get_decision_tree(random_state=42):
    return DecisionTreeClassifier(random_state=random_state)


def plot_boxplot_outliers(df, columns):
    """
    Gera um grid de boxplots para as colunas especificadas, marcando os outliers.
    """
    num_cols = len(columns)
    if num_cols == 0:
        return

    rows = math.ceil(num_cols / 2)
    fig, axes = plt.subplots(rows, 2, figsize=(8, 5 * rows))
    axes_flat = axes.flatten()

    for i, col in enumerate(columns):
        ax = axes_flat[i]
        sb.boxplot(y=df[col].dropna(), ax=ax,
                   flierprops={'markerfacecolor': 'red', 'marker': 'o', 'markersize': 8})
        ax.set_title(f'Boxplot: {col}', fontsize=10)
        ax.set_xlabel('')

    for ax in axes_flat[num_cols:]:
        ax.set_visible(False)

    plt.suptitle('Boxplots com Outliers', fontsize=12)
    plt.tight_layout()
    plt.show()


def get_decision_tree(random_state=42):
    return DecisionTreeClassifier(random_state=random_state)


def detect_outliers_lof(X, n_neighbors=20, contamination=0.1):
    """
    Detecta outliers utilizando o algoritmo Local Outlier Factor (LOF).

    Args:
        X (array-like): Dados de entrada (features) para detecção.
        n_neighbors (int): Número de vizinhos para considerar.
        contamination (float): Proporção esperada de outliers.

    Returns:
        labels (array-like): Rótulos onde 1 é inlier e -1 é outlier.
    """
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    return lof.fit_predict(X)


def detect_outliers_isolation_forest(X_train, contamination=0.1, random_state=42):
    """
    Detecta outliers no conjunto de treinamento usando Isolation Forest.

    Args:
        X_train: Features numéricas de treinamento.
        contamination: Proporção esperada de outliers no dataset (float entre 0 e 0.5).
        random_state: Semente para reprodutibilidade.

    Returns:
        labels: Array com -1 para outliers e 1 para inliers.
        model: Modelo Isolation Forest ajustado.
    """
    model = IsolationForest(contamination=contamination, random_state=random_state)
    labels = model.fit_predict(X_train)
    return labels, model


def map_in_and_outlier_labels(labels, column: str = 'in_and_outlier_labels') -> pd.DataFrame:
    """
    Mapeia rótulos numéricos de outliers (1/-1) para rótulos descritivos ('inlier'/'outlier')
    e adiciona o resultado como uma nova coluna no DataFrame.

    Args:
        labels: Array-like contendo 1 (inlier) e -1 (outlier).
        column (str): Nome da coluna a ser adicionada com os rótulos mapeados.

    Returns:
        pd.DataFrame: Cópia do DataFrame com a coluna de rótulos adicionada.
    """
    mapping = {1: 'inlier', -1: 'outlier'}
    return pd.DataFrame({column: pd.Series(labels).map(mapping)})
