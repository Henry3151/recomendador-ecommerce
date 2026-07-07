"""Metricas de avaliacao do recomendador.

Reune metricas de erro de predicao (RMSE, MAE) e metricas de ranking
(Precision@k, Recall@k), estas ultimas mais adequadas para avaliar
sistemas de recomendacao. Comparam valores previstos vs. reais.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calcula metricas de erro entre valores reais e previstos.

    Args:
        y_true: Ratings reais.
        y_pred: Ratings previstos pelo modelo.

    Returns:
        Dicionario com rmse e mae.
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return {"rmse": rmse, "mae": mae}


def precision_recall_at_k(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: int = 10,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Calcula Precision@k e Recall@k para ranking de recomendacao.

    Ordena os itens pela pontuacao prevista, considera os k melhores
    e verifica quantos sao de fato relevantes (rating real acima do
    limiar).

    Args:
        y_true: Ratings reais (normalizados em [0, 1]).
        y_pred: Pontuacoes previstas pelo modelo.
        k: Numero de itens no topo do ranking.
        threshold: Limiar para considerar um item relevante.

    Returns:
        Dicionario com precision_at_k e recall_at_k.
    """
    k = min(k, len(y_pred))
    if k == 0:
        return {"precision_at_k": 0.0, "recall_at_k": 0.0}

    ordem = np.argsort(y_pred)[::-1]
    top_k = ordem[:k]

    relevantes_no_topo = int(np.sum(y_true[top_k] >= threshold))
    total_relevantes = int(np.sum(y_true >= threshold))

    precision = relevantes_no_topo / k
    recall = relevantes_no_topo / total_relevantes if total_relevantes > 0 else 0.0

    return {"precision_at_k": float(precision), "recall_at_k": float(recall)}


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: int = 10,
) -> dict[str, float]:
    """Reune todas as metricas (erro + ranking) num unico dicionario.

    Args:
        y_true: Ratings reais.
        y_pred: Pontuacoes previstas.
        k: Numero de itens para as metricas de ranking.

    Returns:
        Dicionario com rmse, mae, precision_at_k e recall_at_k.
    """
    erro = compute_metrics(y_true, y_pred)
    ranking = precision_recall_at_k(y_true, y_pred, k=k)
    return {**erro, **ranking}
