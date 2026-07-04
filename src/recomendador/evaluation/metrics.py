"""Metricas de avaliacao do recomendador.

Reune metricas de erro de predicao (RMSE, MAE) usadas para comparar
o modelo neural com baselines. Trabalham sobre valores previstos vs.
valores reais de rating.
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
