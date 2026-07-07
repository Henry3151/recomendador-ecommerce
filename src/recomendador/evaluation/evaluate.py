"""Estagio de avaliacao do pipeline DVC.

Carrega o modelo treinado e o conjunto de teste, treina um baseline
para comparacao, calcula 4 metricas (RMSE, MAE, Precision@k, Recall@k)
para ambos e salva o resultado em metrics.json (rastreado pelo DVC).

Uso:
    uv run python -m recomendador.evaluation.evaluate
"""

import json
from pathlib import Path

import mlflow
import pandas as pd
import torch

from recomendador.evaluation.metrics import compute_all_metrics
from recomendador.models.baseline import MeanBaseline
from recomendador.models.neural import _MLPModule

MODEL_PATH = "models/neural.pt"
TRAIN_INFO_PATH = "models/train_info.json"
TEST_PATH = "data/processed/test.parquet"
FEATURES_PATH = "data/processed/features.parquet"
METRICS_PATH = "metrics.json"
EXPERIMENT_NAME = "recomendador-ecommerce"


def main() -> None:
    """Avalia o modelo neural e o baseline com 4 metricas."""
    info = json.loads(Path(TRAIN_INFO_PATH).read_text())
    teste = pd.read_parquet(TEST_PATH)
    treino_completo = pd.read_parquet(FEATURES_PATH)

    # Reconstroi o modelo neural e carrega os pesos salvos
    modelo = _MLPModule(info["n_users"], info["n_items"], info["embedding_dim"])
    modelo.load_state_dict(torch.load(MODEL_PATH))
    modelo.eval()

    users_t = torch.as_tensor(teste["user_id"].to_numpy(), dtype=torch.long)
    items_t = torch.as_tensor(teste["item_id"].to_numpy(), dtype=torch.long)
    with torch.no_grad():
        pred_neural = modelo(users_t, items_t).numpy()

    # Baseline para comparacao
    baseline = MeanBaseline()
    baseline.fit(
        treino_completo["user_id"].to_numpy(),
        treino_completo["item_id"].to_numpy(),
        treino_completo["rating"].to_numpy(),
    )
    pred_baseline = baseline.predict(
        teste["user_id"].to_numpy(), teste["item_id"].to_numpy()
    )

    y_true = teste["rating"].to_numpy()
    m_neural = compute_all_metrics(y_true, pred_neural, k=10)
    m_baseline = compute_all_metrics(y_true, pred_baseline, k=10)

    metricas = {
        "neural_rmse": m_neural["rmse"],
        "neural_mae": m_neural["mae"],
        "neural_precision_at_k": m_neural["precision_at_k"],
        "neural_recall_at_k": m_neural["recall_at_k"],
        "baseline_rmse": m_baseline["rmse"],
        "baseline_mae": m_baseline["mae"],
        "baseline_precision_at_k": m_baseline["precision_at_k"],
        "baseline_recall_at_k": m_baseline["recall_at_k"],
    }

    Path(METRICS_PATH).write_text(json.dumps(metricas, indent=2))

    # Registra as metricas no MLflow tambem
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run():
        mlflow.log_metrics(metricas)

    print("=" * 55)
    print("AVALIACAO (4 metricas)")
    print("=" * 55)
    print(f"{'Metrica':<18}{'Neural':>12}{'Baseline':>14}")
    print(f"{'RMSE':<18}{m_neural['rmse']:>12.4f}{m_baseline['rmse']:>14.4f}")
    print(f"{'MAE':<18}{m_neural['mae']:>12.4f}{m_baseline['mae']:>14.4f}")
    print(
        f"{'Precision@10':<18}{m_neural['precision_at_k']:>12.4f}"
        f"{m_baseline['precision_at_k']:>14.4f}"
    )
    print(
        f"{'Recall@10':<18}{m_neural['recall_at_k']:>12.4f}"
        f"{m_baseline['recall_at_k']:>14.4f}"
    )
    print("=" * 55)


if __name__ == "__main__":
    main()
