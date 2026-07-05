"""Estagio de treino do pipeline DVC, com MLflow e registro de modelo.

Le os dados processados, treina o modelo neural e o baseline, compara
metricas, registra tudo no MLflow (parametros, metricas e o modelo
treinado) e salva as metricas em metrics.json para o DVC rastrear.

Parametros do modelo podem ser sobrescritos por variaveis de ambiente.

Uso:
    uv run python -m recomendador.training.train
"""

import json
import os
from pathlib import Path

import mlflow
import mlflow.pytorch
import pandas as pd
from sklearn.model_selection import train_test_split

from recomendador.evaluation.metrics import compute_metrics
from recomendador.models.baseline import MeanBaseline
from recomendador.models.neural import NeuralRecommender

PROCESSED_PATH = "data/processed/interactions.parquet"
METRICS_PATH = "metrics.json"
SEED = 42

EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "32"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.001"))
EPOCHS = int(os.getenv("EPOCHS", "5"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "256"))

EXPERIMENT_NAME = "recomendador-ecommerce"
REGISTERED_MODEL_NAME = "recomendador-neural"


def main() -> None:
    """Executa o treino a partir dos dados processados."""
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        mlflow.log_params(
            {
                "embedding_dim": EMBEDDING_DIM,
                "learning_rate": LEARNING_RATE,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "seed": SEED,
            }
        )

        print("Carregando dados processados...")
        df = pd.read_parquet(PROCESSED_PATH)
        n_users = int(df["user_id"].max()) + 1
        n_items = int(df["item_id"].max()) + 1
        print(f"  Interacoes: {len(df)} | Usuarios: {n_users} | Itens: {n_items}")

        treino, teste = train_test_split(df, test_size=0.2, random_state=SEED)

        print("Treinando modelo neural...")
        neural = NeuralRecommender(
            n_users=n_users,
            n_items=n_items,
            embedding_dim=EMBEDDING_DIM,
            learning_rate=LEARNING_RATE,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            seed=SEED,
        )
        neural.fit(
            treino["user_id"].to_numpy(),
            treino["item_id"].to_numpy(),
            treino["rating"].to_numpy(),
        )

        print("Treinando baseline...")
        baseline = MeanBaseline()
        baseline.fit(
            treino["user_id"].to_numpy(),
            treino["item_id"].to_numpy(),
            treino["rating"].to_numpy(),
        )

        print("Avaliando...")
        y_true = teste["rating"].to_numpy()
        pred_neural = neural.predict(
            teste["user_id"].to_numpy(), teste["item_id"].to_numpy()
        )
        pred_baseline = baseline.predict(
            teste["user_id"].to_numpy(), teste["item_id"].to_numpy()
        )

        m_neural = compute_metrics(y_true, pred_neural)
        m_baseline = compute_metrics(y_true, pred_baseline)

        metricas = {
            "neural_rmse": m_neural["rmse"],
            "neural_mae": m_neural["mae"],
            "baseline_rmse": m_baseline["rmse"],
            "baseline_mae": m_baseline["mae"],
        }
        mlflow.log_metrics(metricas)

        # Registra o modelo neural no MLflow (habilita o Model Registry)
        print("Registrando modelo no MLflow...")
        mlflow.pytorch.log_model(
            neural.model,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
        )

        Path(METRICS_PATH).write_text(json.dumps(metricas, indent=2))

        print()
        print("=" * 40)
        print("RESULTADOS (menor = melhor)")
        print("=" * 40)
        print(f"{'Metrica':<10}{'Neural':>12}{'Baseline':>14}")
        print(f"{'RMSE':<10}{m_neural['rmse']:>12.4f}{m_baseline['rmse']:>14.4f}")
        print(f"{'MAE':<10}{m_neural['mae']:>12.4f}{m_baseline['mae']:>14.4f}")
        print("=" * 40)
        print(f"Run: emb={EMBEDDING_DIM}, lr={LEARNING_RATE}, epochs={EPOCHS}")


if __name__ == "__main__":
    main()
