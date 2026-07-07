"""Estagio de treino do pipeline DVC.

Le os dados processados, treina o modelo neural (com early stopping)
e registra no MLflow. Salva o modelo treinado para o estagio de
avaliacao usar. Parametros sobrescreviveis por variaveis de ambiente.

Uso:
    uv run python -m recomendador.training.train
"""

import json
import os
from pathlib import Path

import mlflow
import mlflow.pytorch
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from recomendador.models.neural import NeuralRecommender

FEATURES_PATH = "data/processed/features.parquet"
MODEL_PATH = "models/neural.pt"
TRAIN_INFO_PATH = "models/train_info.json"
SEED = 42

EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "64"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.001"))
EPOCHS = int(os.getenv("EPOCHS", "20"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "256"))
PATIENCE = int(os.getenv("PATIENCE", "3"))

EXPERIMENT_NAME = "recomendador-ecommerce"


def main() -> None:
    """Treina o modelo e salva para avaliacao."""
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        mlflow.log_params(
            {
                "embedding_dim": EMBEDDING_DIM,
                "learning_rate": LEARNING_RATE,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "patience": PATIENCE,
                "seed": SEED,
            }
        )

        print("Carregando features...")
        df = pd.read_parquet(FEATURES_PATH)
        n_users = int(df["user_id"].max()) + 1
        n_items = int(df["item_id"].max()) + 1
        print(f"  Interacoes: {len(df)} | Usuarios: {n_users} | Itens: {n_items}")

        treino, teste = train_test_split(df, test_size=0.2, random_state=SEED)

        print("Treinando modelo neural (com early stopping)...")
        neural = NeuralRecommender(
            n_users=n_users,
            n_items=n_items,
            embedding_dim=EMBEDDING_DIM,
            learning_rate=LEARNING_RATE,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            seed=SEED,
            patience=PATIENCE,
        )
        neural.fit(
            treino["user_id"].to_numpy(),
            treino["item_id"].to_numpy(),
            treino["rating"].to_numpy(),
        )
        print(f"  Early stopping: melhor epoca = {neural.best_epoch}")
        mlflow.log_metric("best_epoch", neural.best_epoch)

        # Salva o modelo e o conjunto de teste para o estagio de avaliacao
        Path("models").mkdir(exist_ok=True)
        torch.save(neural.model.state_dict(), MODEL_PATH)
        info = {
            "n_users": n_users,
            "n_items": n_items,
            "embedding_dim": EMBEDDING_DIM,
            "best_epoch": neural.best_epoch,
        }
        Path(TRAIN_INFO_PATH).write_text(json.dumps(info, indent=2))
        teste.to_parquet("data/processed/test.parquet", index=False)

        # Registra o modelo no MLflow Model Registry
        mlflow.pytorch.log_model(
            neural.model,
            artifact_path="model",
            registered_model_name="recomendador-neural",
        )
        print("Modelo salvo e registrado no MLflow.")


if __name__ == "__main__":
    main()
