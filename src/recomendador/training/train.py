"""Script de treino ponta a ponta do recomendador.

Orquestra o fluxo completo: carrega o RetailRocket, aplica
preprocessamento (k-core + normalizacao), divide treino/teste,
treina o modelo neural e um baseline, e compara as metricas.

Uso:
    uv run python -m recomendador.training.train
"""

import numpy as np
from sklearn.model_selection import train_test_split

from recomendador.data.loader import load_retailrocket
from recomendador.evaluation.metrics import compute_metrics
from recomendador.models.baseline import MeanBaseline
from recomendador.models.neural import NeuralRecommender
from recomendador.preprocessing.pipeline import apply_kcore, reindex_ids
from recomendador.preprocessing.preprocessors import MinMaxRatingStrategy

# --- Parametros (serao migrados para config/.env no refino) ---
DATA_PATH = "C:/Users/PlayHard/Downloads/RetailRocket/events.csv"
KCORE = 5
SAMPLE_USERS = 20000  # None para usar todos os usuarios
SEED = 42
EMBEDDING_DIM = 32
LEARNING_RATE = 0.001
EPOCHS = 5
BATCH_SIZE = 256


def _sample_users(df, n_users, seed):
    """Amostra um subconjunto de usuarios para agilizar o treino.

    Args:
        df: Interacoes com coluna user_id.
        n_users: Quantidade de usuarios a manter (None = todos).
        seed: Semente para amostragem reproduzivel.

    Returns:
        DataFrame contendo apenas os usuarios amostrados.
    """
    if n_users is None:
        return df
    usuarios = df["user_id"].unique()
    if len(usuarios) <= n_users:
        return df
    rng = np.random.default_rng(seed)
    escolhidos = rng.choice(usuarios, size=n_users, replace=False)
    return df[df["user_id"].isin(escolhidos)]


def main() -> None:
    """Executa o pipeline de treino e imprime a comparacao."""
    print("1/5 Carregando dados...")
    df = load_retailrocket(DATA_PATH)

    print("2/5 Aplicando k-core e amostragem...")
    df = apply_kcore(df, k=KCORE)
    df = _sample_users(df, SAMPLE_USERS, SEED)
    df = apply_kcore(df, k=KCORE)  # re-filtra apos amostrar

    print("3/5 Normalizando e reindexando...")
    df = MinMaxRatingStrategy().apply(df)
    df, _, _ = reindex_ids(df)

    n_users = int(df["user_id"].max()) + 1
    n_items = int(df["item_id"].max()) + 1
    print(f"    Interacoes: {len(df)} | Usuarios: {n_users} | Itens: {n_items}")

    print("4/5 Dividindo treino/teste e treinando...")
    treino, teste = train_test_split(df, test_size=0.2, random_state=SEED)

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

    baseline = MeanBaseline()
    baseline.fit(
        treino["user_id"].to_numpy(),
        treino["item_id"].to_numpy(),
        treino["rating"].to_numpy(),
    )

    print("5/5 Avaliando...")
    y_true = teste["rating"].to_numpy()
    pred_neural = neural.predict(teste["user_id"].to_numpy(), teste["item_id"].to_numpy())
    pred_baseline = baseline.predict(teste["user_id"].to_numpy(), teste["item_id"].to_numpy())

    m_neural = compute_metrics(y_true, pred_neural)
    m_baseline = compute_metrics(y_true, pred_baseline)

    print()
    print("=" * 40)
    print("RESULTADOS (menor = melhor)")
    print("=" * 40)
    print(f"{'Metrica':<10}{'Neural':>12}{'Baseline':>14}")
    print(f"{'RMSE':<10}{m_neural['rmse']:>12.4f}{m_baseline['rmse']:>14.4f}")
    print(f"{'MAE':<10}{m_neural['mae']:>12.4f}{m_baseline['mae']:>14.4f}")
    print("=" * 40)


if __name__ == "__main__":
    main()
