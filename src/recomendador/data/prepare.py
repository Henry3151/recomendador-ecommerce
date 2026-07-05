"""Estagio de preparacao de dados do pipeline DVC.

Le o dataset bruto versionado (data/raw/events.csv), aplica
o preprocessamento (k-core, amostragem, normalizacao, reindexacao)
e salva o resultado em data/processed/interactions.parquet.

Este e o primeiro estagio do pipeline reproduzivel (dvc.yaml).

Uso:
    uv run python -m recomendador.data.prepare
"""

from pathlib import Path

import numpy as np

from recomendador.data.loader import load_retailrocket
from recomendador.preprocessing.pipeline import apply_kcore, reindex_ids
from recomendador.preprocessing.preprocessors import MinMaxRatingStrategy

RAW_PATH = "data/raw/events.csv"
PROCESSED_PATH = "data/processed/interactions.parquet"
KCORE = 5
SAMPLE_USERS = 20000
SEED = 42


def _sample_users(df, n_users, seed):
    """Amostra um subconjunto de usuarios.

    Args:
        df: Interacoes com coluna user_id.
        n_users: Quantidade de usuarios a manter (None = todos).
        seed: Semente para amostragem reproduzivel.

    Returns:
        DataFrame apenas com os usuarios amostrados.
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
    """Executa a preparacao e salva os dados processados."""
    print("Carregando dataset bruto...")
    df = load_retailrocket(RAW_PATH)

    print("Aplicando k-core e amostragem...")
    df = apply_kcore(df, k=KCORE)
    df = _sample_users(df, SAMPLE_USERS, SEED)
    df = apply_kcore(df, k=KCORE)

    print("Normalizando e reindexando...")
    df = MinMaxRatingStrategy().apply(df)
    df, _, _ = reindex_ids(df)

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_PATH, index=False)

    print(f"Dados salvos em {PROCESSED_PATH}")
    print(f"  Interacoes: {len(df)}")
    print(f"  Usuarios: {df['user_id'].nunique()}")
    print(f"  Itens: {df['item_id'].nunique()}")


if __name__ == "__main__":
    main()
