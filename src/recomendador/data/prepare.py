"""Estagio 1 do pipeline: preprocessamento (limpeza).

Le o dataset bruto, aplica k-core filtering e amostragem, e salva
as interacoes limpas. A normalizacao e reindexacao ficam no estagio
seguinte (feature_eng).

Uso:
    uv run python -m recomendador.data.prepare
"""

from pathlib import Path

import numpy as np

from recomendador.data.loader import load_retailrocket
from recomendador.preprocessing.pipeline import apply_kcore

RAW_PATH = "data/raw/events.csv"
CLEAN_PATH = "data/processed/clean.parquet"
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
    """Executa a limpeza e salva as interacoes."""
    print("Carregando dataset bruto...")
    df = load_retailrocket(RAW_PATH)

    print("Aplicando k-core e amostragem...")
    df = apply_kcore(df, k=KCORE)
    df = _sample_users(df, SAMPLE_USERS, SEED)
    df = apply_kcore(df, k=KCORE)

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    df.to_parquet(CLEAN_PATH, index=False)
    print(f"Dados limpos salvos em {CLEAN_PATH} ({len(df)} interacoes)")


if __name__ == "__main__":
    main()
