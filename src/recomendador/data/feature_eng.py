"""Estagio 2 do pipeline: engenharia de features.

Le as interacoes limpas, aplica normalizacao do rating (Strategy)
e reindexacao dos IDs para indices contiguos (exigencia dos
embeddings). Salva as features prontas para o treino.

Uso:
    uv run python -m recomendador.data.feature_eng
"""

from pathlib import Path

import pandas as pd

from recomendador.preprocessing.pipeline import reindex_ids
from recomendador.preprocessing.preprocessors import MinMaxRatingStrategy

CLEAN_PATH = "data/processed/clean.parquet"
FEATURES_PATH = "data/processed/features.parquet"


def main() -> None:
    """Aplica normalizacao e reindexacao, salvando as features."""
    print("Carregando dados limpos...")
    df = pd.read_parquet(CLEAN_PATH)

    print("Normalizando rating (Strategy)...")
    df = MinMaxRatingStrategy().apply(df)

    print("Reindexando IDs...")
    df, _, _ = reindex_ids(df)

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    df.to_parquet(FEATURES_PATH, index=False)
    print(f"Features salvas em {FEATURES_PATH}")
    print(f"  Interacoes: {len(df)} | Usuarios: {df['user_id'].nunique()}")
    print(f"  Itens: {df['item_id'].nunique()}")


if __name__ == "__main__":
    main()
