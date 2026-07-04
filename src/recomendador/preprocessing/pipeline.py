"""Preprocessamento de interacoes para treino do recomendador.

Transforma o DataFrame bruto (user_id, item_id, rating) em dados
prontos para o modelo neural. Executa tres tarefas:

1. k-core filtering iterativo: remove usuarios e itens com poucas
   interacoes, reduzindo esparsidade e ruido.
2. Reindexacao: mapeia os IDs originais para indices contiguos
   (0..N-1), formato exigido pelos embeddings do PyTorch.
3. Normalizacao do rating via uma PreprocessingStrategy injetada.
"""

import pandas as pd

from recomendador.preprocessing.preprocessors import PreprocessingStrategy


def apply_kcore(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Aplica k-core filtering iterativo.

    Mantem apenas usuarios e itens com pelo menos k interacoes.
    Repete ate o conjunto estabilizar, pois remover usuarios pode
    tornar itens raros e vice-versa.

    Args:
        df: Interacoes com colunas user_id, item_id.
        k: Minimo de interacoes exigido por usuario e por item.

    Returns:
        DataFrame filtrado.
    """
    atual = df
    while True:
        antes = len(atual)
        contagem_user = atual["user_id"].value_counts()
        users_validos = contagem_user[contagem_user >= k].index
        atual = atual[atual["user_id"].isin(users_validos)]

        contagem_item = atual["item_id"].value_counts()
        itens_validos = contagem_item[contagem_item >= k].index
        atual = atual[atual["item_id"].isin(itens_validos)]

        if len(atual) == antes:
            break
    return atual.reset_index(drop=True)


def reindex_ids(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    """Mapeia IDs originais para indices contiguos 0..N-1.

    Args:
        df: Interacoes com colunas user_id, item_id.

    Returns:
        Tupla (df_reindexado, mapa_users, mapa_items), onde os mapas
        traduzem indice interno de volta para o ID original.
    """
    user_ids = df["user_id"].unique()
    item_ids = df["item_id"].unique()

    user_para_idx = {uid: idx for idx, uid in enumerate(user_ids)}
    item_para_idx = {iid: idx for idx, iid in enumerate(item_ids)}

    resultado = df.copy()
    resultado["user_id"] = resultado["user_id"].map(user_para_idx)
    resultado["item_id"] = resultado["item_id"].map(item_para_idx)

    idx_para_user = {idx: uid for uid, idx in user_para_idx.items()}
    idx_para_item = {idx: iid for iid, idx in item_para_idx.items()}
    return resultado, idx_para_user, idx_para_item


def preprocess(
    df: pd.DataFrame,
    strategy: PreprocessingStrategy,
    k: int = 5,
) -> tuple[pd.DataFrame, dict, dict]:
    """Executa o pipeline completo de preprocessamento.

    Args:
        df: Interacoes brutas (user_id, item_id, rating).
        strategy: Estrategia de normalizacao do rating.
        k: Parametro do k-core filtering.

    Returns:
        Tupla (df_processado, mapa_users, mapa_items).
    """
    filtrado = apply_kcore(df, k=k)
    normalizado = strategy.apply(filtrado)
    reindexado, mapa_users, mapa_items = reindex_ids(normalizado)
    return reindexado, mapa_users, mapa_items
