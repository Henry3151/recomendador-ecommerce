"""Carregamento de interacoes usuario-item a partir de dados brutos.

Este modulo e o unico ponto do sistema que conhece o formato bruto
do dataset. Ele traduz os dados de origem (RetailRocket) para o
contrato interno usado por todo o resto do projeto:

    colunas: user_id, item_id, rating

Isolar essa traducao aqui permite trocar de dataset alterando apenas
este arquivo, sem impacto no preprocessamento, modelo ou treino.
"""

from pathlib import Path

import pandas as pd

# Traducao dos eventos implicitos do RetailRocket para uma escala
# numerica de interesse. Comprar sinaliza mais interesse que so olhar.
EVENT_WEIGHTS = {
    "view": 1.0,
    "addtocart": 2.0,
    "transaction": 3.0,
}


def load_retailrocket(csv_path: str | Path) -> pd.DataFrame:
    """Le o events.csv do RetailRocket no contrato interno.

    Args:
        csv_path: Caminho para o arquivo events.csv.

    Returns:
        DataFrame com as colunas user_id, item_id e rating.

    Raises:
        FileNotFoundError: Se o arquivo nao existir.
        ValueError: Se colunas esperadas estiverem ausentes.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {path}")

    df = pd.read_csv(path, usecols=["visitorid", "event", "itemid"])

    colunas_esperadas = {"visitorid", "event", "itemid"}
    faltando = colunas_esperadas - set(df.columns)
    if faltando:
        raise ValueError(f"Colunas ausentes no CSV: {sorted(faltando)}")

    df["rating"] = df["event"].map(EVENT_WEIGHTS)
    df = df.dropna(subset=["rating"])

    resultado = df.rename(columns={"visitorid": "user_id", "itemid": "item_id"})
    return resultado[["user_id", "item_id", "rating"]].reset_index(drop=True)
