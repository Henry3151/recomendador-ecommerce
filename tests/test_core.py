"""Testes basicos das funcoes centrais do recomendador.

Cobrem o contrato de dados (loader), o preprocessamento (k-core,
reindexacao), a factory de modelos e o baseline. Sao testes rapidos
que validam o comportamento essencial sem depender do dataset real.
"""

import numpy as np
import pandas as pd

from recomendador.models.baseline import MeanBaseline
from recomendador.models.factory import ModelFactory
from recomendador.preprocessing.pipeline import apply_kcore, reindex_ids
from recomendador.preprocessing.preprocessors import (
    BinaryRatingStrategy,
    MinMaxRatingStrategy,
)


def _df_exemplo() -> pd.DataFrame:
    """Cria um DataFrame de interacoes para os testes."""
    return pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4],
            "item_id": [10, 20, 30, 10, 20, 30, 10, 20, 30, 99],
            "rating": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0],
        }
    )


def test_minmax_normaliza_entre_zero_e_um():
    """MinMaxRatingStrategy deve escalar rating para [0, 1]."""
    df = _df_exemplo()
    resultado = MinMaxRatingStrategy().apply(df)
    assert resultado["rating"].min() == 0.0
    assert resultado["rating"].max() == 1.0


def test_binary_strategy_gera_apenas_zero_e_um():
    """BinaryRatingStrategy deve produzir apenas 0.0 e 1.0."""
    df = _df_exemplo()
    resultado = BinaryRatingStrategy(threshold=2.5).apply(df)
    assert set(resultado["rating"].unique()).issubset({0.0, 1.0})


def test_kcore_remove_usuarios_raros():
    """k-core (k=3) deve remover o usuario 4 (apenas 1 interacao)."""
    df = _df_exemplo()
    resultado = apply_kcore(df, k=3)
    assert 4 not in resultado["user_id"].to_numpy()
    assert 99 not in resultado["item_id"].to_numpy()


def test_reindex_gera_indices_contiguos():
    """reindex_ids deve mapear IDs para 0..N-1 sem buracos."""
    df = _df_exemplo()
    reindexado, mapa_users, mapa_items = reindex_ids(df)
    n_users = reindexado["user_id"].nunique()
    assert set(reindexado["user_id"].unique()) == set(range(n_users))
    assert len(mapa_users) == n_users


def test_factory_cria_baseline():
    """A factory deve criar uma instancia de MeanBaseline."""
    modelo = ModelFactory.create("baseline")
    assert isinstance(modelo, MeanBaseline)


def test_factory_rejeita_tipo_desconhecido():
    """A factory deve levantar ValueError para tipo invalido."""
    try:
        ModelFactory.create("inexistente")
        raise AssertionError("Deveria ter levantado ValueError")
    except ValueError:
        pass


def test_baseline_preve_a_media():
    """MeanBaseline deve prever a media dos ratings de treino."""
    modelo = MeanBaseline()
    ratings = np.array([2.0, 4.0])
    modelo.fit(np.array([0, 1]), np.array([0, 1]), ratings)
    pred = modelo.predict(np.array([0]), np.array([0]))
    assert abs(pred[0] - 3.0) < 1e-6
