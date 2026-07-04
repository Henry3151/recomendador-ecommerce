"""Estrategias de pre-processamento de interacoes usuario-item.

Implementa o padrao Strategy: cada estrategia encapsula uma forma
diferente de transformar os dados, e todas seguem a mesma interface.
Isso permite trocar o pre-processamento sem alterar o codigo de treino.
"""

from abc import ABC, abstractmethod

import pandas as pd


class PreprocessingStrategy(ABC):
    """Interface comum a todas as estrategias de pre-processamento."""

    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica a transformacao e devolve um novo DataFrame.

        Args:
            df: Interacoes com colunas userId, movieId, rating.

        Returns:
            DataFrame transformado (a entrada nao e modificada).
        """
        raise NotImplementedError


class MinMaxRatingStrategy(PreprocessingStrategy):
    """Normaliza a coluna rating para o intervalo [0, 1]."""

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Escala o rating usando min-max.

        Args:
            df: Interacoes com a coluna rating.

        Returns:
            DataFrame com rating em [0, 1].
        """
        result = df.copy()
        min_rating = result["rating"].min()
        max_rating = result["rating"].max()
        span = max_rating - min_rating
        if span == 0:
            result["rating"] = 0.0
        else:
            result["rating"] = (result["rating"] - min_rating) / span
        return result


class BinaryRatingStrategy(PreprocessingStrategy):
    """Converte rating em sinal binario de interesse (1 = gostou)."""

    def __init__(self, threshold: float = 3.5) -> None:
        """Inicializa a estrategia.

        Args:
            threshold: Nota minima para o item ser considerado positivo.
        """
        self.threshold = threshold

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Binariza o rating a partir do limiar.

        Args:
            df: Interacoes com a coluna rating.

        Returns:
            DataFrame com rating em {0.0, 1.0}.
        """
        result = df.copy()
        result["rating"] = (result["rating"] >= self.threshold).astype(float)
        return result
