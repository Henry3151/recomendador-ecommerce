"""Modelo baseline de recomendacao (referencia de comparacao).

Preve sempre a media global dos ratings de treino. Serve como piso:
qualquer modelo util deve superar este baseline. Segue a mesma
interface BaseRecommender para ser intercambiavel com o modelo neural.
"""

import numpy as np

from recomendador.models.base import BaseRecommender


class MeanBaseline(BaseRecommender):
    """Baseline que preve a media global dos ratings."""

    def __init__(self) -> None:
        """Inicializa o baseline sem media definida."""
        self.media: float = 0.0

    def fit(self, users: np.ndarray, items: np.ndarray, ratings: np.ndarray) -> None:
        """Calcula a media dos ratings de treino.

        Args:
            users: Indices de usuario (nao usados).
            items: Indices de item (nao usados).
            ratings: Ratings de treino.
        """
        self.media = float(np.mean(ratings))

    def predict(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        """Preve a media global para todos os pares.

        Args:
            users: Indices de usuario.
            items: Indices de item.

        Returns:
            Vetor preenchido com a media global.
        """
        return np.full(len(users), self.media, dtype=np.float32)
