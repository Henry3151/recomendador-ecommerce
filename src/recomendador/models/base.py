"""Contrato base para modelos de recomendacao.

Define a interface comum que todo modelo (neural ou baseline) deve
seguir. Isso permite que o restante do sistema use qualquer modelo
sem conhecer sua implementacao interna (principio da abstracao).
"""

from abc import ABC, abstractmethod

import numpy as np


class BaseRecommender(ABC):
    """Interface comum a todos os modelos de recomendacao."""

    @abstractmethod
    def fit(self, users: np.ndarray, items: np.ndarray, ratings: np.ndarray) -> None:
        """Treina o modelo com as interacoes fornecidas.

        Args:
            users: Indices de usuario.
            items: Indices de item.
            ratings: Alvo (nota ou sinal de interesse).
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        """Preve a pontuacao para pares usuario-item.

        Args:
            users: Indices de usuario.
            items: Indices de item.

        Returns:
            Vetor de pontuacoes previstas.
        """
        raise NotImplementedError
