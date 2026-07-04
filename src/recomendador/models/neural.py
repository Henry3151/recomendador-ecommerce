"""Modelo neural de recomendacao baseado em embeddings.

Aprende um vetor (embedding) para cada usuario e cada item. A afinidade
entre um usuario e um item e estimada passando a concatenacao dos dois
vetores por uma pequena MLP. E o nucleo dos recomendadores neurais
modernos, em forma enxuta.
"""

import numpy as np
import torch
from torch import nn

from recomendador.models.base import BaseRecommender


class _MLPModule(nn.Module):
    """Rede interna: embeddings de usuario/item + MLP de pontuacao."""

    def __init__(self, n_users: int, n_items: int, embedding_dim: int) -> None:
        """Inicializa as camadas da rede.

        Args:
            n_users: Numero de usuarios distintos.
            n_items: Numero de itens distintos.
            embedding_dim: Dimensao dos vetores de embedding.
        """
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """Calcula a pontuacao para pares usuario-item.

        Args:
            users: Tensor de indices de usuario.
            items: Tensor de indices de item.

        Returns:
            Tensor de pontuacoes previstas.
        """
        u = self.user_embedding(users)
        i = self.item_embedding(items)
        x = torch.cat([u, i], dim=1)
        return self.mlp(x).squeeze(1)


class NeuralRecommender(BaseRecommender):
    """Recomendador neural com API estilo fit/predict."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 32,
        learning_rate: float = 0.001,
        epochs: int = 10,
        batch_size: int = 256,
        seed: int = 42,
    ) -> None:
        """Configura o recomendador.

        Args:
            n_users: Numero de usuarios distintos.
            n_items: Numero de itens distintos.
            embedding_dim: Dimensao dos embeddings.
            learning_rate: Taxa de aprendizado.
            epochs: Numero de epocas de treino.
            batch_size: Tamanho do lote.
            seed: Semente para reprodutibilidade.
        """
        torch.manual_seed(seed)
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = _MLPModule(n_users, n_items, embedding_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

    def fit(self, users: np.ndarray, items: np.ndarray, ratings: np.ndarray) -> None:
        """Treina o modelo por minibatches.

        Args:
            users: Indices de usuario.
            items: Indices de item.
            ratings: Alvo (rating normalizado).
        """
        users_t = torch.as_tensor(users, dtype=torch.long)
        items_t = torch.as_tensor(items, dtype=torch.long)
        ratings_t = torch.as_tensor(ratings, dtype=torch.float32)

        n = len(users_t)
        self.model.train()
        for _ in range(self.epochs):
            ordem = torch.randperm(n)
            for inicio in range(0, n, self.batch_size):
                idx = ordem[inicio : inicio + self.batch_size]
                self.optimizer.zero_grad()
                pred = self.model(users_t[idx], items_t[idx])
                loss = self.loss_fn(pred, ratings_t[idx])
                loss.backward()
                self.optimizer.step()

    def predict(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        """Preve pontuacoes para pares usuario-item.

        Args:
            users: Indices de usuario.
            items: Indices de item.

        Returns:
            Vetor de pontuacoes previstas.
        """
        users_t = torch.as_tensor(users, dtype=torch.long)
        items_t = torch.as_tensor(items, dtype=torch.long)
        self.model.eval()
        with torch.no_grad():
            return self.model(users_t, items_t).numpy()
