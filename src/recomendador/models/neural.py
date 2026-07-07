"""Modelo neural de recomendacao baseado em embeddings.

Aprende um vetor (embedding) para cada usuario e cada item. A afinidade
entre um usuario e um item e estimada passando a concatenacao dos dois
vetores por uma pequena MLP. Inclui early stopping para evitar
overfitting: o treino para quando a perda de validacao nao melhora.
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
    """Recomendador neural com API estilo fit/predict e early stopping."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 32,
        learning_rate: float = 0.001,
        epochs: int = 10,
        batch_size: int = 256,
        seed: int = 42,
        patience: int = 2,
        val_fraction: float = 0.1,
    ) -> None:
        """Configura o recomendador.

        Args:
            n_users: Numero de usuarios distintos.
            n_items: Numero de itens distintos.
            embedding_dim: Dimensao dos embeddings.
            learning_rate: Taxa de aprendizado.
            epochs: Numero maximo de epocas de treino.
            batch_size: Tamanho do lote.
            seed: Semente para reprodutibilidade.
            patience: Epocas sem melhora antes de parar (early stopping).
            val_fraction: Fracao dos dados usada para validacao interna.
        """
        torch.manual_seed(seed)
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.val_fraction = val_fraction
        self.seed = seed
        self.model = _MLPModule(n_users, n_items, embedding_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.best_epoch = 0

    def fit(self, users: np.ndarray, items: np.ndarray, ratings: np.ndarray) -> None:
        """Treina o modelo com early stopping baseado na validacao.

        Args:
            users: Indices de usuario.
            items: Indices de item.
            ratings: Alvo (rating normalizado).
        """
        users_t = torch.as_tensor(users, dtype=torch.long)
        items_t = torch.as_tensor(items, dtype=torch.long)
        ratings_t = torch.as_tensor(ratings, dtype=torch.float32)

        # Separa uma fracao para validacao interna (early stopping)
        n = len(users_t)
        gerador = torch.Generator().manual_seed(self.seed)
        perm = torch.randperm(n, generator=gerador)
        n_val = max(1, int(n * self.val_fraction))
        idx_val = perm[:n_val]
        idx_treino = perm[n_val:]

        melhor_val = float("inf")
        epocas_sem_melhora = 0
        melhor_estado = None

        for epoca in range(self.epochs):
            # --- Treino ---
            self.model.train()
            ordem = idx_treino[torch.randperm(len(idx_treino), generator=gerador)]
            for inicio in range(0, len(ordem), self.batch_size):
                lote = ordem[inicio : inicio + self.batch_size]
                self.optimizer.zero_grad()
                pred = self.model(users_t[lote], items_t[lote])
                loss = self.loss_fn(pred, ratings_t[lote])
                loss.backward()
                self.optimizer.step()

            # --- Validacao ---
            self.model.eval()
            with torch.no_grad():
                pred_val = self.model(users_t[idx_val], items_t[idx_val])
                val_loss = self.loss_fn(pred_val, ratings_t[idx_val]).item()

            # --- Early stopping ---
            if val_loss < melhor_val:
                melhor_val = val_loss
                epocas_sem_melhora = 0
                melhor_estado = {k: v.clone() for k, v in self.model.state_dict().items()}
                self.best_epoch = epoca + 1
            else:
                epocas_sem_melhora += 1
                if epocas_sem_melhora >= self.patience:
                    break

        # Restaura o melhor estado encontrado
        if melhor_estado is not None:
            self.model.load_state_dict(melhor_estado)

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
