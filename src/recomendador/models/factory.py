"""Fabrica de modelos de recomendacao (padrao Factory Method).

Centraliza a criacao de modelos: em vez de espalhar if/else pelo
codigo, um unico ponto decide qual classe concreta instanciar com
base no tipo solicitado. Facilita adicionar novos modelos e testar.
"""

from recomendador.models.base import BaseRecommender


class ModelFactory:
    """Cria instancias de modelos a partir de um identificador."""

    @staticmethod
    def create(model_type: str, **kwargs: object) -> BaseRecommender:
        """Cria o modelo correspondente ao tipo informado.

        Args:
            model_type: Identificador do modelo ("neural" ou "baseline").
            **kwargs: Parametros repassados ao construtor do modelo.

        Returns:
            Instancia de um modelo que implementa BaseRecommender.

        Raises:
            ValueError: Se o model_type nao for reconhecido.
        """
        registry = _build_registry()
        if model_type not in registry:
            disponiveis = ", ".join(sorted(registry))
            raise ValueError(f"Modelo desconhecido: {model_type!r}. Disponiveis: {disponiveis}.")
        model_cls = registry[model_type]
        return model_cls(**kwargs)


def _build_registry() -> dict[str, type[BaseRecommender]]:
    """Monta o mapa de tipos de modelo para suas classes.

    O import e feito aqui dentro (import tardio) para evitar
    dependencias circulares e para so carregar o torch quando
    um modelo neural for realmente solicitado.

    Returns:
        Mapa de identificador para classe de modelo.
    """
    from recomendador.models.baseline import MeanBaseline
    from recomendador.models.neural import NeuralRecommender

    return {
        "neural": NeuralRecommender,
        "baseline": MeanBaseline,
    }
