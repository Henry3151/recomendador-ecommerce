"""Configuracao central do projeto (padrao Settings).

Le as variaveis de ambiente (do arquivo .env ou do sistema), valida
seus tipos com Pydantic e as expoe como um objeto unico. Todo o
projeto importa daqui em vez de ler variaveis soltas, garantindo
consistencia e validacao automatica.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Parametros de configuracao do projeto, lidos do ambiente."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ----- Dados -----
    raw_data_path: str = "data/raw/ratings.csv"
    processed_data_path: str = "data/processed/interactions.parquet"

    # ----- MLflow -----
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "recomendador-ecommerce"

    # ----- Modelo -----
    model_type: str = "neural"
    embedding_dim: int = 32
    learning_rate: float = 0.001
    epochs: int = 10
    batch_size: int = 256

    # ----- Reprodutibilidade -----
    random_seed: int = 42


def get_settings() -> Settings:
    """Cria e devolve a configuracao do projeto.

    Returns:
        Objeto Settings com todos os parametros validados.
    """
    return Settings()
