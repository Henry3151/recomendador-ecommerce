"""Promove a melhor versao do modelo para Production no MLflow.

Percorre todas as versoes registradas do modelo, identifica a de
menor RMSE (melhor desempenho) e atribui a ela o alias "production".
Isso automatiza a decisao de governanca: o melhor modelo disponivel
e sempre o que vai para producao, de forma rastreavel.

Uso:
    uv run python -m recomendador.training.promote_model
"""

from mlflow.tracking import MlflowClient

MODEL_NAME = "recomendador-neural"
METRIC = "neural_rmse"
ALIAS = "production"


def main() -> None:
    """Encontra a melhor versao por RMSE e a promove a producao."""
    client = MlflowClient()

    versoes = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not versoes:
        print(f"Nenhuma versao encontrada para o modelo '{MODEL_NAME}'.")
        return

    melhor_versao = None
    melhor_rmse = float("inf")

    print(f"Avaliando versoes do modelo '{MODEL_NAME}':")
    for versao in versoes:
        run = client.get_run(versao.run_id)
        rmse = run.data.metrics.get(METRIC)
        if rmse is None:
            continue
        print(f"  Versao {versao.version}: {METRIC}={rmse:.4f}")
        if rmse < melhor_rmse:
            melhor_rmse = rmse
            melhor_versao = versao.version

    if melhor_versao is None:
        print("Nenhuma versao possui a metrica de RMSE registrada.")
        return

    client.set_registered_model_alias(MODEL_NAME, ALIAS, melhor_versao)

    print()
    print("=" * 50)
    print("Modelo promovido a PRODUCTION:")
    print(f"  Versao: {melhor_versao}")
    print(f"  {METRIC}: {melhor_rmse:.4f}")
    print(f"  Alias: '{ALIAS}'")
    print("=" * 50)


if __name__ == "__main__":
    main()
