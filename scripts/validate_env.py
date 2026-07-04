"""Script de validacao do ambiente de execucao.

Verifica se os pre-requisitos para rodar o projeto estao satisfeitos:
versao do Python, dependencias criticas instaladas e configuracao
carregando corretamente. Serve como diagnostico rapido apos clonar
o repositorio (uv run python scripts/validate_env.py).
"""

import importlib
import sys

# Versao minima de Python exigida pelo projeto.
MIN_PYTHON = (3, 11)

# Dependencias criticas que precisam estar instaladas.
REQUIRED_PACKAGES = [
    "torch",
    "sklearn",
    "mlflow",
    "dvc",
    "pandas",
    "numpy",
    "pydantic_settings",
]


def check_python_version() -> bool:
    """Verifica se a versao do Python atende ao minimo exigido.

    Returns:
        True se a versao for suficiente, False caso contrario.
    """
    atual = sys.version_info[:2]
    ok = atual >= MIN_PYTHON
    esperado = f"{MIN_PYTHON[0]}.{MIN_PYTHON[1]}+"
    encontrado = f"{atual[0]}.{atual[1]}"
    status = "OK" if ok else "FALHA"
    print(f"[{status}] Python {encontrado} (esperado: {esperado})")
    return ok


def check_packages() -> bool:
    """Verifica se as dependencias criticas estao instaladas.

    Returns:
        True se todas estiverem disponiveis, False caso contrario.
    """
    todos_ok = True
    for pacote in REQUIRED_PACKAGES:
        try:
            importlib.import_module(pacote)
            print(f"[OK] Pacote '{pacote}' instalado")
        except ImportError:
            print(f"[FALHA] Pacote '{pacote}' NAO encontrado")
            todos_ok = False
    return todos_ok


def check_config() -> bool:
    """Verifica se a configuracao carrega sem erros.

    Returns:
        True se a configuracao for valida, False caso contrario.
    """
    try:
        from recomendador.config import get_settings

        settings = get_settings()
        print(f"[OK] Configuracao carregada (modelo: {settings.model_type})")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[FALHA] Erro ao carregar configuracao: {exc}")
        return False


def main() -> int:
    """Executa todas as verificacoes e resume o resultado.

    Returns:
        Codigo de saida: 0 se tudo passou, 1 se algo falhou.
    """
    print("=" * 50)
    print("Validacao do ambiente")
    print("=" * 50)
    resultados = [
        check_python_version(),
        check_packages(),
        check_config(),
    ]
    print("=" * 50)
    if all(resultados):
        print("Ambiente OK: pronto para uso.")
        return 0
    print("Ambiente com problemas: revise os itens marcados como FALHA.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
