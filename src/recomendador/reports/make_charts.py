"""Gera graficos de analise de dados e resultados para o README.

Produz quatro visualizacoes a partir dos dados reais do projeto:
1. Distribuicao de eventos (funil de e-commerce)
2. Impacto do k-core filtering (antes vs depois)
3. Comparacao de modelos (RMSE das 3 versoes + baseline)
4. Distribuicao de interacoes por usuario (cauda longa)

As imagens sao salvas em reports/figures/.

Uso:
    uv run python -m recomendador.reports.make_charts
"""

from pathlib import Path

import matplotlib.pyplot as plt

from recomendador.data.loader import load_retailrocket
from recomendador.preprocessing.pipeline import apply_kcore

RAW_PATH = "data/raw/events.csv"
FIGURES_DIR = Path("reports/figures")

COR_PRIMARIA = "#945DD6"
COR_SECUNDARIA = "#0194E2"
COR_DESTAQUE = "#DE5FE9"
COR_NEUTRA = "#B0B0B0"


def grafico_funil_eventos(df) -> None:
    """Distribuicao dos tipos de evento (funil de e-commerce)."""
    contagem = df["rating"].value_counts().sort_index()
    labels = ["View (1.0)", "Add to Cart (2.0)", "Transaction (3.0)"]
    valores = [contagem.get(1.0, 0), contagem.get(2.0, 0), contagem.get(3.0, 0)]

    fig, ax = plt.subplots(figsize=(8, 5))
    barras = ax.bar(labels, valores, color=[COR_SECUNDARIA, COR_PRIMARIA, COR_DESTAQUE])
    ax.set_title("Distribuicao de Eventos - Funil de E-commerce", fontweight="bold")
    ax.set_ylabel("Numero de interacoes")
    ax.bar_label(barras, fmt="{:,.0f}", padding=3)
    total = sum(valores)
    for barra, valor in zip(barras, valores, strict=False):
        pct = 100 * valor / total
        ax.text(
            barra.get_x() + barra.get_width() / 2,
            barra.get_height() / 2,
            f"{pct:.1f}%",
            ha="center",
            color="white",
            fontweight="bold",
        )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "01_funil_eventos.png", dpi=120)
    plt.close()


def grafico_impacto_kcore(df_antes, df_depois) -> None:
    """Compara volume antes e depois do k-core filtering."""
    categorias = ["Interacoes", "Usuarios", "Itens"]
    antes = [
        len(df_antes),
        df_antes["user_id"].nunique(),
        df_antes["item_id"].nunique(),
    ]
    depois = [
        len(df_depois),
        df_depois["user_id"].nunique(),
        df_depois["item_id"].nunique(),
    ]

    x = range(len(categorias))
    largura = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar([i - largura / 2 for i in x], antes, largura, label="Antes", color=COR_NEUTRA)
    b2 = ax.bar(
        [i + largura / 2 for i in x], depois, largura, label="Depois (k=5)", color=COR_PRIMARIA
    )
    ax.set_title("Impacto do k-core Filtering", fontweight="bold")
    ax.set_ylabel("Contagem (escala log)")
    ax.set_yscale("log")
    ax.set_xticks(list(x))
    ax.set_xticklabels(categorias)
    ax.legend()
    ax.bar_label(b1, fmt="{:,.0f}", padding=3, fontsize=8)
    ax.bar_label(b2, fmt="{:,.0f}", padding=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "02_impacto_kcore.png", dpi=120)
    plt.close()


def grafico_comparacao_modelos() -> None:
    """Compara o RMSE das 3 versoes e do baseline."""
    modelos = ["v1\n(emb=32)", "v2\n(emb=64)", "v3\n(ep=10)", "Baseline"]
    rmse = [0.1851, 0.1843, 0.1887, 0.1893]
    cores = [COR_NEUTRA, COR_DESTAQUE, COR_NEUTRA, COR_SECUNDARIA]

    fig, ax = plt.subplots(figsize=(8, 5))
    barras = ax.bar(modelos, rmse, color=cores)
    ax.set_title("Comparacao de Modelos - RMSE (menor = melhor)", fontweight="bold")
    ax.set_ylabel("Neural RMSE")
    ax.set_ylim(0.180, 0.191)
    ax.bar_label(barras, fmt="{:.4f}", padding=3)
    ax.text(1, 0.1843, "MELHOR", ha="center", va="bottom", color=COR_DESTAQUE, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "03_comparacao_modelos.png", dpi=120)
    plt.close()


def grafico_cauda_longa(df) -> None:
    """Distribuicao de interacoes por usuario (cauda longa)."""
    por_usuario = df["user_id"].value_counts()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(por_usuario, bins=50, color=COR_PRIMARIA, edgecolor="white")
    ax.set_title("Distribuicao de Interacoes por Usuario", fontweight="bold")
    ax.set_xlabel("Numero de interacoes por usuario")
    ax.set_ylabel("Numero de usuarios (escala log)")
    ax.set_yscale("log")
    mediana = por_usuario.median()
    ax.axvline(mediana, color=COR_DESTAQUE, linestyle="--", linewidth=2)
    ax.text(mediana, ax.get_ylim()[1] * 0.5, f" mediana={mediana:.0f}", color=COR_DESTAQUE)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "04_cauda_longa.png", dpi=120)
    plt.close()


def main() -> None:
    """Gera todos os graficos."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Carregando dados brutos...")
    df_bruto = load_retailrocket(RAW_PATH)

    print("Gerando grafico 1: funil de eventos...")
    grafico_funil_eventos(df_bruto)

    print("Aplicando k-core para comparacao...")
    df_filtrado = apply_kcore(df_bruto, k=5)

    print("Gerando grafico 2: impacto do k-core...")
    grafico_impacto_kcore(df_bruto, df_filtrado)

    print("Gerando grafico 3: comparacao de modelos...")
    grafico_comparacao_modelos()

    print("Gerando grafico 4: cauda longa...")
    grafico_cauda_longa(df_filtrado)

    print(f"Graficos salvos em {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
