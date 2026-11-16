
import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


def read_upo_csv(filename):
    """Lê o CSV produzido pelo código de busca das UPOs."""

    with open(filename, "r", encoding="utf-8-sig", newline="") as file:
        sample = file.read(2048)
        file.seek(0)

        try:
            dialect = csv.Sniffer().sniff(
                sample,
                delimiters=",;\t"
            )
        except csv.Error:
            dialect = csv.excel

        reader = csv.DictReader(file, dialect=dialect)
        data = list(reader)

    required = [
        "p",
        "|mu_par|<1",
        "|mu_perp|<1",
        "|mu_par|>1",
        "|mu_perp|>1",
    ]

    missing = [col for col in required if col not in reader.fieldnames]

    if missing:
        raise ValueError(
            f"Colunas ausentes no CSV: {missing}"
        )

    return {
        col: np.array([int(row[col]) for row in data])
        for col in required
    }



import csv

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator


def plot_upo_udv(filename="upo.csv"):
    """
    Gráfico de estabilidade/instabilidade das UPOs.

    Barras positivas:
        UPOs estáveis na direção correspondente.

    Barras negativas:
        UPOs instáveis na direção correspondente.

    Cores:
        Azul: tangencial.
        Laranja: transversal.

    Linha preta:
        Fração de UPOs estáveis transversalmente.

    Retorna:
        fig, ax1, ax2
    """

    # --------------------------------------------------
    # Leitura do CSV
    # --------------------------------------------------
    with open(filename, "r", encoding="utf-8-sig") as file:
        sample = file.read(2048)
        file.seek(0)

        try:
            dialect = csv.Sniffer().sniff(
                sample,
                delimiters=",;\t"
            )
        except csv.Error:
            dialect = csv.excel

        reader = csv.DictReader(file, dialect=dialect)
        rows = list(reader)

    if not rows:
        raise ValueError("O arquivo CSV está vazio.")

    # Ordenação por período
    rows.sort(key=lambda row: int(row["p"]))

    p = np.array([int(row["p"]) for row in rows])
    n_upos = np.array([int(row["UPOs"]) for row in rows])

    par_stable = np.array([
        int(row["|mu_par|<1"]) for row in rows
    ])

    perp_stable = np.array([
        int(row["|mu_perp|<1"]) for row in rows
    ])

    par_unstable = np.array([
        int(row["|mu_par|>1"]) for row in rows
    ])

    perp_unstable = np.array([
        int(row["|mu_perp|>1"]) for row in rows
    ])

    # Fração de UPOs transversalmente estáveis
    fraction_perp = np.divide(
        perp_stable,
        n_upos,
        out=np.full(len(p), np.nan),
        where=n_upos > 0
    )

    # --------------------------------------------------
    # Construção do gráfico
    # --------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(14, 7))

    ax2 = ax1.twinx()

    color_par = "tab:blue"
    color_perp = "tab:orange"

    width = 0.36

    x_par = p - width / 2
    x_perp = p + width / 2

    # --------------------------------------------------
    # Direção tangencial
    # --------------------------------------------------

    # Estáveis: positivas
    ax1.bar(
        x_par,
        par_stable,
        width,
        color=color_par,
        edgecolor="black",
        linewidth=0.4
    )

    # Instáveis: negativas
    ax1.bar(
        x_par,
        -par_unstable,
        width,
        color=color_par,
        edgecolor="black",
        linewidth=0.4,
        alpha=0.45,
        hatch="//"
    )

    # --------------------------------------------------
    # Direção transversal
    # --------------------------------------------------

    # Estáveis: positivas
    ax1.bar(
        x_perp,
        perp_stable,
        width,
        color=color_perp,
        edgecolor="black",
        linewidth=0.4
    )

    # Instáveis: negativas
    ax1.bar(
        x_perp,
        -perp_unstable,
        width,
        color=color_perp,
        edgecolor="black",
        linewidth=0.4,
        alpha=0.45,
        hatch="//"
    )

    # --------------------------------------------------
    # Fração de estabilidade transversal
    # --------------------------------------------------
    ax2.plot(
        p,
        fraction_perp,
        color="black",
        marker="o",
        markersize=4,
        linewidth=1.8,
        label=r"$f_{\perp}^{s}$",
        zorder=10
    )

    ax2.set_ylim(-0.03, 1.05)

    # --------------------------------------------------
    # Eixos e formatação
    # --------------------------------------------------
    ax1.axhline(
        0,
        color="black",
        linewidth=0.9
    )

    ax1.set_xlabel(
        r"Período $p$",
        fontsize=13
    )

    ax1.set_ylabel(
        "Quantidade de UPOs",
        fontsize=13
    )

    ax2.set_ylabel(
        r"Fração transversalmente estável $f_{\perp}^{s}$",
        fontsize=12,
        color="black"
    )

    ax1.set_title(
        "Estabilidade tangencial e transversal das UPOs",
        fontsize=14
    )

    ax1.set_xlim(p.min() - 0.7, p.max() + 0.7)

    max_count = max(
        np.max(par_stable),
        np.max(perp_stable),
        np.max(par_unstable),
        np.max(perp_unstable),
        1
    )

    ax1.set_ylim(
        -1.12 * max_count,
        1.12 * max_count
    )

    ax1.xaxis.set_major_locator(
        MaxNLocator(integer=True, nbins=20)
    )

    ax1.grid(
        axis="y",
        linestyle="--",
        alpha=0.25
    )

    ax1.set_axisbelow(True)

    # --------------------------------------------------
    # Legenda conjunta
    # --------------------------------------------------
    legend = [
        Patch(
            facecolor=color_par,
            label=r"$|\mu_{par}|<1$"
        ),
        Patch(
            facecolor=color_par,
            alpha=0.45,
            hatch="//",
            label=r"$|\mu_{par}|>1$"
        ),
        Patch(
            facecolor=color_perp,
            label=r"$|\mu_{perp}|<1$"
        ),
        Patch(
            facecolor=color_perp,
            alpha=0.45,
            hatch="//",
            label=r"$|\mu_{perp}|>1$"
        ),
        Line2D(
            [0],
            [0],
            color="black",
            marker="o",
            linewidth=1.8,
            label=r"$f_{\perp}^{s}$"
        )
    ]

    ax1.legend(
        handles=legend,
        loc="upper left",
        ncol=3,
        fontsize=10
    )

    fig.tight_layout()

    return fig, ax1, ax2


def plot_upo(data, grouped=True):
    """Plota estabilidade positiva e instabilidade negativa.

    Eixo esquerdo: multiplicadores tangenciais.
    Eixo direito: multiplicadores transversais.

    grouped=True: barras deslocadas lateralmente.
    grouped=False: barras alinhadas no mesmo período.
    """

    p = data["p"]

    par_stable = data["|mu_par|<1"]
    par_unstable = data["|mu_par|>1"]

    perp_stable = data["|mu_perp|<1"]
    perp_unstable = data["|mu_perp|>1"]

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    color_par = "tab:blue"
    color_perp = "tab:orange"

    width = 0.32

    if grouped:
        x_par = p - width / 2
        x_perp = p + width / 2
    else:
        x_par = p
        x_perp = p

    # Multiplicadores tangenciais: eixo esquerdo
    ax1.bar(
        x_par,
        par_stable,
        width,
        color=color_par,
        edgecolor="black",
        linewidth=0.5
    )

    ax1.bar(
        x_par,
        -par_unstable,
        width,
        color=color_par,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.45,
        hatch="//"
    )

    # Multiplicadores transversais: eixo direito
    ax2.bar(
        x_perp,
        perp_stable,
        width,
        color=color_perp,
        edgecolor="black",
        linewidth=0.5
    )

    ax2.bar(
        x_perp,
        -perp_unstable,
        width,
        color=color_perp,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.45,
        hatch="//"
    )

    # Limites simétricos em torno de zero
    max_par = max(
        np.max(par_stable),
        np.max(par_unstable),
        1
    )

    max_perp = max(
        np.max(perp_stable),
        np.max(perp_unstable),
        1
    )

    ax1.set_ylim(-1.15 * max_par, 1.15 * max_par)
    ax2.set_ylim(-1.15 * max_perp, 1.15 * max_perp)

    # Linha de referência
    ax1.axhline(
        0,
        color="black",
        linewidth=0.9
    )

    # Eixos
    ax1.set_xlabel(r"Período $p$", fontsize=13)

    ax1.set_ylabel(
        r"Tangencial: $\mu_{par}$",
        color=color_par,
        fontsize=12
    )

    ax2.set_ylabel(
        r"Transversal: $\mu_{perp}$",
        color=color_perp,
        fontsize=12
    )

    ax1.tick_params(axis="y", labelcolor=color_par)
    ax2.tick_params(axis="y", labelcolor=color_perp)

    ax1.set_xticks(p)
    ax1.set_xlim(p.min() - 0.7, p.max() + 0.7)

    ax1.grid(axis="y", linestyle="--", alpha=0.25)
    ax1.set_axisbelow(True)

    # Legenda
    legend = [
        Patch(
            facecolor=color_par,
            label=r"$|\mu_{par}|<1$"
        ),
        Patch(
            facecolor=color_par,
            alpha=0.45,
            hatch="//",
            label=r"$|\mu_{par}|>1$"
        ),
        Patch(
            facecolor=color_perp,
            label=r"$|\mu_{perp}|<1$"
        ),
        Patch(
            facecolor=color_perp,
            alpha=0.45,
            hatch="//",
            label=r"$|\mu_{perp}|>1$"
        )
    ]

    ax1.legend(
        handles=legend,
        loc="upper left",
        ncol=2,
        fontsize=10
    )

    ax1.set_title(
        "Estabilidade tangencial e transversal das UPOs",
        fontsize=14
    )

    fig.tight_layout()

    return fig, ax1, ax2


def main():

    parser = argparse.ArgumentParser(
        description="Visualização dos multiplicadores de Floquet."
    )

    parser.add_argument(
        "filename",
        nargs="?",
        default="upo.csv",
        help="Arquivo CSV (padrão: upo.csv)"
    )

    args = parser.parse_args()

    data = read_upo_csv("upo.csv")

    # Barras deslocadas
    plot_upo(data, grouped=True)

    # # Barras alinhadas
    # plot_upo(data, grouped=False)

    plt.show()


if __name__ == "__main__":
    from pynamicalsys import PlotStyler
    ps = PlotStyler()
    ps.apply_style()

    main()