import pickle
from pathlib import Path
from string import ascii_lowercase

import numpy as np
import matplotlib.pyplot as plt
from fttle_window_analysis import update_reliability_metadata

# --- parametros: devem bater com os usados para gerar os caches ---
A = 1.5250
B = 0.2250
SIGMA = 0.0219

# fttle_window_analysis.py
N_ICS = 1000
X_MIN, X_MAX = -1.5, 1.5
Y_MIN, Y_MAX = -1.5, 1.5

HISTOGRAM_WINDOWS = [7, 10, 13]
HISTOGRAM_BINS = 20
FTTLE_DATA_PATH = Path(f"fttle_window_multiIC_a{A}_b{B}_sigma{SIGMA}_N{N_ICS}_data.pkl")

# find_upos.py
UPO_DATA_PATH = Path(f"upo_transverse_a{A}_b{B}_sigma{SIGMA}_data.npz")

OUTPUT_PATH = "combined_fttle_upo.png"


COLOR = ['red', 'green', 'blue']
TRANSVERSE_DIRECTION_COLORS = {0: "tab:blue", 1: "tab:red", 2: "tab:orange"}

def classify(eigvals, tol=1.0):
    max_abs = np.max(np.abs(eigvals))
    return ("estavel" if max_abs < tol else "instavel"), max_abs


def load_fttle_results(FTTLE_DATA_PATH=FTTLE_DATA_PATH):
    # if not FTTLE_DATA_PATH.exists():
    #     raise FileNotFoundError(
    #         f"Cache do FTTLE nao encontrado: {FTTLE_DATA_PATH}. "
    #         "Rode fttle_window_analysis.py com RECALCULATE=True primeiro."
    #     )
    with open(FTTLE_DATA_PATH, "rb") as f:
        data = pickle.load(f)
    for result in data["results_all"].values():
        update_reliability_metadata(result)
    return data["results_all"]


def load_upo_rows():
    if not UPO_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Cache de UPOs nao encontrado: {UPO_DATA_PATH}. "
            "Rode find_upos.py com RECALCULATE=True primeiro."
        )
    with np.load(UPO_DATA_PATH, allow_pickle=False) as data:
        periods = data["periods"].copy()
        offsets = data["offsets"].copy()
        orbits = data["orbits"].copy()
        long_eigs = data["long_eigs"].copy()
        trans_eigs = data["trans_eigs"].copy()
    rows = []
    for i, p in enumerate(periods):
        orbit = orbits[offsets[i]:offsets[i + 1]]
        long_class, long_mag = classify(long_eigs[i])
        trans_class, trans_mag = classify(trans_eigs[i])
        trans_n_unstable = int(np.count_nonzero(np.abs(trans_eigs[i]) > 1.0))
        rows.append({
            "period": int(p), "orbit": orbit,
            "long_class": long_class, "long_mag": long_mag,
            "trans_class": trans_class, "trans_mag": trans_mag,
            "trans_n_unstable": trans_n_unstable,
        })
    return rows


def reliable_windows(results, windows):
    return [w for w in windows if w in results and results[w]["reliable"]]


def plot_histogram_panel(ax, r, w, idx, xlim, bins, show_ylabel, color=None):
    density, edges = np.histogram(r["lambdas"], bins=bins, density=True)
    # density = counts / len(r["lambdas"]) 
    centers = 0.5 * (edges[:-1] + edges[1:])
    ax.bar(centers, density, width=np.diff(edges), color="gray",
           edgecolor="black", linewidth=0.3, alpha=0.85)
    ax.axvline(0.0, color="crimson" if color is None else color, linestyle="--", linewidth=1.2)
    ax.text(0.03, 0.92, f"({ascii_lowercase[idx]}) w = {w}", transform=ax.transAxes,
            fontsize=11, bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=2))
    frac = r["fplus"]
    if 0 < frac < 1e-3:
        mantissa, exponent = f"{frac:.1e}".split("e")
        label = rf"$f_w^+ = {mantissa}\times10^{{{int(exponent)}}}$"
    else:
        label = rf"$f_w^+ = {frac:.2f}$"
    ax.text(0.62, 0.92, label, transform=ax.transAxes, fontsize=11,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=2))
    # ax.set_yscale("log")
    # ax.set_ylim(0, 0.5)
    ax.set_xlabel(r"$\lambda_{\perp}^{w}$", fontsize=14)
    ax.set_xlim(*xlim)
    ax.tick_params(labelsize=12)

    if show_ylabel:
        ax.set_ylabel(r"$P(\lambda_{\perp}^{w})$", fontsize=14)


def plot_variance_panel(ax, results):
    var_windows = sorted(results.keys())
    good_w = reliable_windows(results, var_windows)
    bad_w = [w for w in var_windows if w in results and w not in good_w]
    even_w = [w for w in good_w if w % 2 == 0]
    odd_w = [w for w in good_w if w % 2 == 1]
    if even_w:
        ax.plot(even_w, [results[w]["variance"] for w in even_w], marker="o", markersize=5, linestyle="none", label="even w")
    if odd_w:
        ax.plot(odd_w, [results[w]["variance"] for w in odd_w], marker="s", markersize=5, linestyle="none", label="odd w")
    if bad_w:
        ax.plot(bad_w, [results[w]["variance"] for w in bad_w], marker="x", markersize=4, linestyle="none", color="gray", alpha=0.6, label="insufficient statistics")
    ax.set_yscale("log")
    ax.set_xlabel(r"$w$")
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylabel(r"$\mathrm{Var}(\lambda_{\perp}^{w})$", fontsize=14)
    ax.text(0.01, 0.82, "(d)", transform=ax.transAxes, fontsize=14)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_ylim(1.9e-5,1e0)

    for i, window in enumerate(HISTOGRAM_WINDOWS):
        ax.axvline(window, color=COLOR[i], linestyle="--", linewidth=0.5)
        # ax.text(window, 0.92, f"({ascii_lowercase[i]}) w = {window}",
        #     transform=ax.get_xaxis_transform(), ha="left", va="top",
        #     fontsize=11, bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=2))



def plot_manifold_panel(ax, upo_rows):
    # p = 6
    selected_upos = [r for r in upo_rows if r["period"] == 6][:2]
    for r in selected_upos:
        color = TRANSVERSE_DIRECTION_COLORS[r["trans_n_unstable"]]
        marker = "o" if r["long_class"] == "estavel" else "^"
        ax.scatter(r["orbit"][:, 0], r["orbit"][:, 1], color=color, marker=marker,
                   edgecolor="black", linewidth=0.3, # alpha=1.0 if r["trans_class"] == "estavel" else 0.3,
                   s=10 if r["trans_class"] == "instavel" else 8,
                   zorder=50 if r["trans_class"] == "instavel" else 100)

    for r in selected_upos:
        ax.plot(r["orbit"][:, 0], r["orbit"][:, 1], ',k', alpha=0.3)
    # ax.scatter([], [], color="gray", marker="o", edgecolor="black",
    #            label=r"Attracting in $\mathcal{S}$ ($\lambda_{\parallel}<0$)")
    # ax.scatter([], [], color="gray", marker="^", edgecolor="black",
    #            label=r"Repelling in $\mathcal{S}$ ($\lambda_{\parallel}>0$)")
    # for n_unstable, color in TRANSVERSE_DIRECTION_COLORS.items():
    #     ax.scatter([], [], color=color, marker="o", edgecolor="black",
    #                 label=f"{n_unstable} transverse unstable direction(s)")

    ax.scatter([], [], color='tab:blue', marker="o", edgecolor="black",
               label="Transversally Stable ($\lambda_{\perp} < 0$)")
    ax.scatter([], [], color='tab:red', marker="o", edgecolor="black",
                   label="Transversally Unstable ($\lambda_{\perp} > 0$)")
    ax.scatter([], [], color="gray", marker="o", edgecolor="black", label=r"Attracting in $\mathcal{S}$ ($\lambda_{\parallel} < 0$)")
    ax.scatter([], [], color="gray", marker="^", edgecolor="black", label=r"Repelling in $\mathcal{S}$ ($\lambda_{\parallel} > 0$)")
    ax.set_xlabel(r"$X_t^{[S]}$", fontsize=12)
    ax.set_ylabel(r"$Y_t^{[S]}$", fontsize=12)
    ax.text(0.03, 0.82, "(e)", transform=ax.transAxes, fontsize=12)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=7.2, loc="center left",)# framealpha=0.9)#, edgecolor="none")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)


def main():
    try:
        from pynamicalsys import PlotStyler
        PlotStyler().apply_style()
    except ImportError:
        pass

    results = load_fttle_results()
    upo_rows = load_upo_rows()

    hist_windows = [w for w in HISTOGRAM_WINDOWS if w in results]
    all_lambdas = np.concatenate([results[w]["lambdas"] for w in hist_windows])
    x_min, x_max = all_lambdas.min(), all_lambdas.max()
    margin = max(abs(x_min) * 0.05, 1e-12) if x_min == x_max else 0.05 * (x_max - x_min)
    bins = np.linspace(x_min - margin, x_max + margin, HISTOGRAM_BINS + 1)
    xlim = (-0.45, 0.45)

    cm = 1 / 2.54
    fig = plt.figure(figsize=(26 * cm, 16 * cm))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.2], hspace=0.35, wspace=0.32)

    ax0 = fig.add_subplot(gs[0, 0])
    hist_axes = [ax0] + [fig.add_subplot(gs[0, i], sharey=ax0) for i in range(1, 3)]

    for idx, w in enumerate(hist_windows):
        plot_histogram_panel(hist_axes[idx], results[w], w, idx, xlim, bins, show_ylabel=(idx == 0), color=COLOR[idx])
    
    ax_var = fig.add_subplot(gs[1, 0:2])
    plot_variance_panel(ax_var, results)

    ax_map = fig.add_subplot(gs[1, 2])
    plot_manifold_panel(ax_map, upo_rows)

    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    print(f"Figura combinada salva em: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
