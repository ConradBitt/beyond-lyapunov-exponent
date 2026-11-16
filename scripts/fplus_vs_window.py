"""f+ em função da janela w para Hénon acoplados, com (a, b) fixos."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit, prange

B_FIX, SIGMA = 0.225, 0.0219
A_R1, A_R2, A_R3 = 1.4600, 1.5250, 1.475
N_IC = 500
TRANSIENT = 5_000
TOTAL_TIME = 1_000_000
ESCAPE = 4.0
SEED = 4756345
WINDOWS_1 = np.unique(np.logspace(0, np.log10(TOTAL_TIME // 10), 150).astype(int))
prime_p = 2
teto_maximo = max(prime_p * 100, TOTAL_TIME // 10)
WINDOWS_2 = np.unique(np.arange(prime_p, teto_maximo + 1, prime_p, dtype=int))

WINDOWS = np.unique(np.concatenate((WINDOWS_1, WINDOWS_2)))


@njit(parallel=True)
def tle_series(ics, a, b, s, transient, TOTAL_TIME, esc):
    """Série de log(normt) por iteração (NaN se a órbita escapar)."""
    n_ic = ics.shape[0]
    out = np.full((n_ic, TOTAL_TIME), np.nan)
    for k in prange(n_ic):
        x0, y0, x1, y1 = ics[k]
        xs, ys = x0, y0  # manifold sincronizado parte de (x0, y0)
        u0, u1 = 1.0, 0.0
        for t in range(transient + TOTAL_TIME):
            n0 = (-2.0 * xs - s) * u0 + b * u1
            n1 = u0
            norm = np.sqrt(n0 * n0 + n1 * n1)
            if norm == 0.0:
                norm = 1e-16
            u0, u1 = n0 / norm, n1 / norm
            if t >= transient:
                out[k, t - transient] = np.log(norm)

            nx0 = a - x0 * x0 + b * y0 + s * x1
            nx1 = a - x1 * x1 + b * y1 + s * x0
            y0, y1, x0, x1 = x0, x1, nx0, nx1
            xs, ys = a - xs * xs + b * ys + s * xs, xs

            if (abs(x0) > esc or abs(y0) > esc or abs(x1) > esc
                    or abs(y1) > esc or abs(xs) > esc):
                out[k, :] = np.nan
                break
    return out


def f_plus(lt, w):
    """Fração de janelas disjuntas de tamanho w com expoente > 0."""
    n = lt.shape[1] // w
    seg = lt[:, :n * w].reshape(lt.shape[0], n, w).mean(axis=2)
    return (seg > 0).mean(axis=1)

def save_mean_std_to_csv(filename, data_list, label_list, a_list, b, sigma, transient, TOTAL_TIME, n_ic, escape):
    """Salva médias e desvios padrão em CSV."""
    # Garante que o arquivo comece limpo a cada nova execução
    if os.path.exists(filename):
        os.remove(filename)

    # Cria o DataFrame concatenando as séries
    df = pd.concat([pd.Series(d, name=label) for d, label in zip(data_list, label_list)], axis=1)
        
    # Prepara o texto do cabeçalho
    header_text = (
        "# =========================================================================\n"
        "# Sim Config\n"
        f"# a_values={a_list}, b={b}, sigma={sigma}, N_IC={n_ic}, "
        f"# transient={transient}, TOTAL_TIME={TOTAL_TIME}\n"
        f"# total_time={transient + TOTAL_TIME}, ESCAPE={escape}\n"
        f"# SEED = {SEED}\n"
        "# =========================================================================\n"
    )
    
    # Abre o arquivo em modo de escrita ('w') para colocar o cabeçalho no TOPO
    with open(filename, mode='w', encoding='utf-8') as f:
        # 1. Escreve primeiro o comentário descritivo
        f.write(header_text)
        
        # 2. Usa o pandas para salvar o DataFrame LOGO ABAIXO do cabeçalho
        # Passamos o ponteiro do arquivo 'f' em vez do nome do arquivo
        df.to_csv(f, index=False, float_format="%.18e")
            


RECALCULATE = False

if __name__ == "__main__":

    if RECALCULATE:
        rng = np.random.default_rng(SEED)
        ics = rng.uniform(-1.5, 1.5, (N_IC, 4))
        lt_R1 = tle_series(ics, A_R1, B_FIX, SIGMA, TRANSIENT, TOTAL_TIME, ESCAPE)
        lt_R2 = tle_series(ics, A_R2, B_FIX, SIGMA, TRANSIENT, TOTAL_TIME, ESCAPE)
        lt_R3 = tle_series(ics, A_R3, B_FIX, SIGMA, TRANSIENT, TOTAL_TIME, ESCAPE)

        lt_R1 = lt_R1[~np.isnan(lt_R1[:, 0])]
        lt_R2 = lt_R2[~np.isnan(lt_R2[:, 0])]
        lt_R3 = lt_R3[~np.isnan(lt_R3[:, 0])]
        print(f"CIs válidas R1 (sem escape): {lt_R1.shape[0]}/{N_IC}")
        print(f"CIs válidas R2 (sem escape): {lt_R2.shape[0]}/{N_IC}")
        print(f"CIs válidas R3 (sem escape): {lt_R3.shape[0]}/{N_IC}")

        fp_R1 = np.array([f_plus(lt_R1, w) for w in WINDOWS])  # (n_w, n_validas)
        fp_R2 = np.array([f_plus(lt_R2, w) for w in WINDOWS])  # (n_w, n_validas)
        fp_R3 = np.array([f_plus(lt_R3, w) for w in WINDOWS])  # (n_w, n_validas)

        mean_R1, std_R1 = fp_R1.mean(axis=1), fp_R1.std(axis=1)
        mean_R2, std_R2 = fp_R2.mean(axis=1), fp_R2.std(axis=1)
        mean_R3, std_R3 = fp_R3.mean(axis=1), fp_R3.std(axis=1)

        lista_de_dados = [WINDOWS, mean_R1, std_R1, mean_R2, std_R2, mean_R3, std_R3] 
        lista_de_labels = ["w", "mean_R1", "std_R1", "mean_R2", "std_R2", "mean_R3", "std_R3"]
        lista_de_a = [A_R1, A_R2, A_R3]

        output_file = "fplus_by_win.csv"

        # Chamada única para salvar tudo com as estatísticas incorporadas
        save_mean_std_to_csv(
            output_file, lista_de_dados, lista_de_labels,
            lista_de_a, B_FIX, SIGMA, TRANSIENT, TOTAL_TIME, N_IC, ESCAPE
        )
    else:
        # Carrega os dados do CSV
        df = pd.read_csv("fplus_by_win.csv", comment="#")
        WINDOWS = df["w"].values
        mean_R1, std_R1 = df["mean_R1"].values, df["std_R1"].values
        mean_R2, std_R2 = df["mean_R2"].values, df["std_R2"].values
        mean_R3, std_R3 = df["mean_R3"].values, df["std_R3"].values



    from pynamicalsys import PlotStyler
    ps = PlotStyler()
    ps.apply_style()
    CM = 1 / 2.54
    fig, ax = plt.subplots(figsize=(22*CM, 11*CM))
    ax.errorbar(WINDOWS, mean_R1, yerr=std_R1, fmt="o-", capsize=3, label=rf"$a = {A_R1:.3f}$ (R1)", color='tab:green')
    ax.errorbar(WINDOWS, mean_R2, yerr=std_R2, fmt="s-", capsize=3, label=rf"$a = {A_R2:.3f}$ (R2)", color='tab:blue')
    ax.errorbar(WINDOWS, mean_R3, yerr=std_R3, fmt="^-", capsize=3, label=rf"$a = {A_R3:.3f}$", color='tab:orange')
    # # marca x-values que são múltiplos de prime_p
    # x_mult = WINDOWS[WINDOWS % prime_p == 0]
    # if x_mult.size:
    #     ax.scatter(x_mult, np.ones_like(x_mult, dtype=float) * 1.02,
    #                marker='+', s=30, color='black', zorder=5)

    ax.set_xscale("log")
    ax.set_xlabel(r"$w$")
    ax.set_ylabel(r"$\langle f^+_w \rangle$")
    # ax.set_title(rf"$b={B_FIX},\ \sigma={SIGMA}$ ({lt_R1.shape[0]} CIs)")
    _bbox = {"facecolor": "w", "alpha": 1., "linewidth": 0.0, "pad": 1}
    from string import ascii_lowercase
    ax.text(0.01, 0.92, rf"({ascii_lowercase[0]})", bbox=_bbox, transform=ax.transAxes)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(WINDOWS[0], WINDOWS[-1])
    ax.axhline(0.5, color="red", ls="--", lw=1, zorder=0)
    ax.legend(bbox_to_anchor=(0.70, 0.95), loc='upper left')

    fig.tight_layout()
    fig.savefig("fplus_vs_window.png", dpi=300, bbox_inches='tight')
    print('Figura salva: fplus_vs_window.png')