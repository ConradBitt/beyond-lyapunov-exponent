"""
entropies_by_aparam.py
======================
Scan do parâmetro de controle 'a' do mapa de Hénon acoplado.

Para cada valor de 'a', computa via GPU (CUDA):
  - Bacia de escape (8 saídas + não-escape)
  - Entropia de bacia normalizada  (Sb)
  - Entropia de fronteira normalizada (Sbb)
  - Dimensão fractal da fronteira (d = 2 - alpha)
  - Estabilidade de bacia (fração de CIs que não escapam)

Parâmetros fixos: b = 0.225, sigma = 0.0219, delta = 1e-15
Marcadores especiais: região λ₁<0 com Lyap. transv. distintos
  • Azul  (quadrado) : a ≈ 1.450
  • Vermelho (triângulo): a ≈ 1.530
"""

# ─── imports ──────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from numba import cuda
from tqdm import tqdm
from pynamicalsys import BasinMetrics, PlotStyler
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines

# ─── CUDA kernel (device functions) ──────────────────────────────────────────

@cuda.jit(device=True)
def henon_coupled_map(x0, y0, x1, y1, a, b, sigma):
    xn0 = a - x0 * x0 + b * y0 + sigma * x1
    yn0 = x0
    xn1 = a - x1 * x1 + b * y1 + sigma * x0
    yn1 = x1
    return xn0, yn0, xn1, yn1


@cuda.jit(device=True)
def find_exit(x0, y0, x1, y1, bounds):
    if x0 < bounds[0, 0]: return 0
    if x0 > bounds[0, 1]: return 1
    if y0 < bounds[1, 0]: return 2
    if y0 > bounds[1, 1]: return 3
    if x1 < bounds[2, 0]: return 4
    if x1 > bounds[2, 1]: return 5
    if y1 < bounds[3, 0]: return 6
    if y1 > bounds[3, 1]: return 7
    return -1


@cuda.jit
def escape_basin_kernel(X, Y, delta, a, b, sigma,
                         max_time, bounds, out_exit, out_time):
    ix, iy = cuda.grid(2)
    nx, ny  = X.shape[0], Y.shape[0]
    if ix >= nx or iy >= ny:
        return

    x0, y0 = X[ix], Y[iy]
    x1, y1 = x0 + delta, y0 + delta

    for step in range(max_time):
        idx = find_exit(x0, y0, x1, y1, bounds)
        if idx != -1:
            out_exit[ix, iy] = idx
            out_time[ix, iy] = step
            return
        x0, y0, x1, y1 = henon_coupled_map(x0, y0, x1, y1, a, b, sigma)

    out_exit[ix, iy] = -1
    out_time[ix, iy] = max_time


# ─── helper ───────────────────────────────────────────────────────────────────

def to_device_f64(arr):
    return cuda.to_device(np.ascontiguousarray(arr, dtype=np.float64))


def compute_escape_basin_cuda(X, Y, delta, a, b, sigma,
                               max_time, bounds, threads=(16, 16)):
    nx, ny   = len(X), len(Y)
    X_d      = to_device_f64(X)
    Y_d      = to_device_f64(Y)
    bounds_d = to_device_f64(bounds)
    out_exit = cuda.device_array((nx, ny), dtype=np.int32)
    out_time = cuda.device_array((nx, ny), dtype=np.int32)
    blocks   = ((nx + threads[0] - 1) // threads[0],
                (ny + threads[1] - 1) // threads[1])
    escape_basin_kernel[blocks, threads](
        X_d, Y_d, delta, a, b, sigma,
        max_time, bounds_d, out_exit, out_time
    )
    return np.stack(
        (out_exit.copy_to_host(), out_time.copy_to_host()), axis=-1
    )

# ─── PARÂMETROS ───────────────────────────────────────────────────────────────

b         = 0.225
sigma     = 0.0219
delta     = np.float64(1e-15)
total_time = 10_000
grid_size  = 1000
# Pontos de interesse (marcadores)
A_GREEN = 1.460 
A_RED = 1.510
A_BLUE  = 1.528

RECALCULATE = True

if RECALCULATE:

    # Range de 'a' — ajuste conforme necessário
    N_A       = 200
    a_values  = np.linspace(1.45, 1.56, N_A)

   
    X = np.linspace(-2.0, 2.0, grid_size)
    Y = np.linspace(-2.0, 2.0, grid_size)

    x_grid, y_grid = np.meshgrid(X, Y, indexing="ij")

    bounds = np.array([
        (-10.0, 10.0),
        (-10.0, 10.0),
        (-10.0, 10.0),
        (-10.0, 10.0),
    ], dtype=np.float64)

    # ─── ARRAYS DE SAÍDA ──────────────────────────────────────────────────────────

    # FIX: Use lists instead of pre-allocated arrays so that a mid-loop crash
    # does not leave mismatched lengths. Convert to arrays only after the loop.
    Sb_list            = []
    Sbb_list           = []
    D_list             = []
    # Stability_list     = []
    n_states_list      = []
    prob_no_escape_list = []
    a_completed        = []   # track which 'a' values actually finished

    # ─── LOOP PRINCIPAL ───────────────────────────────────────────────────────────
    bacias_selecionadas = {
        "bacia_caotica_1.csv": [1.4650, b, sigma],
        "bacia_caotica_2.csv": [1.5500, b, sigma]
    }

    # FIX: use np.isclose for float comparison instead of 'a in dict'
    a_to_save     = [params[0] for params in bacias_selecionadas.values()]
    a_to_filename = {fname: params[0] for fname, params in bacias_selecionadas.items()}

    def should_save(a_val, targets, tol=1e-6):
        for fname, target in a_to_filename.items():
            if np.isclose(a_val, target, atol=tol):
                return fname
        return None

    for i, a in enumerate(tqdm(a_values, desc="Scan a")):
        try:
            escape = compute_escape_basin_cuda(
                X, Y, delta, a, b, sigma, total_time, bounds
            )
            basin_labels = escape[:, :, 0]          # (grid, grid) int32

            # ── SALVAR BACIAS ESPECÍFICAS ─────────────────────────────────────────
            fname = should_save(a, a_to_save)
            if fname is not None:
                df_bacia = pd.DataFrame({
                    'x': X.flatten(),
                    'y': Y.flatten(),
                    'label': basin_labels.flatten()
                })
                df_bacia.to_csv(fname, index=False)
                print(f"\n[INFO] Bacia correspondente a a={a:.4f} salva em: {fname}")

            # ── estabilidade de bacia ──
            total_pts = grid_size * grid_size
            prob_ne = np.sum(basin_labels == -1) / total_pts

            # ── métricas de bacia ──
            bm = BasinMetrics(basin_labels)

            be       = bm.basin_entropy(5, log_base=2)
            n_states = len(np.unique(basin_labels))

            log_n = np.log2(n_states) if n_states > 1 else 1.0
            sb_val  = be[0] / log_n
            sbb_val = be[1] / log_n

            # ── dimensão fractal (uncertainty exponent) ──
            try:
                eps, f = bm.uncertainty_fraction(x_grid, y_grid)
                valid  = f > 0
                alpha, _ = np.polyfit(np.log(eps[valid]), np.log(f[valid]), 1)
                d_val = 2.0 - alpha
            except Exception:
                d_val = np.nan

            # Append results only if everything succeeded
            a_completed.append(a)
            Sb_list.append(sb_val)
            Sbb_list.append(sbb_val)
            D_list.append(d_val)
            n_states_list.append(n_states)
            prob_no_escape_list.append(prob_ne)
            # Stability_list.append(basin_stability)

        except Exception as e:
            print(f"\n[WARNING] a={a:.6f} falhou com erro: {e}. Pulando.")
            continue

    # Convert to arrays — lengths are guaranteed equal because they come from
    # the same loop body and are appended together.
    a_values       = np.array(a_completed)
    Sb             = np.array(Sb_list)
    Sbb            = np.array(Sbb_list)
    D              = np.array(D_list)
    n_states_arr   = np.array(n_states_list)
    prob_no_escape = np.array(prob_no_escape_list)
    prob_escape    = 1.0 - prob_no_escape

    # ─── SALVAR CSV ───────────────────────────────────────────────────────────────
    df = pd.DataFrame({
        "a":              a_values,
        "Sb_norm":        Sb,
        "Sbb_norm":       Sbb,
        "D":              D,
        "prob_no_escape": prob_no_escape,
        "prob_escape":    prob_escape,
        "n_states":       n_states_arr.astype(int),
    })
    df.to_csv(
        f"../data/basin_metrics/entropies_by_aparam_{b}_{sigma}.csv",
        index=False,
        float_format="%.8f",
    )
    print(f"CSV salvo em ../data/basin_metrics/entropies_by_aparam_{b}_{sigma}.csv")

else:
    # FIX: each column was incorrectly reading 'Sb_norm' for every variable

    try:
        df = pd.read_csv(f"../data/basin_metrics/entropies_by_aparam_{b}_{sigma}.csv")
        a_values       = df["a"].values
        Sb             = df["Sb_norm"].values
        Sbb            = df["Sbb_norm"].values
        D              = df["D"].values
        prob_no_escape = df["prob_no_escape"].values
        prob_escape    = df["prob_escape"].values
        n_states_arr   = df["n_states"].values

        b     = 0.225
        sigma = 0.0219
        delta = np.float64(1e-15)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        raise


# ─── PLOT ─────────────────────────────────────────────────────────────────────
# Layout: 2 rows (entropias / dimensão fractal)

ps = PlotStyler(ticks_on_all_sides=False)
ps.apply_style()

cm = 1 / 2.54
fig = plt.figure(figsize=(30 * cm, 10 * cm))

# O GridSpec agora controla o layout de forma absoluta e sem avisos
gs = gridspec.GridSpec(
    2, 1,
    hspace=0.08,
    left=0.05, right=0.97,
    top=0.93,  bottom=0.15,
)

ax1 = fig.add_subplot(gs[0])   # entropias
ax2 = fig.add_subplot(gs[1])   # dimensão fractal

# ── cores dos marcadores ──
C_GREEN = "#35CE21"
C_RED = "#F93E25"
C_BLUE  = "#1f77b4"

for ax in (ax1, ax2):
    ax.axvline(A_GREEN, color=C_GREEN, lw=1.2, ls="--", alpha=0.75, zorder=1)
    ax.axvline(A_RED, color=C_RED, lw=1.2, ls="--", alpha=0.75, zorder=1)
    ax.axvline(A_BLUE,  color=C_BLUE,  lw=1.2, ls="--", alpha=0.75, zorder=1)
    ax.set_xlim(a_values[0], a_values[-1])

# ── marcadores flutuantes acima do eixo (como em entropys.py) ──
ax2.scatter([A_GREEN], [2.25], marker="o", color=C_GREEN,
            s=50, edgecolors="black", linewidths=0.7,
            zorder=25, clip_on=False)
ax2.scatter([A_RED], [2.25], marker="^", color=C_RED,
            s=80, edgecolors="black", linewidths=0.7,
            zorder=25, clip_on=False)
ax2.scatter([A_BLUE],  [2.25], marker="s", color=C_BLUE,
            s=50, edgecolors="black", linewidths=0.7,
            zorder=25, clip_on=False)

# ── (ax1) entropias ──
ax1.plot(a_values, Sb,'--o',color='blue', ms=4, lw=1.2,
         label=r"$S_b$") # / \log_2 n
ax1.plot(a_values, Sbb, '--s',color='orange', ms=4, lw=1.2,
         label=r"$S_{bb}$") # / \log_2 n

ax1.set_ylim(-0.06, 1.50)
# ax1.set_ylabel("Basin entropy (norm.)", fontsize=11)
ax1.set_xticklabels([])
ax1.legend(
    loc="upper left", frameon=True,
    facecolor="white", framealpha=0.9,
    bbox_to_anchor=(0.1, 1.06)
)


# ── (ax2) dimensão fractal ──
ax2.plot(a_values, D, "o-", ms=4, lw=1.2, color="blueviolet", label=r'$d\;(= 2 - \alpha)$')
ax2.set_ylim(-0.05, 2.05)
# ax2.set_ylabel(r"$d\;(= 2 - \alpha)$", fontsize=11)
ax2.set_xlabel(r"$a$")

# ── (ax2) Basin stability ──
# ax2_twin = ax2.twinx()
ax2.plot(a_values, prob_no_escape, "s-", ms=4, lw=1.2, color="gray", label='Basin Stability')
# ax2.set_ylabel('Basin Stability', fontsize=11)
# ax2.set_ylim(0-0.05,1.05)

# ax2.plot(a_values, 1 - (abs(Sb - Sbb))/Sbb, "^-", ms=4, lw=1.2,
#          color="lime",     label=r"$\frac{|S_b - S_{bb}|}{S_{bb}}$")
ax2.legend(bbox_to_anchor=(0.1, 0.98), loc='upper left')

# ── legenda global dos marcadores ──
leg_blue = mlines.Line2D([], [], color=C_GREEN, marker="o", ms=6,
                          linestyle="--", lw=1, zorder=5,
                          label=rf"$a={A_GREEN:.3f}$ ($\lambda_1<0$, $f^+ \approx 0$)")
leg_green = mlines.Line2D([], [], color=C_RED, marker="^", ms=8,
                          linestyle="--", lw=1, zorder=5,
                          label=rf"$a={A_RED:.3f}$ (Escape)")
leg_red  = mlines.Line2D([], [], color=C_BLUE,  marker="s", ms=7,
                          linestyle="--", lw=1, zorder=5,
                          label=rf"$a={A_BLUE:.3f}$ ($\lambda_1<0$, $f^+ \approx 0.5$)")
fig.legend(
    handles=[leg_blue, leg_green, leg_red],
    loc="lower left",
    bbox_to_anchor=(0.79, 0.22),
    frameon=True, facecolor="white", framealpha=0.9,
    fontsize=10,
)


# letras de painel 
from string import ascii_lowercase
bbox_style = {"facecolor": "w", "alpha": 0.75, "linewidth": 0.0, "pad": 1}
for idx, ax in enumerate((ax1, ax2)):
    ax.text(0.01, 0.85, f"({ascii_lowercase[idx+3]})",
            bbox=bbox_style, transform=ax.transAxes, fontsize=20)
    ax.set_xlim(a_values[0], a_values[-1])

    # hatches 
    ax.axvspan(a_values[0], 1.491, color='green', alpha=0.1, hatch='//')
    ax.axvspan(1.498, 1.515, color='red', alpha=0.1, hatch='\\')
    ax.axvspan(1.539, a_values[-1], color='red', alpha=0.1, hatch='\\')
    ax.axvspan(1.520, 1.534, color='blue', alpha=0.1, hatch='.')


ax1.text(1.470, 0.2, 'R1', fontsize=15, color='black', ha='center')
ax1.text(1.525, 0.2, 'R2', fontsize=15, color='black', ha='center')
ax1.text(1.550, 0.2, 'R3', fontsize=15, color='black', ha='center')

ax2.set_ylim(-0.05, 2.49)
# plt.tight_layout()
plt.savefig(
    "../figures/entropies_by_aparam.png",
    dpi=300,
)
plt.close()
print("Plot salvo em ../figures/entropies_by_aparam.png")