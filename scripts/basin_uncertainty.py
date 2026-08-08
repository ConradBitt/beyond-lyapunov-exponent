# =============================================================================
# henon_basin_analysis.py
#
# Bacia de escape do mapa de Henon acoplado (CUDA) + entropia de bacia
# (SB, SBB) + expoente de incerteza (alpha, joblib+numba CPU).
# =============================================================================

import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from numba import cuda, njit, prange
from joblib import Parallel, delayed


# =============================================================================
# Device functions (GPU) - calculo da bacia completa
# =============================================================================

@cuda.jit(device=True)
def henon_coupled_map(x0, y0, x1, y1, a, b, sigma):

    xn0 = a - x0*x0 + b*y0 + sigma*x1
    yn0 = x0

    xn1 = a - x1*x1 + b*y1 + sigma*x0
    yn1 = x1

    return xn0, yn0, xn1, yn1


@cuda.jit(device=True)
def find_exit(x0, y0, x1, y1, bounds):

    if x0 < bounds[0,0]:
        return 0

    if x0 > bounds[0,1]:
        return 1

    if y0 < bounds[1,0]:
        return 2

    if y0 > bounds[1,1]:
        return 3

    if x1 < bounds[2,0]:
        return 4

    if x1 > bounds[2,1]:
        return 5

    if y1 < bounds[3,0]:
        return 6

    if y1 > bounds[3,1]:
        return 7

    return -1


@cuda.jit
def escape_basin_kernel(
    X,
    Y,
    delta,
    a,
    b,
    sigma,
    max_time,
    bounds,
    out_exit,
    out_time,
):

    ix, iy = cuda.grid(2)

    nx = X.shape[0]
    ny = Y.shape[0]

    if ix >= nx or iy >= ny:
        return

    x0 = X[ix]
    y0 = Y[iy]

    x1 = x0 + delta
    y1 = y0 + delta

    for step in range(max_time):

        label = find_exit(x0, y0, x1, y1, bounds)

        if label != -1:
            out_exit[ix, iy] = label
            out_time[ix, iy] = step
            return

        x0, y0, x1, y1 = henon_coupled_map(x0, y0, x1, y1, a, b, sigma)

    out_exit[ix, iy] = -1
    out_time[ix, iy] = max_time


@cuda.jit
def escape_points_kernel(
    X0,
    Y0,
    delta,
    a,
    b,
    sigma,
    max_time,
    bounds,
    out_label,
):

    i = cuda.grid(1)

    if i >= X0.shape[0]:
        return

    x0 = X0[i]
    y0 = Y0[i]

    x1 = x0 + delta
    y1 = y0 + delta

    for step in range(max_time):

        label = find_exit(x0, y0, x1, y1, bounds)

        if label != -1:
            out_label[i] = label
            return

        x0, y0, x1, y1 = henon_coupled_map(x0, y0, x1, y1, a, b, sigma)

    out_label[i] = -1


# =============================================================================
# Helpers CUDA
# =============================================================================

def to_device_f64(arr):
    return cuda.to_device(np.ascontiguousarray(arr, dtype=np.float64))


def compute_escape_basin_cuda(X, Y, delta, a, b, sigma, max_time, bounds, threads=(16,16)):

    nx = len(X)
    ny = len(Y)

    X_d = to_device_f64(X)
    Y_d = to_device_f64(Y)
    bounds_d = to_device_f64(bounds)

    out_exit = cuda.device_array((nx, ny), dtype=np.int32)
    out_time = cuda.device_array((nx, ny), dtype=np.int32)

    blocks = (
        (nx + threads[0]-1)//threads[0],
        (ny + threads[1]-1)//threads[1],
    )

    escape_basin_kernel[blocks, threads](
        X_d, Y_d, delta, a, b, sigma, max_time, bounds_d, out_exit, out_time,
    )

    cuda.synchronize()

    return np.stack(
        (out_exit.copy_to_host(), out_time.copy_to_host()),
        axis=-1,
    )


def compute_escape_points_cuda(X, Y, delta, a, b, sigma, max_time, bounds, threads=256):

    n = len(X)

    X_d = to_device_f64(X)
    Y_d = to_device_f64(Y)
    bounds_d = to_device_f64(bounds)

    out_label = cuda.device_array(n, dtype=np.int32)

    blocks = (n + threads - 1)//threads

    escape_points_kernel[blocks, threads](
        X_d, Y_d, delta, a, b, sigma, max_time, bounds_d, out_label,
    )

    cuda.synchronize()

    return out_label.copy_to_host()


# =============================================================================
# Funcoes CPU (numba njit) - mesma dinamica, usada no expoente de incerteza
# =============================================================================

@njit(cache=True)
def henon_coupled_map_cpu(x0, y0, x1, y1, a, b, sigma):

    xn0 = a - x0*x0 + b*y0 + sigma*x1
    yn0 = x0

    xn1 = a - x1*x1 + b*y1 + sigma*x0
    yn1 = x1

    return xn0, yn0, xn1, yn1


@njit(cache=True)
def find_exit_cpu(x0, y0, x1, y1, bounds):

    if x0 < bounds[0,0]:
        return 0
    if x0 > bounds[0,1]:
        return 1
    if y0 < bounds[1,0]:
        return 2
    if y0 > bounds[1,1]:
        return 3
    if x1 < bounds[2,0]:
        return 4
    if x1 > bounds[2,1]:
        return 5
    if y1 < bounds[3,0]:
        return 6
    if y1 > bounds[3,1]:
        return 7

    return -1


@njit(cache=True)
def escape_label_cpu(x0, y0, delta, a, b, sigma, max_time, bounds):

    x1 = x0 + delta
    y1 = y0 + delta

    for step in range(max_time):

        label = find_exit_cpu(x0, y0, x1, y1, bounds)

        if label != -1:
            return label

        x0, y0, x1, y1 = henon_coupled_map_cpu(x0, y0, x1, y1, a, b, sigma)

    return -1


@njit(cache=True, nogil=True)
def _uncertain_count_single_epsilon(
    xs, ys, labels_center, eps, dcos, dsin, u_rand, needed,
    delta, a, b, sigma, max_time, bounds,
):

    n_samples = xs.shape[0]
    p_samples = dcos.shape[0]

    uncertain = 0

    for i in range(n_samples):

        different = 0

        for j in range(p_samples):

            r = eps * np.sqrt(u_rand[i, j])

            xp = xs[i] + r*dcos[j]
            yp = ys[i] + r*dsin[j]

            lbl = escape_label_cpu(xp, yp, delta, a, b, sigma, max_time, bounds)
            lbl_bin = 1 if lbl >= 0 else 0          # <<< FIX: mesma convenção de labels_center

            if lbl_bin != labels_center[i]:
                different += 1
                if different >= needed:
                    break

        if different >= needed:
            uncertain += 1

    return uncertain

def _process_single_epsilon(
    eps, xs, ys, labels_center, p_samples, dcos, dsin, needed,
    delta, a, b, sigma, max_time, bounds, n_samples, child_seed,
):

    rng = np.random.default_rng(child_seed)
    u_rand = rng.random((n_samples, p_samples))

    uncertain = _uncertain_count_single_epsilon(
        xs, ys, labels_center, eps, dcos, dsin, u_rand, needed,
        delta, a, b, sigma, max_time, bounds,
    )

    return uncertain / n_samples


# =============================================================================
# Amostragem restrita a fronteira (mais eficiente que amostrar tudo)
# =============================================================================

def boundary_candidate_mask(labels, k=1):
    """
    Marca como candidatos os pontos cuja vizinhanca k-imediata na grade
    ja contem rotulo diferente (proxy de proximidade a fronteira fractal).
    Inclui transicoes entre saidas de escape distintas E entre escape/preso.
    """

    center = labels[k:-k, k:-k]
    diff = np.zeros_like(center, dtype=bool)

    for di in range(-k, k+1):
        for dj in range(-k, k+1):
            if di == 0 and dj == 0:
                continue
            neighbor = labels[k+di:labels.shape[0]-k+di, k+dj:labels.shape[1]-k+dj]
            diff |= (neighbor != center)

    mask = np.zeros(labels.shape, dtype=bool)
    mask[k:-k, k:-k] = diff

    return mask


# =============================================================================
# Expoente de incerteza (joblib + numba CPU)
# =============================================================================

def uncertainty_fraction_mapping(
    X,
    Y,
    labels,
    delta,
    a,
    b,
    sigma,
    max_time,
    bounds,
    n_samples=100_000,
    p_samples=7,
    threshold=0.1,
    n_eps=100,
    eps_max=1e-2,
    eps_min=1e-7,
    seed=13,
    n_jobs=-1,
    restrict_to_boundary=True,
):

    rng = np.random.default_rng(seed)

    if restrict_to_boundary:
        mask = boundary_candidate_mask(labels, k=1)
        flat_x = X[mask]
        flat_y = Y[mask]
        flat_labels = labels[mask]
    else:
        flat_x = X.ravel()
        flat_y = Y.ravel()
        flat_labels = labels.ravel()

    n_samples = min(n_samples, flat_x.size)

    idx = rng.choice(flat_x.size, size=n_samples, replace=False)

    xs = flat_x[idx]
    ys = flat_y[idx]
    labels_center = flat_labels[idx].astype(np.int32)

    epsilons = np.logspace(np.log10(eps_max), np.log10(eps_min), n_eps)

    angles = np.linspace(0, 2*np.pi, p_samples, endpoint=False)
    dcos = np.cos(angles)
    dsin = np.sin(angles)

    needed = max(1, int(np.ceil(threshold * p_samples)))

    ss = np.random.SeedSequence(seed)
    child_seeds = ss.spawn(n_eps)

    f_eps = Parallel(n_jobs=n_jobs)(
        delayed(_process_single_epsilon)(
            epsilons[k], xs, ys, labels_center, p_samples, dcos, dsin, needed,
            delta, a, b, sigma, max_time, bounds, n_samples, child_seeds[k],
        )
        for k in tqdm(range(n_eps), desc="Calculando f(epsilon)")
    )

    return epsilons, np.array(f_eps)


# =============================================================================
# Entropia de bacia (SB, SBB)
# =============================================================================

def basin_entropy(basin, n, log_base=np.e):

    Nx, Ny = basin.shape

    if Nx % n != 0 or Ny % n != 0:
        raise ValueError(
            f"Tamanho de sub-caixa ({n},{n}) deve dividir as dimensoes da bacia ({Nx},{Ny})"
        )

    Mx = Nx // n
    My = Ny // n

    S = np.zeros((Mx, My))
    boundary_mask = np.zeros((Mx, My), dtype=bool)

    for i in range(Mx):
        for j in range(My):

            box = basin[i*n:(i+1)*n, j*n:(j+1)*n]
            unique_states, counts = np.unique(box, return_counts=True)

            if len(unique_states) > 1:
                boundary_mask[i, j] = True

            probs = counts / counts.sum()
            S[i, j] = -np.sum(probs * np.log(probs) / np.log(log_base))

    sb = np.mean(S)
    sbb = np.mean(S[boundary_mask]) if boundary_mask.any() else 0.0

    return float(sb), float(sbb)


# =============================================================================
# Ajuste log-log: expoente alpha, dimensao D e R^2
# =============================================================================

def fit_uncertainty_exponent(epsilons, f_eps, dimension=2):

    mask = f_eps > 0

    log_eps = np.log10(epsilons[mask])
    log_f = np.log10(f_eps[mask])

    coef, cov = np.polyfit(log_eps, log_f, 1, cov=True)

    alpha = coef[0]
    alpha_err = np.sqrt(cov[0, 0])

    log_f_pred = np.polyval(coef, log_eps)
    ss_res = np.sum((log_f - log_f_pred)**2)
    ss_tot = np.sum((log_f - np.mean(log_f))**2)
    r2 = 1 - ss_res/ss_tot

    D = dimension - alpha

    return alpha, alpha_err, D, r2


# =============================================================================
# Plot no estilo de dois paineis (bacia + f(epsilon))
# =============================================================================

def plot_basin_and_uncertainty(
    X,
    Y,
    labels,
    epsilons,
    f_eps,
    alpha,
    alpha_err,
    D,
    r2,
    sb,
    sbb,
    outfile="basin_uncertainty.png",
):
    from pynamicalsys import PlotStyler
    ps = PlotStyler()
    ps.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]

    im = ax.imshow(
        labels.T,
        origin="lower",
        extent=[X.min(), X.max(), Y.min(), Y.max()],
        aspect="auto",
        cmap="binary_r",
        interpolation="nearest",
    )

    ax.set_xlabel(r"$x_0$")
    ax.set_ylabel(r"$y_0$")
    ax.text(
        0.03, 0.97,
        "(a)",
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=14,
        bbox= dict(boxstyle="round", facecolor="0.9", edgecolor="0.5")
    )

    ratio = sb/sbb if sbb != 0 else 0.0

    textstr = (
        rf"$S_B$: {sb:.4f}"
        "\n"
        rf"$S_{{BB}}$: {sbb:.4f}"
    )

    ax.text(
        0.97, 0.97,
        textstr,
        transform=ax.transAxes,
        va="top", ha="right",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="0.9", edgecolor="0.5"),
    )

    ax2 = axes[1]

    ax2.loglog(epsilons, f_eps, "o", ms=5, color="tab:blue", label="Data")

    eps_fit = np.logspace(np.log10(epsilons.min()), np.log10(epsilons.max()), 50)
    coef = np.polyfit(np.log10(epsilons[f_eps > 0]), np.log10(f_eps[f_eps > 0]), 1)
    f_fit = 10**np.polyval(coef, np.log10(eps_fit))

    ax2.loglog(
        eps_fit, f_fit, "-", color="red", lw=2,
        label=rf"fit: $f \sim \epsilon^{{{alpha:.3f}}}$",
    )

    ax2.set_xlabel(r"$\epsilon$")
    ax2.set_ylabel(r"$f(\epsilon)$")
    ax2.invert_xaxis()
    ax2.legend(loc="lower left", frameon=True)

    ax2.text(
        0.03, 0.97,
        "(b)",
        transform=ax2.transAxes,
        va="top", ha="left",
        fontsize=14,
        bbox= dict(boxstyle="round", facecolor="0.9", edgecolor="0.5")
    )

    textstr2 = (
        rf"$\alpha = {alpha:.4f}$"
        "\n"
        rf"$d = {D:.4f}$"
        "\n"
        rf"$R^2 = {r2:.4f}$"
    )

    ax2.text(
        0.97, 0.97,
        textstr2,
        transform=ax2.transAxes,
        va="top", ha="right",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="navajowhite", edgecolor="0.5"),
    )

    fig.tight_layout()
    fig.savefig(outfile, dpi=200)

    return fig


# =============================================================================
# Parametros
# =============================================================================

a = 1.5250
b = 0.2250
sigma = 0.0219
OUTPUT = f'basin_a{a:.4f}_b{b:.4f}_sigma{sigma:.4f}'

params = [a, b, sigma]

delta = np.float64(1e-15)

total_time = 10_000       # tempo maximo para a bacia completa (GPU)
unc_max_time = 2_000      # tempo maximo para o expoente de incerteza (CPU)

grid_size = 4000
bound_limit = 2
bounds = np.array([
    [-bound_limit, bound_limit],
    [-bound_limit, bound_limit],
    [-bound_limit, bound_limit],
    [-bound_limit, bound_limit],
], dtype=np.float64)

X = np.linspace(-bound_limit, bound_limit, grid_size)
Y = np.linspace(-bound_limit, bound_limit, grid_size)

XX, YY = np.meshgrid(X, Y, indexing="ij")

rng = np.random.default_rng(1234)

RECALCULATE = True


# =============================================================================
# Calcula a bacia (GPU)
# =============================================================================

if RECALCULATE:

    print("\nCalculando bacia (CUDA)...\n")
    t0 = time.time()
    basin = compute_escape_basin_cuda(
        X, Y, delta, a, b, sigma, total_time, bounds,
    )
    t1 = time.time()
    time_gerate_basin = t1 - t0

    labels = basin[:, :, 0].astype(np.int32)
    labels_binary = (labels >= 0).astype(np.int32)
    labels_flat = labels_binary.ravel()

    basin_df = pd.DataFrame({
        "x": XX.ravel(),
        "y": YY.ravel(),
        "label": labels_flat,
    })

    basin_df.to_csv(f"../data/basins/{OUTPUT}.csv", index=False)

else:

    basin_df = pd.read_csv(f"../data/basins/{OUTPUT}.csv")

    labels_flat = basin_df["label"].to_numpy().astype(np.int32)
    XX = basin_df["x"].to_numpy().reshape(grid_size, grid_size)
    YY = basin_df["y"].to_numpy().reshape(grid_size, grid_size)
    labels = labels_flat.reshape(grid_size, grid_size)

labels = (labels >= 0).astype(np.int32)

# =============================================================================
# Entropia de bacia
# =============================================================================

print("\nCalculando entropia de bacia...\n")

n_box = 5   # deve dividir grid_size

sb, sbb = basin_entropy(labels, n=n_box, log_base=2)

print(f"SB  = {sb:.4f}")
print(f"SBB = {sbb:.4f}")


# =============================================================================
# Expoente de incerteza (CPU, joblib)
# =============================================================================

print("\nCalculando expoente de incerteza (joblib + numba CPU)...\n")

epsilons, f_eps = uncertainty_fraction_mapping(
    XX,
    YY,
    labels,
    delta,
    a,
    b,
    sigma,
    unc_max_time,
    bounds,
    n_samples=100_000,
    p_samples=7,
    threshold=0.1,
    n_eps=40,
    eps_max=1e-3,
    eps_min=1e-8,
    seed=23847,
    n_jobs=-1,
    restrict_to_boundary=True,
)

alpha, alpha_err, D, r2 = fit_uncertainty_exponent(epsilons, f_eps, dimension=2)

print(f"alpha = {alpha:.4f} +/- {alpha_err:.4f}")
print(f"D     = {D:.4f}")
print(f"R^2   = {r2:.4f}\n")


# Fit power-law f ~ C * eps^alpha
valid_mask = (f_eps > 0) & np.isfinite(f_eps)
if np.sum(valid_mask) < 3:
    alpha = np.nan
    fit_info = {'alpha': np.nan, 'dimension': np.nan, 'r_squared': np.nan,
                    'valid_points': np.sum(valid_mask), 'log_C': np.nan}
    fit_info['sb'] = sb
    fit_info['sbb'] = sbb
    
else:
    log_eps = np.log10(epsilons[valid_mask])
    log_f = np.log10(f_eps[valid_mask])
    coeffs = np.polyfit(log_eps, log_f, 1)
    alpha = coeffs[0]
    log_C = coeffs[1]
    log_f_pred = alpha * log_eps + log_C
    ss_res = np.sum((log_f - log_f_pred) ** 2)
    ss_tot = np.sum((log_f - np.mean(log_f)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    dimension = 2.0 - alpha
    fit_info = {'alpha': alpha, 'dimension': dimension, 'r_squared': r2, 'valid_points': np.sum(valid_mask), 'log_C': log_C}
    fit_info['sb'] = sb
    fit_info['sbb'] = sbb

# uncertainty_df = pd.DataFrame({"epsilon": epsilons, "f_epsilon": f_eps})
# uncertainty_df.to_csv("uncertainty_fraction.csv", index=False)

# =============================================================================
# Salvando dados
# =============================================================================
from utils import salvar_basin_labels_csv, salvar_uncertainty_csv
caminho_basin = f'../data/basins/basin_labels_a{a:.4f}_b{b:.4f}_eps{sigma:.4f}_delta{delta:.4f}.csv'
caminho_uncert_exp = f'../data/uncertainty/uncertainty_data_a{a:.4f}_b{b:.4f}_eps{sigma:.4f}_delta{delta:.4f}.csv'

salvar_basin_labels_csv(
    basin_labels=labels,
    x_vals=X,
    y_vals=Y,
    parameters=params,
    caminho=caminho_basin,
    autor='Conrado F. Bittencourt',
    tempo_calculo=time_gerate_basin
)

salvar_uncertainty_csv(
    epsilons=epsilons,
    fractions=f_eps,
    alpha=alpha,
    fit_info=fit_info,
    parameters=params,
    caminho=caminho_uncert_exp,
    autor='Conrado F. Bittencourt',
    direction='polar',
    n_samples=100_000,
    grid_size=grid_size
)

# =============================================================================
# Figura final (dois paineis)
# =============================================================================

print("\nGerando figura...\n")
print(np.unique(labels))
fig = plot_basin_and_uncertainty(
    X, Y, labels, epsilons, f_eps, alpha, alpha_err, D, r2, sb, sbb,
    outfile=f"../figures/basins/{OUTPUT}.png",
)

print(f"Concluido. Figura salva em ../figures/basins/{OUTPUT}.png\n")