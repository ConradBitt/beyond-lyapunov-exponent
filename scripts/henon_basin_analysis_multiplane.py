import itertools
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, to_rgba
from numba import cuda, float64
from tqdm import tqdm

# ------------------------------------------------------------- parametros
REGIONS = {"R1": 1.460, "R2": 1.525}
COLORS = {"R1": "tab:green", "R2": "tab:blue"}
B = 0.225
SIGMA = 0.0219
DELTA = 1e-15        # deslocamento transversal da fatia (x0, y0)
DELTA_SIX = 0.0      # 0.0: condicoes iniciais exatamente sobre S
BOUND = 2.0
MAX_TIME = 10_000    # constantes globais congeladas na compilacao
N_GRID = 4000
N_BOX = 5
N_SAMPLES = 500_000
P_SAMPLES = 7
THRESHOLD = 0.1
MIN_COUNT = 3       # minimo de pontos incertos aceito no ajuste
RESTRICT = True      # True: populacao na fronteira; False: uniforme
REF_POINT = (0.0, 0.0)
EPS = np.logspace(-3, -8, 40)
OUT = Path(__file__).resolve().parent.parent / "figures" / "basins"
RECALCULATE = False  # False: relê os dados; True: recalcula toda a análise

PLANES = ((0, 1), (0, 2), (0, 3), (1, 3))
NAMES = ("x^{[0]}", "y^{[0]}", "x^{[1]}", "y^{[1]}")
AXIS = np.linspace(-BOUND, BOUND, N_GRID)
EXTENT = [-BOUND, BOUND, -BOUND, BOUND]
THREADS_GRID = (16, 16)
THREADS_PTS = (32, 8)


# ------------------------------------------------------------------- GPU
@cuda.jit(device=True)
def escape_label(z, tie, delta, a, b, sigma, max_time, bound):
    """Rotulo binario: 1 se alguma componente excede bound, 0 se nao."""
    if tie:
        z[2] = z[0] + delta
        z[3] = z[1] + delta
    x0, y0, x1, y1 = z[0], z[1], z[2], z[3]
    for _ in range(max_time):
        if (abs(x0) > bound or abs(y0) > bound
                or abs(x1) > bound or abs(y1) > bound):
            return 1
        x0, y0, x1, y1 = (a - x0*x0 + b*y0 + sigma*x1, x0,
                          a - x1*x1 + b*y1 + sigma*x0, x1)
    return 0


@cuda.jit
def basin_kernel(u, v, base, ci, cj, tie, delta, a, b, sigma, out):
    iu, iv = cuda.grid(2)
    if iu >= u.shape[0] or iv >= v.shape[0]:
        return
    z = cuda.local.array(4, float64)
    for k in range(4):
        z[k] = base[k]
    z[ci] = u[iu]
    z[cj] = v[iv]
    out[iu, iv] = escape_label(z, tie, delta, a, b, sigma, MAX_TIME, BOUND)


@cuda.jit
def perturb_kernel(base, ci, cj, tie, delta, eps, cs, sn, rad, a, b,
                   sigma, out):
    i, j = cuda.grid(2)
    if i >= cs.shape[0] or j >= cs.shape[1]:
        return
    z = cuda.local.array(4, float64)
    for k in range(4):
        z[k] = base[i, k]
    z[ci] += eps*rad[i, j]*cs[i, j]
    z[cj] += eps*rad[i, j]*sn[i, j]
    out[i, j] = escape_label(z, tie, delta, a, b, sigma, MAX_TIME, BOUND)


def blocks_for(nx, ny, threads):
    return -(-nx // threads[0]), -(-ny // threads[1])


def compute_basin(plane, base, tie, delta, a, sigma):
    """Bacia binaria (1 = escape) no plano de componentes `plane`."""
    out = cuda.device_array((N_GRID, N_GRID), dtype=np.int8)
    ax = cuda.to_device(AXIS)
    base_d = cuda.to_device(np.asarray(base, dtype=np.float64))
    basin_kernel[blocks_for(N_GRID, N_GRID, THREADS_GRID), THREADS_GRID](
        ax, ax, base_d, plane[0], plane[1], tie, delta, a, B, sigma, out)
    return out.copy_to_host()


# ------------------------------------------------- populacao e incerteza
def boundary_candidates(labels):
    """Pontos com ao menos um vizinho (8-conexo) de rotulo distinto."""
    c = labels[1:-1, 1:-1]
    diff = np.zeros(c.shape, dtype=bool)
    nx, ny = labels.shape
    for di, dj in itertools.product((-1, 0, 1), repeat=2):
        if di or dj:
            diff |= labels[1 + di:nx - 1 + di, 1 + dj:ny - 1 + dj] != c
    mask = np.zeros(labels.shape, dtype=bool)
    mask[1:-1, 1:-1] = diff
    return mask


def sample_population(labels, rng):
    """Pontos (xs, ys) sobre S, na fronteira ou uniformes na fatia."""
    if RESTRICT:
        mask = boundary_candidates(labels)
    else:
        mask = np.ones(labels.shape, dtype=bool)
    iu, iv = np.nonzero(mask)
    k = rng.choice(iu.size, size=min(N_SAMPLES, iu.size), replace=False)
    return AXIS[iu[k]], AXIS[iv[k]]


def make_samples(n, rng, p=P_SAMPLES):
    """Direcoes (fase aleatoria por ponto) e raios uniformes no disco."""
    ang = rng.uniform(0, 2*np.pi, (n, 1)) + 2*np.pi*np.arange(p)/p
    rad = np.sqrt(rng.random((n, p)))
    return np.cos(ang), np.sin(ang), rad


def uncertainty_curve(base, plane, tie, delta, a, sigma, samples):
    """f(eps): fracao de pontos com ao menos `needed` vizinhos de rotulo
    distinto; centro e vizinhos usam o mesmo kernel (mesma aritmetica).
    """
    cs, sn, rad = (cuda.to_device(s) for s in samples)
    base_d = cuda.to_device(np.ascontiguousarray(base, dtype=np.float64))
    n, p = cs.shape
    needed = max(1, math.ceil(THRESHOLD*p))
    grid = blocks_for(n, p, THREADS_PTS)

    def labels_at(eps):
        out = cuda.device_array((n, p), dtype=np.int8)
        perturb_kernel[grid, THREADS_PTS](
            base_d, plane[0], plane[1], tie, delta, eps, cs, sn, rad,
            a, B, sigma, out)
        return out.copy_to_host()

    center = labels_at(0.0)[:, :1]
    f = [np.mean((labels_at(e) != center).sum(axis=1) >= needed)
         for e in tqdm(EPS, leave=False)]
    return np.array(f)


def fit_exponent(eps, f, n):
    """Ajuste log-log f ~ eps**alpha, excluindo f*n < MIN_COUNT."""
    ok = f*n >= MIN_COUNT
    if ok.sum() < 3:
        return np.nan, np.nan, np.nan
    x, y = np.log10(eps[ok]), np.log10(f[ok])
    alpha, c = np.polyfit(x, y, 1)
    ss_tot = ((y - y.mean())**2).sum()
    r2 = 1 - ((y - alpha*x - c)**2).sum()/ss_tot if ss_tot > 0 else np.nan
    return alpha, c, r2


def basin_entropy(labels, n=N_BOX):
    """S_B e S_BB (log2) para bacia binaria, sub-caixas n x n."""
    mx, my = labels.shape[0] // n, labels.shape[1] // n
    p = labels[:mx*n, :my*n].reshape(mx, n, my, n).mean(axis=(1, 3))
    inner = (p > 0) & (p < 1)
    s = np.zeros_like(p)
    q = p[inner]
    s[inner] = -(q*np.log2(q) + (1 - q)*np.log2(1 - q))
    return s.mean(), (s[inner].mean() if inner.any() else 0.0)


# --------------------------------------------------------------- analises
def slice_analysis(a, sigma, delta, rng):
    """Fatia (x0, y0) com x1 = x0 + delta e y1 = y0 + delta."""
    labels = compute_basin((0, 1), np.zeros(4), True, delta, a, sigma)
    xs, ys = sample_population(labels, rng)
    samples = make_samples(xs.size, rng)
    f = uncertainty_curve(np.column_stack((xs, ys, xs, ys)), (0, 1), True,
                          delta, a, sigma, samples)
    alpha, c, r2 = fit_exponent(EPS, f, xs.size)
    sb, sbb = basin_entropy(labels)
    return dict(labels=labels, f=f, n=xs.size, alpha=alpha, c=c, r2=r2,
                sb=sb, sbb=sbb, xs=xs, ys=ys)


def four_planes(a, ref, rng):
    """Quatro planos, mesma populacao (xs, ys) e mesmas perturbacoes."""
    xs, ys = ref["xs"], ref["ys"]
    d = DELTA_SIX
    base = np.column_stack((xs, ys, xs + d, ys + d))
    xr, yr = REF_POINT
    base_ref = (xr, yr, xr + d, yr + d)
    samples = make_samples(xs.size, rng)
    basins = [compute_basin(pl, base_ref, False, d, a, SIGMA)
              for pl in PLANES]
    f = np.array([uncertainty_curve(base, pl, False, d, a, SIGMA, samples)
                  for pl in PLANES])
    return dict(basins=np.asarray(basins), f=f, n=xs.size)


# ---------------------------------------------------------------- figuras
def show_basin(ax, labels, color='white', alpha=1):
    # Define o preto com alpha=1 e a segunda cor com o alpha desejado
    cmap = ListedColormap([
        to_rgba("black", alpha=1.0),
        to_rgba(color, alpha=alpha)
    ])
    
    ax.imshow(labels.T, origin="lower", extent=EXTENT, aspect="auto",
              cmap=cmap, vmin=0, vmax=1)


def plot_four(tag, a, res, outfile):
    from string import ascii_lowercase
    bbox = {'facecolor': 'w','alpha': 0.75,'linewidth': 0.0,'pad': 1}

    CM = 1/2.54
    fig = plt.figure(figsize=(30*CM, 14*CM))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 2.4], hspace=0.3, wspace=0.52)
    cmap = plt.get_cmap("tab10")
    ax0 = None
    for k, (pl, basin) in enumerate(zip(PLANES, res["basins"])):
        row, col = divmod(k, 2)
        if k == 0:
            # Cria o primeiro subplot normalmente
            ax = fig.add_subplot(gs[row, col])
            ax0 = ax  # Guarda a referência
        else:
            # Cria os subplots seguintes partilhando os eixos com ax0
            ax = fig.add_subplot(gs[row, col], sharex=ax0, sharey=ax0)
            # ax.tick_params(labelbottom=False, labelleft=False)
            
        color = cmap(k)
        show_basin(ax, basin, color)
        ax.set_xlabel(f"${NAMES[pl[0]]}$")
        ax.set_ylabel(f"${NAMES[pl[1]]}$", rotation=0, labelpad=7)
        letter = np.sum(k)
        ax.text(0.02,0.89,f'({ascii_lowercase[letter]})',bbox=bbox, transform=ax.transAxes)

    axc = fig.add_subplot(gs[:, 2])        
    for k, (pl, f) in enumerate(zip(PLANES, res["f"])):
        alpha, c, _ = fit_exponent(EPS, f, res["n"])
        fit_mask = f * res["n"] >= MIN_COUNT
        if alpha > 0.3:
            axc.loglog(EPS, f, "o", ms=5, lw=1.2, color=cmap(k),
                       label=rf"$({NAMES[pl[0]]},\,{NAMES[pl[1]]})$: "
                             rf"$\alpha={alpha:.3f}$", zorder=10)
        else:
            axc.semilogx(EPS, f, "o", ms=5, lw=1.2, color=cmap(k),
                         label=rf"$({NAMES[pl[0]]},\,{NAMES[pl[1]]})$: "
                               rf"$\alpha={alpha:.3f}$", zorder=10)
        if fit_mask.sum() >= 3 and np.isfinite(alpha) and np.isfinite(c):
            fit_eps = EPS[fit_mask]
            fit = 10 ** (alpha * np.log10(fit_eps) + c)
            if alpha > 0.3:
                axc.loglog(fit_eps, fit, "--", lw=1.1, color=cmap(k), zorder=9)
            else:
                axc.semilogx(fit_eps, fit, "--", lw=1.1, color=cmap(k), zorder=9)
            
    f_mean = res["f"].mean(axis=0)
    alpha, c, _ = fit_exponent(EPS, f_mean, res["n"]) 
    fit_mask = f_mean * res["n"] >= MIN_COUNT

    if alpha > 0.3:
        axc.loglog(EPS, f_mean, "ok", lw=2.5, alpha=1.0,
                   label=rf"mean: $\alpha={alpha:.3f}$", zorder=8)
    else:
        axc.semilogx(EPS, f_mean, "ok", lw=2.5, alpha=1.0,
                     label=rf"mean: $\alpha={alpha:.3f}$", zorder=8)

    if fit_mask.sum() >= 3 and np.isfinite(alpha) and np.isfinite(c):
        fit_eps = EPS[fit_mask]
        fit = 10 ** (alpha * np.log10(fit_eps) + c)
        if alpha > 0.3:
            axc.loglog(fit_eps, fit, "-", lw=1.5, color='k', zorder=7)
        else:
            axc.semilogx(fit_eps, fit, "-", lw=1.5, color='k', zorder=7)

    axc.text(0.06,0.95,f'({ascii_lowercase[letter+1]})',bbox=bbox, transform=axc.transAxes)
    axc.invert_xaxis()
    axc.set_xlabel(r"$\epsilon$")
    axc.set_ylabel(r"$f(\epsilon)$", rotation=0, labelpad=20)
    axc.legend(loc="upper right", fontsize=14)
    axc.minorticks_on()
    axc.grid(which="both", alpha=0.25)
    # fig.suptitle(f"{tag}: a = {a:.3f}")
    fig.tight_layout()
    fig.savefig(outfile, dpi=300,  bbox_inches='tight')#, pad_inches=1)
    print(f'FIG SAVED: {outfile}')
    plt.close(fig)


def box(ax, xy, r, k, color):
    txt = "\n".join([f"$R{k}$",
                     rf"$\alpha_{{{k}}}={r['alpha']:.3f}$",
                     rf"$R^2={r['r2']:.4f}$",
                     rf"$S_b={r['sb']:.4f}$",
                     rf"$S_{{bb}}={r['sbb']:.4f}$"])
    ax.text(*xy, txt, transform=ax.transAxes, color=color, ha="right",
            va="center", bbox=dict(boxstyle="round", fc="white", ec=color))


def plot_reference(res, outfile, tags=("R1", "R2")):
    CM = 1/2.54
    fig = plt.figure(figsize=(35*CM, 15*CM))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 2.4], hspace=0.09,
                          wspace=0.25)
    axc = fig.add_subplot(gs[:, 1])
    inset = axc.inset_axes([0.50, 0.54, 0.50, 0.36])
    for k, (tag, panel, mk) in enumerate(zip(tags, "ab", "os"), start=1):
        r, col = res[tag], COLORS[tag]
        ax = fig.add_subplot(gs[k - 1, 0])
        show_basin(ax, r["labels"])
        ax.text(0.04, 0.95, f"({panel})", transform=ax.transAxes, va="top",
                bbox=dict(boxstyle="round", fc="0.9", ec="0.9"))
        ax.text(0.96, 0.95, tag, transform=ax.transAxes, va="top",
                ha="right", color=col,
                bbox=dict(boxstyle="round", fc="0.9", ec="0.5"))
        ax.set_ylabel(r"$y^{[0]}$", rotation=0,labelpad=12)
        if k == 2:
            ax.set_xlabel(r"$x^{[0]}$")
        else:
            ax.set_xticklabels([])
        fit = 10**(r["alpha"]*np.log10(EPS) + r["c"])
        axc.loglog(EPS, r["f"], mk, color=col, mec="k", ms=8, label=tag)
        axc.loglog(EPS, fit, "--", color=col, lw=2)
        if k == 1:
            box(axc, (0.26, 0.34), r, k, col)
        else:
            inset.semilogx(EPS, r["f"], mk, color=col, ms=3)
            inset.semilogx(EPS, fit, "-", color=col, lw=1.5)
            box(inset, (0.20, -0.95), r, k, col)
    for ax in (axc, inset):
        ax.invert_xaxis()
        ax.minorticks_on()
        ax.set_xlabel(r"$\epsilon$", )
        ax.grid(which='both', alpha=0.25)
    inset.set_ylim(None, 1.0)
    inset.set_ylabel(r"$f(\epsilon)$", rotation=0, labelpad=18)
    axc.set_ylabel(r"$f(\epsilon)$", rotation=0, labelpad=10)
    axc.legend(loc="lower left")
    axc.text(0.05, 0.85, "(c)", transform=axc.transAxes, va="top")
    inset.text(0.20, 0.95, "(d)", transform=inset.transAxes, va="top")
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    print(f"FIG SAVED: {outfile}")
    plt.close(fig)


def save_npz(res, outfile):
    np.savez_compressed(outfile, eps=EPS, f=res["f"], n=res["n"],
                        labels=res["labels"], alpha=res["alpha"],
                        r2=res["r2"], sb=res["sb"], sbb=res["sbb"])


def load_npz(path, required):
    """Carrega um cache e valida os campos e a grade de eps esperados."""
    if not path.exists():
        raise FileNotFoundError(
            f"Cache ausente: {path}. Execute com RECALCULATE=True.")
    with np.load(path, allow_pickle=False) as data:
        missing = set(required) - set(data.files)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(
                f"Cache incompatível ({path}): faltam {names}. "
                "Execute com RECALCULATE=True para regenerá-lo.")
        if not np.array_equal(data["eps"], EPS):
            raise ValueError(
                f"Cache incompatível ({path}): EPS difere da configuração "
                "atual. Execute com RECALCULATE=True.")
        return {key: data[key].copy() for key in required}


def load_slice(path):
    """Carrega resultados de uma análise de fatia."""
    res = load_npz(path, ("eps", "f", "n", "labels", "alpha", "r2",
                          "sb", "sbb"))
    if res["f"].shape != EPS.shape or res["labels"].shape != (N_GRID, N_GRID):
        raise ValueError(
            f"Cache incompatível ({path}): dimensões inesperadas. "
            "Execute com RECALCULATE=True.")
    _, res["c"], _ = fit_exponent(EPS, res["f"], int(res["n"]))
    return res


def load_four(tag):
    """Carrega as curvas e as bacias dos quatro planos."""
    path = OUT / f"four_curves_{tag}.npz"
    res = load_npz(path, ("eps", "f", "n", "basins"))
    if (res["f"].shape != (len(PLANES), EPS.size)
            or res["basins"].shape != (len(PLANES), N_GRID, N_GRID)):
        raise ValueError(
            f"Cache incompatível ({path}): faltam dados dos mapas dos "
            "quatro planos ou as dimensões são inesperadas. "
            "Execute com RECALCULATE=True para regenerá-lo.")
    return res


# ------------------------------------------------------------------ main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        from pynamicalsys import PlotStyler
        PlotStyler().apply_style()
    except ImportError:
        pass
    rng = np.random.default_rng(23847)

    controls = {"sigma0": (0.0, DELTA), "onS": (SIGMA, 0.0)}
    if RECALCULATE:
        ref = {}
        for tag, a in REGIONS.items():
            ref[tag] = slice_analysis(a, SIGMA, DELTA, rng)
            save_npz(ref[tag], OUT / f"slice_{tag}.npz")
            four = four_planes(a, ref[tag], rng)
            plot_four(tag, a, four, OUT / f"four_planes_{tag}.png")
            np.savez_compressed(OUT / f"four_curves_{tag}.npz", eps=EPS,
                                f=four["f"], n=four["n"],
                                basins=four["basins"])
        plot_reference(ref, OUT / "reference_R1_R2.png")

        for name, (sigma, delta) in controls.items():
            res = {t: slice_analysis(a, sigma, delta, rng)
                   for t, a in REGIONS.items()}
            plot_reference(res, OUT / f"control_{name}.png")
            for t, r in res.items():
                save_npz(r, OUT / f"control_{name}_{t}.npz")
    else:
        ref = {tag: load_slice(OUT / f"slice_{tag}.npz")
               for tag in REGIONS}
        four_results = {tag: load_four(tag) for tag in REGIONS}
        control_results = {
            name: {
                tag: load_slice(OUT / f"control_{name}_{tag}.npz")
                for tag in REGIONS
            }
            for name in controls
        }
        for tag, a in REGIONS.items():
            four = four_results[tag]
            plot_four(tag, a, four, OUT / f"four_planes_{tag}.png")
        plot_reference(ref, OUT / "reference_R1_R2.png")

        for name in controls:
            plot_reference(control_results[name],
                           OUT / f"control_{name}.png")


if __name__ == "__main__":
    main()