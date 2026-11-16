"""Janela w comum para f+_w no espaço (a, b) de Hénon acoplados (CUDA).

Um thread = uma tripla (a, b, CI). Cada thread acumula, em paralelo,
os expoentes transversais de tempo finito de todas as janelas w.
A análise posterior (CPU) estima, para cada (a, b), a menor janela a
partir da qual f+_w deixa de variar, e extrai o máximo sobre a grade.
"""
import math
import warnings

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from numba import cuda, float64, int32

# --------------------------------------------------------------------
# Configuração
# --------------------------------------------------------------------
SIGMA = 0.0219
A_VALS = np.linspace(1.450, 1.55, 200)
B_VALS = np.linspace(0.200, 0.25, 200)
N_IC = 500 
TRANSIENT = 5_000
TOTAL_TIME = 100_000           # n_janelas = TOTAL_TIME // w
ESCAPE = 4.0
SEED = 756345
WINDOWS = np.unique(np.round(np.logspace(0, 4, 30)).astype(np.int32))
NW = len(WINDOWS)

TOL_A = 0.05                   # tolerância sobre a média das CIs
TOL_B = 0.05                   # tolerância de f+ por CI (vs. w máx.)
FRAC_MAX = 0.10                # fração de CIs que pode ainda variar
K_TAIL = 5                     # nº de w finais usados como referência
OUT = "fplus_w_param_space_full"


@cuda.jit
def kernel(a_vals, b_vals, ics, windows, sigma, transient, total,
           esc, out):
    idx = cuda.grid(1)
    n_b = b_vals.shape[0]
    n_ic = ics.shape[0]
    if idx >= a_vals.shape[0] * n_b * n_ic:
        return
    ic = idx % n_ic
    j = (idx // n_ic) % n_b
    i = idx // (n_ic * n_b)
    a = a_vals[i]
    b = b_vals[j]

    x0, y0 = ics[ic, 0], ics[ic, 1]
    x1, y1 = ics[ic, 2], ics[ic, 3]
    xs = x0
    ys = y0
    u0, u1 = 1.0, 0.0

    acc = cuda.local.array(NW, float64)
    cnt = cuda.local.array(NW, int32)
    pos = cuda.local.array(NW, int32)
    tot = cuda.local.array(NW, int32)
    for k in range(NW):
        acc[k] = 0.0
        cnt[k] = 0
        pos[k] = 0
        tot[k] = 0

    escaped = False
    for t in range(transient + total):
        n0 = (-2.0 * xs - sigma) * u0 + b * u1
        n1 = u0
        norm = math.sqrt(n0 * n0 + n1 * n1)
        if norm == 0.0:
            norm = 1e-16
        u0, u1 = n0 / norm, n1 / norm

        if t >= transient:
            lt = math.log(norm)
            for k in range(NW):
                acc[k] += lt
                cnt[k] += 1
                if cnt[k] == windows[k]:
                    if acc[k] > 0.0:
                        pos[k] += 1
                    tot[k] += 1
                    acc[k] = 0.0
                    cnt[k] = 0

        nx0 = a - x0 * x0 + b * y0 + sigma * x1
        nx1 = a - x1 * x1 + b * y1 + sigma * x0
        y0, y1, x0, x1 = x0, x1, nx0, nx1
        xs, ys = a - xs * xs + b * ys + sigma * xs, xs

        if not (abs(x0) < esc and abs(y0) < esc and abs(x1) < esc
                and abs(y1) < esc and abs(xs) < esc):
            escaped = True
            break

    for k in range(NW):
        if escaped or tot[k] == 0:
            out[i, j, ic, k] = math.nan
        else:
            out[i, j, ic, k] = pos[k] / tot[k]


def run_gpu():
    rng = np.random.default_rng(SEED)
    ics = rng.uniform(-1.5, 1.5, (N_IC, 4))
    out = np.full((len(A_VALS), len(B_VALS), N_IC, NW), np.nan,
                  dtype=np.float32)
    d_out = cuda.to_device(out)
    threads = 128
    blocks = (out.shape[0] * out.shape[1] * N_IC + threads - 1) // threads
    kernel[blocks, threads](
        cuda.to_device(A_VALS), cuda.to_device(B_VALS),
        cuda.to_device(ics), cuda.to_device(WINDOWS),
        SIGMA, TRANSIENT, TOTAL_TIME, ESCAPE, d_out)
    cuda.synchronize()
    return d_out.copy_to_host()


def first_stable_index(dev, tol):
    ok = dev <= tol
    ok = np.flip(np.cumprod(np.flip(ok, -1), -1), -1).astype(bool)
    return np.where(ok.any(axis=-1), np.argmax(ok, axis=-1), NW)


def analyse(fp):
    valid = ~np.isnan(fp[..., 0])                  # (nA, nB, nIC)
    n_valid = valid.sum(axis=2)
    bad = n_valid == 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        # Critério A: curva média das CIs.
        mean = np.nanmean(fp, axis=2)              # (nA, nB, nW)
        ref = mean[..., -K_TAIL:].mean(axis=-1, keepdims=True)
        dev_a = np.abs(mean - ref)
        # Critério B: cada CI estabiliza (para sempre) em w_c,ic;
        # tolera-se FRAC_MAX de CIs ainda instáveis em w.
        dev_ic = np.abs(fp - fp[..., -1:])         # (nA, nB, nIC, nW)
        idx_ic = first_stable_index(dev_ic, TOL_B)
        k = np.arange(NW)
        unstable = (idx_ic[..., None] > k) & valid[..., None]
        frac_b = unstable.sum(axis=2) / n_valid[..., None]
    res = {}
    for name, dev, tol in (("A", dev_a, TOL_A), ("B", frac_b, FRAC_MAX)):
        idx = first_stable_index(dev, tol)
        # sem convergência: estável só nos últimos K_TAIL pontos ou nunca
        unres = (idx > NW - 1 - K_TAIL) & ~bad
        w_c = WINDOWS[np.minimum(idx, NW - 1)].astype(float)
        w_c[unres] = WINDOWS[-1]
        w_c[bad] = np.nan
        dev[bad] = np.nan
        res[name] = (w_c, dev, unres)
    return res, mean, n_valid


def report(res):
    for name, (w_c, dev, unres) in res.items():
        w_common = np.nanmax(w_c)
        i, j = np.unravel_index(np.nanargmax(w_c), w_c.shape)
        n_ok = np.isfinite(w_c).sum()
        sign = ">=" if unres.any() else "="
        print(f"[{name}] w comum {sign} {w_common:.0f}; "
              f"sem convergência: {unres.sum()}/{n_ok}; "
              f"crítico: a={A_VALS[i]:.4f}, b={B_VALS[j]:.4f}")
        for q in (50, 90, 99):
            print(f"    percentil {q}: w_c = "
                  f"{np.nanpercentile(w_c, q):.0f}")
        for w in (5, 10, 20, 50, 100):
            frac = np.mean(w_c[np.isfinite(w_c)] <= w)
            print(f"    fração com w_c <= {w}: {frac:.3f}")


def plot(res):
    from pynamicalsys import PlotStyler
    ps = PlotStyler(fontsize=18)
    ps.apply_style()

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for k, name in enumerate(("A", "B")):
        w_c = res[name][0]
        hm = ax[k].pcolormesh(
            A_VALS, B_VALS, w_c.T, shading="nearest",
            norm=mcolors.LogNorm(vmin=1, vmax=WINDOWS.max()),
            cmap="nipy_spectral")
        cb = fig.colorbar(hm, ax=ax[k])
        cb.ax.set_title(r"$w_c(a,b)$", pad=10)
        ax[k].set(xlabel="$a$", ylabel="$b$", title=f"Critério {name}")
    for name, tol, c in (("A", TOL_A, "tab:blue"),
                         ("B", FRAC_MAX, "tab:orange")):
        dev = res[name][1]
        ax[2].plot(WINDOWS, np.nanmax(dev, axis=(0, 1)), "o-",
                   color=c, label=f"pior caso {name}")
        ax[2].axhline(tol, color=c, ls="--", lw=1)
    ax[2].set(xscale="log", xlabel="$w$", ylabel="pior caso em (a, b)")
    ax[2].legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}.png", dpi=200)

def plot_space_param(res):
    from pynamicalsys import PlotStyler
    ps = PlotStyler(fontsize=18)
    ps.apply_style()
    CM = 1/2.54
    fig, ax = plt.subplots(figsize=(12*CM, 12*CM))

    w_c = res["A"][0]
    hm = ax.pcolormesh(
        A_VALS, B_VALS, w_c.T, shading="nearest",
        norm=mcolors.LogNorm(vmin=1, vmax=WINDOWS.max()),
        cmap="CMRmap")
    cb = fig.colorbar(hm, ax=ax, orientation='horizontal', location='top')
    cb.ax.set_title(r"$w_c(a,b)$", pad=20)
    ax.set(xlabel="$a$", ylabel="$b$")
    _bbox = {"facecolor": "w", "alpha": 1., "linewidth": 0.0, "pad": 1}
    ax.text(
        0.02, 0.98, "(b)", transform=ax.transAxes, fontsize=18, verticalalignment="top",
        bbox=_bbox
    )

    scatter_infos = {
        'R1' : {
            'x':1.460,
            'y':0.225,
            's':100,
            'color': "#35CE21",
            'marker':'o',
            'linewidths': 1.1,
            'alpha':0.65
            },
        'R2':{
            'x':1.528,
            'y':0.225,
            's':100,
            'color': "#1f77b4",
            'marker': 's',
            'linewidths': 1.1,
            'alpha':0.65
            },
        'Chaos':{
            'x':1.475,
            'y':0.225,
            's':100,
            'color': "#d3860a",
            'marker': '^',
            'linewidths': 1.1,
            'alpha':0.65
        }
    }

    for key, info in scatter_infos.items():
        ax.scatter(
            info['x'], info['y'], s=info['s'], color=info['color'],
             marker=info['marker'], edgecolors="black", linewidths=info['linewidths'],
             zorder=10, clip_on=False,alpha=info['alpha'], label=key)
    ax.axhline(0.225, color="gray", linestyle="--", linewidth=1.1)
    ax.legend(loc='upper right', fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{OUT}_A.png", dpi=300)
    print(f"Figura salva: {OUT}_A.png")

def criterion_a(mean, n_valid):
    """Critério A a partir de <f+>(a, b, w): devolve (w_c, dev, unres)."""
    bad = n_valid == 0
    ref = mean[..., -K_TAIL:].mean(axis=-1, keepdims=True)
    dev = np.abs(mean - ref)
    idx = first_stable_index(dev, TOL_A)
    unres = (idx > NW - 1 - K_TAIL) & ~bad
    w_c = WINDOWS[np.minimum(idx, NW - 1)].astype(float)
    w_c[unres] = WINDOWS[-1]
    w_c[bad] = np.nan
    dev[bad] = np.nan
    return w_c, dev, unres


RECALCULATE = False

if __name__ == "__main__":

    if RECALCULATE:
        fp = run_gpu()
        res, mean, n_valid = analyse(fp)
        print(f"CIs válidas: min={n_valid.min()}, max={n_valid.max()} "
            f"de {N_IC}")
        report(res)
        np.savez_compressed(
            f"{OUT}.npz", a=A_VALS, b=B_VALS, w=WINDOWS, mean=mean,
            n_valid=n_valid, wc_A=res["A"][0], wc_B=res["B"][0])
    else:
        # lê os dados salvos anteriormente
        data_path = f"{OUT}.npz"
        try:
            with np.load(data_path) as data:
                A_VALS, B_VALS = data["a"], data["b"]
                WINDOWS = data["w"]
                mean, n_valid = data["mean"], data["n_valid"]
            NW = WINDOWS.size
            res = {"A": criterion_a(mean, n_valid)}
            print(f"Dados carregados de {data_path}")
            report(res)
            print(f"Dados carregados de {data_path}")
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Arquivo {data_path} não encontrado. "
                "Execute com RECALCULATE=True para regenerar os dados."
            ) from exc

    # plot(res)
    plot_space_param(res)
    print(f"Figura salva: {OUT}.png")