"""UPOs de período primo (mínimo) de dois mapas de Hénon acoplados (4D).

    x_i' = a - x_i^2 + b*y_i + sigma*x_j ,   y_i' = x_i   (i != j)

Newton multi-start para F^p(v) = v, com v sorteado em [-1.5, 1.5]^4.
O laço quente (Newton + filtro de sub-harmônicos) roda em GPU (numba.cuda);
deduplicação, multiplicadores de Floquet, tabelas e figuras rodam na CPU.
"""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import pandas as pd
from numba import cuda
from scipy.spatial import cKDTree

RECALCULATE = False  # False: lê o cache salvo; True: refaz a busca

A, B, SIGMA = 1.5250, 0.225, 0.0219
P_MAX = 100
N_TRIES = 50_000       # chutes iniciais por lote
MAX_BATCHES = 10          # máximo de lotes por período
PATIENCE = 3              # lotes seguidos sem UPO nova para encerrar p
DOMAIN = (-1.5, 1.5)
NEWTON_TOL = 1e-10        # tolerância mínima em |F^p(v) - v|_1
NOISE_FACTOR = 100.0      # piso de ruído: NOISE_FACTOR * eps * |M|
TOL_CAP = 1e-7            # teto da tolerância efetiva
NEWTON_MAX_ITER = 100
MAX_STEP = 1.0            # passo de Newton máximo (norma-1)
DEDUP_TOL = 1e-6
PRIME_TOL = 1e-7
SYNC_TOL = 1e-6           # órbita na variedade x0 = x1, y0 = y1
UNSTABLE_TOL = 1.0
SEED = 72348
THREADS = 128
SCHEMA = 1

pre = f''
PREFIX = f"upo_a{A}_b{B}_sigma{SIGMA}"
DATA_PATH = Path(f"{PREFIX}_data.npz")
CSV_PATH = Path(f"{PREFIX}_upos.csv")
SUMMARY_CSV = Path(f"{PREFIX}_summary.csv")
TEX_PATH = Path(f"{PREFIX}_summary.tex")
FIG_PATH = Path(f"{PREFIX}_summary.png")


def search_config():
    """Parâmetros que definem os dados salvos (validação do cache)."""
    return {
        "SCHEMA": SCHEMA, "A": A, "B": B, "SIGMA": SIGMA, "P_MAX": P_MAX,
        "N_TRIES": N_TRIES, "MAX_BATCHES": MAX_BATCHES,
        "PATIENCE": PATIENCE, "DOMAIN": list(DOMAIN),
        "NEWTON_TOL": NEWTON_TOL, "NOISE_FACTOR": NOISE_FACTOR,
        "TOL_CAP": TOL_CAP, "NEWTON_MAX_ITER": NEWTON_MAX_ITER,
        "MAX_STEP": MAX_STEP, "DEDUP_TOL": DEDUP_TOL,
        "PRIME_TOL": PRIME_TOL, "SYNC_TOL": SYNC_TOL,
        "UNSTABLE_TOL": UNSTABLE_TOL, "SEED": SEED,
    }


# ---------------------------------------------------------------------------
# GPU: Newton + filtro de sub-harmônicos
# ---------------------------------------------------------------------------
@cuda.jit(device=True)
def _fmap(x0, y0, x1, y1, a, b, s):
    return (a - x0 * x0 + b * y0 + s * x1, x0,
            a - x1 * x1 + b * y1 + s * x0, x1)


@cuda.jit(device=True)
def _lex_less(x0, y0, x1, y1, c0, c1, c2, c3):
    if x0 != c0:
        return x0 < c0
    if y0 != c1:
        return y0 < c1
    if x1 != c2:
        return x1 < c2
    return y1 < c3


@cuda.jit(device=True)
def _newton(v, p, a, b, s, tol, cap, nf, max_it, max_step, M, g):
    """Newton em G(v) = F^p(v) - v. `v` é atualizado in-place."""
    for _ in range(max_it):
        x0, y0, x1, y1 = v[0], v[1], v[2], v[3]
        for r in range(4):
            for c in range(4):
                M[r, c] = 1.0 if r == c else 0.0
        for _k in range(p):  # M <- J(v_k) M  (J esparso 4x4)
            for c in range(4):
                m0, m1, m2, m3 = M[0, c], M[1, c], M[2, c], M[3, c]
                M[0, c] = -2.0 * x0 * m0 + b * m1 + s * m2
                M[1, c] = m0
                M[2, c] = s * m0 - 2.0 * x1 * m2 + b * m3
                M[3, c] = m2
            x0, y0, x1, y1 = _fmap(x0, y0, x1, y1, a, b, s)
        g[0] = x0 - v[0]
        g[1] = y0 - v[1]
        g[2] = x1 - v[2]
        g[3] = y1 - v[3]
        nm = 0.0
        for r in range(4):
            for c in range(4):
                nm += abs(M[r, c])
        gs = abs(g[0]) + abs(g[1]) + abs(g[2]) + abs(g[3])
        if not math.isfinite(gs + nm):
            return False
        # tolerância efetiva: acompanha o ruído amplificado por |M|
        if gs < min(max(tol, nf * 2.2e-16 * nm), cap):
            return True
        for r in range(4):  # resolve (M - I) d = G (eliminação de Gauss)
            M[r, r] -= 1.0
        for col in range(4):
            piv = col
            big = abs(M[col, col])
            for r in range(col + 1, 4):
                if abs(M[r, col]) > big:
                    big = abs(M[r, col])
                    piv = r
            if big == 0.0:
                return False
            if piv != col:
                for c in range(4):
                    tmp = M[col, c]
                    M[col, c] = M[piv, c]
                    M[piv, c] = tmp
                tmp = g[col]
                g[col] = g[piv]
                g[piv] = tmp
            for r in range(col + 1, 4):
                f = M[r, col] / M[col, col]
                for c in range(col, 4):
                    M[r, c] -= f * M[col, c]
                g[r] -= f * g[col]
        for r in range(3, -1, -1):
            acc = g[r]
            for c in range(r + 1, 4):
                acc -= M[r, c] * g[c]
            g[r] = acc / M[r, r]
        dn = abs(g[0]) + abs(g[1]) + abs(g[2]) + abs(g[3])
        scale = 1.0 if dn <= max_step else max_step / dn
        for k in range(4):
            v[k] -= scale * g[k]
            if not abs(v[k]) < 1e3:  # também captura NaN
                return False
    return False


@cuda.jit
def search_kernel(starts, p, divs, a, b, s, tol, cap, nf, max_it, max_step,
                  prime_tol, ok, keys):
    """Um chute por thread: Newton, filtro de período mínimo e chave
    canônica (ponto lexicograficamente menor da órbita)."""
    i = cuda.grid(1)
    if i >= starts.shape[0]:
        return
    v = cuda.local.array(4, nb.float64)
    g = cuda.local.array(4, nb.float64)
    M = cuda.local.array((4, 4), nb.float64)
    for k in range(4):
        v[k] = starts[i, k]
    ok[i] = False
    if not _newton(v, p, a, b, s, tol, cap, nf, max_it, max_step, M, g):
        return
    for d in range(divs.shape[0]):  # sub-harmônicos F^q(v) = v, q | p
        q = divs[d]
        if q > 0:
            x0, y0, x1, y1 = v[0], v[1], v[2], v[3]
            for _k in range(q):
                x0, y0, x1, y1 = _fmap(x0, y0, x1, y1, a, b, s)
            dist = (abs(x0 - v[0]) + abs(y0 - v[1])
                    + abs(x1 - v[2]) + abs(y1 - v[3]))
            if dist < prime_tol:
                return
    x0, y0, x1, y1 = v[0], v[1], v[2], v[3]
    c0, c1, c2, c3 = x0, y0, x1, y1
    for _k in range(p - 1):
        x0, y0, x1, y1 = _fmap(x0, y0, x1, y1, a, b, s)
        if _lex_less(x0, y0, x1, y1, c0, c1, c2, c3):
            c0, c1, c2, c3 = x0, y0, x1, y1
    keys[i, 0] = c0
    keys[i, 1] = c1
    keys[i, 2] = c2
    keys[i, 3] = c3
    ok[i] = True


# ---------------------------------------------------------------------------
# CPU: lotes, deduplicação, multiplicadores de Floquet
# ---------------------------------------------------------------------------
def divisors(p):
    d = [q for q in range(1, p) if p % q == 0]
    return np.array(d or [0], dtype=np.int64)  # 0 = sentinela (p = 1)


def fmap(v):
    x0, y0, x1, y1 = v.T
    return np.column_stack((A - x0**2 + B * y0 + SIGMA * x1, x0,
                            A - x1**2 + B * y1 + SIGMA * x0, x1))


def orbits_from(keys, p):
    pts = np.empty((len(keys), p, 4))
    v = keys.copy()
    for k in range(p):
        pts[:, k] = v
        v = fmap(v)
    return pts


def run_batch(starts, p, divs):
    n = len(starts)
    ok = cuda.device_array(n, dtype=np.bool_)
    keys = cuda.device_array((n, 4), dtype=np.float64)
    search_kernel[(n + THREADS - 1) // THREADS, THREADS](
        cuda.to_device(starts), p, divs, A, B, SIGMA, NEWTON_TOL, TOL_CAP,
        NOISE_FACTOR, NEWTON_MAX_ITER, MAX_STEP, PRIME_TOL, ok, keys)
    ok = ok.copy_to_host()
    return keys.copy_to_host()[ok]


def dedupe(keys, p, tree):
    """Órbitas novas: distintas entre si e das já conhecidas (`tree`)."""
    if not len(keys):
        return np.empty((0, p, 4))
    code = np.round(keys / DEDUP_TOL).astype(np.int64)
    _, idx = np.unique(code, axis=0, return_index=True)
    orbs = orbits_from(keys[idx], p)
    if tree is not None:
        d, _ = tree.query(orbs.reshape(-1, 4))
        orbs = orbs[d.reshape(-1, p).min(axis=1) >= DEDUP_TOL]
    if len(orbs) > 1:  # duplicatas no lote (ruído na chave canônica)
        pairs = cKDTree(orbs.reshape(-1, 4)).query_pairs(
            DEDUP_TOL, output_type="ndarray")
        dup = np.zeros(len(orbs), dtype=bool)
        if len(pairs):
            for i, j in np.unique(np.sort(pairs // p, axis=1), axis=0):
                if i != j and not dup[i]:
                    dup[j] = True
        orbs = orbs[~dup]
    return orbs


def find_upos(p, pts, rng):
    divs = cuda.to_device(divisors(p))
    found, stale = [], 0
    for batch in range(MAX_BATCHES):
        tree = cKDTree(pts) if len(pts) else None
        starts = rng.uniform(DOMAIN[0], DOMAIN[1], size=(N_TRIES, 4))
        new = dedupe(run_batch(starts, p, divs), p, tree)
        if len(new):
            found.append(new)
            pts = np.concatenate([pts, new.reshape(-1, 4)])
        stale = 0 if len(new) else stale + 1
        print(f"    p={p:2d} lote {batch + 1:2d}: +{len(new)} "
              f"(total {sum(len(f) for f in found)})")
        if stale >= PATIENCE:
            break
    return np.concatenate(found) if found else np.empty((0, p, 4))


def analyse(orbs):
    """Multiplicadores de Floquet: 4D completo e, para órbitas na variedade
    sincronizada, blocos longitudinal (+sigma) e transversal (-sigma)."""
    n, p, _ = orbs.shape
    eye4 = np.broadcast_to(np.eye(4), (n, 4, 4))
    M, J = eye4.copy(), np.zeros((n, 4, 4))
    J[:, 0, 1], J[:, 0, 2], J[:, 1, 0] = B, SIGMA, 1.0
    J[:, 2, 0], J[:, 2, 3], J[:, 3, 2] = SIGMA, B, 1.0
    eye2 = np.broadcast_to(np.eye(2), (n, 2, 2))
    L, T = eye2.copy(), eye2.copy()
    JL = np.zeros((n, 2, 2))
    JL[:, 0, 1], JL[:, 1, 0] = B, 1.0
    JT = JL.copy()
    for k in range(p):
        x0, x1 = orbs[:, k, 0], orbs[:, k, 2]
        J[:, 0, 0], J[:, 2, 2] = -2.0 * x0, -2.0 * x1
        JL[:, 0, 0], JT[:, 0, 0] = -2.0 * x0 + SIGMA, -2.0 * x0 - SIGMA
        M, L, T = J @ M, JL @ L, JT @ T
    sync = np.abs(orbs[..., :2] - orbs[..., 2:]).max(axis=(1, 2)) < SYNC_TOL
    return {"mult": np.linalg.eigvals(M), "long": np.linalg.eigvals(L),
            "trans": np.linalg.eigvals(T), "sync": sync}


def calculate():
    if not cuda.is_available():
        raise RuntimeError("GPU CUDA não encontrada.")
    pts = np.empty((0, 4))
    parts = []
    for p in range(1, P_MAX + 1):
        rng = np.random.default_rng(SEED + p)
        orbs = find_upos(p, pts, rng)
        pts = np.concatenate([pts, orbs.reshape(-1, 4)])
        print(f"  período {p:2d}: {len(orbs)} órbita(s) de período primo")
        if len(orbs):
            parts.append((p, orbs, analyse(orbs)))
    if not parts:
        raise RuntimeError("Nenhuma UPO encontrada.")
    return {
        "period": np.concatenate([np.full(len(o), p) for p, o, _ in parts]),
        "orbits": np.concatenate([o.reshape(-1, 4) for _, o, _ in parts]),
        **{k: np.concatenate([r[k] for _, _, r in parts])
           for k in ("mult", "long", "trans", "sync")},
    }


def save(res):
    np.savez_compressed(
        DATA_PATH, config=np.asarray(json.dumps(search_config(),
                                                sort_keys=True)), **res)
    print(f"Dados salvos em: {DATA_PATH}")


def load():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} ausente: use RECALCULATE=True")
    with np.load(DATA_PATH, allow_pickle=False) as data:
        if json.loads(str(data["config"].item())) != search_config():
            raise ValueError("Cache incompatível: use RECALCULATE=True")
        return {k: data[k].copy() for k in data.files if k != "config"}


# ---------------------------------------------------------------------------
# Tabelas e figura
# ---------------------------------------------------------------------------
def build_table(res):
    per = res["period"]
    first = np.concatenate(([0], np.cumsum(per)))[:-1]
    rep = res["orbits"][first]  # primeiro ponto = chave canônica
    mod = np.sort(np.abs(res["mult"]), axis=1)[:, ::-1]
    sync = res["sync"]
    n_long = (np.abs(res["long"]) > UNSTABLE_TOL).sum(axis=1)
    n_trans = (np.abs(res["trans"]) > UNSTABLE_TOL).sum(axis=1)
    return pd.DataFrame({
        "period": per, "x0": rep[:, 0], "y0": rep[:, 1],
        "x1": rep[:, 2], "y1": rep[:, 3], "sync": sync,
        "n_unstable": (mod > UNSTABLE_TOL).sum(axis=1),
        "n_unstable_long": np.where(sync, n_long, -1),
        "n_unstable_trans": np.where(sync, n_trans, -1),
        **{f"mu{k + 1}_abs": mod[:, k] for k in range(4)},
    })


def build_summary(df):
    cnt = pd.crosstab(df["period"], df["n_unstable"]).reindex(
        index=range(1, P_MAX + 1), columns=range(5), fill_value=0)
    cnt.columns = [f"d{k}" for k in cnt.columns]
    cnt.insert(0, "n_upos", cnt.sum(axis=1))
    cnt["n_sync"] = df.groupby("period")["sync"].sum().reindex(
        cnt.index, fill_value=0)
    return cnt.rename_axis("period").reset_index()


def save_latex(summary):
    names = {"period": r"$p$", "n_upos": "No. UPOs", "n_sync": "On $\\mathcal{S}$"}
    names.update({f"d{k}": rf"$d_u={k}$" for k in range(5)})
    caption = (
        rf"Prime-period orbits found for $a={A}$, $b={B}$ and "
        rf"$\sigma={SIGMA}$ in the 4D coupled H\'enon map, using a "
        r"multi-start Newton-Raphson method for $F^p(\mathbf{X})=\mathbf{X}$ "
        rf"with up to {N_TRIES:,} initial guesses per batch in "
        rf"$[{DOMAIN[0]},{DOMAIN[1]}]^4$ (at most {MAX_BATCHES} batches per "
        rf"period; stop after {PATIENCE} batches without new orbits). "
        r"$d_u$ is the number of Floquet multipliers with modulus greater "
        r"than one. The search is not exhaustive: counts are lower bounds."
    )
    tex = summary.rename(columns=names).to_latex(
        index=False, caption=caption, label="tab:upos-4d", escape=False,
        column_format="r" * len(summary.columns))
    TEX_PATH.write_text(tex, encoding="utf-8")
    print(f"Tabela LaTeX salva em: {TEX_PATH}")


def plot_summary(res, df, summary):
    from pynamicalsys import PlotStyler
    ps = PlotStyler()
    ps.apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for k in range(5):
        ax1.semilogy(summary["period"], summary[f"d{k}"].replace(0, np.nan),
                     "o-", ms=3, label=rf"$d_u={k}$")
    ax1.set_xlabel(r"$p$")
    ax1.set_ylabel("No. UPOs")
    ax1.legend(fontsize=8)
    col = np.repeat(df["n_unstable"].values, df["period"].values)
    pts = res["orbits"]
    idx = np.random.default_rng(0).choice(
        len(pts), size=min(len(pts), 200_000), replace=False)
    sc = ax2.scatter(pts[idx, 0], pts[idx, 2], c=col[idx], s=1, marker=".", edgecolor='none',
                     cmap="vanimo", vmin=0, vmax=4, rasterized=True)
    fig.colorbar(sc, ax=ax2, label=r"$d_u$")
    ax2.set_xlabel(r"$x^{(0)}$")
    ax2.set_ylabel(r"$x^{(1)}$")
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=300, bbox_inches="tight")
    print(f"Figura salva em: {FIG_PATH}")


def main():
    print(f"UPOs 4D: a={A}, b={B}, sigma={SIGMA}, P_MAX={P_MAX}, "
          f"N_TRIES={N_TRIES:,}, domínio={DOMAIN}^4")
    res = calculate() if RECALCULATE else load()
    if RECALCULATE:
        save(res)
    df = build_table(res)
    df.to_csv(CSV_PATH, index=False)
    summary = build_summary(df)
    summary.to_csv(SUMMARY_CSV, index=False)
    print(summary.to_string(index=False))
    print(f"\nTotal de UPOs: {len(df)} | CSV: {CSV_PATH}")
    save_latex(summary)
    plot_summary(res, df, summary)


if __name__ == "__main__":
    main()
