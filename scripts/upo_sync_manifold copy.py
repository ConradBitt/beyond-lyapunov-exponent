"""UPOs na variedade de sincronização S (mapa 2D) e sua estabilidade.

Mapa em S:  x' = a - x^2 + b*y + sigma*x,  y' = x.

Para cada UPO de período mínimo p, o produto M das Jacobianas ao longo
da órbita fornece os multiplicadores de Floquet mu (autovalores de M)
nos blocos tangencial (J: -2x + sigma) e transversal (J: -2x - sigma).
O expoente por iteração é  lambda = ln|mu_max| / p, e
sign(lambda) = sign(ln|mu_max|). Como det J = -b, vale
|mu_1 mu_2| = b^p, logo lambda_1 + lambda_2 = ln b.
"""
import json
from pathlib import Path

import numba as nb
import numpy as np
import pandas as pd

np.seterr(over="ignore", invalid="ignore")

RECALCULATE = True        # False: lê o cache; True: refaz a busca

A, B, SIGMA = 1.5250, 0.2250, 0.0219
P_MAX = 100
N_TRIES = 1_000_000       # chutes iniciais por lote
MAX_BATCHES = 10          # máximo de lotes por período
PATIENCE = 5              # lotes seguidos sem órbita nova para parar
DOMAIN = (-1.5, 1.5)
NEWTON_TOL = 1e-10
NEWTON_MAX_ITER = 1000
DEDUP_TOL = 1e-6
PRIME_TOL = 1e-7
MARGINAL_TOL = 1e-9       # |lambda| abaixo disto: marginal
SEED = 72348
SCHEMA = 3

PREFIX = f"upo_sync_a{A}_b{B}_sigma{SIGMA}"
DATA_PATH = Path(f"{PREFIX}_data.npz")
CSV_PATH = Path(f"{PREFIX}_upos.csv")
TEX_PATH = Path(f"{PREFIX}_table.tex")


def search_config():
    """Parâmetros que definem os dados salvos (validação do cache)."""
    return {
        "SCHEMA": SCHEMA, "A": A, "B": B, "SIGMA": SIGMA, "P_MAX": P_MAX,
        "N_TRIES": N_TRIES, "MAX_BATCHES": MAX_BATCHES,
        "PATIENCE": PATIENCE, "DOMAIN": list(DOMAIN),
        "NEWTON_TOL": NEWTON_TOL, "NEWTON_MAX_ITER": NEWTON_MAX_ITER,
        "DEDUP_TOL": DEDUP_TOL, "PRIME_TOL": PRIME_TOL, "SEED": SEED,
    }


# --------------------------------------------------------------------
# Núcleo Numba: Newton multi-start + filtro de subharmônicos
# --------------------------------------------------------------------
@nb.njit(cache=True)
def _newton(x, y, p, a, b, s, tol, max_iter):
    """Newton para G(v) = F^p(v) - v = 0 (Jacobiana pela regra da cadeia)."""
    for _ in range(max_iter):
        vx, vy = x, y
        m00, m01, m10, m11 = 1.0, 0.0, 0.0, 1.0
        for _k in range(p):
            j00 = -2.0 * vx + s
            n00 = j00 * m00 + b * m10
            n01 = j00 * m01 + b * m11
            m10, m11 = m00, m01
            m00, m01 = n00, n01
            vx, vy = a - vx * vx + b * vy + s * vx, vx
        gx, gy = vx - x, vy - y
        if not (np.isfinite(gx) and np.isfinite(gy)):
            return False, x, y
        if np.sqrt(gx * gx + gy * gy) < tol:
            return True, x, y
        j00, j01, j10, j11 = m00 - 1.0, m01, m10, m11 - 1.0
        det = j00 * j11 - j01 * j10
        if det == 0.0 or not np.isfinite(det):
            return False, x, y
        x -= (j11 * gx - j01 * gy) / det
        y -= (-j10 * gx + j00 * gy) / det
        if not (abs(x) < 1e3 and abs(y) < 1e3):
            return False, x, y
    return False, x, y


@nb.njit(cache=True)
def _is_primitive(x, y, divs, a, b, s, prime_tol):
    """False se algum divisor q < p satisfaz F^q(v) = v (subharmônico)."""
    for idx in range(divs.shape[0]):
        vx, vy = x, y
        for _k in range(divs[idx]):
            vx, vy = a - vx * vx + b * vy + s * vx, vx
        if np.sqrt((vx - x) ** 2 + (vy - y) ** 2) < prime_tol:
            return False
    return True


@nb.njit(cache=True)
def _canonical(x, y, p, a, b, s):
    """Ponto da órbita com menor x (desempate por y): chave canônica."""
    cx, cy, vx, vy = x, y, x, y
    for _k in range(p - 1):
        vx, vy = a - vx * vx + b * vy + s * vx, vx
        if vx < cx or (vx == cx and vy < cy):
            cx, cy = vx, vy
    return cx, cy


@nb.njit(parallel=True, cache=True)
def _newton_batch(starts, p, divs, a, b, s, tol, max_iter, prime_tol):
    n = starts.shape[0]
    ok = np.zeros(n, dtype=np.bool_)
    keys = np.full((n, 2), np.nan)
    for i in nb.prange(n):
        good, x, y = _newton(starts[i, 0], starts[i, 1], p, a, b, s,
                             tol, max_iter)
        if good and _is_primitive(x, y, divs, a, b, s, prime_tol):
            keys[i, 0], keys[i, 1] = _canonical(x, y, p, a, b, s)
            ok[i] = True
    return ok, keys


@nb.njit(cache=True)
def _min_dist(orbit, pts):
    best = np.inf
    for i in range(orbit.shape[0]):
        for j in range(pts.shape[0]):
            d = np.sqrt((orbit[i, 0] - pts[j, 0]) ** 2
                        + (orbit[i, 1] - pts[j, 1]) ** 2)
            best = min(best, d)
    return best


def _unique_keys(keys, tol):
    keys = keys[np.lexsort((keys[:, 1], keys[:, 0]))]
    reps = []
    for k in keys:
        dup = False
        for r in reversed(reps):
            if k[0] - r[0] > tol:
                break
            if abs(k[1] - r[1]) <= tol:
                dup = True
                break
        if not dup:
            reps.append(k)
    return np.asarray(reps)


def iterate_orbit(v0, p):
    pts = np.empty((p, 2))
    x, y = v0
    for k in range(p):
        pts[k] = x, y
        x, y = A - x * x + B * y + SIGMA * x, x
    return pts


def find_upos(p, known, rng):
    """Órbitas de período mínimo p, em lotes de N_TRIES chutes."""
    divs = np.array([q for q in range(1, p) if p % q == 0],
                    dtype=np.int64)
    found, found_pts, stale = [], np.empty((0, 2)), 0
    for batch in range(MAX_BATCHES):
        starts = rng.uniform(DOMAIN[0], DOMAIN[1], size=(N_TRIES, 2))
        ok, keys = _newton_batch(starts, p, divs, A, B, SIGMA, NEWTON_TOL,
                                 NEWTON_MAX_ITER, PRIME_TOL)
        n_new = 0
        if ok.any():
            for key in _unique_keys(keys[ok], DEDUP_TOL):
                orbit = iterate_orbit(key, p)
                if len(known) and _min_dist(orbit, known) < DEDUP_TOL:
                    continue
                if len(found_pts) and _min_dist(orbit, found_pts) < DEDUP_TOL:
                    continue
                found.append(orbit)
                found_pts = np.concatenate([found_pts, orbit])
                n_new += 1
        stale = 0 if n_new else stale + 1
        print(f"    p={p:3d} lote {batch + 1:2d}: +{n_new} "
              f"(total {len(found)})")
        if stale >= PATIENCE:
            break
    return found


# --------------------------------------------------------------------
# Estabilidade: multiplicadores de Floquet e expoentes por iteração
# --------------------------------------------------------------------
def stability(orbit, shift):
    """(lambda_max, lambda_min) do bloco com J = [[-2x + shift, b], [1, 0]].

    O produto é reescalado a cada passo (log_scale) para evitar overflow
    em períodos grandes; lambda_min usa det M = (-b)^p.
    """
    p = len(orbit)
    m, log_scale = np.eye(2), 0.0
    for x in orbit[:, 0]:
        m = np.array([[-2.0 * x + shift, B], [1.0, 0.0]]) @ m
        scale = np.abs(m).max()
        m /= scale
        log_scale += np.log(scale)
    log_mu_max = np.log(np.abs(np.linalg.eigvals(m)).max()) + log_scale
    lam_max = log_mu_max / p
    return lam_max, np.log(B) - lam_max


def calculate():
    known, rows = np.empty((0, 2)), []
    for p in range(1, P_MAX + 1):
        orbits = find_upos(p, known, np.random.default_rng(SEED + p))
        for orbit in orbits:
            par = stability(orbit, +SIGMA)
            perp = stability(orbit, -SIGMA)
            rows.append((p, *orbit[0], par[0], par[1], perp[0], perp[1]))
        if orbits:
            known = np.concatenate([known, *orbits])
        print(f"  período {p:3d}: {len(orbits)} órbita(s) primitiva(s)")
    cols = ["period", "x0", "y0", "lam_par", "lam_par_min",
            "lam_perp", "lam_perp_min"]
    return pd.DataFrame(rows, columns=cols)


def save(df):
    np.savez_compressed(
        DATA_PATH, config=np.asarray(json.dumps(search_config(),
                                                sort_keys=True)),
        **{c: df[c].to_numpy() for c in df.columns})
    print(f"Dados salvos em: {DATA_PATH}")


def load():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} ausente: RECALCULATE=True")
    with np.load(DATA_PATH, allow_pickle=False) as data:
        if json.loads(str(data["config"].item())) != search_config():
            raise ValueError("Cache incompatível com os parâmetros atuais.")
        return pd.DataFrame({k: data[k] for k in data.files
                             if k != "config"})


# --------------------------------------------------------------------
# Tabela
# --------------------------------------------------------------------
def build_summary(df):
    tol = MARGINAL_TOL
    flags = pd.DataFrame({
        "period": df["period"],
        "par_s": df["lam_par"] < -tol, "perp_s": df["lam_perp"] < -tol,
        "par_u": df["lam_par"] > tol, "perp_u": df["lam_perp"] > tol,
    })
    out = flags.groupby("period").sum().astype(int)
    out.insert(0, "n_upos", flags.groupby("period").size())
    # "Yes": há órbita atratora em S (lambda_par < 0) neste período.
    out["observed"] = np.where(out["par_s"] > 0, "Yes", "No")
    n_marg = int(((df["lam_par"].abs() <= tol)
                  | (df["lam_perp"].abs() <= tol)).sum())
    if n_marg:
        print(f"Aviso: {n_marg} órbita(s) com |lambda| <= {tol:g} "
              "(marginal) fora das colunas da tabela.")
    return out.reset_index()


def make_caption():
    return (
        rf"Primitive UPOs found on the synchronization manifold for "
        rf"$a={A}$, $b={B}$ and $\sigma={SIGMA}$. Each orbit is classified "
        r"by the tangential ($\lambda_{\parallel}$) and transverse "
        r"($\lambda_{\perp}$) Lyapunov exponents per iteration, "
        r"$\lambda=\ln|\mu_{\max}|/p$, where $\mu_{\max}$ is the Floquet "
        r"multiplier of largest modulus in that direction; an orbit is "
        r"stable in a direction if $\lambda<0$ and unstable if "
        r"$\lambda>0$. The last column indicates whether an orbit "
        r"attracting in the manifold ($\lambda_{\parallel}<0$) exists for "
        rf"that period. Periods $p=1,\ldots,{P_MAX}$ were searched with a "
        r"multi-start Newton-Raphson method for "
        r"$F^p(\mathbf{X}^{[S]})=\mathbf{X}^{[S]}$, using up to "
        rf"{N_TRIES:,} initial guesses per batch in "
        rf"$[{DOMAIN[0]},{DOMAIN[1]}]^2$, at most {MAX_BATCHES} batches per "
        rf"period, and stopping after {PATIENCE} consecutive batches "
        r"without new orbits. Subharmonic solutions and duplicates were "
        r"discarded, and periods with no orbit found are omitted. The "
        r"search is not exhaustive, so the counts are lower bounds."
    )


def save_latex(summary):
    names = {
        "period": r"$p$", "n_upos": "No. UPOs",
        "par_s": r"$\lambda_{\parallel}<0$",
        "perp_s": r"$\lambda_{\perp}<0$",
        "par_u": r"$\lambda_{\parallel}>0$",
        "perp_u": r"$\lambda_{\perp}>0$",
        "observed": "View on the bif. diagram?",
    }
    table = summary[["period", "n_upos", "par_s", "perp_s", "par_u",
                     "perp_u", "observed"]].rename(columns=names)
    tex = table.to_latex(index=False, caption=make_caption(),
                         label="tab:upos-por-periodo", escape=False,
                         column_format="r" * len(table.columns))
    TEX_PATH.write_text(tex, encoding="utf-8")
    print(f"Tabela LaTeX salva em: {TEX_PATH}")


def main():
    print(f"UPOs em S: a={A}, b={B}, sigma={SIGMA}, P_MAX={P_MAX}, "
          f"N_TRIES={N_TRIES:,}, domínio={DOMAIN}^2")
    if RECALCULATE:
        df = calculate()
        save(df)
    else:
        df = load()
    df.to_csv(CSV_PATH, index=False)
    summary = build_summary(df)
    print(summary.to_string(index=False))
    n_pos = int((df["lam_perp"] > MARGINAL_TOL).sum())
    n_neg = int((df["lam_perp"] < -MARGINAL_TOL).sum())
    print(f"\nTotal: {len(df)} UPOs | lambda_perp > 0: {n_pos} | "
          f"lambda_perp < 0: {n_neg}")
    print(f"lambda_perp em [{df['lam_perp'].min():.3f}, "
          f"{df['lam_perp'].max():.3f}]")
    save_latex(summary)


if __name__ == "__main__":
    main()
