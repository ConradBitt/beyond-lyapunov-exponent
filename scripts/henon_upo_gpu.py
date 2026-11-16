"""
Busca de órbitas periódicas (UPOs) no manifold sincronizado de dois mapas de
Hénon acoplados, com a grelha de condições iniciais paralelizada na GPU
(Numba-CUDA).

Resolve-se, para cada período p = 1..p_max,   G(X) = F^p(X) - X = 0
com X = (xs, ys) no manifold sincronizado (domínio 2D).

HIPÓTESE sobre o mapa 4D completo (necessária só para os multiplicadores
transversos):
    x_i' = a - x_i^2 + b*y_i + sigma*x_j ,   y_i' = x_i        (i != j)
Em x0 = x1 = xs, o Jacobiano decompõe-se em dois blocos 2x2:
    tangencial  (modo simétrico)     : [[-2x + sigma, b], [1, 0]]
    transverso  (modo antissimétrico): [[-2x - sigma, b], [1, 0]]
Se o acoplamento real for outro, bastaria alterar `floquet_blocks`
(host) e o sinal de sigma no bloco transverso.

Duas estratégias de busca (mesmo núcleo Newton, mesma deduplicação):
    "newton"      : multi-start aleatório + Newton-Raphson (um thread por palpite)
    "stabilized"  : grelha determinística deslocada + transformações
                    estabilizadoras (8 matrizes C) + polimento por Newton
                    (um thread por par palpite/matriz C)
"""
import argparse
import csv
import json
import math
import os
from dataclasses import dataclass

import numpy as np
from numba import cuda
from numba.cuda.random import (create_xoroshiro128p_states,
                               xoroshiro128p_uniform_float64)

TPB = 256  # threads por bloco

# índices do tuplo de parâmetros `par` (passado por valor aos kernels)
(IA, IB, ISIG, ITOLSTEP, ITOLRES, IMAXSTEP, IESC, ITOLPRIM, ILAM, ITRIG) = range(10)


# --------------------------------------------------------------------------
# Configuração
# --------------------------------------------------------------------------
@dataclass
class Cfg:
    a: float = 1.525
    b: float = 0.225
    sigma: float = 0.0219
    lo: float = -1.5                 # domínio dos palpites: [lo, hi]^2
    hi: float = 1.5
    n_guesses: int = 5_000_000       # palpites (threads) por lote
    max_batches: int = 20
    patience: int = 5                # lotes consecutivos sem novidade
    p_max: int = 15
    seed: int = 12345
    # Newton
    newton_iter: int = 60
    tol_step: float = 1e-12          # critério no passo de Newton
    tol_res: float = 1e-8            # validação do resíduo |F^p(X)-X|
    max_step: float = 0.5            # limita o passo de Newton / do fluxo
    escape: float = 50.0             # |x|,|y| acima disto => diverge
    tol_prim: float = 1e-7           # rejeita subharmónicos
    tol_dup: float = 1e-6            # duplicados (host)
    # Estratégia "stabilized"
    lam: float = 0.05                # passo do fluxo estabilizador
    n_flow: int = 400                # iterações máximas do fluxo
    trigger: float = 1e-2            # |G| abaixo disto => passa ao Newton


def make_par(cfg):
    return (cfg.a, cfg.b, cfg.sigma, cfg.tol_step, cfg.tol_res, cfg.max_step,
            cfg.escape, cfg.tol_prim, cfg.lam, cfg.trigger)


# --------------------------------------------------------------------------
# Dispositivo (GPU)
# --------------------------------------------------------------------------
@cuda.jit(device=True)
def henon_sync_step(xs, ys, a, b, sigma):
    """
    Dinâmica no manifold sincronizado (x0=x1=xs, y0=y1=ys).
        xs_{n+1} = a - xs² + b*ys + sigma*xs
        ys_{n+1} = xs
    """
    xsn = a - xs * xs + b * ys + sigma * xs
    ysn = xs
    return xsn, ysn


@cuda.jit(device=True)
def orbit_end(x0, y0, p, par):
    """F^p(X0). ok=False se a trajetória escapar (ou der NaN)."""
    x = x0
    y = y0
    for _ in range(p):
        x, y = henon_sync_step(x, y, par[IA], par[IB], par[ISIG])
        if not (abs(x) < par[IESC] and abs(y) < par[IESC]):
            return x, y, False
    return x, y, True


@cuda.jit(device=True)
def orbit_with_monodromy(x0, y0, p, par):
    """F^p(X0) e M = DF^p(X0) = J_{p-1} ... J_0 (tangencial, 2x2)."""
    x = x0
    y = y0
    m11 = 1.0
    m12 = 0.0
    m21 = 0.0
    m22 = 1.0
    for _ in range(p):
        j11 = -2.0 * x + par[ISIG]                # J = [[j11, b], [1, 0]]
        n11 = j11 * m11 + par[IB] * m21
        n12 = j11 * m12 + par[IB] * m22
        n21 = m11
        n22 = m12
        m11 = n11
        m12 = n12
        m21 = n21
        m22 = n22
        x, y = henon_sync_step(x, y, par[IA], par[IB], par[ISIG])
        if not (abs(x) < par[IESC] and abs(y) < par[IESC]):
            return x, y, m11, m12, m21, m22, False
    return x, y, m11, m12, m21, m22, True


@cuda.jit(device=True)
def newton_refine(x, y, p, par, max_iter):
    """Newton 2x2 em G(X) = F^p(X) - X, com passo limitado."""
    for _ in range(max_iter):
        fx, fy, m11, m12, m21, m22, ok = orbit_with_monodromy(x, y, p, par)
        if not ok:
            return x, y, False
        gx = fx - x
        gy = fy - y
        d11 = m11 - 1.0
        d22 = m22 - 1.0
        det = d11 * d22 - m12 * m21
        if abs(det) < 1e-14:
            return x, y, False
        dx = -(d22 * gx - m12 * gy) / det
        dy = -(-m21 * gx + d11 * gy) / det
        sn = math.sqrt(dx * dx + dy * dy)
        if sn > par[IMAXSTEP]:
            s = par[IMAXSTEP] / sn
            dx *= s
            dy *= s
            sn = par[IMAXSTEP]
        x += dx
        y += dy
        if sn < par[ITOLSTEP]:
            fx, fy, ok = orbit_end(x, y, p, par)
            if not ok:
                return x, y, False
            r = math.sqrt((fx - x) ** 2 + (fy - y) ** 2)
            return x, y, r < par[ITOLRES]
    return x, y, False


@cuda.jit(device=True)
def is_primitive(x0, y0, p, par):
    """False se F^k(X0) ~ X0 para algum divisor k < p (subharmónico)."""
    x = x0
    y = y0
    tol2 = par[ITOLPRIM] * par[ITOLPRIM]
    for k in range(1, p):
        x, y = henon_sync_step(x, y, par[IA], par[IB], par[ISIG])
        if p % k == 0:
            if (x - x0) ** 2 + (y - y0) ** 2 < tol2:
                return False
    return True


@cuda.jit
def kernel_multistart_newton(rng, n, p, par, lo, hi, max_iter, out_pts, out_count):
    """Estratégia 'newton': um thread = um palpite aleatório uniforme."""
    i = cuda.grid(1)
    if i >= n:
        return
    x = lo + (hi - lo) * xoroshiro128p_uniform_float64(rng, i)
    y = lo + (hi - lo) * xoroshiro128p_uniform_float64(rng, i)
    x, y, ok = newton_refine(x, y, p, par, max_iter)
    if ok and is_primitive(x, y, p, par):
        k = cuda.atomic.add(out_count, 0, 1)
        if k < out_pts.shape[0]:
            out_pts[k, 0] = x
            out_pts[k, 1] = y


@cuda.jit
def kernel_grid_stabilized(n_total, ng, off_x, off_y, cmats, p, par, lo, hi,
                           n_flow, max_iter, out_pts, out_count):
    """
    Estratégia 'stabilized': thread = (célula da grelha, matriz C).
    Fluxo  X <- X + lam * C * G(X)  (C em {matrizes de permutação com sinais})
    seguido de polimento por Newton quando |G| < trigger.
    """
    tid = cuda.grid(1)
    if tid >= n_total:
        return
    c = tid % 8
    cell = tid // 8
    h = (hi - lo) / ng
    x = lo + ((cell % ng) + off_x) * h
    y = lo + ((cell // ng) + off_y) * h
    c11 = cmats[c, 0]
    c12 = cmats[c, 1]
    c21 = cmats[c, 2]
    c22 = cmats[c, 3]

    near = False
    for _ in range(n_flow):
        fx, fy, ok = orbit_end(x, y, p, par)
        if not ok:
            return
        gx = fx - x
        gy = fy - y
        if math.sqrt(gx * gx + gy * gy) < par[ITRIG]:
            near = True
            break
        sx = par[ILAM] * (c11 * gx + c12 * gy)
        sy = par[ILAM] * (c21 * gx + c22 * gy)
        sn = math.sqrt(sx * sx + sy * sy)
        if sn > par[IMAXSTEP]:
            s = par[IMAXSTEP] / sn
            sx *= s
            sy *= s
        x += sx
        y += sy
    if not near:
        return

    x, y, ok = newton_refine(x, y, p, par, max_iter)
    if ok and is_primitive(x, y, p, par):
        k = cuda.atomic.add(out_count, 0, 1)
        if k < out_pts.shape[0]:
            out_pts[k, 0] = x
            out_pts[k, 1] = y


# --------------------------------------------------------------------------
# Host: lotes, deduplicação, multiplicadores de Floquet
# --------------------------------------------------------------------------
def signed_permutations_2d():
    mats = []
    for perm in (np.array([[1, 0], [0, 1]]), np.array([[0, 1], [1, 0]])):
        for s0 in (1, -1):
            for s1 in (1, -1):
                mats.append((perm * np.array([[s0], [s1]])).ravel())
    return np.array(mats, dtype=np.float64)  # (8, 4)


class Buffers:
    def __init__(self, cap):
        self.cap = cap
        self.d_out = cuda.device_array((cap, 2), dtype=np.float64)
        self.d_count = cuda.to_device(np.zeros(1, dtype=np.int32))

    def reset(self):
        self.d_count.copy_to_device(np.zeros(1, dtype=np.int32))

    def fetch(self):
        k = min(int(self.d_count.copy_to_host()[0]), self.cap)
        if k == 0:
            return np.empty((0, 2))
        return self.d_out[:k].copy_to_host()


def batch_newton(p, cfg, buf, rng):
    buf.reset()
    blocks = (cfg.n_guesses + TPB - 1) // TPB
    kernel_multistart_newton[blocks, TPB](
        rng, cfg.n_guesses, p, make_par(cfg), cfg.lo, cfg.hi,
        cfg.newton_iter, buf.d_out, buf.d_count)
    cuda.synchronize()
    return buf.fetch()


def batch_stabilized(p, batch, cfg, buf, d_cmats):
    buf.reset()
    ng = max(1, int(math.sqrt(cfg.n_guesses / 8)))      # mesmo orçamento de threads
    n_total = ng * ng * 8
    # deslocamento da grelha por lote (sequência R2, quasi-aleatória determinística)
    off_x = (0.5 + batch * 0.7548776662466927) % 1.0
    off_y = (0.5 + batch * 0.5698402909980532) % 1.0
    blocks = (n_total + TPB - 1) // TPB
    kernel_grid_stabilized[blocks, TPB](
        n_total, ng, off_x, off_y, d_cmats, p, make_par(cfg), cfg.lo, cfg.hi,
        cfg.n_flow, cfg.newton_iter, buf.d_out, buf.d_count)
    cuda.synchronize()
    return buf.fetch()


def floquet_blocks(pt, p, a, b, sigma):
    """Pontos da órbita e |multiplicadores| tangenciais / transversos."""
    x, y = pt
    Mpar = np.eye(2)
    Mperp = np.eye(2)
    orb = []
    for _ in range(p):
        orb.append((x, y))
        Mpar = np.array([[-2 * x + sigma, b], [1.0, 0.0]]) @ Mpar
        Mperp = np.array([[-2 * x - sigma, b], [1.0, 0.0]]) @ Mperp
        x, y = a - x * x + b * y + sigma * x, x
    return (np.array(orb), np.abs(np.linalg.eigvals(Mpar)),
            np.abs(np.linalg.eigvals(Mperp)))


def merge_new_orbits(raw, p, known, cfg):
    """
    Descarta duplicados: um ponto é conhecido se estiver (tol_dup) sobre
    qualquer ponto de uma órbita já registada (cobre permutações cíclicas).
    """
    if len(raw) == 0:
        return [], known
    rounded = np.round(raw, 7)                      # só para agrupar
    _, idx = np.unique(rounded, axis=0, return_index=True)
    new = []
    for pt in raw[idx]:                              # ponto original, não arredondado
        if len(known):
            d = np.hypot(known[:, 0] - pt[0], known[:, 1] - pt[1])
            if d.min() < cfg.tol_dup:
                continue
        orb, _, _ = floquet_blocks(pt, p, cfg.a, cfg.b, cfg.sigma)
        new.append(pt)
        known = np.vstack([known, orb])
    return new, known


def search_period(p, strategy, cfg, ctx, verbose=True):
    known = np.empty((0, 2))
    reps = []
    stale = 0
    for batch in range(cfg.max_batches):
        if strategy == "newton":
            raw = batch_newton(p, cfg, ctx["buf"], ctx["rng"])
        else:
            raw = batch_stabilized(p, batch, cfg, ctx["buf"], ctx["cmats"])
        new, known = merge_new_orbits(raw, p, known, cfg)
        reps.extend(new)
        stale = 0 if new else stale + 1
        if verbose:
            print(f"  p={p:2d} lote {batch + 1:2d}: convergiram {len(raw):8d} | "
                  f"novas {len(new):3d} | total {len(reps):3d}")
        if stale >= cfg.patience:
            break
    return reps


def table_rows(results, cfg):
    header = [
        "p", "UPOs", "|mu_par|<1", "|mu_perp|<1",
        "|mu_par|>1", "|mu_perp|>1", "no diagrama?"
    ]
    rows = [header]
    for p, reps in results.items():
        sp = su = tp = tu = 0
        visible = False
        for pt in reps:
            _, mpar, mperp = floquet_blocks(pt, p, cfg.a, cfg.b, cfg.sigma)
            s_par, s_perp = mpar.max() < 1, mperp.max() < 1
            sp += int(s_par)
            tp += int(s_perp)
            su += int(mpar.max() > 1)
            tu += int(mperp.max() > 1)
            visible |= (s_par and s_perp)       # estável nas duas direções
        rows.append([p, len(reps), sp, tp, su, tu, "Sim" if visible else "Não"])
    return rows


def table_lines(results, cfg):
    rows = table_rows(results, cfg)
    lines = [
        "\n p | UPOs | |mu_par|<1 | |mu_perp|<1 | |mu_par|>1 | |mu_perp|>1 | no diagrama?",
        "---+------+------------+-------------+------------+-------------+-------------",
    ]
    for p, reps, sp, tp, su, tu, visible in [tuple(r) for r in rows[1:]]:
        lines.append(
            f"{p:2d} | {reps:4d} | {sp:10d} | {tp:11d} | {su:10d} | "
            f"{tu:11d} | {visible}"
        )
    return "\n".join(lines)


def write_table_csv(results, cfg, out_path):
    rows = table_rows(results, cfg)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerows(rows)


def state_path_for(out_path):
    if not out_path:
        return None
    if out_path.lower().endswith(".json"):
        return out_path
    return out_path + ".state.json"


def save_results_state(results, out_path):
    state_path = state_path_for(out_path)
    if not state_path:
        return
    payload = {
        str(p): [list(map(float, pt)) for pt in reps]
        for p, reps in sorted(results.items())
    }
    with open(state_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def load_results_state(out_path):
    state_path = state_path_for(out_path)
    if not state_path or not os.path.exists(state_path):
        return {}
    with open(state_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return {
        int(p): np.asarray(points, dtype=np.float64)
        for p, points in payload.items()
    }


def save_table(results, cfg, out_path):
    if not out_path or out_path.lower().endswith(".json"):
        return
    text = table_lines(results, cfg)
    if out_path.lower().endswith(".csv"):
        write_table_csv(results, cfg, out_path)
    else:
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")


def print_table(results, cfg, out_path=None):
    text = table_lines(results, cfg)
    print(f"\n{text}")
    save_table(results, cfg, out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=["newton", "stabilized"], default="newton")
    ap.add_argument("--pmax", type=int, default=25)
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--batches", type=int, default=20)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--output", type=str, default=None,
                    help=("Caminho do arquivo de resumo. Quando informado, o script "
                          "salva também um estado JSON persistente para reinício "
                          "em caso de queda do sistema; use .json para gravar só o "
                          "estado ou .txt/.csv para resumo legível."))
    args = ap.parse_args()

    cfg = Cfg(p_max=args.pmax, n_guesses=args.n,
              max_batches=args.batches, patience=args.patience)
    if not cuda.is_available():
        raise SystemExit("CUDA indisponível.")

    ctx = {
        "buf": Buffers(cfg.n_guesses),
        "rng": create_xoroshiro128p_states(cfg.n_guesses, seed=cfg.seed),
        "cmats": cuda.to_device(signed_permutations_2d()),
    }
    results = load_results_state(args.output) if args.output else {}
    for p in range(1, cfg.p_max + 1):
        if p in results:
            continue
        results[p] = search_period(p, args.strategy, cfg, ctx)
        save_results_state(results, args.output)
        if args.output and not args.output.lower().endswith(".json"):
            save_table(results, cfg, args.output)
    print_table(results, cfg, args.output)


if __name__ == "__main__":
    main()
