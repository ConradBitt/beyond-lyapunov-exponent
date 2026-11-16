import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange

from pynamicalsys import PlotStyler

ps = PlotStyler(
    fontsize=18
    )
ps.apply_style()

A, B, SIGMA, XLIM = 1.5250, 0.2250, 0.0219, 4


@njit(parallel=True)
def t_esc(xs, ys, delta, tmax):
    out = np.empty((ys.size, xs.size), np.int32)
    for i in prange(ys.size):
        for j in range(xs.size):
            x0, y0 = xs[j], ys[i]
            x1, y1 = x0 + delta, y0 + delta
            t = tmax
            for n in range(tmax):
                xn0 = A - x0 * x0 + B * y0 + SIGMA * x1
                xn1 = A - x1 * x1 + B * y1 + SIGMA * x0
                x0, y0, x1, y1 = xn0, x0, xn1, x1
                if abs(x0) > XLIM or abs(x1) > XLIM:
                    t = n + 1
                    break
            out[i, j] = t
    return out


TMAX, N = 10**8, 200

xs = ys = np.linspace(-2, 2, N)
ts = np.unique(np.logspace(1, 8, 200).astype(int))
deltas = (1e-14, 1e-2, 1)
curves = {}
for d in deltas:
    T = t_esc(xs, ys, d, TMAX).ravel()
    curves[d] = np.array([(T >= t).mean() for t in ts])

CM = 1/2.54
fig, ax = plt.subplots(figsize=(26*CM, 5*CM), constrained_layout=True)
colors = ['blue', 'red', 'black']
ls = ['dotted', 'dashed', 'solid']
for d, bs in curves.items():
    ax.semilogx(ts, bs, label=rf"$\delta=10^{{{int(np.log10(d))}}}$",color=colors[list(deltas).index(d)], ls=ls[list(deltas).index(d)])

ax.axvline(1e6, ls=":", c="gray")
ax.set_xlabel(r"$T_{max}$")
ax.set_xlim(1e1, TMAX)
ax.set_ylabel(r"$B_s$")
ax.set_ylim(-0.05, 1.05)
ax.legend(ncols=6, fontsize=14, loc="upper center", framealpha=0.8)
_bbox = {"facecolor": "w", "alpha": 0.8, "linewidth": 0.0, "pad": 1}
from string import ascii_lowercase
ax.text(
    0.01, 0.85,
    rf"({ascii_lowercase[3]})",
    fontsize=18,
    bbox=_bbox, transform=ax.transAxes
)
plt.savefig("basin_stability_tmax.png", dpi=300, bbox_inches="tight")