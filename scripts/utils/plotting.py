"""
Colormaps e helpers de plotagem reutilizáveis.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns


# -----------------------------------------------------------------------
# Colormap para expoente de Lyapunov
# (preto → ciano [negativo], amarelo → branco [positivo])
# -----------------------------------------------------------------------

def make_lyapunov_cmap(vmin: float = -0.4, vmax: float = 0.6,
                       delta: float = 1e-3) -> LinearSegmentedColormap:
    zero_pos = abs(vmin) / (vmax - vmin)
    cores = {
        "red": [
            (0.0,                    1.0, 1.0),
            (zero_pos - delta,       0.0, 0.0),
            (zero_pos,               0.0, 0.0),
            (zero_pos + delta,       0.0, 1.0),
            (1.0,                    1.0, 1.0),
        ],
        "green": [
            (0.0,                    0.0, 0.0),
            (zero_pos - delta,       0.0, 1.0),
            (zero_pos,               1.0, 1.0),
            (zero_pos + delta,       1.0, 0.70),
            (1.0,                    1.0, 1.0),
        ],
        "blue": [
            (0.0,                    0.0, 0.0),
            (zero_pos - delta,       0.0, 1.0),
            (zero_pos,               1.0, 1.0),
            (zero_pos + delta,       0.0, 0.0),
            (1.0,                    1.0, 1.0),
        ],
    }
    cmap = LinearSegmentedColormap("lyapunov", segmentdata=cores, N=256)
    cmap.set_over("white")
    cmap.set_under("royalblue")
    return cmap


CMAP_LYAPUNOV = make_lyapunov_cmap()


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def plot_basin(basin: np.ndarray,
               x_range: tuple, y_range: tuple,
               ax: plt.Axes | None = None,
               title: str = "",
               xlabel: str = r"$x_0$",
               ylabel: str = r"$y_0$",
               **imshow_kw) -> plt.Axes:
    """Plota a bacia de atração com imshow."""
    if ax is None:
        _, ax = plt.subplots()
    defaults = dict(origin="lower", cmap="binary",
                    extent=(*x_range, *y_range), aspect="equal")
    defaults.update(imshow_kw)
    ax.imshow(basin.T, **defaults)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax


def plot_parameter_space(grid: np.ndarray,
                         a_vals: np.ndarray, b_vals: np.ndarray,
                         ax: plt.Axes | None = None,
                         vmin: float = -0.4, vmax: float = 0.6,
                         title: str = "Espaço de parâmetros (MLE)",
                         xlabel: str = r"$a$",
                         ylabel: str = r"$b$") -> plt.Axes:
    """Plota o grid de MLE no espaço (a, b)."""
    if ax is None:
        _, ax = plt.subplots()
    cmap = make_lyapunov_cmap(vmin=vmin, vmax=vmax)
    extent = (a_vals[0], a_vals[-1], b_vals[0], b_vals[-1])
    im = ax.imshow(grid, origin="lower", extent=extent,
                   cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, label="MLE")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax


from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm
import matplotlib as mpl
import seaborn as sns
import numpy as np
from dataclasses import dataclass

@dataclass
class PeriodCmap:
    cmap: ListedColormap
    norm: BoundaryNorm
    ticks: np.ndarray
    tick_labels: list
    p_min: int
    p_max: int


def make_period_cmap(N: int = 10, p_min: int = 1) -> PeriodCmap:
    p_max = p_min + N - 1

    colors = sns.color_palette("Set1", N)
    colors.append((0, 0, 0))  # preto para >= p_max
    cmap = mpl.colors.ListedColormap(colors)
    cmap.set_bad(color="white")   # <-- NaN → branco
    cmap.set_under(color="white") # <-- abaixo do p_min → branco

    bounds = np.arange(p_min - 0.5, p_max + 1.5, 1.0)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N, clip=False) # <-- clip=False

    ticks = np.arange(p_min, p_max + 1, dtype=int)
    tick_labels = [str(t) for t in ticks]
    tick_labels[-1] = f"${p_max}".replace('.0', '') + r"\leq$"

    return PeriodCmap(cmap=cmap, norm=norm, ticks=ticks,
                      tick_labels=tick_labels, p_min=p_min, p_max=p_max)


# def make_period_cmap(N: int = 10) -> LinearSegmentedColormap:
#     # Criando o Colormap (Set1 com 10 cores + Preto)
#     cmaplist = sns.color_palette("Set1", N)
#     cmaplist.append((0, 0, 0)) # Preto para >= 14
#     cmap = mpl.colors.ListedColormap(cmaplist)
#     return cmap