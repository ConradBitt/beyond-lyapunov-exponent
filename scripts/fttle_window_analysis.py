import math
import pickle
import numpy as np
import matplotlib.pyplot as plt
from numba import njit

RECALCULATE = True   # True: roda a simulação e salva os dados; False: só lê o cache e plota

A = 1.5250
B = 0.2250
SIGMA = 0.0219
DELTA = 1e-12

N_ICS = 1000
X_MIN, X_MAX = -1.5, 1.5
Y_MIN, Y_MAX = -1.5, 1.5
SEED = 498234

N_ITER_PER_IC = 10_000
TRANSIENT_PER_IC = 5_000
ESCAPE_THRESHOLD = 2

WINDOW_OVERLAP = 0.25
WINDOW_MIN = 2
WINDOW_MAX = 101
HISTOGRAM_WINDOWS = [7, 10, 13]
N_VARIANCE_WINDOWS = 100
VARIANCE_WINDOWS = np.unique(np.geomspace(WINDOW_MIN, WINDOW_MAX, N_VARIANCE_WINDOWS).astype(int))
WINDOWS = np.unique(np.concatenate((HISTOGRAM_WINDOWS, VARIANCE_WINDOWS)))
HISTOGRAM_BINS = 25
MIN_N_POS = 100
MIN_N_POSITIVE_ICS = 30
MIN_N_VALID_ICS = 100
N_BOOT = 5000
FPLUS_THRESHOLD = 0.5
FPLUS_ALPHA = 0.05
MIN_CONSECUTIVE_WINDOWS = 3

OUTPUT_PREFIX = f"fttle_window_multiIC_a{A}_b{B}_sigma{SIGMA}_N{N_ICS}"
DATA_PATH = f"{OUTPUT_PREFIX}_data.pkl"
CACHE_SCHEMA_VERSION = 2


def cache_config(a=None):
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "A": A if a is None else float(a),
        "B": B,
        "SIGMA": SIGMA,
        "DELTA": DELTA,
        "N_ICS": N_ICS,
        "X_RANGE": [X_MIN, X_MAX],
        "Y_RANGE": [Y_MIN, Y_MAX],
        "SEED": SEED,
        "N_ITER_PER_IC": N_ITER_PER_IC,
        "TRANSIENT_PER_IC": TRANSIENT_PER_IC,
        "ESCAPE_THRESHOLD": ESCAPE_THRESHOLD,
        "WINDOWS": WINDOWS.tolist(),
        "HISTOGRAM_WINDOWS": list(HISTOGRAM_WINDOWS),
        "VARIANCE_WINDOWS": VARIANCE_WINDOWS.tolist(),
        "HISTOGRAM_BINS": HISTOGRAM_BINS,
        "MIN_N_POS": MIN_N_POS,
        "MIN_N_POSITIVE_ICS": MIN_N_POSITIVE_ICS,
        "MIN_N_VALID_ICS": MIN_N_VALID_ICS,
        "N_BOOT": N_BOOT,
        "FPLUS_THRESHOLD": FPLUS_THRESHOLD,
        "FPLUS_ALPHA": FPLUS_ALPHA,
    }


def validate_cache_config(data, a=None):
    saved_config = data.get("config")
    if saved_config != cache_config(a):
        raise ValueError(
            f"O cache em {DATA_PATH} não corresponde à configuração atual "
            "(ou é legado e não possui metadados). Defina RECALCULATE=True "
            "para gerar um cache compatível."
        )


@njit(cache=True, fastmath=True)
def henon_step(x0, y0, x1, y1, a, b, sigma):
    xn0 = a - x0 * x0 + b * y0 + sigma * x1
    yn0 = x0
    xn1 = a - x1 * x1 + b * y1 + sigma * x0
    yn1 = x1
    return xn0, yn0, xn1, yn1


@njit(cache=True, fastmath=True)
def henon_sync_step(xs, ys, a, b, sigma):
    xsn = a - xs * xs + b * ys + sigma * xs
    ysn = xs
    return xsn, ysn


@njit(cache=True, fastmath=True)
def simulate_system_safe(a, b, sigma, x0, y0, x1, y1, n_iter, transient, escape_threshold):
    xs, ys = x0, y0
    ut0, ut1 = 1.0, 0.0
    n_valid = n_iter - transient
    logs = np.empty(n_valid, dtype=np.float64)
    k = 0
    diverged = False
    for t in range(n_iter):
        if abs(xs) > escape_threshold or abs(ys) > escape_threshold or abs(x0) > escape_threshold or abs(x1) > escape_threshold:
            diverged = True
            break
        jt00 = -2.0 * xs - sigma
        jt01 = b
        jt10 = 1.0
        jt11 = 0.0
        nut0 = jt00 * ut0 + jt01 * ut1
        nut1 = jt10 * ut0 + jt11 * ut1
        normt = math.sqrt(nut0 * nut0 + nut1 * nut1)
        if normt == 0.0:
            normt = 1e-16
        ut0 = nut0 / normt
        ut1 = nut1 / normt
        if t >= transient:
            logs[k] = math.log(normt)
            k += 1
        x0, y0, x1, y1 = henon_step(x0, y0, x1, y1, a, b, sigma)
        xs, ys = henon_sync_step(xs, ys, a, b, sigma)
    return logs[:k], diverged


@njit(cache=True, fastmath=True)
def detect_period(a, b, sigma, x0, y0, n_transient=5000, n_check=300, tol=1e-6, max_period=24):
    """Detecta se a órbita da variedade de sincronização (xs,ys) é periódica.
    Retorna o período p (1..max_period) se detectado, 0 se aparenta caótica,
    -1 se divergiu durante a checagem."""
    xs, ys = x0, y0
    for _ in range(n_transient):
        xs, ys = henon_sync_step(xs, ys, a, b, sigma)
        if abs(xs) > 1e6:
            return -1
    buf = np.empty(n_check)
    for i in range(n_check):
        buf[i] = xs
        xs, ys = henon_sync_step(xs, ys, a, b, sigma)
    for p in range(1, max_period + 1):
        max_dev = 0.0
        for i in range(n_check - p):
            d = abs(buf[i + p] - buf[i])
            if d > max_dev:
                max_dev = d
        if max_dev < tol:
            return p
    return 0


def compute_finite_time_exponents(logs, windows, overlap=WINDOW_OVERLAP):
    results = {}
    for w in windows:
        step = max(1, int(round(w * (1.0 - overlap))))
        starts = range(0, logs.shape[0] - w + 1, step)

        results[w] = np.array(
            [logs[start:start + w].mean() for start in starts],
            dtype=np.float64,
        )
    return results


def lag1_autocorr(logs):
    """Autocorrelação em lag-1 de logs; usada para checar componente de período-2."""
    if logs.shape[0] < 2:
        return np.nan
    x, y = logs[:-1], logs[1:]
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def simultaneous_fplus_ci_over_ics(per_ic_lambdas, windows, n_boot=N_BOOT, alpha=FPLUS_ALPHA, seed=0):
    """Intervalos simultâneos via bootstrap de ICs, preservando a dependência entre janelas."""
    valid_windows = [w for w in windows if w in per_ic_lambdas and per_ic_lambdas[w]]
    if not valid_windows:
        return {}

    n_ics = len(per_ic_lambdas[valid_windows[0]])
    if any(len(per_ic_lambdas[w]) != n_ics for w in valid_windows):
        raise ValueError("Cada janela deve conter o mesmo conjunto de condições iniciais.")

    per_ic_fplus = np.array([
        [np.mean(lambdas > 0.0) for lambdas in per_ic_lambdas[w]]
        for w in valid_windows
    ])
    estimates = per_ic_fplus.mean(axis=1)
    rng = np.random.default_rng(seed)
    max_deviations = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n_ics, n_ics)
        bootstrap_estimates = per_ic_fplus[:, idx].mean(axis=1)
        max_deviations[b] = np.max(np.abs(bootstrap_estimates - estimates))

    critical_value = float(np.quantile(max_deviations, 1.0 - alpha))
    return {
        w: (max(0.0, float(estimate - critical_value)),
            min(1.0, float(estimate + critical_value)))
        for w, estimate in zip(valid_windows, estimates)
    }


def add_fplus_confidence_intervals(results, windows, seed):
    per_ic_lambdas = {
        w: results[w]["per_ic_lambdas"]
        for w in windows if w in results
    }
    intervals = simultaneous_fplus_ci_over_ics(per_ic_lambdas, windows, seed=seed)
    for w, interval in intervals.items():
        results[w]["fplus_ci"] = interval


def update_reliability_metadata(result):
    per_ic_lambdas = result.get("per_ic_lambdas", [])
    n_ics = len(per_ic_lambdas)
    n_positive_ics = sum(np.any(lambdas > 0.0) for lambdas in per_ic_lambdas)
    result["n_ics"] = n_ics
    result["n_positive_ics"] = int(n_positive_ics)
    result["reliable"] = (
        n_ics >= MIN_N_VALID_ICS
        and result["n_pos"] >= MIN_N_POS
        and n_positive_ics >= MIN_N_POSITIVE_ICS
    )
    return result


def aggregate_results(per_ic_lambdas, windows, seed):
    results = {}
    for w in windows:
        if not per_ic_lambdas[w]:
            continue
        all_lambdas = np.concatenate(per_ic_lambdas[w])
        n_total = int(all_lambdas.shape[0])
        n_pos = int(np.sum(all_lambdas > 0.0))
        fplus = n_pos / n_total if n_total > 0 else np.nan
        results[w] = {
            'lambdas': all_lambdas,
            'per_ic_lambdas': per_ic_lambdas[w],
            'mean': float(np.mean(all_lambdas)),
            'std': float(np.std(all_lambdas)),
            'fplus': float(fplus),
            'fplus_ci': (np.nan, np.nan),
            'n_pos': n_pos,
            'n_total': n_total,
            'variance': float(np.var(all_lambdas, ddof=1)) if n_total > 1 else np.nan,
            'n_windows': n_total,
        }
    add_fplus_confidence_intervals(results, windows, seed)
    for result in results.values():
        update_reliability_metadata(result)
    return results


def run_multi_ic(a, b, sigma, n_ics, x_range, y_range, n_iter_per_ic, transient_per_ic, windows, seed, escape_threshold, delta):
    rng = np.random.default_rng(seed)
    x0s = rng.uniform(x_range[0], x_range[1], n_ics)
    y0s = rng.uniform(y_range[0], y_range[1], n_ics)
    per_ic_lambdas_all = {w: [] for w in windows}
    per_ic_lambdas_chaotic = {w: [] for w in windows}
    n_valid = 0
    n_diverged = 0
    ics_used = []
    lag1_list = []
    period_counts = {}
    for i in range(n_ics):
        x0, y0 = x0s[i], y0s[i]
        x1, y1 = x0 + delta, y0 + delta
        logs, diverged = simulate_system_safe(a, b, sigma, x0, y0, x1, y1, n_iter_per_ic, transient_per_ic, escape_threshold)
        if diverged or logs.shape[0] < max(windows):
            n_diverged += 1
            continue
        n_valid += 1
        ics_used.append((x0, y0))
        lag1_list.append(lag1_autocorr(logs))
        period = detect_period(a, b, sigma, x0, y0)
        period_counts[period] = period_counts.get(period, 0) + 1
        lambdas_by_w = compute_finite_time_exponents(logs, windows)
        for w in windows:
            per_ic_lambdas_all[w].append(lambdas_by_w[w])
            if period == 0:
                per_ic_lambdas_chaotic[w].append(lambdas_by_w[w])
    mean_lag1 = float(np.nanmean(lag1_list)) if lag1_list else np.nan

    results_all = aggregate_results(per_ic_lambdas_all, windows, seed)
    results_chaotic = aggregate_results(per_ic_lambdas_chaotic, windows, seed + 1)
    return results_all, results_chaotic, n_valid, n_diverged, np.array(ics_used), mean_lag1, period_counts


def reliable_windows(results, windows):
    for result in results.values():
        update_reliability_metadata(result)
    return [w for w in windows if w in results and results[w]['reliable']]


def fplus_classification(result):
    update_reliability_metadata(result)
    if not result["reliable"]:
        return "insufficient data"
    lower, upper = result["fplus_ci"]
    if lower > FPLUS_THRESHOLD:
        return "above 0.5"
    if upper < FPLUS_THRESHOLD:
        return "below 0.5"
    return "inconclusive"


def select_stable_window(results, windows, n_consecutive=MIN_CONSECUTIVE_WINDOWS):
    tested_windows = [w for w in windows if w in results]
    classifications = {w: fplus_classification(results[w]) for w in tested_windows}
    for start in range(len(tested_windows) - n_consecutive + 1):
        candidate_windows = tested_windows[start:start + n_consecutive]
        classification = classifications[candidate_windows[0]]
        if classification in ("above 0.5", "below 0.5") and all(
            classifications[w] == classification for w in candidate_windows
        ):
            return candidate_windows[0], classification
    return None, None


def print_best_window(results, windows, label):
    best_w, classification = select_stable_window(results, windows)
    if best_w is None:
        print(f"{label}: nenhuma janela teve classificação significativa e estável em "
              f"{MIN_CONSECUTIVE_WINDOWS} janelas avaliadas consecutivas.")
        return
    lower, upper = results[best_w]["fplus_ci"]
    print(f"{label}: menor janela estável w={best_w} ({classification}; "
          f"IC simultâneo 95%=[{lower:.4f}, {upper:.4f}]).")


def print_period_breakdown(period_counts, n_valid):
    print(f"\nClassificação por período da variedade de sincronização ({n_valid} ICs válidas):")
    n_chaotic = period_counts.get(0, 0)
    n_diverged_check = period_counts.get(-1, 0)
    print(f"  caóticas (período não detectado): {n_chaotic}")
    for p in sorted(k for k in period_counts if k > 0):
        print(f"  período {p}: {period_counts[p]}")
    if n_diverged_check:
        print(f"  divergiu durante checagem de período: {n_diverged_check}")
    print()


def print_table(results, windows, n_valid, n_diverged, label=""):
    if label:
        print(f"--- {label} ---")
    print(f"\nICs válidas: {n_valid}   |   ICs descartadas (divergência): {n_diverged}\n")
    print(f"{'w':>8} | {'mean':>10} | {'std':>10} | {'f+':>10} | {'n+ blocos':>12} | {'IC+':>7} | {'IC95% simultâneo':>23} | {'decisão':>18}")
    print("-" * 120)
    for w in windows:
        if w not in results:
            print(f"{w:>8d} | {'—':>10} | {'—':>10} | {'—':>10} | {'—':>12} | {'—':>7} | {'—':>23} | {'—':>18}")
            continue
        r = results[w]
        update_reliability_metadata(r)
        lower, upper = r["fplus_ci"]
        interval = f"[{lower:.4f}, {upper:.4f}]"
        positive_blocks = f"{r['n_pos']}/{r['n_total']}"
        positive_ics = f"{r['n_positive_ics']}/{r['n_ics']}"
        print(f"{w:>8d} | {r['mean']:>10.4f} | {r['std']:>10.4f} | {r['fplus']:>10.4g} | {positive_blocks:>12} | {positive_ics:>7} | {interval:>23} | {fplus_classification(r):>18}")
    print()


def plot_histograms(results, windows, output_path):
    from string import ascii_lowercase
    try:
        from pynamicalsys import PlotStyler
        PlotStyler().apply_style()
    except ImportError:
        pass
    valid_windows = [w for w in windows if w in results]
    all_lambdas = np.concatenate([results[w]["lambdas"] for w in valid_windows])
    x_min, x_max = all_lambdas.min(), all_lambdas.max()
    margin = max(abs(x_min) * 0.05, 1e-12) if x_min == x_max else 0.05 * (x_max - x_min)
    common_bins = np.linspace(x_min - margin, x_max + margin, HISTOGRAM_BINS + 1)
    cm = 1 / 2.54
    fig, axes = plt.subplots(2, 3, figsize=(25 * cm, 12 * cm), sharey=True)
    axes = axes.flatten()
    for idx, w in enumerate(valid_windows):
        ax = axes[idx]
        r = results[w]
        density, edges = np.histogram(r["lambdas"], bins=common_bins, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.bar(centers, density, width=np.diff(edges), color="gray", edgecolor="black", linewidth=0.3, alpha=0.85)
        ax.axvline(0.0, color="crimson", linestyle="--", linewidth=1.2)
        ax.text(0.03, 0.85, f"({ascii_lowercase[idx]}) w = {w}", transform=ax.transAxes, fontsize=14, bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=2))
        frac = r["fplus"]
        update_reliability_metadata(r)
        tag = "" if r["reliable"] else " (tail unresolved)"
        if 0 < frac < 1e-3:
            mantissa, exponent = f"{frac:.1e}".split("e")
            label = rf"$f_w^+ = {mantissa}\times10^{{{int(exponent)}}}${tag}"
        else:
            label = rf"$f_w^+ = {frac:.2f}${tag}"
        ax.text(0.1, 0.75, label, transform=ax.transAxes, fontsize=11, bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=2))
        ax.set_yscale("log")
        ax.tick_params(axis="y", labelsize=10)
        if idx % 3 == 0:
            ax.set_ylabel(r"$P(\lambda_{\perp}^{w})$", fontsize=15)
        ax.set_xlabel(r"$\lambda_{\perp}^{w}$", fontsize=15)
        # ax.set_xlim(x_min - margin, 1.05)
        ax.tick_params(axis="x", labelsize=12)
    for idx in range(len(valid_windows), 6):
        fig.delaxes(axes[idx])
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Figura salva: {output_path}")
    return fig


def plot_combined(results, hist_windows, var_windows, output_path):
    """Figura única: histogramas de lambda_w (a,b,c) + variância vs w (d)."""
    from string import ascii_lowercase
    try:
        from pynamicalsys import PlotStyler
        PlotStyler().apply_style()
    except ImportError:
        pass

    valid_hist_w = [w for w in hist_windows if w in results]
    all_lambdas = np.concatenate([results[w]["lambdas"] for w in valid_hist_w])
    x_min, x_max = all_lambdas.min(), all_lambdas.max()
    margin = max(abs(x_min) * 0.05, 1e-12) if x_min == x_max else 0.05 * (x_max - x_min)
    common_bins = np.linspace(x_min - margin, x_max + margin, HISTOGRAM_BINS + 1)

    cm = 1 / 2.54
    fig = plt.figure(figsize=(25 * cm, 15 * cm))
    gs = fig.add_gridspec(2, len(valid_hist_w), height_ratios=[1.3, 1])
    hist_axes = [fig.add_subplot(gs[0, 0])]
    for i in range(1, len(valid_hist_w)):
        hist_axes.append(fig.add_subplot(gs[0, i], sharey=hist_axes[0]))

    ax_var = fig.add_subplot(gs[1, :])

    for idx, w in enumerate(valid_hist_w):
        ax = hist_axes[idx]
        ax.set_xlabel(r"$\lambda_{\perp}^{w}$", fontsize=14)
        # ax.set_xlim(x_min - margin, 1.05)
        # ax.set_yscale("log")
        # ax.set_ylim(-1,1)
        ax.tick_params(axis="x", labelsize=12)
        ax.tick_params(axis="y", labelsize=12)
        ax.grid(True, which="both", alpha=0.3)
        r = results[w]
        density, edges = np.histogram(r["lambdas"], bins=common_bins, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.bar(centers, density, width=np.diff(edges), color="gray", edgecolor="black", linewidth=0.3, alpha=0.85)
        ax.axvline(0.0, color="crimson", linestyle="--", linewidth=1.2)
        ax.text(0.03, 0.92, f"({ascii_lowercase[idx]}) w = {w}", transform=ax.transAxes, fontsize=14,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=2))
        frac = r["fplus"]
        update_reliability_metadata(r)
        if 0 < frac < 1e-3:
            mantissa, exponent = f"{frac:.1e}".split("e")
            label = rf"$f_w^+ = {mantissa}\times10^{{{int(exponent)}}}$"
        else:
            label = rf"$f_w^+ = {frac:.2f}$"
        label = f"{label} (IC+ {r['n_positive_ics']}/{r['n_ics']})"
        ax.text(0.05, 0.85, label, transform=ax.transAxes, fontsize=14,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=2))
        if idx == 0:
            ax.set_ylabel(r"$P(\lambda_{\perp}^{w})$", fontsize=14)
        

    good_w = reliable_windows(results, var_windows)
    bad_w = [w for w in var_windows if w in results and w not in good_w]
    even_w = [w for w in good_w if w % 2 == 0]
    odd_w = [w for w in good_w if w % 2 == 1]
    if even_w:
        ax_var.plot(even_w, [results[w]["variance"] for w in even_w], marker="o", markersize=5, linestyle="none", label="even w")
    if odd_w:
        ax_var.plot(odd_w, [results[w]["variance"] for w in odd_w], marker="s", markersize=5, linestyle="none", label="odd w")
    if bad_w:
        ax_var.plot(bad_w, [results[w]["variance"] for w in bad_w], marker="x", markersize=4, linestyle="none", color="gray", alpha=0.6, label="insufficient statistics")
    ax_var.set_yscale("log")
    ax_var.set_xlabel(r"$w$")
    ax_var.set_ylim(1e-5, None)
    ax_var.tick_params(axis="y", labelsize=12, which="both")
    ax_var.set_ylabel(r"$\mathrm{Var}(\lambda_{\perp}^{w})$", fontsize=14)
    ax_var.text(0.01, 0.9, "(d)", transform=ax_var.transAxes, fontsize=14)
    ax_var.grid(True, which="both", alpha=0.3)
    ax_var.legend(fontsize=12)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Figura salva: {output_path}")
    return fig


def plot_fplus(results, windows, output_path):
    try:
        from pynamicalsys import PlotStyler
        PlotStyler().apply_style()
    except ImportError:
        pass
    good_w = reliable_windows(results, windows)
    bad_w = [w for w in windows if w in results and w not in good_w]
    cm = 1 / 2.54
    fig, ax = plt.subplots(figsize=(25 * cm, 8 * cm))
    if good_w:
        fplus_good = [results[w]["fplus"] for w in good_w]
        yerr = np.array([(results[w]["fplus"] - results[w]["fplus_ci"][0], results[w]["fplus_ci"][1] - results[w]["fplus"]) for w in good_w]).T
        yerr = np.clip(yerr, 0, None)
        ax.errorbar(good_w, fplus_good, yerr=yerr, fmt='o', markersize=3, linewidth=1, capsize=2,
                label=f"cauda resolvida: n+≥{MIN_N_POS}, IC+≥{MIN_N_POSITIVE_ICS}")
    if bad_w:
        ax.scatter(bad_w, [results[w]["fplus"] for w in bad_w], marker='x', color='gray', alpha=0.6, label="estatística insuficiente")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.axhline(FPLUS_THRESHOLD, color="crimson", linestyle="--", linewidth=1,
               label=rf"$f_w^+={FPLUS_THRESHOLD}$")
    ax.set_xlabel(r"tamanho da janela $w$")
    ax.set_ylabel(r"$f_{w}^{+}$")
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    print(f"Figura salva: {output_path}")
    return fig


def plot_variance(results, windows, output_path):
    try:
        from pynamicalsys import PlotStyler
        PlotStyler().apply_style()
    except ImportError:
        pass
    good_w = reliable_windows(results, windows)
    bad_w = [w for w in windows if w in results and w not in good_w]
    cm = 1 / 2.54
    fig, ax = plt.subplots(figsize=(25 * cm, 8 * cm))
    if good_w:
        ax.plot(good_w, [results[w]["variance"] for w in good_w], marker="o", markersize=3, linestyle="none", label=f"n_pos ≥ {MIN_N_POS}")
    if bad_w:
        ax.plot(bad_w, [results[w]["variance"] for w in bad_w], marker="x", markersize=4, linestyle="none", color="gray", alpha=0.6, label="estatística insuficiente")
    ax.set_yscale("log")
    ax.set_xlabel(r"$w$")
    ax.set_ylabel(r"$\mathrm{Var}(\lambda_{\perp}^{w})$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    print(f"Figura salva: {output_path}")
    return fig


if __name__ == "__main__":
    if RECALCULATE:
        print("Sorteando e simulando condições iniciais...")
        print(f"  a={A}, b={B}, sigma={SIGMA}")
        print(f"  N_ICS={N_ICS}, range=[{X_MIN},{X_MAX}]x[{Y_MIN},{Y_MAX}], seed={SEED}")
        print(f"  N_ITER_PER_IC={N_ITER_PER_IC}, TRANSIENT_PER_IC={TRANSIENT_PER_IC}")
        print(f"  WINDOWS=[{WINDOWS.min()}, {WINDOWS.max()}] ({len(WINDOWS)} tamanhos testados)")
        print(f"  ESCAPE_THRESHOLD={ESCAPE_THRESHOLD:.0e}")
        print(f"  Resolução de cauda: n+ blocos ≥ {MIN_N_POS}, ICs com positivos ≥ {MIN_N_POSITIVE_ICS}, ICs válidas ≥ {MIN_N_VALID_ICS}")

        results_all, results_chaotic, n_valid, n_diverged, ics_used, mean_lag1, period_counts = run_multi_ic(
            A, B, SIGMA, N_ICS, (X_MIN, X_MAX), (Y_MIN, Y_MAX),
            N_ITER_PER_IC, TRANSIENT_PER_IC, WINDOWS, SEED, ESCAPE_THRESHOLD, DELTA
        )

        with open(DATA_PATH, "wb") as f:
            pickle.dump({
                "config": cache_config(),
                "results_all": results_all, "results_chaotic": results_chaotic,
                "n_valid": n_valid, "n_diverged": n_diverged, "ics_used": ics_used,
                "mean_lag1": mean_lag1, "period_counts": period_counts,
            }, f)
        print(f"Dados salvos em: {DATA_PATH}")
    else:
        print(f"RECALCULATE=False -> lendo dados salvos de: {DATA_PATH}")
        with open(DATA_PATH, "rb") as f:
            data = pickle.load(f)
        validate_cache_config(data)
        results_all = data["results_all"]
        results_chaotic = data["results_chaotic"]
        n_valid = data["n_valid"]
        n_diverged = data["n_diverged"]
        ics_used = data["ics_used"]
        mean_lag1 = data["mean_lag1"]
        period_counts = data["period_counts"]

    print_period_breakdown(period_counts, n_valid)
    n_chaotic_ics = period_counts.get(0, 0)
    if n_valid - n_chaotic_ics > 0:
        print(f"AVISO: {n_valid - n_chaotic_ics} de {n_valid} ICs caíram em janelas "
              "periódicas da variedade de sincronização (não caos). Para essas ICs, "
              "lambda_w em blocos com w múltiplo do período tem variância ~0 (média "
              "exata sobre um ciclo completo), inflando artificialmente ramos de baixa "
              "variância no gráfico Var(lambda_w) vs w e criando picos afiados nos "
              "histogramas. Geradas figuras separadas: todas as ICs vs. só caóticas.")

    print_table(results_all, WINDOWS, n_valid, n_diverged, label="Todas as ICs")
    print_table(results_chaotic, WINDOWS, n_chaotic_ics, n_diverged, label="Só ICs caóticas")
    print(f"Autocorrelação média em lag-1 de log(fator de esticamento): {mean_lag1:.4f}")

    plot_combined(results_all, HISTOGRAM_WINDOWS, VARIANCE_WINDOWS, f"{OUTPUT_PREFIX}_combined_all.png")
    plot_fplus(results_all, VARIANCE_WINDOWS, f"{OUTPUT_PREFIX}_fplus_all.png")

    # if n_chaotic_ics > 0:
    #     plot_combined(results_chaotic, HISTOGRAM_WINDOWS, VARIANCE_WINDOWS, f"{OUTPUT_PREFIX}_combined_chaotic.png")
    #     plot_fplus(results_chaotic, VARIANCE_WINDOWS, f"{OUTPUT_PREFIX}_fplus_chaotic.png")