"""
CUDA Bifurcation Diagram for Coupled Hénon Maps
===============================================

Calcula:

    - Diagrama de bifurcação de X0 e X1
    - Diagrama do manifold sincronizado
    - 4 expoentes de Lyapunov
    - Expoente transversal
    - Fração positiva f+
    - Tempo de escape
    - Período

Versão:
    a -> métricas
    b fixo

"""

import math
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from string import ascii_lowercase

from numba import cuda
from numba import njit
from numba import prange


# ============================================================
# CONFIGURAÇÕES
# ============================================================

RECALCULATE = True
EPSILON = 0.0219
A_VALS = np.linspace(1.40, 1.65, 4000)
B_FIXED = 0.225
N_ITER = 60_000
TRANSIENT = 50_000
WINDOW = 100
ESCAPE_RADIUS = 1e6
DELTA = 1e-15
MAX_PERIOD = 16
TOTAL_TIME = int(1e6)
BIF_POINTS = 300

OUTPUT_PREFIX = (
    f"../figures/bifurcation_cuda_sigma{EPSILON}_b{B_FIXED}"
)

# ============================================================
# RECALCULAR
# ============================================================

if RECALCULATE:
    # ========================================================
    # EVOLUÇÃO 4D
    # ========================================================
    @cuda.jit(device=True)
    def henon_step(x0, y0, x1, y1, a, b, sigma):

        xn0 = a - x0*x0 + b*y0 + sigma*x1
        yn0 = x0

        xn1 = a - x1*x1 + b*y1 + sigma*x0
        yn1 = x1

        return xn0, yn0, xn1, yn1

    # ========================================================
    # MANIFOLD SINCRONIZAÇÃO
    # ========================================================
    @cuda.jit(device=True)
    def henon_sync_step(xs, ys, a, b, sigma):

        xsn = a - xs*xs + b*ys + sigma*xs
        ysn = xs

        return xsn, ysn


    # ========================================================
    # KERNEL CUDA
    # ========================================================
    @cuda.jit
    def bifurcation_kernel(
        a_vals,
        lyap_1_arr,
        lyap_2_arr,
        lyap_3_arr,
        lyap_4_arr,

        tle_arr,
        frac_arr,
        esc_arr,

        sigma,
        b,

        n_iter,
        transient,
        window,
        escape_radius
    ):

        idx = cuda.grid(1)

        nA = a_vals.shape[0]

        if idx >= nA:
            return

        a = a_vals[idx]

        # ====================================================
        # CONDIÇÕES INICIAIS
        # ====================================================

        x0 = -1.0
        y0 = -1.0

        x1 = -1.0 + DELTA
        y1 = -1.0 + DELTA

        xs = -1.0
        ys = -1.0

        # ====================================================
        # BASE ORTONORMAL
        # ====================================================

        q1 = cuda.local.array(4, dtype=np.float64)
        q2 = cuda.local.array(4, dtype=np.float64)
        q3 = cuda.local.array(4, dtype=np.float64)
        q4 = cuda.local.array(4, dtype=np.float64)

        q1[0], q1[1], q1[2], q1[3] = 1., 0., 0., 0.
        q2[0], q2[1], q2[2], q2[3] = 0., 1., 0., 0.
        q3[0], q3[1], q3[2], q3[3] = 0., 0., 1., 0.
        q4[0], q4[1], q4[2], q4[3] = 0., 0., 0., 1.

        # ====================================================
        # TRANSVERSAL
        # ====================================================

        ut0 = 1.0
        ut1 = 0.0

        # ====================================================
        # ACUMULADORES
        # ====================================================

        lyap1_sum = 0.0
        lyap2_sum = 0.0
        lyap3_sum = 0.0
        lyap4_sum = 0.0

        tle_sum = 0.0

        local_tle = 0.0
        local_count = 0

        positive_count = 0
        total_windows = 0

        escaped = False
        esc_time = n_iter

        # ====================================================
        # LOOP TEMPORAL
        # ====================================================

        for t in range(n_iter):

            # =================================================
            # JACOBIANA
            # =================================================

            J = cuda.local.array((4, 4), dtype=np.float64)

            J[0,0] = -2.0*x0
            J[0,1] = b
            J[0,2] = sigma
            J[0,3] = 0.0

            J[1,0] = 1.0
            J[1,1] = 0.0
            J[1,2] = 0.0
            J[1,3] = 0.0

            J[2,0] = sigma
            J[2,1] = 0.0
            J[2,2] = -2.0*x1
            J[2,3] = b

            J[3,0] = 0.0
            J[3,1] = 0.0
            J[3,2] = 1.0
            J[3,3] = 0.0

            # =================================================
            # MULTIPLICAÇÃO
            # =================================================

            z1 = cuda.local.array(4, dtype=np.float64)
            z2 = cuda.local.array(4, dtype=np.float64)
            z3 = cuda.local.array(4, dtype=np.float64)
            z4 = cuda.local.array(4, dtype=np.float64)

            for r in range(4):

                z1[r] = (
                    J[r,0]*q1[0] +
                    J[r,1]*q1[1] +
                    J[r,2]*q1[2] +
                    J[r,3]*q1[3]
                )

                z2[r] = (
                    J[r,0]*q2[0] +
                    J[r,1]*q2[1] +
                    J[r,2]*q2[2] +
                    J[r,3]*q2[3]
                )

                z3[r] = (
                    J[r,0]*q3[0] +
                    J[r,1]*q3[1] +
                    J[r,2]*q3[2] +
                    J[r,3]*q3[3]
                )

                z4[r] = (
                    J[r,0]*q4[0] +
                    J[r,1]*q4[1] +
                    J[r,2]*q4[2] +
                    J[r,3]*q4[3]
                )

            # =================================================
            # PRODUTO INTERNO
            # =================================================

            def dot4(a, b):
                return (
                    a[0]*b[0] +
                    a[1]*b[1] +
                    a[2]*b[2] +
                    a[3]*b[3]
                )

            # =================================================
            # QR / GRAM SCHMIDT
            # =================================================

            r11 = math.sqrt(dot4(z1, z1))

            if r11 == 0.0:
                r11 = 1e-16

            for k in range(4):
                q1[k] = z1[k] / r11

            proj21 = dot4(z2, q1)

            for k in range(4):
                z2[k] -= proj21*q1[k]

            r22 = math.sqrt(dot4(z2, z2))

            if r22 == 0.0:
                r22 = 1e-16

            for k in range(4):
                q2[k] = z2[k] / r22

            proj31 = dot4(z3, q1)
            proj32 = dot4(z3, q2)

            for k in range(4):
                z3[k] -= (
                    proj31*q1[k] +
                    proj32*q2[k]
                )

            r33 = math.sqrt(dot4(z3, z3))

            if r33 == 0.0:
                r33 = 1e-16

            for k in range(4):
                q3[k] = z3[k] / r33

            proj41 = dot4(z4, q1)
            proj42 = dot4(z4, q2)
            proj43 = dot4(z4, q3)

            for k in range(4):
                z4[k] -= (
                    proj41*q1[k] +
                    proj42*q2[k] +
                    proj43*q3[k]
                )

            r44 = math.sqrt(dot4(z4, z4))

            if r44 == 0.0:
                r44 = 1e-16

            for k in range(4):
                q4[k] = z4[k] / r44

            # =================================================
            # TRANSVERSAL
            # =================================================

            jt00 = -2.0*xs - sigma
            jt01 = b

            jt10 = 1.0
            jt11 = 0.0

            nut0 = jt00*ut0 + jt01*ut1
            nut1 = jt10*ut0 + jt11*ut1

            normt = math.sqrt(
                nut0*nut0 +
                nut1*nut1
            )

            if normt == 0.0:
                normt = 1e-16

            ut0 = nut0 / normt
            ut1 = nut1 / normt

            # =================================================
            # ACUMULAÇÃO
            # =================================================

            if t >= transient:

                l1 = math.log(r11)
                l2 = math.log(r22)
                l3 = math.log(r33)
                l4 = math.log(r44)

                lt = math.log(normt)

                lyap1_sum += l1
                lyap2_sum += l2
                lyap3_sum += l3
                lyap4_sum += l4

                tle_sum += lt

                local_tle += lt
                local_count += 1

                if local_count >= window:

                    finite_time_tle = (
                        local_tle / window
                    )

                    if finite_time_tle > 0.0:
                        positive_count += 1

                    total_windows += 1

                    local_tle = 0.0
                    local_count = 0

            # =================================================
            # EVOLUÇÃO
            # =================================================

            x0, y0, x1, y1 = henon_step(
                x0,
                y0,
                x1,
                y1,
                a,
                b,
                sigma
            )

            xs, ys = henon_sync_step(
                xs,
                ys,
                a,
                b,
                sigma
            )

            # =================================================
            # ESCAPE
            # =================================================

            if (
                abs(x0) > escape_radius or
                abs(y0) > escape_radius or
                abs(x1) > escape_radius or
                abs(y1) > escape_radius
            ):

                escaped = True
                esc_time = t

                break

        # ====================================================
        # RESULTADOS
        # ====================================================

        if escaped:

            lyap_1_arr[idx] = math.nan
            lyap_2_arr[idx] = math.nan
            lyap_3_arr[idx] = math.nan
            lyap_4_arr[idx] = math.nan

            tle_arr[idx] = math.nan
            frac_arr[idx] = math.nan

            esc_arr[idx] = esc_time

        else:

            denom = n_iter - transient

            lyap_1_arr[idx] = (
                lyap1_sum / denom
            )

            lyap_2_arr[idx] = (
                lyap2_sum / denom
            )

            lyap_3_arr[idx] = (
                lyap3_sum / denom
            )

            lyap_4_arr[idx] = (
                lyap4_sum / denom
            )

            tle_arr[idx] = (
                tle_sum / denom
            )

            if total_windows > 0:

                frac_arr[idx] = (
                    positive_count /
                    total_windows
                )

            else:

                frac_arr[idx] = 0.0

            esc_arr[idx] = math.nan


    # ========================================================
    # DETECÇÃO DE PERÍODO
    # ========================================================
    @njit
    def detect_period(
        a,
        b,
        sigma,
        n_iter,
        transient,
        max_period
    ):

        x0 = -1.0
        y0 = -1.0

        x1 = -1.0 + DELTA
        y1 = -1.0 + DELTA

        history = np.zeros((max_period, 4))

        idx = 0

        for t in range(n_iter):

            xn0 = a - x0*x0 + b*y0 + sigma*x1
            yn0 = x0

            xn1 = a - x1*x1 + b*y1 + sigma*x0
            yn1 = x1

            x0, y0, x1, y1 = (
                xn0,
                yn0,
                xn1,
                yn1
            )

            if t < transient:
                continue

            history[idx, 0] = x0
            history[idx, 1] = y0
            history[idx, 2] = x1
            history[idx, 3] = y1

            idx += 1

            if idx >= max_period:
                break

        tol = 1e-8

        for p in range(1, max_period+1):

            ok = True

            for k in range(max_period-p):

                d0 = abs(
                    history[k,0] -
                    history[k+p,0]
                )

                d1 = abs(
                    history[k,1] -
                    history[k+p,1]
                )

                d2 = abs(
                    history[k,2] -
                    history[k+p,2]
                )

                d3 = abs(
                    history[k,3] -
                    history[k+p,3]
                )

                if (
                    d0 > tol or
                    d1 > tol or
                    d2 > tol or
                    d3 > tol
                ):
                    ok = False
                    break

            if ok:
                return p

        return -1


    @njit(parallel=True)
    def compute_periods(
        a_vals,
        b_fixed,
        lyap_1_arr,
        sigma,
        n_iter,
        transient,
        max_period
    ):

        nA = len(a_vals)

        out = np.full(nA, -1)

        for i in prange(nA):

            mle = lyap_1_arr[i]

            if np.isnan(mle):
                continue

            if mle < 0.0:

                out[i] = detect_period(
                    a_vals[i],
                    b_fixed,
                    sigma,
                    n_iter,
                    transient,
                    max_period
                )

        return out


    # ========================================================
    # EXECUÇÃO CUDA
    # ========================================================
    print("Executando CUDA...")
    nA = len(A_VALS)
    lyap_1_arr = np.zeros(nA)
    lyap_2_arr = np.zeros(nA)
    lyap_3_arr = np.zeros(nA)
    lyap_4_arr = np.zeros(nA)

    tle_arr = np.zeros(nA)

    frac_arr = np.zeros(nA)

    esc_arr = np.zeros(nA)

    d_a = cuda.to_device(A_VALS)

    d_lyap_1 = cuda.to_device(lyap_1_arr)
    d_lyap_2 = cuda.to_device(lyap_2_arr)
    d_lyap_3 = cuda.to_device(lyap_3_arr)
    d_lyap_4 = cuda.to_device(lyap_4_arr)

    d_tle = cuda.to_device(tle_arr)

    d_frac = cuda.to_device(frac_arr)

    d_esc = cuda.to_device(esc_arr)

    threads = 256

    blocks = (
        nA + threads - 1
    ) // threads

    bifurcation_kernel[
        blocks,
        threads
    ](

        d_a,

        d_lyap_1,
        d_lyap_2,
        d_lyap_3,
        d_lyap_4,

        d_tle,
        d_frac,
        d_esc,

        EPSILON,
        B_FIXED,

        N_ITER,
        TRANSIENT,
        WINDOW,
        ESCAPE_RADIUS
    )

    cuda.synchronize()

    lyap_1_arr = d_lyap_1.copy_to_host()
    lyap_2_arr = d_lyap_2.copy_to_host()
    lyap_3_arr = d_lyap_3.copy_to_host()
    lyap_4_arr = d_lyap_4.copy_to_host()

    tle_arr = d_tle.copy_to_host()

    frac_arr = d_frac.copy_to_host()

    esc_arr = d_esc.copy_to_host()

    esc_arr[np.isnan(esc_arr)] = TOTAL_TIME

    print("CUDA finalizado.")

    # ========================================================
    # PERÍODOS
    # ========================================================

    print("Calculando períodos...")

    period_arr = compute_periods(
        A_VALS,
        B_FIXED,
        lyap_1_arr,
        EPSILON,
        N_ITER,
        TRANSIENT,
        MAX_PERIOD
    )

    print("Períodos finalizados.")

else:

    print("Modo leitura CSV ainda não implementado.")
    raise SystemExit


# ============================================================
# BIFURCAÇÃO X0/X1
# ============================================================

@njit(parallel=True)
def compute_bifurcation_states(
    a_vals,
    b,
    sigma,
    n_iter,
    bif_points
):

    nA = len(a_vals)

    out_x0 = np.zeros((nA, bif_points))
    out_x1 = np.zeros((nA, bif_points))

    for i in prange(nA):

        a = a_vals[i]

        x0 = -1.0
        y0 = -1.0

        x1 = -1.0 + DELTA
        y1 = -1.0 + DELTA

        save_idx = 0

        for t in range(n_iter):

            xn0 = (
                a - x0*x0 +
                b*y0 +
                sigma*x1
            )

            yn0 = x0

            xn1 = (
                a - x1*x1 +
                b*y1 +
                sigma*x0
            )

            yn1 = x1

            x0, y0, x1, y1 = (
                xn0,
                yn0,
                xn1,
                yn1
            )

            if t >= (n_iter - bif_points):

                out_x0[i, save_idx] = x0
                out_x1[i, save_idx] = x1

                save_idx += 1

    return out_x0, out_x1


# ============================================================
# MANIFOLD SINCRONIZAÇÃO
# ============================================================

@njit(parallel=True)
def compute_sync_bifurcation(
    a_vals,
    b,
    sigma,
    n_iter,
    bif_points
):

    nA = len(a_vals)

    out = np.zeros((nA, bif_points))

    for i in prange(nA):

        a = a_vals[i]

        xs = -1.0
        ys = -1.0

        save_idx = 0

        for t in range(n_iter):

            xsn = (
                a - xs*xs +
                b*ys +
                sigma*xs
            )

            ysn = xs

            xs, ys = xsn, ysn

            if t >= (n_iter - bif_points):

                out[i, save_idx] = xs

                save_idx += 1

    return out


print("Calculando órbitas...")

states_x0, states_x1 = (
    compute_bifurcation_states(
        A_VALS,
        B_FIXED,
        EPSILON,
        N_ITER,
        BIF_POINTS
    )
)

sync_states = (
    compute_sync_bifurcation(
        A_VALS,
        B_FIXED,
        EPSILON,
        N_ITER,
        BIF_POINTS
    )
)

print("Órbitas finalizadas.")


# ============================================================
# DATAFRAMES
# ============================================================

a_rep = np.repeat(
    A_VALS,
    BIF_POINTS
)

df_states_x0 = pd.DataFrame({
    "param": a_rep,
    "state": states_x0.ravel(),
    "component": "x0"
})

df_states_x1 = pd.DataFrame({
    "param": a_rep,
    "state": states_x1.ravel(),
    "component": "x1"
})

df_states = pd.concat(
    [
        df_states_x0,
        df_states_x1
    ],
    ignore_index=True
)

df_sync = pd.DataFrame({
    "param": a_rep,
    "state": sync_states.ravel()
})

df_metrics = pd.DataFrame({
    "param": A_VALS,
    "lyap_1": lyap_1_arr,
    "lyap_2": lyap_2_arr,
    "lyap_3": lyap_3_arr,
    "lyap_4": lyap_4_arr,
    "lyap_transversal": tle_arr,
    "frac_pos": frac_arr,
    "esc_time": esc_arr,
    "period": period_arr
})


# ============================================================
# PLOT
# ============================================================

def plot_bifurcation(
    df_states,
    df_sync,
    df_metrics,
    figsize=(20, 16)
):

    try:
        from pynamicalsys import PlotStyler
        PlotStyler().apply_style()
    except ImportError:
        pass

    cm = 1 / 2.54

    fig, axes = plt.subplots(
        3, 1,
        figsize=np.array(figsize) * cm,
        sharex=True,
        gridspec_kw={
            'height_ratios': [3, 2, 2]
        },
        constrained_layout=True,
    )

    ax_bif = axes[0]
    ax_lyap = axes[1]
    ax_frac = axes[2]

    a_min = df_metrics['param'].min()
    a_max = df_metrics['param'].max()

    # ========================================================
    # LINHA 1
    # ========================================================

    state_colors = {
        'x0': 'orange',
        'x1': 'black'
    }

    state_legend = {
        'x0': '$X^{[0]}$',
        'x1': '$X^{[1]}$'
    }

    for component in ('x0', 'x1'):

        df_comp = df_states[
            df_states['component'] == component
        ]

        ax_bif.scatter(
            df_comp['param'],
            df_comp['state'],
            marker='.',
            s=0.5,
            alpha=0.2,
            color=state_colors[component],
            edgecolor='none',
            label=state_legend[component]
        )

    ax_bif.set_ylabel(
        r'$X^{[n]}$',
        rotation=0,
        labelpad=15
    )

    ax_bif.set_ylim(-2.1, 2.4)

    ax_bif.legend(
        loc='upper right',
        markerscale=25
    )

    # ========================================================
    # LINHA 3
    # ========================================================
    # ax_lyap = ax_bif.twinx()
    ax_lyap.set_ylim(-2.5, 2.)
    lyap_series = [

        ('lyap_1', r'$\lambda_1$', 'blue'),
        ('lyap_2', r'$\lambda_2$', 'darkorange'),
        ('lyap_3', r'$\lambda_3$', 'green'),
        ('lyap_4', r'$\lambda_4$', 'red'),
    ]

    for col, label, color in lyap_series:

        ax_lyap.plot(
            df_metrics['param'],
            df_metrics[col],
            '.-',
            lw=0.9,
            alpha=0.5,
            color=color,
            label=label,
            markeredgewidth=0.1,
        )

    ax_lyap.axhline(
        0,
        color='gray',
        ls='--',
        lw=0.8
    )

    ax_lyap.legend(ncol=2,loc='upper right',fontsize=9)
    ax_lyap.set_ylabel(r'$\lambda_i$',rotation=0)

    # ========================================================
    # LINHA 4
    # ========================================================
    ax_frac.set_ylim(-0.05, 1.25)
    ax_frac.plot(
        df_metrics['param'],
        df_metrics['frac_pos'],
        '.-',
        markersize=3,
        markeredgewidth=0.1,
        color='darkorange',
        zorder=5
    )

    ax_frac.set_ylabel(
        rf'$f_{{{WINDOW}}}^+$',
        rotation=0,
        labelpad=20,
        color='darkorange'
    )

    ax_frac.tick_params(
        axis='y',
        labelcolor='darkorange'
    )

    ax_twin = ax_frac.twinx()
    ax_twin.set_yscale('log')
    ax_twin.plot(
        df_metrics['param'],
        df_metrics['esc_time'],
        '-',
        color='blue',
        alpha=0.35
    )
    ax_twin.set_ylabel(
        r'$T_{esc}$',
        color='blue'
    )
    ax_twin.tick_params(
        axis='y',
        labelcolor='blue'
    )
    ax_twin.set_ylim(1e0, 1.5e6)

    # ========================================================
    # VLINES
    # ========================================================
    extra_markers = [

        (1.4500, 'lime', 'v'),
        (1.4825, 'magenta', '^'),
        (1.5350, 'red', '<'),
        (1.5655, 'blue', '>'),
        (1.6000, 'goldenrod', '*')
    ]
    if extra_markers:
        for x, color, marker in extra_markers:
            for ax in axes:
                ax.axvline(
                    x,
                    color=color,
                    linestyle='--',
                    linewidth=0.5,
                    alpha=0.7
                )
                ax.scatter(x, [2.15], marker=marker, color=color,s=50, edgecolors="black", linewidths=0.7,zorder=10, clip_on=False)

    # ========================================================
    # GLOBAL
    # ========================================================

    bbox = {
        'facecolor': 'w',
        'alpha': 0.75,
        'linewidth': 0.0,
        'pad': 1
    }

    for i, ax in enumerate(axes):
        y = 0.84 if i > 0 else 0.88
        ax.text(
            0.005,
            y,
            f'({ascii_lowercase[i]})',
            bbox=bbox,
            transform=ax.transAxes
        )

        ax.set_xlim(a_min, a_max)

        ax.grid(alpha=0.1)

    axes[-1].set_xlabel(r'$a$')

    return fig, axes

def plot_only_bifurcation(
    df_states,
    df_metrics=None,
    figsize=(35, 6)
):

    try:
        from pynamicalsys import PlotStyler
        PlotStyler().apply_style()
    except ImportError:
        pass

    cm = 1 / 2.54

    fig, ax_bif = plt.subplots(
        1, 1,
        figsize=np.array(figsize) * cm,
        constrained_layout=True,
    )

    a_min = df_states['param'].min()
    a_max = df_states['param'].max()

    # ========================================================
    # PAINEL ÚNICO: BIFURCAÇÃO
    # ========================================================

    state_colors = {
        'x0': 'red',
        'x1': 'black'
    }

    state_legend = {
        'x0': '$X^{[0]}$',
        'x1': '$X^{[1]}$'
    }

    for component in ('x0', 'x1'):

        df_comp = df_states[
            df_states['component'] == component
        ]

        ax_bif.scatter(
            df_comp['param'],
            df_comp['state'],
            marker='.',
            s=0.5,
            alpha=0.2,
            color=state_colors[component],
            edgecolor='none',
            label=state_legend[component]
        )

    ax_bif.set_ylabel(
        r'$X^{[n]}$',
        rotation=0,
        labelpad=15
    )

    ax_bif.set_ylim(-2.1, 2.4)

    # ========================================================
    # TWINX: TEMPO DE ESCAPE (LOG)
    # ========================================================

    # ax_esc = None

    if df_metrics is not None:

        ax_esc = ax_bif.twinx()

        ax_esc.set_yscale('log')

        ax_esc.plot(
            df_metrics['param'],
            df_metrics['esc_time'],
            '.-',
            color='blue',
            alpha=0.15,
            zorder=1,
            markeredgecolor='none',
        )

        ax_esc.set_ylabel(
            r'$T_{esc}$',
            color='blue'
        )

        ax_esc.tick_params(
            axis='y',
            labelcolor='blue'
        )

        ax_esc.set_ylim(1e0, 8.0e6)

    ax_bif.legend(
        loc='upper right',
        markerscale=25
    )

    # ========================================================
    # VLINES
    # ========================================================

    extra_markers = [
        # (1.4500, 'lime', 'v'),
        # (1.4825, 'magenta', '^'),
        # (1.5350, 'red', '<'),
        # (1.5655, 'blue', '>'),
        # (1.6000, 'goldenrod', '*')
    ]

    # scatter_infos = {
        # 'filled' : {
        #     'x':1.460,
        #     'y':2.75,
        #     's':100,
        #     'color': "#35CE21",
        #     'marker':'o',
        #     'lw': 0.5
        # },
        # # 'escape' : {
        # #     'x':1.510,
        # #     'y':2.75,
        # #     's':150,
        # #     'color': "#F93E25",
        # #     'marker':'^',
        # #     'lw': 0.5
        # # },
        # 'riddled':{
        #     'x':1.528,
        #     'y':2.75,
        #     's':100,
        #     'color': "#1f77b4",
        #     'marker': 's',
        #     'lw': 0.5
        # },
    # }

    scatter_infos = {
        'marker_1': {
            'x': 1.4500,
            'y': 2.25,
            's': 50,
            'color': 'lime',
            'marker': 'v',
            'label': 'Periodic',
            'lw': 0.5

        },
        'marker_2': {
            'x': 1.4825,
            'y': 2.25,
            's': 50,
            'color': 'magenta',
            'marker': '^',
            'label': 'Chaotic' ,
            'lw': 0.5
        },
        'marker_3': {
            'x': 1.5350,
            'y': 2.25,
            's': 50,
            'color': 'red',
            'marker': '<',
            'label': 'Int. Crisis',
            'lw': 0.5
        },
        'marker_4': {
            'x': 1.5655,
            'y': 2.25,
            's': 50,
            'color': 'blue',
            'marker': '>',
            'label': 'Ext. Crisis',
            'lw': 0.5
        },
        'marker_5': {
            'x': 1.6000,
            'y': 2.25,
            's': 50,
            'color': 'goldenrod',
            'marker': '*',
            'label': 'Escape',
            'lw': 0.5
        }
    }

    if scatter_infos:
        from matplotlib.legend import Legend
        points = []
        for key, info in scatter_infos.items():
            y_val = info.get('y', 3.0)

            point = ax_bif.scatter(
                info['x'],
                y_val,
                c=info['color'],
                s=info.get('s', 50),
                marker=info['marker'],
                edgecolors='black',
                linewidths=0.7 if 'lw' in key else 0.5,
                zorder=10 if 'lw' in key else 5,
                clip_on=False if 'lw' in key else True,
                label=info.get('label')
            )
            points.append(point)
            ax_bif.vlines(
                    x=info['x'],
                    ymin=-2.5,
                    ymax=5.,
                    color=info['color'],
                    linestyle='--',
                    linewidth=info['lw'],
                    alpha=0.7
                )
            # ax_bif.legend(loc='lower left', frameon=True, ncol=len(scatter_infos)/2, fontsize=7.5)
        legend2 = Legend(ax_bif, points, labels=[info.get('label') for info in scatter_infos.values()],
                         fontsize=10, frameon=True, loc='lower right')
    ax_bif.add_artist(legend2)

    # ========================================================
    # GLOBAL
    # ========================================================

    bbox = {
        'facecolor': 'w',
        'alpha': 0.75,
        'linewidth': 0.0,
        'pad': 1
    }

    ax_bif.text(
        0.005,
        0.90,
        f'({ascii_lowercase[2]})',
        bbox=bbox,
        transform=ax_bif.transAxes
    )

    ax_bif.set_xlim(a_min, a_max)
    ax_bif.set_ylim(-2.05, 3.)
    ax_bif.grid(alpha=0.1)
    ax_bif.set_xlabel(r'$a$')

    if df_metrics is not None:
        return fig, ax_bif, ax_esc

    return fig, ax_bif


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':

    plot_bifurcation(
        df_states,
        df_sync,
        df_metrics,
        figsize=(30, 16)
    )

    plot_only_bifurcation(
        df_states,
        # df_metrics,
        figsize=(30, 8)
    )

    plt.savefig(
        f'{OUTPUT_PREFIX}.png',
        dpi=300,
        bbox_inches='tight'
    )

    print(
        f'Salvo: {OUTPUT_PREFIX}.png'
    )