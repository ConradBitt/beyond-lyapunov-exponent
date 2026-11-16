"""
CUDA Parameter Space for Coupled Hénon Maps
===========================================

Calcula:
    - Maior expoente de Lyapunov (MLE)
    - Expoente transversal (TLE)
    - Fração positiva f+
    - Tempo de escape
    - Período da órbita

Utiliza:
    - numba.cuda
    - numpy
    - matplotlib
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from numba import cuda
from numba import njit
from numba import prange


# ============================================================
# CONFIGURAÇÕES
# ============================================================
EPSILON = 0.0219

A_VALS = np.linspace(1.450, 1.650, 1000)
B_VALS = np.linspace(0.200, 0.230, 1000)
B_FIX = 0.225
N_ITER = 60_000
TRANSIENT = 50_000
TOTAL_TIME = int(1e6)
WINDOW = 200

ESCAPE_RADIUS = 10

DELTA = 1e-15

STATE0 = np.array([
    -1.0,
    -1.0,
    -1.0 + DELTA,
    -1.0 + DELTA
], dtype=np.float64)

MAX_PERIOD = 16

OUTPUT_PREFIX = f"teste_space_param"
RECALCULATE = True
if RECALCULATE:
    # ============================================================
    # EVOLUÇÃO DO SISTEMA 4D
    # ============================================================

    @cuda.jit(device=True)
    def henon_step(x0, y0, x1, y1, a, b, sigma):
        """
        Evolui um passo do sistema acoplado 4D.
        xn_i = a - xi² + b*yi + sigma*x_j
        yn_i = xi
        """
        xn0 = a - x0*x0 + b*y0 + sigma*x1
        yn0 = x0

        xn1 = a - x1*x1 + b*y1 + sigma*x0
        yn1 = x1

        return xn0, yn0, xn1, yn1


    # ============================================================
    # DINÂMICA NO MANIFOLD SINCRONIZADO
    # ============================================================

    @cuda.jit(device=True)
    def henon_sync_step(xs, ys, a, b, sigma):
        """
        Dinâmica no manifold sincronizado (x0=x1=xs, y0=y1=ys).

        No manifold sincronizado o sistema 4D colapsa para:
            xs_{n+1} = a - xs² + b*ys + sigma*xs
            ys_{n+1} = xs

        NOTA: sigma*xs é correto pois x1=xs no manifold.
        """
        xsn = a - xs*xs + b*ys + sigma*xs
        ysn = xs
        return xsn, ysn


    # ============================================================
    # KERNEL PRINCIPAL  (CORRIGIDO)
    # ============================================================

    @cuda.jit
    def parameter_space_kernel(
        a_vals,
        b_vals,
        mle_grid,
        tle_grid,
        frac_grid,
        esc_grid,
        sigma,
        n_iter,
        transient,
        window,
        escape_radius
    ):
        """
        Kernel CUDA principal.

        CORREÇÕES APLICADAS
        -------------------
        1. Ordem do loop: todos os jacobianos e acumulações são
           calculados com o estado do instante t; só ao final do
           corpo do loop os estados são avançados para t+1.
           Isso garante que xs e (x0,x1) estejam sempre no mesmo
           instante de tempo quando a jacobiana transversal é
           avaliada.

        2. Cálculo de f⁺ alinhado com a definição dos artigos:
           a fração positiva F é calculada sobre expoentes de
           tempo finito τ=window, acumulando log(normt) em
           segmentos consecutivos não sobrepostos e verificando
           se a média do segmento (= expoente de tempo-τ) é
           positiva. Isso reproduz Eq.(22) de Viana & Grebogi
           (2000) e Eq.(16) de Czajkowski & Viana (2024).
        """

        idx = cuda.grid(1)

        nA = a_vals.shape[0]
        nB = b_vals.shape[0]

        total = nA * nB

        if idx >= total:
            return

        i = idx // nB
        j = idx % nB

        a = a_vals[i]
        b = b_vals[j]

        # =====================================================
        # CONDIÇÕES INICIAIS — sistema 4D
        # =====================================================
        x0 = -1.0
        y0 = -1.0

        x1 = -1.0 + 1e-15
        y1 = -1.0 + 1e-15

        # =====================================================
        # CONDIÇÕES INICIAIS — manifold sincronizado 2D
        # Inicia no mesmo ponto que (x0, y0) para que
        # xs e x0 estejam sincronizados no t=0.
        # =====================================================
        xs = -1.0
        ys = -1.0

        # =====================================================
        # VETOR TANGENTE 4D (para o MLE)
        # =====================================================
        v0 = 1.0
        v1 = 0.0
        v2 = 0.0
        v3 = 0.0

        # =====================================================
        # VETOR TRANSVERSAL 2D (para o TLE)
        # =====================================================
        ut0 = 1.0
        ut1 = 0.0

        # =====================================================
        # ACUMULADORES
        # =====================================================
        mle_sum = 0.0
        tle_sum = 0.0

        # Segmento atual para f⁺ (expoente de tempo-τ)
        local_tle  = 0.0   # soma de log(normt) no segmento
        local_count = 0    # iterações no segmento atual

        positive_count = 0
        total_windows  = 0

        escaped  = False
        esc_time = n_iter

        # =====================================================
        # LOOP TEMPORAL
        # =====================================================
        for t in range(n_iter):

            # -------------------------------------------------
            # BLOCO 1 — Jacobiana 4D com estado ATUAL (x0, x1)
            # -------------------------------------------------
            j00 = -2.0*x0
            j01 = b
            j02 = sigma
            j03 = 0.0

            j10 = 1.0
            j11 = 0.0
            j12 = 0.0
            j13 = 0.0

            j20 = sigma
            j21 = 0.0
            j22 = -2.0*x1
            j23 = b

            j30 = 0.0
            j31 = 0.0
            j32 = 1.0
            j33 = 0.0

            # -------------------------------------------------
            # BLOCO 2 — Evolução do vetor tangente 4D
            # -------------------------------------------------
            nv0 = j00*v0 + j01*v1 + j02*v2 + j03*v3
            nv1 = j10*v0 + j11*v1 + j12*v2 + j13*v3
            nv2 = j20*v0 + j21*v1 + j22*v2 + j23*v3
            nv3 = j30*v0 + j31*v1 + j32*v2 + j33*v3

            normv = math.sqrt(
                nv0*nv0 + nv1*nv1 + nv2*nv2 + nv3*nv3
            )
            if normv == 0.0:
                normv = 1e-16

            v0 = nv0 / normv
            v1 = nv1 / normv
            v2 = nv2 / normv
            v3 = nv3 / normv

            # -------------------------------------------------
            # BLOCO 3 — Jacobiana transversal com xs ATUAL
            #
            # CORREÇÃO: xs é avaliado ANTES de ser atualizado,
            # garantindo consistência temporal com (x0, x1).
            #
            # A jacobiana da direção transversal δv = x0 - x1
            # no manifold sincronizado (x0=x1=xs) é:
            #   ∂(xn0-xn1)/∂(x0-x1)|_{x0=x1=xs} = -2*xs - sigma
            #   ∂(xn0-xn1)/∂(y0-y1)|_{x0=x1=xs} = b
            #   ∂(yn0-yn1)/∂(x0-x1)|_{x0=x1=xs} = 1
            #   ∂(yn0-yn1)/∂(y0-y1)|_{x0=x1=xs} = 0
            # -------------------------------------------------
            jt00 = -2.0*xs - sigma
            jt01 = b
            jt10 = 1.0
            jt11 = 0.0

            # -------------------------------------------------
            # BLOCO 4 — Evolução do vetor transversal
            # -------------------------------------------------
            nut0 = jt00*ut0 + jt01*ut1
            nut1 = jt10*ut0 + jt11*ut1

            normt = math.sqrt(nut0*nut0 + nut1*nut1)
            if normt == 0.0:
                normt = 1e-16

            ut0 = nut0 / normt
            ut1 = nut1 / normt

            # -------------------------------------------------
            # BLOCO 5 — Acumulação (apenas após o transiente)
            # -------------------------------------------------
            if t >= transient:
                lv = math.log(normv)
                lt = math.log(normt)

                mle_sum += lv
                tle_sum += lt

                # ---- f⁺: expoente de tempo-τ ---------------
                # Acumula log(normt) no segmento atual.
                # Quando o segmento completa `window` iterações,
                # o expoente de tempo-τ do segmento é
                #   λ̃⊥(τ) = local_tle / window
                # Contamos quantos segmentos têm λ̃⊥(τ) > 0,
                # reproduzindo F = ∫₀^∞ P(λ̃⊥(τ)) dλ̃⊥(τ).
                local_tle   += lt
                local_count += 1

                if local_count >= window:
                    # expoente de tempo finito do segmento
                    finite_time_tle = local_tle / window

                    if finite_time_tle > 0.0:
                        positive_count += 1

                    total_windows += 1

                    # reinicia segmento
                    local_tle   = 0.0
                    local_count = 0

            # -------------------------------------------------
            # BLOCO 6 — Avança o sistema 4D para t+1
            # CORREÇÃO: feito DEPOIS de todos os cálculos acima.
            # -------------------------------------------------
            x0, y0, x1, y1 = henon_step(
                x0, y0, x1, y1, a, b, sigma
            )

            # -------------------------------------------------
            # BLOCO 7 — Avança o manifold sincronizado para t+1
            # CORREÇÃO: feito no MESMO passo que o bloco 6,
            # mantendo xs e (x0,x1) sempre no mesmo instante.
            # -------------------------------------------------
            xs, ys = henon_sync_step(xs, ys, a, b, sigma)

            # -------------------------------------------------
            # BLOCO 8 — Verificação de escape
            # -------------------------------------------------
            if (
                abs(x0) > escape_radius or
                abs(y0) > escape_radius or
                abs(x1) > escape_radius or
                abs(y1) > escape_radius
            ):
                escaped  = True
                esc_time = t
                break

        # =====================================================
        # RESULTADOS
        # =====================================================
        if escaped:
            mle_grid[i, j]  = math.nan
            tle_grid[i, j]  = math.nan
            frac_grid[i, j] = math.nan
            esc_grid[i, j]  = esc_time

        else:
            denom = n_iter - transient

            mle_grid[i, j] = mle_sum / denom
            tle_grid[i, j] = tle_sum / denom

            if total_windows > 0:
                frac_grid[i, j] = positive_count / total_windows
            else:
                frac_grid[i, j] = 0.0

            esc_grid[i, j] = math.nan


    # ============================================================
    # PERIOD DETECTION CPU  (inalterado)
    # ============================================================

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

        x1 = -1.0 + 1e-15
        y1 = -1.0 + 1e-15

        history = np.zeros((max_period, 4))

        idx = 0

        for t in range(n_iter):

            xn0 = a - x0*x0 + b*y0 + sigma*x1
            yn0 = x0

            xn1 = a - x1*x1 + b*y1 + sigma*x0
            yn1 = x1

            x0, y0, x1, y1 = xn0, yn0, xn1, yn1

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

                d0 = abs(history[k,0] - history[k+p,0])
                d1 = abs(history[k,1] - history[k+p,1])
                d2 = abs(history[k,2] - history[k+p,2])
                d3 = abs(history[k,3] - history[k+p,3])

                if d0 > tol or d1 > tol or d2 > tol or d3 > tol:
                    ok = False
                    break

            if ok:
                return p

        return -1


    @njit(parallel=True)
    def compute_periods(
        a_vals,
        b_vals,
        mle_grid,
        sigma,
        n_iter,
        transient,
        max_period
    ):

        nA = len(a_vals)
        nB = len(b_vals)

        out = np.full((nA, nB), -1)

        for i in prange(nA):

            for j in range(nB):

                mle = mle_grid[i, j]

                if np.isnan(mle):
                    continue

                if mle < 0.0:

                    out[i, j] = detect_period(
                        a_vals[i],
                        b_vals[j],
                        sigma,
                        n_iter,
                        transient,
                        max_period
                    )

        return out


    # ============================================================
    # EXECUÇÃO CUDA
    # ============================================================

    print("Executando CUDA...")

    nA = len(A_VALS)
    nB = len(B_VALS)

    mle_grid  = np.zeros((nA, nB), dtype=np.float64)
    tle_grid  = np.zeros((nA, nB), dtype=np.float64)
    frac_grid = np.zeros((nA, nB), dtype=np.float64)
    esc_grid  = np.zeros((nA, nB), dtype=np.float64)

    d_a    = cuda.to_device(A_VALS)
    d_b    = cuda.to_device(B_VALS)
    d_mle  = cuda.to_device(mle_grid)
    d_tle  = cuda.to_device(tle_grid)
    d_frac = cuda.to_device(frac_grid)
    d_esc  = cuda.to_device(esc_grid)

    threads = 256
    blocks  = (nA*nB + threads - 1) // threads

    parameter_space_kernel[blocks, threads](
        d_a,
        d_b,
        d_mle,
        d_tle,
        d_frac,
        d_esc,
        EPSILON,
        N_ITER,
        TRANSIENT,
        WINDOW,
        ESCAPE_RADIUS
    )

    cuda.synchronize()

    mle_grid  = d_mle.copy_to_host()
    tle_grid  = d_tle.copy_to_host()
    frac_grid = d_frac.copy_to_host()
    esc_grid  = d_esc.copy_to_host()

    esc_grid[np.isnan(esc_grid)] = TOTAL_TIME

    print("CUDA finalizado.")

    # ============================================================
    # PERÍODOS
    # ============================================================

    print("Calculando períodos...")

    period_grid = compute_periods(
        A_VALS,
        B_VALS,
        mle_grid,
        EPSILON,
        N_ITER,
        TRANSIENT,
        MAX_PERIOD
    )

    print("Períodos finalizados.")


    # ============================================================
    # SALVAR CSV
    # ============================================================

    aa, bb = np.meshgrid(A_VALS, B_VALS, indexing='ij')

    df = pd.DataFrame({
        "a":           aa.ravel(),
        "b":           bb.ravel(),
        "mle":         mle_grid.ravel(),
        "tle":         tle_grid.ravel(),
        "frac_pos":    frac_grid.ravel(),
        "escape_time": esc_grid.ravel(),
        "period":      period_grid.ravel()
    })

    csv_name = f"{OUTPUT_PREFIX}.csv"
    df.to_csv(csv_name, index=False)
    print(f"CSV salvo: {csv_name}")

else:
    # -----------------------------------------------------------------------
    # Read data
    # -----------------------------------------------------------------------
    print("Lendo os dados...")
    df = pd.read_csv(f"{OUTPUT_PREFIX}.csv", comment='#')
    A_VALS = np.unique(df['a'].values)
    B_VALS = np.unique(df['b'].values)
    mle_grid  = df['mle'].values.reshape(len(A_VALS), len(B_VALS))
    tle_grid  = df['tle'].values.reshape(len(A_VALS), len(B_VALS))
    period_grid = df['period'].values.reshape(len(A_VALS), len(B_VALS))
    esc_grid  = df['escape_time'].values.reshape(len(A_VALS), len(B_VALS))
    frac_grid = df['frac_pos'].values.reshape(len(A_VALS), len(B_VALS))

if __name__ == '__main__':
    # -----------------------------------------------------------------------
    # Plot  (inalterado)
    # -----------------------------------------------------------------------
    import os
    import sys
    from string import ascii_lowercase
    from pynamicalsys import PlotStyler

    # Permite executar o script diretamente sem pacote pai.
    # A raiz do repositório fica um nível acima de Revisor7Analysis.
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    # Importa o helper do repositório; não há pacote pai quando o arquivo é
    # executado como script standalone.
    from scripts.utils.plotting import make_period_cmap

    import matplotlib.gridspec as gridspec
    CM = 1 / 2.54

    cmap_period = make_period_cmap(N=9, p_min=5)

    ps = PlotStyler(fontsize=20, ticks_on_all_sides=False)
    ps.apply_style()
    fig = plt.figure(figsize=((35*CM, 10*CM)))
    gs = gridspec.GridSpec(
        1, 3,
        hspace=0.08,
        left=0.05, right=0.97,
        top=0.93,  bottom=0.15,
    )

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharey=ax1)
    ax3 = fig.add_subplot(gs[2], sharey=ax1)
    axes = [ax1, ax2, ax3]

    # cmap_orig = plt.get_cmap('berlin')
    # pontos = [0.0, 0.5 - 0.005, 0.5 - 0.005, 0.5 + 0.005, 0.5 + 0.005, 1.0]
    # cores = [cmap_orig(0.0), cmap_orig(0.495), 'cyan', 'cyan', cmap_orig(0.505), cmap_orig(1.0)]
    # cmap_custom = mcolors.LinearSegmentedColormap.from_list('berlin_cyan', list(zip(pontos, cores)))
    # hm1 = axes[0].pcolor(A_VALS, B_VALS, mle_grid.T, vmin=-0.501, vmax=0.501, cmap=cmap_custom)

    # 3. Usa no seu gráfico
    hm1 = axes[0].pcolor(A_VALS, B_VALS, mle_grid.T, vmin=-0.501, vmax=0.501, cmap='berlin')
    plt.colorbar(hm1, aspect=40, pad=0.02, location="top")
    axes[0].set_title(r'$\lambda_1$', loc='left', x=-0.01, y=1.15, pad=10)
    axes[0].set_xlabel(r'$a$')
    axes[0].set_ylabel(r'$b$')

    hm2 = axes[2].pcolor(A_VALS, B_VALS, period_grid.T, vmin=5, vmax=15, cmap=cmap_period.cmap)
    cbar = plt.colorbar(hm2, aspect=40, pad=0.02, location="top")
    bounds = np.linspace(5, 15, 11)
    ticks = (bounds[:-1] + bounds[1:]) / 2
    tick_labels = [f"{int(t)}" for t in ticks]
    tick_labels[-1] = f"${int(bounds[-2] + 0.5)}\\leq $"
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(tick_labels)
    
    # axes[1].set_title(r'$p$', loc='left', x=-0.01, y=1.15, pad=10)
    # axes[1].set_xlabel(r'$a$')

    # hm2 = axes[1].pcolor(A_VALS, B_VALS, tle_grid.T, vmin=-0.501, vmax=0.501, cmap='berlin')
    # plt.colorbar(hm2, aspect=40, pad=0.02, location="top")
    # axes[1].set_title(r'$\lambda_\perp$', loc='left', x=-0.01, y=1.15, pad=10)
    # axes[1].set_xlabel(r'$a$')

    hm2 = axes[1].pcolor(A_VALS, B_VALS, frac_grid.T, vmin=0., vmax=1., cmap='nipy_spectral_r')
    plt.colorbar(hm2, aspect=40, pad=0.02, location="top")
    axes[1].set_title(rf'$f_{{{(WINDOW)}}}^+$', loc='left', x=-0.01, y=1.15, pad=10)
    axes[1].set_xlabel(r'$a$')
    
    # escape_plot = np.asarray(esc_grid, dtype=np.float64)
    # hm = axes[2].pcolormesh(
    #     A_VALS,
    #     B_VALS,
    #     escape_plot.T,
    #     cmap='nipy_spectral',
    #     norm=mcolors.LogNorm(vmin=1, vmax=max(TOTAL_TIME, 10)),
    # )
    # plt.colorbar(hm, aspect=40, pad=0.02, location="top")
    # axes[2].set_title(r'$T_{\mathrm{esc}}$', loc='left', x=-0.01, y=1.15, pad=10)
    # axes[2].set_xlabel(r'$a$')

    _bbox = {"facecolor": "w", "alpha": 1., "linewidth": 0.0, "pad": 1}
    if EPSILON == 0.005:
        idx_fig = 0
    elif EPSILON == 0.05:
        idx_fig = 1
    elif EPSILON == 0.1:
        idx_fig = 2
    else:
        idx_fig = 3

    # Pontos de interesse (marcadores)
    # A_BLUE = 1.48 
    # A_GREEN = 1.53
    # A_RED  = 1.545


    scatter_infos = {
        'filled' : {
            'x':1.4510,
            'y':B_FIX,
            's':100,
            'color': "#35CE21",
            'marker':'o'
        },
        'blowout_1' : {
            'x':1.4638527705541107,
            'y':B_FIX,
            's':100,
            'color': "#AA2816",
            'marker':'v'
        },
        'blowout_2' : {
            'x':1.5390326065213041,
            'y':B_FIX,
            's':100,
            'color': "#AA2816",
            'marker':'^'
        },
        # 'escape' : {
        #     'x':1.510,
        #     'y':B_FIX,
        #     's':150,
        #     'color': "#F93E25",
        #     'marker':'^'
        # },
        'riddled':{
            'x':1.550,
            'y':B_FIX,
            's':100,
            'color': "#1f77b4",
            'marker': 's'
        },
    }

    extra_markers = [
        # (1.4500, 'lime', 'v', 'p=5'),
        # (1.4825, 'magenta', '^', 'chaos'),
        # (1.5350, 'red', '<', 'int. crisis'),
        # (1.5655, 'blue', '>', 'ext. crisis'),
        # (1.6000, 'goldenrod', '*', 'escape')
    ]

    for i, ax in enumerate(axes):
        ax.text(0.01, 0.92, rf"({ascii_lowercase[i]})", bbox=_bbox, transform=ax.transAxes)
        ax.set_xlim(A_VALS.min(), A_VALS.max())
        ax.set_ylim(B_VALS.min(), B_VALS.max())
        ax.tick_params(axis='both', which='major', pad=10)
        if scatter_infos:
            ax.axhline(
                scatter_infos['filled']['y'],
                color='gray',
                ls='--',
                lw=1.1,
                zorder=1
            )

        if extra_markers:
            ax.axhline(
                0.225,
                color='gray',
                ls='--',
                lw=1.1,
                zorder=1
            )
            for x, color, marker, label in extra_markers:
                ax.scatter(
                    x, 0.225,
                    c=color,
                    marker=marker,
                    s=70,
                    edgecolors='black',
                    linewidth=0.5,
                    zorder=6,
                    label=label
                )
                # ax.annotate(
                #     label,
                #     (x, 0.225),
                #     xytext=(4, 4),
                #     textcoords='offset points',
                #     fontsize=9,
                #     color='black',
                #     zorder=7
                # )

        for key, info in scatter_infos.items():
            print(key, info)
            ax.scatter(info['x'], info['y'], c=info['color'], s=info['s'], marker=info['marker'], edgecolors='black', linewidth=0.5, zorder=5)
    axes[1].tick_params(axis='y', labelleft=False)
    axes[2].tick_params(axis='y', labelleft=False)
    plt.savefig(f'{OUTPUT_PREFIX}.png', dpi=300, bbox_inches='tight')
    print(f"FIGURA salva: {OUTPUT_PREFIX}.png")