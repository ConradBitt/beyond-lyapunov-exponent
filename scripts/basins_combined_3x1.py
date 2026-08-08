"""
plot_basins_combined.py
-----------------------
Lê os CSVs de bacia e expoente de incerteza já calculados e produz
uma figura de 4 colunas:
  [R1 basin] | [R2 basin] | [R3 basin] | [f(eps) combinado]

Uso:
    python3 plot_basins_combined.py

Dependências: numpy, pandas, matplotlib, pynamicalsys (PlotStyler), utils
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from string import ascii_lowercase
import sys
import os

#  utilitário de leitura de metadados (mesmo usado no basin_analysis.py) 
try:
    from utils import parse_metadata_params
except ImportError:
    def parse_metadata_params(path):
        """
        Lê as linhas de comentário do CSV (prefixo '#') e devolve dict
        com chaves/valores extraídos no formato  '# chave: valor'.
        """
        meta = {}
        with open(path, 'r') as fh:
            for line in fh:
                line = line.strip()
                if not line.startswith('#'):
                    break
                if ':' in line:
                    key, _, val = line.lstrip('#').partition(':')
                    key = key.strip().lower().replace(' ', '_')
                    val = val.strip()
                    try:
                        meta[key] = float(val)
                    except ValueError:
                        meta[key] = val
        return meta

try:
    from pynamicalsys import PlotStyler
    PlotStyler().apply_style()
except ImportError:
    pass  # sem o pynamicalsys apenas ignoramos o estilo customizado

#  Parâmetros dos três regimes
b        = 0.225
delta    = 0.0000#np.float64(1e-15)

runs = [
    dict(label=r'$R1$', a=1.4600, coupling=0.0219, color='#2ca02c', marker='o'),
    dict(label=r'$R2$', a=1.5250,  coupling=0.0219,  color="#1f77b4", marker='s'),
    dict(label=r'$R3$', a=1.5000,  coupling=0.0219,  color='#d62728', marker='^'),
    # color='#d62728' vermelho
    # color='#1f77b4' azul 
    # color='#2ca02c' verde
]

data_dir_basin   = '../data/basins'
data_dir_uncert  = '../data/uncertainty'

#  Helpers de caminhos (mesmo padrão do basin_analysis.py) 
def basin_path(a, b, eps, delta):
    return os.path.join(
        data_dir_basin,
        # data/basins/basin_a1.4600_b0.225_sigma0.0219.csv
        f'basin_labels_a{a:.4f}_b{b:.4f}_eps{eps:.4f}_delta{delta:.4f}.csv'
    )

def uncert_path(a, b, eps, delta):
    return os.path.join(
        data_dir_uncert,
        f'uncertainty_data_a{a:.4f}_b{b:.4f}_eps{eps:.4f}_delta{delta:.4f}.csv'
    )

#  Leitura dos dados
def load_basin(a, coupling):
    path = basin_path(a, b, coupling, delta)
    meta = parse_metadata_params(path)

    grid_nx = int(meta.get('grid_nx', meta.get('grid_size', 1000)))
    x_min   = float(meta.get('x_min', -2.0))
    x_max   = float(meta.get('x_max',  2.0))
    y_min   = float(meta.get('y_min', -2.0))
    y_max   = float(meta.get('y_max',  2.0))

    raw = pd.read_csv(path, comment='#', header=0).to_numpy()
    # colunas esperadas: x, y, label  (shape N²×3)
    labels = raw[:, 2].reshape((grid_nx, grid_nx)).astype(np.int32).T
    return labels, x_min, x_max, y_min, y_max, meta


def load_uncertainty(a, coupling):
    path = uncert_path(a, b, coupling, delta)
    meta = parse_metadata_params(path)

    df = pd.read_csv(path, comment='#', header=0)
    epsilons = df['epsilon'].to_numpy(dtype=float)
    fracs    = df['fraction'].to_numpy(dtype=float)

    # Refit no-log (para consistência, mesmo que o CSV já guarde o alpha)
    valid = (fracs > 0) & np.isfinite(fracs) & np.isfinite(epsilons)
    if valid.sum() >= 3:
        log_eps  = np.log10(epsilons[valid])
        log_f    = np.log10(fracs[valid])
        coeffs   = np.polyfit(log_eps, log_f, 1)
        alpha    = coeffs[0]
        log_C    = coeffs[1]
        log_pred = alpha * log_eps + log_C
        ss_res   = np.sum((log_f - log_pred) ** 2)
        ss_tot   = np.sum((log_f - log_f.mean()) ** 2)
        r2       = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    else:
        alpha, log_C, r2 = np.nan, np.nan, np.nan

    sb  = float(meta.get('sb',  np.nan))
    sbb = float(meta.get('sbb', np.nan))

    return epsilons, fracs, alpha, log_C, r2, sb, sbb

# Configuração Geral da Figura (GridSpec)
cm = 1 / 2.54
fig_w, fig_h = 30 * cm, 15 * cm

fig = plt.figure(figsize=(fig_w, fig_h))
gs = fig.add_gridspec(nrows=3, ncols=2, width_ratios=[1, 1.8], hspace=0.05, wspace=0.01)

ax_basins = [fig.add_subplot(gs[i, 0]) for i in range(3)]
ax_uncert = fig.add_subplot(gs[:, 1])

# ==============================================================================
# PAINÉIS (a, b, c): BACIAS DE ATRAÇÃO
# ==============================================================================
all_fracs = []

for idx, run in enumerate(runs):
    ax = ax_basins[idx]
    a, eps = run['a'], run['coupling']

    # Tenta carregar dados
    try:
        labels, x_min, x_max, y_min, y_max, _ = load_basin(a, eps)
        epsilons, fracs, alpha, log_C, r2, sb, sbb = load_uncertainty(a, eps)
    except FileNotFoundError as e:
        print(f"[AVISO] Faltando dados para {run['label']}: {e}")
        ax.set_visible(False)
        continue
    
    # Guarda os dados na estrutura do regime para usar no painel (d)
    run.update({'epsilons': epsilons, 'fracs': fracs, 'alpha': alpha, 'log_C': log_C, 'r2': r2})
    if (fracs > 0).sum(): all_fracs.append(fracs[fracs > 0])

    # Plota bacia
    ax.imshow(labels, extent=(x_min, x_max, y_min, y_max), origin='lower',
              cmap='binary_r', aspect='equal', interpolation='nearest', vmin=0, vmax=1)
    
    # Label do Regime (inferior direito)
    ax.text(0.95, 0.95, run['label'], color=run['color'], transform=ax.transAxes,
            va='top', ha='right', fontsize=12,
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor='gray', alpha=0.92, linewidth=0.6))
    
    # Formatação dos eixos (compartilhados)
    if idx == 2: ax.set_xlabel(r'$X_{0}^{[0]}$', fontsize=12)
    else: ax.set_xticklabels([])
    
    if idx == 0: ax.set_ylabel(r'$Y_{0}^{[0]}$', fontsize=12, rotation=0)
    else: ax.set_yticklabels([])
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

# ==============================================================================
# PAINEL (d): FRAÇÃO DE INCERTEZA + INSET (ZOOM)
# ==============================================================================
ax = ax_uncert

# Cria o Eixo de Zoom (Inset) posicionado para não tapar as caixas de texto
plot_insert = True
ax_ins = ax.inset_axes([0.55, 0.5, 0.42, 0.32]) if plot_insert else None

# Loop único para plotar tudo no painel principal e no inset
# R1, R2, R3
y_pos = [0.58, 0.68, 0.77]
x_pos = [0.31, 0.67, 0.95]
for i, run in enumerate(runs):
     # Tenta carregar dados
    try:
        _, _, _, _, _, sb, sbb = load_uncertainty(run['a'], run['coupling'])
    except FileNotFoundError as e:
        print(f"[AVISO] Faltando dados para {run['label']}: {e}")
        ax.set_visible(False)
        continue

    eps_r = run.get('epsilons', np.array([]))
    frac_r = run.get('fracs', np.array([]))
    alpha, log_C, r2 = run.get('alpha', np.nan), run.get('log_C', np.nan), run.get('r2', np.nan)

    valid = (frac_r > 0) & np.isfinite(frac_r)
    if not valid.any(): continue

    # 1. Plota os Dados (Marcadores) no gráfico principal
    ax.loglog(eps_r[valid], frac_r[valid], linestyle='none', marker=run['marker'],
              color=run['color'], markeredgecolor='black', markeredgewidth=0.7, # Adicionado contorno preto
              markersize=8, alpha=0.85, label=run['label'], zorder=3)

    # 2. Plota as Linhas de Ajuste e Textos
    if not np.isnan(alpha):
        eps_fit = np.logspace(np.log10(eps_r[valid].min()), np.log10(eps_r[valid].max()), 300)
        f_fit = (10 ** log_C) * eps_fit ** alpha
        
        # Plota linha de ajuste no eixo principal
        ax.loglog(eps_fit, f_fit, color=run['color'], linewidth=1.8, linestyle='--', zorder=2)
        
        # Plota linha de ajuste no eixo INSET (Apenas R2 e R3)
        if plot_insert:
            if run['label'] in [r'$R2$', r'$R3$']:
                ax_ins.plot(eps_fit, f_fit, color=run['color'])# marker=run['marker'],
                    #markeredgecolor='black', markeredgewidth=0.7, # Adicionado contorno preto
                    #markersize=8, alpha=0.85, label=run['label'])

        # 1. Cálculo prévio da razão de entropia
        if not (np.isnan(sb) or np.isnan(sbb)):
            ratio_str = f'{sb/sbb:.4f}' if sbb != 0 else r'$\infty$'
            entropy_txt = f"\n$S_b = {sb:.4f}$\n$S_{{bb}} = {sbb:.4f}$"
        else:
            print('Nenhuma entropia disponível')
            entropy_txt = ""

        # 2. Construção da string final e renderização do texto
        txt = f"{run['label']}\n$\\alpha_{run['label'][-2:-1]}={abs(alpha):.3f}$\n$R^2={r2:.4f}$" + entropy_txt

        ax.text(x_pos[i], y_pos[i], txt, transform=ax.transAxes, fontsize=9, va='top', ha='right',
                color=run['color'], zorder=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                        edgecolor=run['color'], alpha=0.88, linewidth=0.7))


# Formatação do Eixo Principal (ax_uncert)
ax.set_xlabel(r'$\varepsilon$', fontsize=16)
ax.set_ylabel(r'$f(\varepsilon)$', fontsize=16, rotation=0, labelpad=20)
ax.grid(True, which='both', linestyle='-', alpha=0.25)
ax.minorticks_on()
ax.legend(loc='lower left', fontsize=10, framealpha=0.9, handlelength=1.2)

# Ajuste automático do Y limite e Inversão limpa do eixo X
if all_fracs:
    f_all = np.concatenate(all_fracs)
    ax.set_ylim(10 ** (np.log10(f_all.min()) - 0.5), 5)
# Usar set_xlim com (max, min) inverte o eixo X de forma segura
ax.set_xlim(1e-3, 0.5e-7) 

# Formatação do Eixo de Zoom (ax_ins)
if plot_insert:
    ax_ins.set_xscale('log')
    # Foco exclusivo nos epsilons pequenos (passando min, max de forma decrescente para inverter)
    ax_ins.set_xlim(1e-3, 1e-7)
    ax_ins.set_ylim(0.55, 1.05) # Escala Y linear para evidenciar a leve queda
    ax_ins.tick_params(axis='both', which='both', labelsize=8)
    ax_ins.grid(True, which="both", ls=":", alpha=0.5)

# Desenha as linhas conectando a área do zoom
# ax.indicate_inset_zoom(ax_ins, edgecolor="gray", alpha=0.8, linewidth=1.5)

# ==============================================================================
# FINALIZAÇÃO (Letras a, b, c, d) E SALVAMENTO
# ==============================================================================
bbox_letter = dict(facecolor='white', alpha=0.80, linewidth=0.0, pad=1.5)
for i, ax_i in enumerate(ax_basins + [ax_uncert]):
    ax_i.text(0.04, 0.95, f'({ascii_lowercase[i]})',
              bbox=bbox_letter, transform=ax_i.transAxes, fontsize=14, va='top')

out_png = '../figures/basins_combined_R1R2R3.png'
plt.savefig(out_png, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figura gerada com sucesso: {out_png}")