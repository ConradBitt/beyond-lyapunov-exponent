"""
Utilitários de I/O: salva e lê CSVs com preâmbulo de metadados.
"""
import os
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def salvar_csv_com_metadados(
    df: pd.DataFrame,
    caminho: str,
    descricao: str = "",
    autor: str = "",
    parametros: Optional[Dict[str, Any]] = None,
    dicionario_variaveis: Optional[Dict[str, str]] = None,
    metadados_extras: Optional[Dict[str, Any]] = None,
    **kwargs_csv,
) -> int:
    """
    Salva `df` em CSV precedido de um bloco de comentários '#' com metadados.

    Retorna o número de linhas de metadados escritas (útil para `skiprows`).
    """
    os.makedirs(os.path.dirname(caminho) if os.path.dirname(caminho) else ".", exist_ok=True)
    linhas: list[str] = []

    def _add(chave: str, valor: Any) -> None:
        linhas.append(f"# {chave}: {valor}\n")

    _add("descricao", descricao)
    _add("autor", autor)
    _add("data_geracao", datetime.now().isoformat(timespec="seconds"))

    if parametros:
        linhas.append("# --- parametros ---\n")
        for k, v in parametros.items():
            _add(k, v)

    if dicionario_variaveis:
        linhas.append("# --- variaveis ---\n")
        for k, v in dicionario_variaveis.items():
            _add(k, v)

    if metadados_extras:
        linhas.append("# --- extras ---\n")
        for k, v in metadados_extras.items():
            _add(k, v)

    n_meta = len(linhas)
    with open(caminho, "w", encoding="utf-8") as f:
        f.writelines(linhas)
        df.to_csv(f, index=False, **kwargs_csv)

    return n_meta


def parse_metadata_params(caminho: str) -> Dict[str, Any]:
    """
    Lê o preâmbulo de comentários de um CSV salvo por `salvar_csv_com_metadados`
    e devolve um dicionário {chave: valor} com conversão automática de tipo.
    """
    meta: Dict[str, Any] = {}
    with open(caminho, "r", encoding="utf-8") as f:
        for line in f:
            if not line.startswith("#"):
                break
            line = line[1:].strip()
            if ":" not in line or line.startswith("-"):
                continue
            chave, _, valor = line.partition(":")
            chave = chave.strip().lower().replace(" ", "_")
            valor = valor.strip()
            # tentativa de conversão numérica
            for conv in (int, float):
                try:
                    valor = conv(valor)  # type: ignore
                    break
                except ValueError:
                    pass
            meta[chave] = valor
    return meta


def salvar_basin_labels_csv(basin_labels, x_vals, y_vals, parameters, 
                            caminho='output/basin_labels.csv',
                            autor="", descricao="",
                            tempo_calculo=None):
    """
    Converte basin_labels para formato long (melted) e salva com metadados.
    
    Parâmetros:
    -----------
    basin_labels : np.ndarray (n_y, n_x)
        Array 2D com os labels das bacias
    x_vals : np.ndarray (n_x,)
        Valores de x (colunas)
    y_vals : np.ndarray (n_y,)
        Valores de y (linhas)
    parameters : np.ndarray
        Parâmetros [a, b, epsilon]
    caminho : str
        Caminho do arquivo CSV
    autor : str
        Nome do autor
    tempo_calculo : float
        Tempo de cálculo em segundos
    """
    # Criar meshgrid para fazer o melt
    X, Y = np.meshgrid(x_vals, y_vals)
    
    # Flatten arrays para formato long
    x_flat = X.flatten()
    y_flat = Y.flatten()
    label_flat = basin_labels.flatten()
    
    # Criar DataFrame
    df = pd.DataFrame({
        'x': x_flat,
        'y': y_flat,
        'label': label_flat
    })
    
    # Descrição
    n_pontos = len(df)
    n_b0 = np.sum(basin_labels == 0)
    n_b1 = np.sum(basin_labels == 1)
    n_indet = np.sum(basin_labels == -1)
    
    # Parâmetros
    params_dict = {
        'a': float(parameters[0]),
        'b': float(parameters[1]),
        'epsilon_coupling': float(parameters[2]),
        'max_iter': 1000,
        'transient': 500,
        'grid_nx': len(x_vals),
        'grid_ny': len(y_vals)
    }
    
    # Dicionário de variáveis
    dic_vars = {
        'x': 'Coordenada x_0',
        'y': 'Coordenada y_0',
        'label': 'Label da bacia'
    }
    
    # Metadados extras
    metadados_extras = {
        
    }
    
    if tempo_calculo is not None:
        metadados_extras['tempo_calculo_s'] = f"{tempo_calculo:.2f}"
    
    # Salvar usando a função do utils
    n_linhas = salvar_csv_com_metadados(
        df=df,
        caminho=caminho,
        descricao=descricao,
        autor=autor,
        parametros=params_dict,
        dicionario_variaveis=dic_vars,
        metadados_extras=metadados_extras
    )
    
    return n_linhas

def extrair_metadados(caminho_arquivo):
    metadados = {}
    secao_atual = None
    conteudo_secao = []

    with open(caminho_arquivo, 'r', encoding='utf-8') as f:
        for linha in f:
            linha = linha.strip()
            
            # Pula linhas de separação ou vazias
            if not linha.startswith('#') or '===' in linha:
                continue
            
            # Limpa o prefixo '#' e espaços extras
            conteudo = linha.lstrip('#').strip()
            if not conteudo: continue

            # Identifica se a linha é um título de seção (termina com ':')
            if conteudo.endswith(':'):
                # Se já estávamos em uma seção, salva o que foi acumulado
                if secao_atual:
                    metadados[secao_atual] = "\n".join(conteudo_secao)
                
                secao_atual = conteudo[:-1].lower() # Ex: "parâmetros do sistema"
                conteudo_secao = []
            else:
                # Se estivermos dentro de uma seção, adiciona a linha à lista
                if secao_atual:
                    conteudo_secao.append(conteudo)

        # Salva a última seção processada
        if secao_atual:
            metadados[secao_atual] = "\n".join(conteudo_secao)

    return metadados


def salvar_uncertainty_csv(epsilons, fractions, alpha, fit_info, parameters,
                           caminho='output/uncertainty_data.csv',
                           autor="",
                           direction='both',
                           n_samples=None,
                           grid_size=None):
    """
    Salva dados de incerteza em CSV com metadados, incluindo fit_info.
    
    Parâmetros:
    -----------
    epsilons : np.ndarray
        Valores de epsilon testados
    fractions : np.ndarray
        Frações de incerteza f(epsilon)
    alpha : float
        Expoente do ajuste
    fit_info : dict
        Dicionário com informações do ajuste
    parameters : np.ndarray
        Parâmetros [a, b, epsilon]
    caminho : str
        Caminho do arquivo CSV
    autor : str
        Nome do autor
    direction : str
        Direção das perturbações
    n_samples : int ou None
        Número de amostras usadas
    grid_size : int ou None
        Tamanho do grid
    """
    # Criar DataFrame
    df = pd.DataFrame({
        'epsilon': epsilons,
        'fraction': fractions
    })
    
    # Descrição com informações do ajuste
    descricao = f"""Fração de incerteza f(epsilon) vs tamanho da perturbação epsilon
Análise da fronteira das bacias de atração do mapa de Hénon acoplado 4D

AJUSTE DE LEI DE POTÊNCIA: f(ε) ~ C·ε^alpha
  alpha (expoente): {fit_info['alpha']:.6f}
  Dimensão fractal: d = 2 - alpha = {fit_info['dimension']:.6f}
  R² (qualidade do ajuste): {fit_info['r_squared']:.6f}
  log₁₀(C): {fit_info['log_C']:.6f}
  Pontos válidos no ajuste: {fit_info['valid_points']}

Interpretação física:
  - alpha ≈ 0: fronteira suave, dimensão fractal ≈ 2 (área)
  - alpha ≈ 1: fronteira muito fractal, dimensão ≈ 1 (curva)
  - Quanto menor alpha, mais complexa e fractal é a fronteira entre bacias"""
    
    # Parâmetros
    params_dict = {
        'a': float(parameters[0]),
        'b': float(parameters[1]),
        'epsilon_coupling': float(parameters[2]),
        'direction': direction,
        'n_samples': n_samples if n_samples else 'grid completo',
        'n_epsilons': len(epsilons),
        'epsilon_min': float(epsilons.min()),
        'epsilon_max': float(epsilons.max()),
        'max_iter': 1000,
        'transient': 500
    }
    
    if grid_size:
        params_dict['grid_size'] = f"{grid_size}x{grid_size}"
    
    # Dicionário de variáveis
    dic_vars = {
        'epsilon': 'Tamanho da perturbação aplicada às condições iniciais (ε)',
        'fraction': 'Fração de pontos incertos: f(ε) = N_incerto / N_total'
    }
    
    # Metadados extras - incluindo TODOS os dados do fit_info
    metadados_extras = {
        'mapa': 'coupled_henon',
        'dimensao': '4D',
        'fit_alpha': fit_info['alpha'],
        'fit_dimension': fit_info['dimension'],
        'fit_r_squared': fit_info['r_squared'],
        'fit_log_C': fit_info['log_C'],
        'fit_valid_points': fit_info['valid_points']
    }

    if 'sb' in list(fit_info.keys()):
        metadados_extras['sb'] = fit_info['sb']
    if 'sbb' in list(fit_info.keys()):
        metadados_extras['sbb'] = fit_info['sbb']
    
    # Salvar usando a função do utils
    n_linhas = salvar_csv_com_metadados(
        df=df,
        caminho=caminho,
        descricao=descricao,
        autor=autor,
        parametros=params_dict,
        dicionario_variaveis=dic_vars,
        metadados_extras=metadados_extras
    )
    
    return n_linhas
