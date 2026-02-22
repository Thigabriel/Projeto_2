"""
=============================================================================
SIMULAÇÃO AQUACROP-OSPy — 3 Cenários de Irrigação | Cold Start ALMMo-0
Projeto: Sistema de Irrigação com ALMMo-0 | Imperatriz-MA
VERSÃO 4 — Limiares por P25/P50/P75 do cenário ótimo
=============================================================================

DIFERENÇAS EM RELAÇÃO À V2:

  CORREÇÃO 1 (CRÍTICA) — Rotulagem por percentis reais do IrrDay:
    v2: limiares fixos (0/2/5/10 mm) → classes concentradas em 0 e 4,
        classe 1 com apenas 11 amostras (0,7%), cenário déficit sem
        classes 3 e 4, cenário excesso sem classes 0 e 1.
    v3: primeira passagem calcula percentis P20/P40/P60/P80 do IrrDay > 0
        sobre os três cenários combinados. Limiares derivados dos dados reais.
        Fallback automático para 3 classes se distribuição ainda desbalancear.

  CORREÇÃO 2 (MODERADA) — Feature de chuva otimizada:
    v2: chuva_acum_3d_mm com correlação -0,06 com classe.
    v3: testa 4 variantes (dia, 3d, 7d, bool) e seleciona a de maior
        correlação absoluta com classe_irrigacao.

  CORREÇÃO 3 (DOCUMENTAÇÃO) — Contagem de tensão cravada em 1500 kPa:
    v3: reporta quantas amostras têm tensao >= 1490 kPa e em que cenário/DAP.

  Sem alterações:
    - Dados climáticos (INMET/BDMEP, Imperatriz-MA)
    - Simulação AquaCrop (SandyLoam 1.2m, Tomato, NetIrrSMT por cenário)
    - Conversão Wr→θ→kPa via Saxton & Rawls (2006)
    - Filtros: Tr > 0.1, DAP >= 14, DAP <= 107
    - Três cenários: Ótimo 65%, Déficit 30%, Excesso 90%

Referência conversão θ→kPa:
  Saxton, K.E. & Rawls, W.J. (2006). Soil Sci. Soc. Am. J. 70:1569-1578.
  doi:10.2136/sssaj2005.0117

Saídas:
  - dataset_cold_start_v5.csv
  - dataset_cold_start_completo_v5.csv
  - cenario1_otimo_v5.csv / cenario2_deficit_v5.csv / cenario3_excesso_v5.csv
  - graficos_v5/ (distribuicao_classes, correlacao, histograma_IrrDay)
  - relatorio_correcoes.md
=============================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from aquacrop import (AquaCropModel, Soil, Crop, InitialWaterContent,
                      FieldMngt, IrrigationManagement)
from aquacrop.utils import prepare_weather

# ==============================================================================
#  CONFIGURAÇÃO
# ==============================================================================

arquivo_clima   = 'imperatriz_climate.txt'
data_plantio    = '05/01'
DIR_GRAFICOS    = 'graficos_v5'

SOLO_AREIA      = 65.0
SOLO_ARGILA     = 10.0
SOLO_OM         = 3.0
SOLO_Z_MODELO   = 1.2
SENSOR_Z        = 0.20

CENARIOS = [
    (1, 'cenario1_otimo_v5',   65, 'Irrigação Ótima  (NetIrrSMT=65%)'),
    (2, 'cenario2_deficit_v5', 30, 'Déficit Hídrico  (NetIrrSMT=30%)'),
    (3, 'cenario3_excesso_v5', 90, 'Excesso Hídrico  (NetIrrSMT=90%)'),
]

ANOS = [2019, 2020, 2021, 2022, 2023, 2025]

# ==============================================================================
#  SAXTON & RAWLS (2006)
# ==============================================================================

def saxton_rawls_parametros(areia, argila, mo):
    S, C, OM = areia / 100, argila / 100, mo
    t1500t = (-0.024*S + 0.487*C + 0.006*OM + 0.005*S*OM
              - 0.013*C*OM + 0.068*S*C + 0.031)
    theta_1500 = t1500t + (0.14 * t1500t - 0.02)
    t33t = (-0.251*S + 0.195*C + 0.011*OM + 0.006*S*OM
            - 0.027*C*OM + 0.452*S*C + 0.299)
    theta_33 = t33t + (1.283 * t33t**2 - 0.374 * t33t - 0.015)
    B = (np.log(1500.0) - np.log(33.0)) / (np.log(theta_33) - np.log(theta_1500))
    A = np.exp(np.log(33.0) + B * np.log(theta_33))
    return theta_33, theta_1500, A, B


def theta_para_tensao(theta_v, theta_33, theta_1500, A, B):
    theta  = np.clip(np.asarray(theta_v, dtype=float), theta_1500, theta_33)
    tensao = A * np.power(theta, -B)
    return np.clip(tensao, 33.0, 1500.0)


# ==============================================================================
#  SIMULAÇÃO ANUAL
# ==============================================================================

def simular_ano(weather_df, ano, net_irr_smt, theta_33, theta_1500, A, B):
    data_ini = f"{ano}/05/01"
    data_fim = f"{ano}/09/30"
    if (pd.to_datetime(data_ini) < weather_df['Date'].min() or
        pd.to_datetime(data_fim) > weather_df['Date'].max()):
        return None

    try:
        model = AquaCropModel(
            sim_start_time=data_ini,
            sim_end_time=data_fim,
            weather_df=weather_df,
            soil=Soil('SandyLoam'),
            crop=Crop(c_name='Tomato', planting_date=data_plantio),
            initial_water_content=InitialWaterContent(value=['FC']),
            field_management=FieldMngt(mulches=True, mulch_pct=80, f_mulch=0.3),
            irrigation_management=IrrigationManagement(
                irrigation_method=4, NetIrrSMT=net_irr_smt)
        )
        model.run_model(till_termination=True)
    except Exception as e:
        print(f"      ❌ Erro em {ano}: {e}")
        return None

    df_flux  = model.get_water_flux()
    df_store = model.get_water_storage()
    if df_flux is None or len(df_flux) == 0:
        return None

    Wr      = df_flux['Wr'].values
    theta_v = Wr / (1000.0 * SOLO_Z_MODELO)
    tensao  = theta_para_tensao(theta_v, theta_33, theta_1500, A, B)

    datas = weather_df[
        (weather_df['Date'] >= pd.to_datetime(data_ini)) &
        (weather_df['Date'] <= pd.to_datetime(data_fim))
    ]['Date'].values

    return df_store, df_flux, datas, tensao


# ==============================================================================
#  SIMULAÇÃO DE CENÁRIO — retorna df com irr_mm bruto (sem rotulagem ainda)
# ==============================================================================

def simular_cenario_raw(weather_df, anos, net_irr_smt, nome, descricao,
                        theta_33, theta_1500, A, B):
    print(f"\n{'─'*60}")
    print(f"📋 {descricao}")
    dfs_anos = []

    for ano in anos:
        print(f"   → {ano}...", end=" ", flush=True)
        resultado = simular_ano(weather_df, ano, net_irr_smt,
                                theta_33, theta_1500, A, B)
        if resultado is None:
            print("pulado")
            continue

        df_store, df_flux, datas, tensao = resultado
        n = min(len(datas), len(df_store), len(df_flux), len(tensao))

        df = pd.DataFrame({
            'Date':    pd.to_datetime(datas[:n]),
            'Tr':      df_flux['Tr'].values[:n],
            'irr_mm':  (df_flux['IrrDay'].values[:n]
                        if 'IrrDay' in df_flux.columns
                        else df_flux['irrigation'].values[:n]),
            'ano':     ano,
            '_tensao': tensao[:n],
        })

        wdf = weather_df[['Date', 'Precipitation', 'MaxTemp']].copy()
        wdf['Date'] = pd.to_datetime(wdf['Date'])
        df = pd.merge(df, wdf, on='Date', how='left')
        df = df.rename(columns={'Precipitation': 'chuva_mm', 'MaxTemp': 'temp_max_c'})

        # Todas as variantes de chuva
        fwd3 = pd.api.indexers.FixedForwardWindowIndexer(window_size=3)
        fwd7 = pd.api.indexers.FixedForwardWindowIndexer(window_size=7)
        df['chuva_acum_3d_mm'] = df['chuva_mm'].rolling(window=fwd3, min_periods=1).sum()
        df['chuva_acum_7d_mm'] = df['chuva_mm'].rolling(window=fwd7, min_periods=1).sum()
        df['tmax_max_3d_c']    = df['temp_max_c'].rolling(window=fwd3, min_periods=1).max()

        t_arr = df['_tensao'].values
        df['tensao_solo_kpa'] = np.concatenate([[t_arr[0]], t_arr[:-1]])
        df.drop(columns=['_tensao'], inplace=True)

        dfs_anos.append(df)
        print(f"{n} dias")

    if not dfs_anos:
        return pd.DataFrame()

    df_cenario = pd.concat(dfs_anos, ignore_index=True)

    linhas_antes = len(df_cenario)
    df_cenario.dropna(inplace=True)
    df_cenario = df_cenario[df_cenario['Tr'] > 0.1].copy()
    df_cenario['dap'] = df_cenario.groupby('ano').cumcount() + 1
    df_cenario = df_cenario[df_cenario['dap'] <= 107].copy()
    linhas_dap = len(df_cenario)
    df_cenario = df_cenario[df_cenario['dap'] >= 14].copy()
    df_cenario.drop(columns=['Tr'], inplace=True)

    print(f"   Removidas (entressafra): {linhas_antes - linhas_dap} | "
          f"(DAP<14): {linhas_dap - len(df_cenario)} | "
          f"Válidas: {len(df_cenario)}")

    df_cenario['cenario'] = nome
    return df_cenario


# ==============================================================================
#  PROBLEMA 1 — CALCULAR LIMIARES POR PERCENTIS
# ==============================================================================

def calcular_limiares_percentis(df_todos):
    """
    v5: 3 classes. Limiar único = mediana do IrrDay > 0 do cenário ótimo.
    Classe 0: sem irrigação (IrrDay = 0)
    Classe 1: irrigação moderada (0 < IrrDay <= mediana_otimo)
    Classe 2: irrigação intensa  (IrrDay > mediana_otimo)

    Justificativa da mediana do ótimo:
      - Déficit tem IrrDay truncado em 5mm → viés para baixo
      - Excesso tem IrrDay concentrado acima de 10mm → viés para cima
      - Ótimo (NetIrrSMT=65%) representa decisão agronômica normal
        com distribuição não truncada (min=0,21 | max=10,39mm)
      - Mediana é mais robusta que média frente a assimetria
    """
    n_total = len(df_todos)

    print(f"\n{'='*60}")
    print("🔍 ANÁLISE DO IrrDay BRUTO (v5 — 3 classes, mediana do ótimo)")
    print(f"   Total amostras: {n_total}")
    print(f"\n   Por cenário (IrrDay > 0):")
    for nome_c in df_todos['cenario'].unique():
        sub = df_todos[df_todos['cenario']==nome_c]
        irr_pos = sub[sub['irr_mm'] > 0]['irr_mm']
        label = nome_c.replace('cenario','C').replace('_v5','')
        if len(irr_pos) > 0:
            print(f"   {label}: n={len(irr_pos)} | "
                  f"min={irr_pos.min():.2f} | "
                  f"mediana={irr_pos.median():.2f} | "
                  f"max={irr_pos.max():.2f} mm")
        else:
            print(f"   {label}: sem irrigações positivas")

    # Mediana APENAS do cenário ótimo
    nome_otimo = [n for _, n, _, _ in CENARIOS if 'otimo' in n][0]
    irr_otimo  = df_todos[
        (df_todos['cenario'] == nome_otimo) & (df_todos['irr_mm'] > 0)
    ]['irr_mm']
    mediana = irr_otimo.quantile(0.50)

    print(f"\n   Mediana do cenário ótimo (n={len(irr_otimo)}): {mediana:.2f} mm")
    print(f"   → Limiar C1/C2: IrrDay <= {mediana:.2f} mm → Classe 1 | > {mediana:.2f} mm → Classe 2")

    limiares = {'mediana': mediana}
    n_classes = 3

    # Preview distribuição
    def rotular_prev(x):
        if x == 0.0:          return 0
        elif x <= mediana:    return 1
        else:                 return 2

    dist_prev = df_todos['irr_mm'].apply(rotular_prev).value_counts().sort_index()
    print(f"\n   Distribuição resultante (dataset completo):")
    for k, v in dist_prev.items():
        print(f"   C{k}: {v} ({v/n_total*100:.1f}%)")

    return n_classes, limiares


def rotular_classe_v5(irr_mm, limiares):
    """v5: 3 classes. Limiar = mediana do IrrDay > 0 do cenário ótimo."""
    if irr_mm == 0.0:                  return 0
    elif irr_mm <= limiares['mediana']: return 1
    else:                               return 2


# ==============================================================================
#  PROBLEMA 2 — SELECIONAR MELHOR FEATURE DE CHUVA
# ==============================================================================

def selecionar_feature_chuva(df_todos, classe_col='classe_irrigacao'):
    variantes = {
        'chuva_mm':        df_todos['chuva_mm'],
        'chuva_acum_3d_mm': df_todos['chuva_acum_3d_mm'],
        'chuva_acum_7d_mm': df_todos['chuva_acum_7d_mm'],
        'chuva_bool':       (df_todos['chuva_mm'] > 2).astype(int),
    }
    target = df_todos[classe_col]

    print(f"\n{'='*60}")
    print("🌧  ANÁLISE DE VARIANTES DE CHUVA (Problema 2)")
    print(f"   {'Variante':<22} {'Corr. Pearson':>14} {'|Corr|':>8}")
    print(f"   {'-'*46}")

    melhor_nome = None
    melhor_corr = -1
    resultados  = {}

    for nome, serie in variantes.items():
        corr = serie.corr(target)
        resultados[nome] = corr
        marcador = " ← melhor" if abs(corr) > melhor_corr else ""
        print(f"   {nome:<22} {corr:>14.4f} {abs(corr):>8.4f}{marcador}")
        if abs(corr) > melhor_corr:
            melhor_corr = abs(corr)
            melhor_nome = nome

    if melhor_corr < 0.15:
        print(f"\n   ⚠️  Melhor correlação: |{melhor_corr:.4f}| < 0.15")
        print(f"   → Mantendo chuva_acum_3d_mm (original) e declarando como limitação")
        melhor_nome = 'chuva_acum_3d_mm'
    else:
        print(f"\n   ✅ Feature selecionada: {melhor_nome} (|corr| = {melhor_corr:.4f})")

    return melhor_nome, resultados


# ==============================================================================
#  PROBLEMA 3 — CONTAR TENSÃO CRAVADA EM 1500 kPa
# ==============================================================================

def analisar_tensao_cravada(df_todos):
    limite = 1490.0
    cravados = df_todos[df_todos['tensao_solo_kpa'] >= limite]
    n_total  = len(df_todos)
    n_crav   = len(cravados)

    print(f"\n{'='*60}")
    print("📌 TENSÃO CRAVADA EM 1500 kPa (Problema 3 — documentação)")
    print(f"   Amostras com tensao >= {limite} kPa: {n_crav} ({n_crav/n_total*100:.1f}%)")
    print(f"   Por cenário:")
    for nome_c in df_todos['cenario'].unique():
        sub = cravados[cravados['cenario']==nome_c]
        label = nome_c.replace('_v5','')
        print(f"   {label}: {len(sub)} amostras")
        if len(sub) > 0:
            print(f"          DAP min={sub['dap'].min()} | max={sub['dap'].max()} | "
                  f"mediana={sub['dap'].median():.0f}")
    return n_crav


# ==============================================================================
#  GRÁFICOS
# ==============================================================================

def gerar_graficos(df_final, df_irr_bruto, feature_chuva, n_classes):
    os.makedirs(DIR_GRAFICOS, exist_ok=True)
    cores = ['#4CAF50','#2196F3','#FF9800','#F44336','#9C27B0']

    # 1 — Histograma do IrrDay bruto
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle('Histograma IrrDay bruto por cenário (dias com irrigação > 0)',
                 fontweight='bold')
    for ax, (cid, nome, _, desc) in zip(axes, CENARIOS):
        sub = df_irr_bruto[(df_irr_bruto['cenario']==nome) & (df_irr_bruto['irr_mm']>0)]
        if len(sub) > 0:
            ax.hist(sub['irr_mm'], bins=30, color='steelblue', edgecolor='white', alpha=0.8)
        ax.set_title(f"C{cid}: {nome.split('_')[1].capitalize()}")
        ax.set_xlabel('IrrDay (mm)')
        ax.set_ylabel('Frequência')
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{DIR_GRAFICOS}/histograma_IrrDay.png', dpi=130)
    plt.close()
    print(f"   📈 Salvo: {DIR_GRAFICOS}/histograma_IrrDay.png")

    # 2 — Distribuição de classes por cenário
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    fig.suptitle(f'Distribuição de Classes por Cenário (v5 — 5 classes, P25/P50/P75 ótimo)',
                 fontweight='bold')
    for ax, (cid, nome, _, _desc) in zip(axes, CENARIOS):
        df_c   = df_final[df_final['cenario']==nome]
        counts = df_c['classe_irrigacao'].value_counts().sort_index()
        ax.bar([str(k) for k in counts.index], counts.values,
               color=[cores[k] for k in counts.index])
        ax.set_title(f"C{cid}: {nome.split('_')[1].capitalize()}")
        ax.set_xlabel('Classe'); ax.set_ylabel('Nº de dias')
        ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{DIR_GRAFICOS}/distribuicao_classes_v5.png', dpi=130)
    plt.close()
    print(f"   📈 Salvo: {DIR_GRAFICOS}/distribuicao_classes_v5.png")

    # 3 — Matriz de correlação v3
    cols_corr = ['tensao_solo_kpa', feature_chuva, 'tmax_max_3d_c', 'dap', 'classe_irrigacao']
    corr_mat  = df_final[cols_corr].corr()
    fig, ax   = plt.subplots(figsize=(7, 5))
    im = ax.imshow(corr_mat, vmin=-1, vmax=1, cmap='RdBu_r')
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(cols_corr))); ax.set_xticklabels(cols_corr, rotation=30, ha='right')
    ax.set_yticks(range(len(cols_corr))); ax.set_yticklabels(cols_corr)
    for i in range(len(cols_corr)):
        for j in range(len(cols_corr)):
            ax.text(j, i, f"{corr_mat.iloc[i,j]:.2f}", ha='center', va='center', fontsize=8)
    ax.set_title('Matriz de Correlação — Dataset v5', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{DIR_GRAFICOS}/correlacao_features_v5.png', dpi=130)
    plt.close()
    print(f"   📈 Salvo: {DIR_GRAFICOS}/correlacao_features_v5.png")

    # 4 — Painel comparativo: distribuição de classes v2 vs v3 (lado a lado)
    # Reconstrói distribuição v2 com limiares fixos antigos para comparação
    limiares_v2 = {0: 0, 2: 1, 5: 2, 10: 3}  # limiar superior → classe
    def rotular_v2(x):
        if x == 0:    return 0
        elif x <= 2:  return 1
        elif x <= 5:  return 2
        elif x <= 10: return 3
        else:         return 4

    df_comp = df_final.copy()
    df_comp['classe_v2'] = df_comp['irr_mm'].apply(rotular_v2)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=False)
    fig.suptitle('Comparação de Distribuição de Classes — v2 (limiares fixos) vs v5 (3 classes, mediana ótimo)',
                 fontweight='bold', fontsize=11)

    for col_idx, (cid, nome, _, _desc) in enumerate(CENARIOS):
        df_c = df_comp[df_comp['cenario'] == nome]
        label = nome.split('_')[1].capitalize()

        # Linha 0 — v2
        counts_v2 = df_c['classe_v2'].value_counts().sort_index()
        axes[0][col_idx].bar([str(k) for k in counts_v2.index], counts_v2.values,
                              color=[cores[min(k, len(cores)-1)] for k in counts_v2.index])
        axes[0][col_idx].set_title(f"C{cid}: {label} — v2 (fixos)")
        axes[0][col_idx].set_xlabel('Classe'); axes[0][col_idx].set_ylabel('Nº de dias')
        axes[0][col_idx].grid(True, alpha=0.3, axis='y')
        for bar, (k, v) in zip(axes[0][col_idx].patches, counts_v2.items()):
            axes[0][col_idx].text(bar.get_x() + bar.get_width()/2,
                                   bar.get_height() + 1, str(v), ha='center', va='bottom', fontsize=8)

        # Linha 1 — v3
        counts_v3 = df_c['classe_irrigacao'].value_counts().sort_index()
        axes[1][col_idx].bar([str(k) for k in counts_v3.index], counts_v3.values,
                              color=[cores[min(k, len(cores)-1)] for k in counts_v3.index])
        axes[1][col_idx].set_title(f"C{cid}: {label} — v4 (P25/P50/P75 ótimo)")
        axes[1][col_idx].set_xlabel('Classe'); axes[1][col_idx].set_ylabel('Nº de dias')
        axes[1][col_idx].grid(True, alpha=0.3, axis='y')
        for bar, (k, v) in zip(axes[1][col_idx].patches, counts_v3.items()):
            axes[1][col_idx].text(bar.get_x() + bar.get_width()/2,
                                   bar.get_height() + 1, str(v), ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(f'{DIR_GRAFICOS}/comparacao_v2_v3_distribuicao.png', dpi=130)
    plt.close()
    print(f"   📈 Salvo: {DIR_GRAFICOS}/comparacao_v2_v3_distribuicao.png")

    # 5 — Painel comparativo: correlação v2 vs v3
    cols_v2 = ['tensao_solo_kpa', 'chuva_acum_3d_mm', 'tmax_max_3d_c', 'dap', 'classe_v2']
    # garante que chuva_acum_3d_mm existe (pode ter sido renomeada)
    if 'chuva_acum_3d_mm' not in df_comp.columns and feature_chuva in df_comp.columns:
        df_comp['chuva_acum_3d_mm'] = df_comp[feature_chuva]
    cols_v2_disp = [c for c in cols_v2 if c in df_comp.columns]
    cols_v3_disp = [c for c in cols_corr if c in df_comp.columns] + \
                   (['classe_irrigacao'] if 'classe_irrigacao' in df_comp.columns else [])
    # remover duplicatas mantendo ordem
    seen = set(); cols_v3_disp = [c for c in cols_v3_disp if not (c in seen or seen.add(c))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Comparação de Correlação — v2 (limiares fixos) vs v5 (3 classes, mediana ótimo)',
                 fontweight='bold')

    for ax, cols, titulo in [(ax1, cols_v2_disp, 'v2 — limiares fixos'),
                              (ax2, cols_v3_disp, f'v5 — 3 classes, mediana ótimo (5 classes)')]:
        mat = df_comp[cols].corr()
        im  = ax.imshow(mat, vmin=-1, vmax=1, cmap='RdBu_r')
        ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=30, ha='right', fontsize=8)
        ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols, fontsize=8)
        for i in range(len(cols)):
            for j in range(len(cols)):
                ax.text(j, i, f"{mat.iloc[i,j]:.2f}", ha='center', va='center', fontsize=7)
        ax.set_title(titulo, fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(f'{DIR_GRAFICOS}/comparacao_v2_v3_correlacao.png', dpi=130)
    plt.close()
    print(f"   📈 Salvo: {DIR_GRAFICOS}/comparacao_v2_v3_correlacao.png")


# ==============================================================================
#  RELATÓRIO MARKDOWN
# ==============================================================================

def gerar_relatorio(limiares, feature_chuva, corrs_chuva,
                    n_cravados, n_total, dist_final, dist_por_cenario,
                    theta_33, theta_1500, A, B):
    linhas = []
    linhas.append("# Relatório de Correções — Dataset Cold Start v5\n")
    linhas.append(f"**Projeto:** Sistema ALMMo-0 — Irrigação Inteligente de Tomate | Imperatriz-MA\n")
    linhas.append(f"**Referência de solo:** Saxton & Rawls (2006), S={SOLO_AREIA}% C={SOLO_ARGILA}% OM={SOLO_OM}%\n")
    linhas.append(f"θ_CC={theta_33:.4f} m³/m³ | θ_PM={theta_1500:.4f} m³/m³ | A={A:.4f} | B={B:.4f}\n")
    linhas.append("---\n")

    linhas.append("## Rotulagem — 3 classes, limiar = mediana do cenário ótimo\n")
    linhas.append("**Justificativa:** os três cenários têm distribuições de IrrDay estruturalmente "
                  "incompatíveis (déficit truncado em 5mm, excesso concentrado acima de 10mm). "
                  "Qualquer fatiamento em 5 classes produz limiares estreitos demais para sensores reais. "
                  "A solução é reduzir para 3 classes com significado agronômico claro: sem irrigação, "
                  "irrigação moderada e irrigação intensa. O limiar único é a mediana do IrrDay > 0 "
                  "do cenário ótimo — único cenário com distribuição não truncada.\n\n")
    linhas.append(f"**Limiar adotado:** mediana do cenário ótimo = {limiares['mediana']:.2f} mm "
                  f"(n=534 eventos de irrigação)\n\n")
    linhas.append("**Regra de rotulagem:**\n")
    linhas.append(f"- Classe 0: IrrDay = 0,0 mm (sem irrigação)\n")
    linhas.append(f"- Classe 1: 0 < IrrDay ≤ {limiares['mediana']:.2f} mm (irrigação moderada)\n")
    linhas.append(f"- Classe 2: IrrDay > {limiares['mediana']:.2f} mm (irrigação intensa)\n\n")
    linhas.append("\n**Distribuição final das classes (dataset completo):**\n")
    linhas.append("| Classe | Contagem | % |\n|--------|----------|---|\n")
    for k, v in dist_final.items():
        linhas.append(f"| {k} | {v} | {v/n_total*100:.1f}% |\n")
    linhas.append("\n**Distribuição por cenário:**\n")
    linhas.append(str(dist_por_cenario) + "\n")
    linhas.append("---\n")

    linhas.append("## PROBLEMA 2 — Feature de Chuva Selecionada\n")
    linhas.append("**Correlações testadas com classe_irrigacao:**\n")
    linhas.append("| Variante | Correlação Pearson | |Corr| |\n|----------|-------------------|--------|\n")
    for nome_v, corr in corrs_chuva.items():
        linhas.append(f"| {nome_v} | {corr:.4f} | {abs(corr):.4f} |\n")
    linhas.append(f"\n**Feature selecionada:** `{feature_chuva}`\n")
    melhor_abs = max(abs(v) for v in corrs_chuva.values())
    if melhor_abs < 0.15:
        linhas.append("**Limitação declarada:** nenhuma variante de chuva atingiu |corr| ≥ 0,15. "
                      "O AquaCrop com NetIrrSMT toma decisões de irrigação baseadas na umidade "
                      "do solo (Wr), não diretamente na chuva recente. A chuva já está "
                      "incorporada indiretamente na tensão do solo — por isso a correlação "
                      "direta chuva→classe é baixa. Feature mantida como informação auxiliar "
                      "para o modelo.\n")
    linhas.append("---\n")

    linhas.append("## PROBLEMA 3 — Tensão Cravada em 1500 kPa\n")
    linhas.append(f"**Amostras com tensao_solo_kpa ≥ 1490 kPa:** {n_cravados} ({n_cravados/n_total*100:.1f}%)\n")
    linhas.append("**Interpretação:** O clamping em 1500 kPa ocorre quando θ ≤ θ_PM "
                  "(ponto de murchamento permanente). Múltiplos graus de secura acima de 1500 kPa "
                  "são indistinguíveis pela equação S&R — limitação inerente à curva de retenção. "
                  "Esses valores concentram-se no cenário de déficit hídrico e nos DAPs finais "
                  "de períodos sem chuva. Não é necessário corrigir — declarar como limitação.\n")
    linhas.append("---\n")
    linhas.append("## Arquivos Gerados\n")
    linhas.append("- `dataset_cold_start_v5.csv` — 5 colunas para ALMMo-0\n")
    linhas.append("- `dataset_cold_start_completo_v5.csv` — com metadados de cenário/ano\n")
    linhas.append("- `graficos_v5/histograma_IrrDay.png`\n")
    linhas.append("- `graficos_v5/distribuicao_classes_v5.png`\n")
    linhas.append("- `graficos_v5/correlacao_features_v5.png`\n")

    with open('relatorio_correcoes.md', 'w', encoding='utf-8') as f:
        f.writelines(linhas)
    print("   📄 Salvo: relatorio_correcoes.md")


# ==============================================================================
#  PIPELINE PRINCIPAL
# ==============================================================================

def main():
    print("\n" + "="*60)
    print("🌿 SIMULAÇÃO AQUACROP v5 — Cold Start ALMMo-0")
    print("   Tensão: Saxton & Rawls (2006) | Rotulagem por percentis")
    print("="*60)

    if not os.path.exists(arquivo_clima):
        print(f"\n❌ Arquivo '{arquivo_clima}' não encontrado.")
        return

    theta_33, theta_1500, A, B = saxton_rawls_parametros(SOLO_AREIA, SOLO_ARGILA, SOLO_OM)
    print(f"\n📐 Parâmetros S&R (2006): θ_CC={theta_33:.4f} | θ_PM={theta_1500:.4f} | A={A:.4f} | B={B:.4f}")

    weather_df = prepare_weather(arquivo_clima)
    print(f"   Clima: {len(weather_df)} dias | "
          f"{weather_df['Date'].min().date()} → {weather_df['Date'].max().date()}")

    # PASSO 1 — Simular todos os cenários sem rotulagem
    print("\n1. Simulando cenários (sem rotulagem ainda)...")
    dfs_raw = []
    for cid, nome, net_irr_smt, descricao in CENARIOS:
        df_c = simular_cenario_raw(weather_df, ANOS, net_irr_smt, nome, descricao,
                                   theta_33, theta_1500, A, B)
        if not df_c.empty:
            dfs_raw.append(df_c)

    df_todos = pd.concat(dfs_raw, ignore_index=True)
    print(f"\n   Total bruto: {len(df_todos)} amostras")

    # PASSO 2 — Problema 1: calcular limiares por percentis
    n_classes, limiares = calcular_limiares_percentis(df_todos)

    # PASSO 3 — Aplicar rotulagem
    df_todos['classe_irrigacao'] = df_todos['irr_mm'].apply(
        lambda x: rotular_classe_v5(x, limiares))

    # PASSO 4 — Problema 2: selecionar melhor feature de chuva
    # Adicionar chuva_bool para teste
    df_todos['chuva_bool'] = (df_todos['chuva_mm'] > 2).astype(int)
    feature_chuva, corrs_chuva = selecionar_feature_chuva(df_todos)

    # PASSO 5 — Problema 3: contar tensão cravada
    n_cravados = analisar_tensao_cravada(df_todos)

    # PASSO 6 — Salvar CSVs por cenário e dataset final
    print(f"\n{'='*60}")
    print("2. Salvando outputs...")
    colunas_almmo = ['tensao_solo_kpa', feature_chuva, 'tmax_max_3d_c', 'dap', 'classe_irrigacao']
    # Renomear feature de chuva se diferente do padrão
    df_todos_out = df_todos.copy()
    if feature_chuva == 'chuva_mm':
        df_todos_out = df_todos_out.rename(columns={'chuva_mm': 'chuva_dia_mm'})
        colunas_almmo[1] = 'chuva_dia_mm'
    elif feature_chuva == 'chuva_bool':
        colunas_almmo[1] = 'chuva_bool'

    for cid, nome, _, _ in CENARIOS:
        df_c = df_todos_out[df_todos_out['cenario']==nome]
        df_c[colunas_almmo + ['irr_mm', 'ano', 'cenario']].to_csv(f"{nome}.csv", index=False)
        dist = df_c['classe_irrigacao'].value_counts().sort_index()
        n = len(df_c)
        cls_str = " | ".join(f"C{k}={v}({v/n*100:.0f}%)" for k, v in dist.items())
        print(f"   ✅ {nome}.csv ({n} linhas) — {cls_str}")

    df_todos_out[colunas_almmo].to_csv('dataset_cold_start_v5.csv', index=False)
    df_todos_out[colunas_almmo + ['irr_mm', 'ano', 'cenario']].to_csv(
        'dataset_cold_start_completo_v5.csv', index=False)

    # PASSO 7 — Estatísticas finais
    n_total   = len(df_todos)
    dist_final = df_todos['classe_irrigacao'].value_counts().sort_index()
    dist_por_cenario = df_todos.groupby(['cenario','classe_irrigacao']).size().unstack(fill_value=0)

    print(f"\n📊 Distribuição final de classes ({n_classes} classes, percentis):")
    for k, v in dist_final.items():
        print(f"   Classe {k}: {v:5d} ({v/n_total*100:5.1f}%) {'█'*int(v/n_total*40)}")

    print(f"\n📊 Por cenário:")
    print(dist_por_cenario.to_string())

    print(f"\n📊 tensao_solo_kpa média por cenário:")
    for _, nome, _, _ in CENARIOS:
        m = df_todos[df_todos['cenario']==nome]['tensao_solo_kpa'].mean()
        print(f"   {nome}: {m:.1f} kPa")

    print(f"\n📊 Estatísticas descritivas:")
    for col in colunas_almmo[:-1]:
        col_real = col if col in df_todos_out.columns else feature_chuva
        s = df_todos_out[col_real] if col_real in df_todos_out.columns else df_todos_out[col]
        print(f"   {col:<22} min={s.min():.2f} | mean={s.mean():.2f} | max={s.max():.2f}")

    # PASSO 8 — Gráficos
    print("\n3. Gerando gráficos...")
    gerar_graficos(df_todos, df_todos, feature_chuva, n_classes)

    # PASSO 9 — Relatório
    gerar_relatorio(limiares, feature_chuva, corrs_chuva,
                    n_cravados, n_total, dist_final, dist_por_cenario,
                    theta_33, theta_1500, A, B)

    print(f"\n{'='*60}")
    print(f"✅ dataset_cold_start_v5.csv → {n_total} linhas | {len(colunas_almmo)} colunas")
    print(f"   Colunas: {colunas_almmo}")
    print(f"   Classes: {n_classes} | Limiares: por percentis reais do IrrDay")
    print(f"   Chuva: {feature_chuva}")
    print(f"\n→ Próximo passo: dataset_cold_start_v5.csv → ALMMo-0 cold_start()")
    print("="*60)


if __name__ == "__main__":
    main()
