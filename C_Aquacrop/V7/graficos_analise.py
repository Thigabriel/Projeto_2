#!/usr/bin/env python3
"""
============================================================================
Gráficos de Análise — Dataset Cold Start ALMMo-0 (v7)
============================================================================

Gera 3 figuras:
  1. Correlação (matriz + barras por cenário)
  2. Distribuição de classes por cenário
  3. Dados climáticos NASA POWER (temperatura, precipitação, ETo, balanço)

Requisitos: pip install matplotlib seaborn pandas numpy
Execução:   python graficos_analise.py

Os gráficos são salvos como PNG na pasta de execução.

NOTA: Este script espera os seguintes arquivos no diretório de execução:
  - dataset_cold_start_v7.csv
  - weather_files/weather_imperatriz_YYYY_full_meta.csv (2019-2023)

Se o dataset_cold_start_v7.csv não tiver coluna 'cenario' ou 'janela',
o gráfico 2 usará apenas as colunas disponíveis.
============================================================================
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use('Agg')  # backend não-interativo
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.gridspec import GridSpec
    import seaborn as sns
except ImportError:
    print("ERRO: pip install matplotlib seaborn")
    sys.exit(1)

# Configuração global
plt.rcParams.update({
    'figure.dpi': 150,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.facecolor': 'white',
})

ANOS = [2019, 2020, 2021, 2022, 2023]
WEATHER_DIR = Path('weather_files')

# ============================================================================
# CARREGAR DADOS
# ============================================================================

def carregar_dataset():
    """Carrega dataset_cold_start_v7.csv."""
    path = Path('dataset_cold_start_v7.csv')
    if not path.exists():
        print(f"ERRO: {path} não encontrado no diretório atual.")
        print(f"  Diretório: {Path.cwd()}")
        sys.exit(1)

    df = pd.read_csv(path)
    print(f"Dataset carregado: {len(df)} linhas, colunas: {list(df.columns)}")
    return df


def carregar_dataset_completo():
    """
    Tenta carregar o dataset com colunas extras (cenario, janela).
    Se não existir, retorna None.
    """
    # O dataset final só tem 5 colunas. Para ter cenario/janela,
    # precisamos do dataset_full que é intermediário.
    # Vamos tentar carregar se existir.
    for candidate in ['dataset_cold_start_v7_full.csv', 'dataset_full.csv']:
        p = Path(candidate)
        if p.exists():
            return pd.read_csv(p)
    return None


def carregar_weather():
    """Carrega todos os meta CSVs do NASA POWER."""
    frames = []
    for year in ANOS:
        path = WEATHER_DIR / f'weather_imperatriz_{year}_full_meta.csv'
        if not path.exists():
            print(f"  AVISO: {path} não encontrado, pulando.")
            continue
        df = pd.read_csv(path, parse_dates=['date'])
        frames.append(df)
        print(f"  Weather {year}: {len(df)} dias")

    if not frames:
        print("ERRO: Nenhum arquivo weather_meta encontrado.")
        return None

    weather = pd.concat(frames, ignore_index=True)
    weather = weather.sort_values('date').reset_index(drop=True)
    print(f"Weather total: {len(weather)} dias ({weather['date'].min().date()} a {weather['date'].max().date()})")
    return weather


# ============================================================================
# GRÁFICO 1: ANÁLISE DE CORRELAÇÃO
# ============================================================================

def grafico_correlacao(df, df_full=None):
    """
    Gera figura com:
      - Esquerda: Matriz de correlação (Pearson) entre features + classe
      - Direita: Barras de correlação feature × classe por cenário
    """
    print("\n[1/3] Gerando gráfico de correlação...")

    features = ['tensao_solo_kpa', 'chuva_acum_3d_mm', 'tmax_max_3d_c', 'dap', 'classe_irrigacao']
    labels_short = ['Tensão\n(kPa)', 'Chuva\n3d (mm)', 'Tmax\n3d (°C)', 'DAP', 'Classe\nIrrig.']

    # Subconjunto
    df_corr = df[features].copy()

    fig = plt.figure(figsize=(16, 6))
    fig.suptitle('Análise de Correlação — Dataset Cold Start ALMMo-0', fontsize=15, fontweight='bold', y=1.02)

    gs = GridSpec(1, 2, width_ratios=[1, 1], wspace=0.35)

    # --- Esquerda: Heatmap ---
    ax1 = fig.add_subplot(gs[0, 0])
    corr_matrix = df_corr.corr()

    # Colormap verde-branco-vermelho
    cmap = sns.diverging_palette(10, 130, s=80, l=55, as_cmap=True)

    mask = np.zeros_like(corr_matrix, dtype=bool)  # sem máscara — mostra tudo

    sns.heatmap(
        corr_matrix, ax=ax1, annot=True, fmt='.2f',
        cmap=cmap, center=0, vmin=-1, vmax=1,
        square=True, linewidths=0.8, linecolor='white',
        xticklabels=labels_short, yticklabels=labels_short,
        cbar_kws={'shrink': 0.8, 'label': ''},
        annot_kws={'fontsize': 11, 'fontweight': 'bold'}
    )
    ax1.set_title('Matriz de Correlação (Pearson)', fontsize=12, pad=10)

    # --- Direita: Barras por cenário ---
    ax2 = fig.add_subplot(gs[0, 1])

    feat_cols = ['tensao_solo_kpa', 'chuva_acum_3d_mm', 'tmax_max_3d_c', 'dap']
    feat_labels = ['Tensão\n(kPa)', 'Chuva\n3d (mm)', 'Tmax\n3d (°C)', 'DAP']

    if df_full is not None and 'cenario' in df_full.columns:
        # Usar dados por cenário do dataset completo
        cenarios_config = {
            'otimo':   {'color': '#2196F3', 'label': 'Ótimo'},
            'deficit': {'color': '#F44336', 'label': 'Déficit'},
            'excesso': {'color': '#4CAF50', 'label': 'Excesso'},
        }
        cenarios_presentes = [c for c in ['otimo', 'deficit', 'excesso'] if c in df_full['cenario'].values]
        n_cenarios = len(cenarios_presentes)
        bar_width = 0.25
        x = np.arange(len(feat_cols))

        for i, cen in enumerate(cenarios_presentes):
            sub = df_full[df_full['cenario'] == cen]
            cfg = cenarios_config.get(cen, {'color': 'gray', 'label': cen})
            corrs = [sub[f].corr(sub['classe_irrigacao']) for f in feat_cols]
            offset = (i - n_cenarios / 2 + 0.5) * bar_width
            bars = ax2.bar(x + offset, corrs, bar_width, label=cfg['label'],
                          color=cfg['color'], alpha=0.85, edgecolor='white', linewidth=0.5)

    else:
        # Sem cenário — barra única global
        bar_width = 0.5
        x = np.arange(len(feat_cols))
        corrs = [df[f].corr(df['classe_irrigacao']) for f in feat_cols]
        ax2.bar(x, corrs, bar_width, label='Global', color='#4CAF50', alpha=0.85)

    ax2.set_xticks(np.arange(len(feat_cols)))
    ax2.set_xticklabels(feat_labels)
    ax2.set_ylabel('Correlação de Pearson com Classe')
    ax2.set_title('Correlação Feature × Classe por Cenário', fontsize=12, pad=10)
    ax2.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
    ax2.set_ylim(-1, 1)
    ax2.legend(loc='upper left', framealpha=0.9)
    ax2.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    out_path = 'grafico1_correlacao_features.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Salvo: {out_path}")
    return out_path


# ============================================================================
# GRÁFICO 2: DISTRIBUIÇÃO DE CLASSES POR CENÁRIO
# ============================================================================

def grafico_distribuicao_classes(df, df_full=None):
    """
    Gera figura com distribuição de classes por cenário (3 subplots).
    Se df_full não disponível, usa o dataset global em um único plot.
    """
    print("\n[2/3] Gerando gráfico de distribuição de classes...")

    # Cores por classe
    cores_classe = {
        0: '#4CAF50',  # verde
        1: '#FF9800',  # laranja
        2: '#F44336',  # vermelho
    }

    cenarios_ordem = ['otimo', 'deficit', 'excesso']
    cenarios_titulo = {'otimo': 'Cenário 1: Ótimo', 'deficit': 'Cenário 2: Déficit', 'excesso': 'Cenário 3: Excesso'}

    if df_full is not None and 'cenario' in df_full.columns:
        cenarios_presentes = [c for c in cenarios_ordem if c in df_full['cenario'].values]
        n_plots = len(cenarios_presentes)
    else:
        cenarios_presentes = None
        n_plots = 1

    fig, axes = plt.subplots(1, max(n_plots, 1), figsize=(6 * n_plots, 5), squeeze=False)
    fig.suptitle('Distribuição de Classes por Cenário', fontsize=15, fontweight='bold', y=1.03)

    classes_possiveis = sorted(df['classe_irrigacao'].unique())

    if cenarios_presentes:
        for idx, cen in enumerate(cenarios_presentes):
            ax = axes[0, idx]
            sub = df_full[df_full['cenario'] == cen]
            vc = sub['classe_irrigacao'].value_counts().sort_index()

            bars_x = []
            bars_h = []
            bars_c = []
            for c in classes_possiveis:
                bars_x.append(c)
                bars_h.append(vc.get(c, 0))
                bars_c.append(cores_classe.get(c, 'gray'))

            bars = ax.bar(bars_x, bars_h, color=bars_c, edgecolor='white', linewidth=1.2, width=0.6)

            # Labels acima das barras
            for bar, h in zip(bars, bars_h):
                if h > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, h + max(bars_h) * 0.02,
                            str(h), ha='center', va='bottom', fontsize=10, fontweight='bold')

            ax.set_xlabel('Classe de Irrigação', fontsize=11)
            ax.set_ylabel('Nº de dias' if idx == 0 else '')
            ax.set_title(cenarios_titulo.get(cen, cen), fontsize=13)
            ax.set_xticks(classes_possiveis)
            ax.set_xlim(-0.5, max(classes_possiveis) + 0.5)
            ax.grid(axis='y', alpha=0.3)

    else:
        ax = axes[0, 0]
        vc = df['classe_irrigacao'].value_counts().sort_index()
        bars = ax.bar(vc.index, vc.values,
                      color=[cores_classe.get(c, 'gray') for c in vc.index],
                      edgecolor='white', linewidth=1.2, width=0.6)
        for bar, h in zip(bars, vc.values):
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + max(vc.values) * 0.02,
                        str(h), ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.set_xlabel('Classe de Irrigação')
        ax.set_ylabel('Nº de dias')
        ax.set_title('Distribuição Global', fontsize=13)
        ax.set_xticks(classes_possiveis)
        ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    out_path = 'grafico2_distribuicao_classes_cenarios.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Salvo: {out_path}")
    return out_path


# ============================================================================
# GRÁFICO 3: DADOS CLIMÁTICOS NASA POWER
# ============================================================================

def grafico_climatico(weather):
    """
    Gera figura com 4 subplots (estilo INMET):
      1. Temperatura (amplitude Tmin-Tmax + Tmédia)
      2. Precipitação diária
      3. ETo (FAO-56 PM)
      4. Balanço hídrico (Prec - ETo)
    """
    print("\n[3/3] Gerando gráfico climático NASA POWER...")

    if weather is None:
        print("  PULANDO: dados weather não disponíveis.")
        return None

    w = weather.copy()
    w['tmean'] = (w['tmax'] + w['tmin']) / 2
    w['balanco'] = w['prec'] - w['eto']

    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
    fig.suptitle('Dados Climáticos — NASA POWER Imperatriz-MA (2019–2023)',
                 fontsize=15, fontweight='bold', y=1.01)

    dates = w['date']

    # --- Subplot 1: Temperatura ---
    ax = axes[0]
    ax.fill_between(dates, w['tmin'], w['tmax'], alpha=0.3, color='#E57373', label='Amplitude (Tmin–Tmax)')
    ax.plot(dates, w['tmean'], color='#D32F2F', linewidth=0.6, alpha=0.8, label='Tmédia')

    # Média móvel 30d para suavizar
    if len(w) > 30:
        tmean_smooth = w['tmean'].rolling(30, center=True, min_periods=5).mean()
        ax.plot(dates, tmean_smooth, color='#B71C1C', linewidth=1.5, label='Tmédia (30d)')

    ax.set_ylabel('Temperatura (°C)')
    ax.set_ylim(15, 42)
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3)

    # --- Subplot 2: Precipitação ---
    ax = axes[1]
    ax.bar(dates, w['prec'], width=1.0, color='#42A5F5', alpha=0.7, linewidth=0)
    ax.set_ylabel('Precipitação (mm/dia)')
    ax.set_ylim(0, w['prec'].max() * 1.1)
    ax.grid(alpha=0.3)

    # Anotar totais anuais
    for year in ANOS:
        sub = w[w['date'].dt.year == year]
        total = sub['prec'].sum()
        mid_date = pd.Timestamp(f'{year}-07-01')
        ax.text(mid_date, w['prec'].max() * 0.95, f'{total:.0f}mm',
                ha='center', fontsize=9, color='#1565C0', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    # --- Subplot 3: ETo ---
    ax = axes[2]
    ax.plot(dates, w['eto'], color='#FF8F00', linewidth=0.5, alpha=0.6, label='ETo diária')

    if len(w) > 30:
        eto_smooth = w['eto'].rolling(30, center=True, min_periods=5).mean()
        ax.plot(dates, eto_smooth, color='#E65100', linewidth=1.8, label='ETo (média 30d)')

    ax.set_ylabel('ETo (mm/dia)')
    ax.set_ylim(0, w['eto'].max() * 1.15)
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3)

    # --- Subplot 4: Balanço hídrico ---
    ax = axes[3]

    positivo = w['balanco'].clip(lower=0)
    negativo = w['balanco'].clip(upper=0)

    ax.fill_between(dates, 0, positivo, alpha=0.5, color='#42A5F5', label='Excesso (chuva > ETo)')
    ax.fill_between(dates, 0, negativo, alpha=0.4, color='#EF9A9A', label='Déficit (ETo > chuva)')
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_ylabel('Balanço Hídrico (mm/dia)')
    ax.set_xlabel('')
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3)

    # Formatar eixo X
    import matplotlib.dates as mdates
    axes[3].xaxis.set_major_locator(mdates.YearLocator())
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    axes[3].xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))

    fig.tight_layout()
    out_path = 'grafico3_climatico_nasapower.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Salvo: {out_path}")
    return out_path


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("Gráficos de Análise — Dataset ALMMo-0 v7")
    print("=" * 60)

    # 1. Carregar dados
    print("\n[DADOS] Carregando...")
    df = carregar_dataset()
    df_full = carregar_dataset_completo()
    weather = carregar_weather()

    if df_full is not None:
        print(f"  Dataset completo: {len(df_full)} linhas, cenarios: {df_full['cenario'].unique() if 'cenario' in df_full.columns else 'N/A'}")
    else:
        print("  Dataset completo (com cenário) não encontrado.")
        print("  DICA: Para ter gráficos por cenário, salve o dataset_full")
        print("        (antes de selecionar apenas as 5 colunas finais).")
        print("  Os gráficos 1 e 2 usarão apenas dados globais.")

    # 2. Gerar gráficos
    paths = []
    paths.append(grafico_correlacao(df, df_full))
    paths.append(grafico_distribuicao_classes(df, df_full))
    p3 = grafico_climatico(weather)
    if p3:
        paths.append(p3)

    print("\n" + "=" * 60)
    print(f"CONCLUÍDO — {len(paths)} gráficos gerados:")
    for p in paths:
        if p:
            print(f"  {p}")
    print("=" * 60)


if __name__ == '__main__':
    main()
