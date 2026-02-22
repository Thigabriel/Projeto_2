"""
=============================================================================
SIMULAÇÃO AQUACROP-OSPy — 3 Cenários de Irrigação | Cold Start ALMMo-0
Projeto: Sistema de Irrigação com ALMMo-0 | Imperatriz-MA
VERSÃO 2 — Tensão via Saxton & Rawls (2006)
=============================================================================

DIFERENÇA EM RELAÇÃO À V1:
  V1: tensao_solo_kpa via curva exponencial 33×exp(3.8×dep_rel)
      Limitação: expoente 3.8 escolhido arbitrariamente por nós.

  V2: tensao_solo_kpa via Saxton & Rawls (2006), Eq. [1],[2],[11],[14],[15]
      Referência: Saxton, K.E. & Rawls, W.J. (2006). Soil Water Characteristic
      Estimates by Texture and Organic Matter for Hydrologic Solutions.
      Soil Sci. Soc. Am. J. 70:1569-1578. doi:10.2136/sssaj2005.0117

      Parâmetros do solo (Franco-Arenoso, Imperatriz-MA):
        S=65% areia | C=10% argila | OM=2.5% matéria orgânica

      Procedimento (equações do artigo):
        1. θ_1500 (PM) via Eq.[1]
        2. θ_33   (CC) via Eq.[2]
        3. B via Eq.[15], A via Eq.[14]
        4. θ_medio = Wr / (1000 × Z_solo)
        5. tensao = A × θ^(-B) via Eq.[11], clampada [33, 1500] kPa

      Vantagem: curva derivada de ~1722 amostras USDA — citável e defensável.
      Limitação remanescente: θ é média da coluna total (1.2m), não pontual.

Cenários:
  1. Irrigação Ótima  — NetIrrSMT=65%
  2. Déficit Hídrico  — NetIrrSMT=30%
  3. Excesso Hídrico  — NetIrrSMT=90%

Saídas:
  - cenario1_otimo_v2.csv / cenario2_deficit_v2.csv / cenario3_excesso_v2.csv
  - dataset_cold_start_v2.csv
  - dataset_cold_start_completo_v2.csv
  - distribuicao_classes_cenarios_v2.png
  - correlacao_features_v2.png

Colunas: tensao_solo_kpa | chuva_acum_3d_mm | tmax_max_3d_c | dap | classe_irrigacao
=============================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from aquacrop import (AquaCropModel, Soil, Crop, InitialWaterContent,
                      FieldMngt, IrrigationManagement)
from aquacrop.utils import prepare_weather

# ==============================================================================
#  CONFIGURAÇÃO
# ==============================================================================

arquivo_clima = 'imperatriz_climate.txt'
data_plantio  = '05/01'

# Parâmetros do solo — Franco-Arenoso (Imperatriz-MA)
SOLO_AREIA  = 65.0   # % areia (S)
SOLO_ARGILA = 10.0   # % argila (C)
SOLO_OM     = 3.0    # % matéria orgânica (OM)

# DECISÃO DE PROJETO — Rota A:
# O dataset de cold start usa campo aberto (SandyLoam padrão AquaCrop, 1.2m).
# SOLO_Z_MODELO = 1.2m → usado na conversão Wr→θ, consistente com o que
#   o AquaCrop calcula internamente. Wr sempre refere-se à coluna total.
# SOLO_Z_CANTEIRO = 0.35m → usado em campo para validação após implantação.
#   O sensor capacitivo/tensiômetro será instalado a 20cm no canteiro.
#   A mesma fórmula S&R (A, B) é usada em campo — mesma grandeza física.
SOLO_Z_MODELO   = 1.2    # m — coluna interna do AquaCrop SandyLoam
SOLO_Z_CANTEIRO = 0.35   # m — profundidade real do canteiro (validação)
SENSOR_Z        = 0.20   # m — profundidade do sensor em campo

CENARIOS = [
    (1, 'cenario1_otimo_v2',   65, 'Irrigação Ótima  (NetIrrSMT=65% — FAO tomate)'),
    (2, 'cenario2_deficit_v2', 30, 'Déficit Hídrico  (NetIrrSMT=30% — estresse severo)'),
    (3, 'cenario3_excesso_v2', 90, 'Excesso Hídrico  (NetIrrSMT=90% — irrigação frequente)'),
]


# ==============================================================================
#  SAXTON & RAWLS (2006)
# ==============================================================================

def saxton_rawls_parametros(areia, argila, mo):
    """
    Calcula parâmetros da curva de retenção hídrica do solo.
    Saxton & Rawls (2006), Soil Sci. Soc. Am. J. 70:1569-1578.

    Entradas: areia (% peso), argila (% peso), mo (% peso)
    Retorna:  theta_33, theta_1500 [m3/m3], A, B [adimensionais]
    """
    S  = areia  / 100.0
    C  = argila / 100.0
    OM = mo  # % peso conforme artigo original

    # Eq.[1] — θ_1500 (ponto de murchamento permanente, PM)
    t1500t     = (-0.024*S + 0.487*C + 0.006*OM
                  + 0.005*(S*OM) - 0.013*(C*OM)
                  + 0.068*(S*C) + 0.031)
    theta_1500 = t1500t + (0.14*t1500t - 0.02)

    # Eq.[2] — θ_33 (capacidade de campo, CC)
    t33t     = (-0.251*S + 0.195*C + 0.011*OM
                + 0.006*(S*OM) - 0.027*(C*OM)
                + 0.452*(S*C) + 0.299)
    theta_33 = t33t + (1.283*t33t**2 - 0.374*t33t - 0.015)

    # Garantia de ordem física
    theta_1500 = max(0.0001, theta_1500)
    theta_33   = max(theta_1500 + 0.01, theta_33)

    # Eq.[15] — coeficiente B (inclinação da curva log-log)
    B = (np.log(1500.0) - np.log(33.0)) / (np.log(theta_33) - np.log(theta_1500))

    # Eq.[14] — coeficiente A
    A = np.exp(np.log(33.0) + B * np.log(theta_33))

    return theta_33, theta_1500, A, B


def theta_para_tensao(theta_v, theta_33, theta_1500, A, B):
    """
    Converte θ [m3/m3] → tensão [kPa].
    Saxton & Rawls (2006), Eq.[11]: Yu = A × θ^(-B)
    θ clampado a [θ_PM, θ_CC] antes do cálculo.
    Tensão clampada a [33, 1500] kPa.
    """
    theta  = np.clip(np.asarray(theta_v, dtype=float), theta_1500, theta_33)
    tensao = A * np.power(theta, -B)
    return np.clip(tensao, 33.0, 1500.0)


# ==============================================================================
#  ROTULAGEM DE CLASSES
# ==============================================================================

def rotular_classe(irr_mm):
    if irr_mm == 0.0:    return 0
    elif irr_mm <= 2.0:  return 1
    elif irr_mm <= 5.0:  return 2
    elif irr_mm <= 10.0: return 3
    else:                return 4


# ==============================================================================
#  SIMULAÇÃO ANUAL
# ==============================================================================

def simular_ano(weather_df, ano, net_irr_smt, theta_33, theta_1500, A, B):
    """
    Simula ciclo anual de tomate (01/mai → 30/set).
    Wr → θ_medio = Wr/(1000×Z) → tensao via Saxton & Rawls Eq.[11].
    """
    data_ini = f"{ano}/05/01"
    data_fim = f"{ano}/09/30"

    if (pd.to_datetime(data_ini) < weather_df['Date'].min() or
        pd.to_datetime(data_fim) > weather_df['Date'].max()):
        print(f"      ⚠️  {ano}: clima insuficiente — pulando")
        return None

    # Solo padrão AquaCrop: SandyLoam, coluna interna de 1.2m
    # DECISÃO DE PROJETO: o dataset de cold start representa campo aberto (1.2m).
    # O canteiro de 35cm será usado para VALIDAÇÃO após implantação em campo,
    # não para geração do dataset de treinamento.
    # A tensão é calculada com Z=1.2m (SOLO_Z_MODELO), consistente com o Wr
    # que o AquaCrop retorna — que sempre refere-se à coluna total simulada.
    solo = Soil('SandyLoam')

    try:
        model = AquaCropModel(
            sim_start_time=data_ini,
            sim_end_time=data_fim,
            weather_df=weather_df,
            soil=solo,
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
        print(f"      ⚠️  {ano}: sem dados de saída")
        return None

    # Wr → θ médio da coluna do canteiro (0.35m) → tensão via S&R
    # Agora Wr vem de um solo configurado para 0.35m — fisicamente consistente
    # θ = Wr / (1000 × 0.35) deve ficar dentro de [θ_PM, θ_CC]
    Wr      = df_flux['Wr'].values
    theta_v = Wr / (1000.0 * SOLO_Z_MODELO)      # m³/m³
    tensao  = theta_para_tensao(theta_v, theta_33, theta_1500, A, B)

    if ano == 2019:
        irr_total = df_flux['IrrDay'].sum()
        # Intervalo esperado de θ para um sensor capacitivo em campo:
        # abaixo de θ_PM = solo praticamente seco (Wr zerado pelo modelo)
        theta_sensor_min = theta_v[theta_v > 0].min() if (theta_v > 0).any() else 0
        print(f"\n   [DIAG] θ coluna: min={theta_v.min():.3f} "
              f"mean={theta_v.mean():.3f} max={theta_v.max():.3f} m³/m³")
        print(f"   [DIAG] θ_CC={theta_33:.3f} θ_PM={theta_1500:.3f} m³/m³  "
              f"(sensor capacitivo leria nesse intervalo)")
        print(f"   [DIAG] tensao: min={tensao.min():.1f} "
              f"mean={tensao.mean():.1f} max={tensao.max():.1f} kPa")
        print(f"   [DIAG] Irrigação 2019: {irr_total:.1f} mm | "
              f"dias: {(df_flux['IrrDay']>0).sum()}")
        print(f"   [DIAG] TAW canteiro = {(theta_33-theta_1500)*1000*SOLO_Z_MODELO:.1f} mm "
              f"{SOLO_Z_MODELO}m coluna)\n")

    datas = weather_df[
        (weather_df['Date'] >= pd.to_datetime(data_ini)) &
        (weather_df['Date'] <= pd.to_datetime(data_fim))
    ]['Date'].values

    return df_store, df_flux, datas, tensao


# ==============================================================================
#  SIMULAÇÃO DE CENÁRIO
# ==============================================================================

def simular_cenario(weather_df, anos, net_irr_smt, nome, descricao,
                    theta_33, theta_1500, A, B):
    print(f"\n{'─'*60}")
    print(f"📋 {descricao}")
    print(f"   Anos: {anos}  |  NetIrrSMT: {net_irr_smt}%")

    dfs_anos = []

    for ano in anos:
        print(f"   → {ano}...", end=" ", flush=True)
        resultado = simular_ano(weather_df, ano, net_irr_smt,
                                theta_33, theta_1500, A, B)
        if resultado is None:
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
        df = df.rename(columns={'Precipitation': 'chuva_mm',
                                 'MaxTemp':       'temp_max_c'})

        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=3)
        df['chuva_acum_3d_mm'] = df['chuva_mm'].rolling(window=indexer, min_periods=1).sum()
        df['tmax_max_3d_c']    = df['temp_max_c'].rolling(window=indexer, min_periods=1).max()

        t_arr = df['_tensao'].values
        df['tensao_solo_kpa'] = np.concatenate([[t_arr[0]], t_arr[:-1]])
        df.drop(columns=['_tensao'], inplace=True)

        if ano == 2019:
            print(f"   [CHECK] tensao_solo_kpa: min={df['tensao_solo_kpa'].min():.1f} "
                  f"mean={df['tensao_solo_kpa'].mean():.1f} "
                  f"max={df['tensao_solo_kpa'].max():.1f} kPa")

        dfs_anos.append(df)
        print(f"{n} dias")

    if not dfs_anos:
        print("   ❌ Nenhum ano simulado")
        return pd.DataFrame()

    df_cenario = pd.concat(dfs_anos, ignore_index=True)

    linhas_antes = len(df_cenario)
    df_cenario.dropna(inplace=True)
    df_cenario = df_cenario[df_cenario['Tr'] > 0.1].copy()
    df_cenario['dap'] = df_cenario.groupby('ano').cumcount() + 1
    df_cenario = df_cenario[df_cenario['dap'] <= 160].copy()

    # Filtro DAP < 14: fase de estabelecimento da raiz
    # Nos primeiros ~13 dias, Zr do AquaCrop é mínimo (~0.15m).
    # O Wr retornado refere-se apenas à zona radicular pequena, não à coluna
    # completa de 1.2m usada na conversão Wr→θ→kPa (S&R).
    # Isso produz θ artificialmente baixo → tensão clampada em 1500 kPa.
    # Esse artefato não é erro da equação S&R — é inconsistência entre
    # Zr interno do modelo e Z_solo=1.2m da conversão.
    # Esses dias também não representam decisão de irrigação real
    # (raiz não estabelecida, planta não consome água significativa).
    linhas_dap = len(df_cenario)
    df_cenario = df_cenario[df_cenario['dap'] >= 14].copy()

    df_cenario.drop(columns=['Tr'], inplace=True)

    print(f"   Linhas removidas (entressafra)  : {linhas_antes - linhas_dap}")
    print(f"   Linhas removidas (DAP < 14)     : {linhas_dap - len(df_cenario)}")
    print(f"   Dados válidos                   : {len(df_cenario)}")

    df_cenario['classe_irrigacao'] = df_cenario['irr_mm'].apply(rotular_classe)
    df_cenario['cenario']          = nome

    dist  = df_cenario['classe_irrigacao'].value_counts().sort_index()
    total = len(df_cenario)
    print("   Classes: " + " | ".join(
        f"C{k}={v}({v/total*100:.0f}%)" for k, v in dist.items()))

    return df_cenario


# ==============================================================================
#  PIPELINE PRINCIPAL
# ==============================================================================

def main():
    print("\n" + "="*60)
    print("🌿 SIMULAÇÃO AQUACROP v2 — Cold Start ALMMo-0")
    print("   Tensão: Saxton & Rawls (2006) | Imperatriz-MA")
    print("="*60)

    if not os.path.exists(arquivo_clima):
        print(f"\n❌ Arquivo '{arquivo_clima}' não encontrado.")
        return

    # Calcular parâmetros S&R uma vez para todos os cenários
    theta_33, theta_1500, A, B = saxton_rawls_parametros(
        SOLO_AREIA, SOLO_ARGILA, SOLO_OM)

    print(f"\n📐 Parâmetros Saxton & Rawls (2006) — Solo do canteiro:")
    print(f"   S={SOLO_AREIA:.0f}% areia | C={SOLO_ARGILA:.0f}% argila | OM={SOLO_OM:.1f}%")
    print(f"   θ_CC  (33 kPa)   = {theta_33:.4f} m³/m³  ({theta_33*100:.1f}%v)")
    print(f"   θ_PM  (1500 kPa) = {theta_1500:.4f} m³/m³  ({theta_1500*100:.1f}%v)")
    print(f"   TAW modelo       = {(theta_33-theta_1500)*1000*SOLO_Z_MODELO:.1f} mm "
          f"(coluna {SOLO_Z_MODELO}m — SandyLoam padrão AquaCrop)")
    print(f"   A={A:.4f}  B={B:.4f}")
    print(f"\n   ⚠️  Sensor em campo a {SENSOR_Z*100:.0f}cm:")
    print(f"   → Tensiômetro  : lê kPa diretamente — comparar com tensao_solo_kpa")
    print(f"   → Capacitivo   : lê θ [m³/m³] → tensao = {A:.4f} × θ^(-{B:.4f})")
    print(f"   → Intervalo θ esperado em campo: [{theta_1500:.3f}, {theta_33:.3f}] m³/m³")

    print(f"\n1. Lendo clima: {arquivo_clima}...")
    weather_df = prepare_weather(arquivo_clima)
    print(f"   → {len(weather_df)} dias | "
          f"{weather_df['Date'].min().date()} → {weather_df['Date'].max().date()}")

    dfs_cenarios = []
    ANOS = [2019, 2020, 2021, 2022, 2023, 2025]

    for cid, nome, net_irr_smt, descricao in CENARIOS:
        df_c = simular_cenario(weather_df, ANOS, net_irr_smt, nome, descricao,
                               theta_33, theta_1500, A, B)
        df_c.to_csv(f"{nome}.csv", index=False)
        print(f"   ✅ Salvo: {nome}.csv ({len(df_c)} linhas)")
        dfs_cenarios.append(df_c)

    print(f"\n{'='*60}")
    print("2. Consolidando dataset final...")

    df_final = pd.concat(dfs_cenarios, ignore_index=True)

    colunas_almmo = ['tensao_solo_kpa', 'chuva_acum_3d_mm',
                     'tmax_max_3d_c', 'dap', 'classe_irrigacao']

    print(f"\n📊 Distribuição de classes:")
    dist  = df_final['classe_irrigacao'].value_counts().sort_index()
    total = len(df_final)
    for cls, cnt in dist.items():
        print(f"   Classe {cls}: {cnt:5d} ({cnt/total*100:5.1f}%) "
              f"{'█'*int(cnt/total*40)}")

    if dist.get(0, 0) / total > 0.70:
        print("⚠️  Classe 0 > 70%. Documente e avalie oversampling.")

    print(f"\n📊 Amostras por cenário e ano:")
    print(df_final.groupby(['cenario','ano']).size().unstack(fill_value=0).to_string())

    print(f"\n📊 Estatísticas descritivas:")
    print(f"   {'Variável':<22} {'Mín':>8} {'Média':>8} {'Máx':>8} {'NaN':>6}")
    print(f"   {'-'*57}")
    for col in ['tensao_solo_kpa', 'chuva_acum_3d_mm', 'tmax_max_3d_c', 'dap']:
        s = df_final[col]
        print(f"   {col:<22} {s.min():>8.2f} {s.mean():>8.2f} "
              f"{s.max():>8.2f} {s.isna().sum():>5d}")

    print(f"\n📊 tensao_solo_kpa média por cenário:")
    for cn in [c[1] for c in CENARIOS]:
        m = df_final[df_final['cenario']==cn]['tensao_solo_kpa'].mean()
        print(f"   {cn}: {m:.1f} kPa")

    df_final[colunas_almmo].to_csv('dataset_cold_start_v2.csv', index=False)
    df_final.to_csv('dataset_cold_start_completo_v2.csv', index=False)

    # Gráfico 1 — Distribuição de classes
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    fig.suptitle('Distribuição de Classes por Cenário (v2 — Saxton & Rawls 2006)',
                 fontsize=11, fontweight='bold')
    cores = ['#4CAF50', '#2196F3', '#FF9800', '#F44336', '#9C27B0']
    for ax, (cid, nome, _, _desc) in zip(axes, CENARIOS):
        df_c   = df_final[df_final['cenario'] == nome]
        counts = df_c['classe_irrigacao'].value_counts().sort_index()
        ax.bar([str(k) for k in counts.index], counts.values,
               color=[cores[k] for k in counts.index])
        ax.set_title(f"Cenário {cid}: {nome.split('_')[1].capitalize()}")
        ax.set_xlabel('Classe de Irrigação')
        ax.set_ylabel('Nº de dias')
        ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('distribuicao_classes_cenarios_v2.png', dpi=150, bbox_inches='tight')
    print("\n📈 Gráfico salvo: distribuicao_classes_cenarios_v2.png")
    plt.close()

    # Gráfico 2 — Correlação
    features = ['tensao_solo_kpa', 'chuva_acum_3d_mm', 'tmax_max_3d_c',
                'dap', 'classe_irrigacao']
    labels   = ['Tensão\n(kPa)', 'Chuva\n3d (mm)', 'Tmax\n3d (°C)',
                'DAP', 'Classe\nIrrig.']

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Análise de Correlação — Dataset Cold Start ALMMo-0 v2\n'
                 '(Tensão via Saxton & Rawls 2006)',
                 fontsize=12, fontweight='bold')

    corr = df_final[features].corr()
    im   = axes[0].imshow(corr.values, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_yticks(range(len(labels)))
    axes[0].set_xticklabels(labels, fontsize=9)
    axes[0].set_yticklabels(labels, fontsize=9)
    axes[0].set_title('Matriz de Correlação (Pearson)', fontsize=11)
    for i in range(len(features)):
        for j in range(len(features)):
            val     = corr.values[i, j]
            cor_txt = 'white' if abs(val) > 0.6 else 'black'
            axes[0].text(j, i, f'{val:.2f}', ha='center', va='center',
                         fontsize=9, color=cor_txt, fontweight='bold')
    plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    feats       = ['tensao_solo_kpa', 'chuva_acum_3d_mm', 'tmax_max_3d_c', 'dap']
    feat_labels = ['Tensão\n(kPa)', 'Chuva\n3d (mm)', 'Tmax\n3d (°C)', 'DAP']
    cn_list     = [c[1] for c in CENARIOS]
    cl_list     = ['Ótimo', 'Déficit', 'Excesso']
    cores_c     = ['#2196F3', '#F44336', '#4CAF50']
    x, width    = np.arange(len(feats)), 0.25

    for i, (cn, cl, cor) in enumerate(zip(cn_list, cl_list, cores_c)):
        df_c  = df_final[df_final['cenario'] == cn]
        corrs = [df_c[f].corr(df_c['classe_irrigacao']) for f in feats]
        axes[1].bar(x + i*width, corrs, width, label=cl, color=cor, alpha=0.85)

    axes[1].axhline(0, color='black', linewidth=0.8, linestyle='--')
    axes[1].set_xticks(x + width)
    axes[1].set_xticklabels(feat_labels, fontsize=9)
    axes[1].set_ylabel('Correlação de Pearson com Classe')
    axes[1].set_title('Correlação Feature × Classe por Cenário', fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].set_ylim(-1, 1)
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_xlabel('Feature')

    plt.tight_layout()
    plt.savefig('correlacao_features_v2.png', dpi=150, bbox_inches='tight')
    print("📈 Gráfico salvo: correlacao_features_v2.png")
    plt.close()

    print("\n" + "="*60)
    print(f"✅ dataset_cold_start_v2.csv          → "
          f"{len(df_final[colunas_almmo])} linhas | {len(colunas_almmo)} colunas")
    print(f"✅ dataset_cold_start_completo_v2.csv → {len(df_final)} linhas")
    print(f"\nColunas: {colunas_almmo}")
    print(f"\nReferência conversão θ→kPa:")
    print(f"  Saxton & Rawls (2006), Soil Sci. Soc. Am. J. 70:1569-1578")
    print(f"  Eq.[1],[2],[14],[15],[11] | S={SOLO_AREIA}% C={SOLO_ARGILA}% OM={SOLO_OM}%")
    print(f"\nConfiguração:")
    print(f"  Dataset (cold start) : SandyLoam AquaCrop, {SOLO_Z_MODELO}m — campo aberto")
    print(f"  Validação em campo   : canteiro {SOLO_Z_CANTEIRO*100:.0f}cm, sensor a {SENSOR_Z*100:.0f}cm")
    print(f"  TAW modelo           : {(theta_33-theta_1500)*1000*SOLO_Z_MODELO:.1f} mm")
    print(f"  θ_CC / θ_PM          : {theta_33:.4f} / {theta_1500:.4f} m³/m³")
    print(f"  Conversão S&R campo  : tensao = {A:.4f} × θ^(-{B:.4f})")
    print(f"  θ esperado canteiro  : [{theta_1500:.3f}, {theta_33:.3f}] m³/m³")
    print("\n→ Próximo passo: dataset_cold_start_v2.csv → ALMMo-0 cold_start()")
    print("="*60)


if __name__ == "__main__":
    main()