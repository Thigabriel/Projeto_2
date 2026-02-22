"""
=============================================================================
SIMULAÇÃO AQUACROP-OSPy — 3 Cenários de Irrigação | Cold Start ALMMo-0
Projeto: Sistema de Irrigação com ALMMo-0 | Imperatriz-MA
=============================================================================
Adaptado do código modelo funcional.

Cenários:
  1. Irrigação Ótima  — SMT 40% (~35-40 kPa, referência FAO para tomate)
  2. Déficit Hídrico  — SMT 25% (~70% da lâmina ótima)
  3. Excesso Hídrico  — SMT 70% (irrigação muito frequente)

Saídas:
  - cenario1_otimo.csv
  - cenario2_deficit.csv
  - cenario3_excesso.csv
  - dataset_cold_start.csv          ← 6 colunas, pronto para o ALMMo-0
  - dataset_cold_start_completo.csv ← com metadados (ano, cenário)

Colunas do dataset final:
  tensao_solo_kpa | chuva_acum_3d_mm | tmax_max_3d_c | dap | classe_irrigacao
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
#  CONFIGURAÇÃO INICIAL
# ==============================================================================

arquivo_clima = 'imperatriz_climate.txt'
data_plantio  = '05/01'   # 01 de maio — igual para todos os anos

# Cenários: (id, nome_arquivo, NetIrrSMT, descrição)
# irrigation_method=4: Net Irrigation
# NetIrrSMT = % da TAW abaixo do qual irriga para repor até CC
# SMT ALTO  = irriga quando solo ainda está relativamente úmido → EXCESSO
# SMT BAIXO = só irriga quando solo está bem seco → DÉFICIT
# A lâmina aplicada repõe exatamente o déficit até CC (controlado pelo modelo)
CENARIOS = [
    (1, 'cenario1_otimo',   65, 'Irrigação Ótima  (NetIrrSMT=65% — FAO tomate)'),
    (2, 'cenario2_deficit', 30, 'Déficit Hídrico  (NetIrrSMT=30% — irriga só sob estresse severo)'),
    (3, 'cenario3_excesso', 90, 'Excesso Hídrico  (NetIrrSMT=90% — irriga muito frequente)'),
]

# ==============================================================================
#  PARTE 1: FUNÇÃO FÍSICA (SAXTON & RAWLS) — tensão em kPa
# ==============================================================================

def converter_theta_para_kpa_saxton_rawls(theta_v, areia, argila, mo=2.5):
    """
    Converte Umidade Volumétrica (m³/m³) → Tensão do solo (kPa).
    Saxton & Rawls (2006) — calibrado para Franco-Arenoso (65% areia, 10% argila).
    Theta é clampado entre o ponto de murchamento e a saturação para evitar
    tensões fisicamente impossíveis quando o AquaCrop retorna th=0.
    """
    theta = np.array(theta_v, dtype=float)
    S  = float(areia)  / 100.0 if float(areia)  > 1.0 else float(areia)
    C  = float(argila) / 100.0 if float(argila) > 1.0 else float(argila)
    MO = float(mo)

    theta_1500t = (-0.024*S + 0.487*C + 0.006*MO
                   + 0.005*(S*MO) - 0.013*(C*MO) + 0.068*(S*C) + 0.031)
    theta_33t   = (-0.251*S + 0.195*C + 0.011*MO
                   + 0.006*(S*MO) - 0.027*(C*MO) + 0.452*(S*C) + 0.299)
    theta_33    = theta_33t + (1.283*theta_33t**2 - 0.374*theta_33t - 0.015)

    theta_33    = max(0.001, theta_33)
    theta_1500t = max(0.0001, min(theta_1500t, theta_33 - 0.01))

    B = (np.log(1500) - np.log(33)) / (np.log(theta_33) - np.log(theta_1500t))
    A = np.exp(np.log(33) + B * np.log(theta_33))

    # CLAMPAR theta ao intervalo físico válido [theta_PM, theta_CC]
    # th=0 do AquaCrop (entressafra/inicialização) produziria tensão → infinito
    theta = np.clip(theta, theta_1500t, theta_33)

    tensao_kPa = A * np.power(theta, -B)

    # Clampar tensão final ao range agronômico real: [1 kPa, 1500 kPa]
    tensao_kPa = np.clip(tensao_kPa, 1.0, 1500.0)

    return tensao_kPa


# ==============================================================================
#  PARTE 2: ROTULAGEM DE CLASSES (irrigação mm → classe 0-4)
# ==============================================================================

def rotular_classe(irr_mm: float) -> int:
    """
    Converte irrigação diária (mm) em classe discreta 0-4.
    Limiares agronômicos para tomate em ambiente protegido.
    """
    if irr_mm == 0.0:    return 0   # Sem irrigação
    elif irr_mm <= 2.0:  return 1   # Muito baixa  (0–2 mm)
    elif irr_mm <= 5.0:  return 2   # Baixa        (2–5 mm)
    elif irr_mm <= 10.0: return 3   # Média        (5–10 mm)
    else:                return 4   # Alta         (> 10 mm)


# ==============================================================================
#  PARTE 3: SIMULAÇÃO DE UM CENÁRIO (período completo 2019→2025)
# ==============================================================================

def simular_ano(weather_df, ano, net_irr_smt):
    """
    Simula um ciclo anual de tomate (01/mai → 30/set).
    Usa till_termination=True para performance, e estima Dr/TAW via Wr
    com Zr do tomate crescendo linearmente de 0.15m → 0.60m ao longo do ciclo.
    """
    data_ini = f"{ano}/05/01"
    data_fim = f"{ano}/09/30"

    clima_min = weather_df['Date'].min()
    clima_max = weather_df['Date'].max()
    if (pd.to_datetime(data_ini) < clima_min or
        pd.to_datetime(data_fim) > clima_max):
        print(f"      ⚠️  {ano}: clima insuficiente — pulando")
        return None

    solo      = Soil('SandyLoam')
    cultura   = Crop(c_name='Tomato', planting_date=data_plantio)
    irrig_mng = IrrigationManagement(irrigation_method=4, NetIrrSMT=net_irr_smt)

    try:
        model = AquaCropModel(
            sim_start_time=data_ini,
            sim_end_time=data_fim,
            weather_df=weather_df,
            soil=solo,
            crop=cultura,
            initial_water_content=InitialWaterContent(value=['FC']),
            field_management=FieldMngt(mulches=True, mulch_pct=80, f_mulch=0.3),
            irrigation_management=irrig_mng
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

    # Calcular dep_rel corretamente
    # Wr do AquaCrop = água na coluna TOTAL do solo (1.2m para SandyLoam)
    # SandyLoam: FC~0.23, WP~0.10 → TAW = 130 mm/m × 1.2m = 156mm
    TAW_TOTAL = 130.0 * 1.2   # = 156 mm (coluna total SandyLoam)
    Wr      = df_flux['Wr'].values
    dep_rel = np.clip(1.0 - Wr / TAW_TOTAL, 0.0, 1.0)

    # Diagnóstico para 2019
    if ano == 2019:
        t_diag = np.clip(33.0 * np.exp(3.8 * dep_rel), 33.0, 1500.0)
        irr_total = df_flux['IrrDay'].sum()
        print(f"\n   [DIAG] dep_rel: min={dep_rel.min():.3f} mean={dep_rel.mean():.3f} max={dep_rel.max():.3f}")
        print(f"   [DIAG] tensao:   min={t_diag.min():.1f} mean={t_diag.mean():.1f} max={t_diag.max():.1f} kPa")
        print(f"   [DIAG] Irrigação total 2019: {irr_total:.1f} mm | dias: {(df_flux['IrrDay']>0).sum()}\n")

    datas = weather_df[
        (weather_df['Date'] >= pd.to_datetime(data_ini)) &
        (weather_df['Date'] <= pd.to_datetime(data_fim))
    ]['Date'].values

    return df_store, df_flux, datas, dep_rel


def simular_cenario(weather_df, anos, net_irr_smt, nome, descricao):
    """
    Roda simulações anuais independentes para cada ano e consolida os resultados.
    """
    print(f"\n{'─'*60}")
    print(f"📋 {descricao}")
    print(f"   Anos : {anos}  |  NetIrrSMT: {net_irr_smt}%")

    dfs_anos = []

    for ano in anos:
        print(f"   → {ano}...", end=" ", flush=True)

        resultado = simular_ano(weather_df, ano, net_irr_smt)
        if resultado is None:
            continue

        df_store, df_flux, datas, dep_rel = resultado
        n = min(len(datas), len(df_store), len(df_flux), len(dep_rel))

        # Montar df SEM tensão primeiro — merge pode alterar índices
        df = pd.DataFrame({
            'Date':   pd.to_datetime(datas[:n]),
            'Tr':     df_flux['Tr'].values[:n],
            'irr_mm': (df_flux['IrrDay'].values[:n]
                       if 'IrrDay' in df_flux.columns
                       else df_flux['irrigation'].values[:n]),
            'ano':    ano,
            # Guardar dep_rel como coluna numérica para sobreviver ao merge
            '_dep':   dep_rel[:n],
        })

        # Dados climáticos
        wdf = weather_df[['Date', 'Precipitation', 'MaxTemp']].copy()
        wdf['Date'] = pd.to_datetime(wdf['Date'])
        df = pd.merge(df, wdf, on='Date', how='left')
        df = df.rename(columns={'Precipitation': 'chuva_mm', 'MaxTemp': 'temp_max_c'})

        # Janela futura de 3 dias
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=3)
        df['chuva_acum_3d_mm'] = df['chuva_mm'].rolling(window=indexer, min_periods=1).sum()
        df['tmax_max_3d_c']    = df['temp_max_c'].rolling(window=indexer, min_periods=1).max()

        # Tensão calculada AQUI — depois do merge, índices estáveis
        dep_arr = df['_dep'].values
        tensao_raw = np.clip(33.0 * np.exp(3.8 * dep_arr), 33.0, 1500.0)
        # shift(1): tensão de hoje = estado do solo de ontem
        df['tensao_solo_kpa'] = np.concatenate([[tensao_raw[0]], tensao_raw[:-1]])
        df.drop(columns=['_dep'], inplace=True)

        # Diagnóstico inline por ano
        if ano == 2019:
            print(f"   [CHECK tensao] min={df['tensao_solo_kpa'].min():.1f} "
                  f"mean={df['tensao_solo_kpa'].mean():.1f} "
                  f"max={df['tensao_solo_kpa'].max():.1f}")

        dfs_anos.append(df)
        print(f"{n} dias")

    if not dfs_anos:
        print(f"   ❌ Nenhum ano simulado com sucesso")
        return pd.DataFrame()

    df_cenario = pd.concat(dfs_anos, ignore_index=True)

    # ==========================================================================
    #  LIMPEZA E DAP
    # ==========================================================================
    linhas_antes = len(df_cenario)

    df_cenario.dropna(inplace=True)
    df_cenario = df_cenario[df_cenario['Tr'] > 0.1].copy()

    # DAP reinicia a cada ano (cada ano é um ciclo independente)
    df_cenario['dap'] = df_cenario.groupby('ano').cumcount() + 1
    df_cenario        = df_cenario[df_cenario['dap'] <= 160].copy()
    df_cenario.drop(columns=['Tr'], inplace=True)

    linhas_depois = len(df_cenario)
    print(f"   Linhas removidas (entressafra): {linhas_antes - linhas_depois}")
    print(f"   Dados válidos                 : {linhas_depois}")

    # Rotular classes e marcar cenário
    df_cenario['classe_irrigacao'] = df_cenario['irr_mm'].apply(rotular_classe)
    df_cenario['cenario']          = nome

    dist  = df_cenario['classe_irrigacao'].value_counts().sort_index()
    total = len(df_cenario)
    print(f"   Classes: " + " | ".join(
        f"C{k}={v}({v/total*100:.0f}%)" for k, v in dist.items()
    ))

    return df_cenario


# ==============================================================================
#  PARTE 4: PIPELINE PRINCIPAL
# ==============================================================================

def main():
    print("\n" + "="*60)
    print("🌿 SIMULAÇÃO AQUACROP — Cold Start ALMMo-0")
    print("   Projeto: Sistema de Irrigação | Imperatriz-MA")
    print("="*60)

    if not os.path.exists(arquivo_clima):
        print(f"\n❌ Arquivo '{arquivo_clima}' não encontrado.")
        print("   Execute primeiro o script converter_clima_aquacrop.py")
        return

    print(f"\n1. Lendo clima: {arquivo_clima}...")
    weather_df = prepare_weather(arquivo_clima)
    print(f"   → {len(weather_df)} dias | "
          f"{weather_df['Date'].min().date()} → {weather_df['Date'].max().date()}")

    # --- Simular os 3 cenários ---
    dfs_cenarios = []

    ANOS = [2019, 2020, 2021, 2022, 2023, 2025]

    for cid, nome, net_irr_smt, descricao in CENARIOS:
        df_c = simular_cenario(weather_df, ANOS, net_irr_smt, nome, descricao)

        arquivo_c = f"{nome}.csv"
        df_c.to_csv(arquivo_c, index=False)
        print(f"   ✅ Salvo: {arquivo_c} ({len(df_c)} linhas)")

        dfs_cenarios.append(df_c)

    # --- Dataset combinado ---
    print(f"\n{'='*60}")
    print("2. Consolidando dataset final...")

    df_final = pd.concat(dfs_cenarios, ignore_index=True)

    # Colunas para o ALMMo-0 (conforme briefing atualizado)
    colunas_almmo = [
        'tensao_solo_kpa',   # input 1: tensão do solo ontem (kPa)
        'chuva_acum_3d_mm',  # input 2: chuva acumulada hoje+amanhã+depois (mm)
        'tmax_max_3d_c',     # input 3: tmax máxima hoje+amanhã+depois (°C)
        'dap',               # input 4: dias após plantio
        'classe_irrigacao',  # target:  classe 0-4
    ]

    # Distribuição global de classes
    print(f"\n📊 Distribuição de classes (dataset combinado):")
    dist  = df_final['classe_irrigacao'].value_counts().sort_index()
    total = len(df_final)
    for cls, cnt in dist.items():
        barra = '█' * int(cnt / total * 40)
        print(f"   Classe {cls}: {cnt:5d} ({cnt/total*100:5.1f}%) {barra}")

    if dist.get(0, 0) / total > 0.70:
        print(f"\n⚠️  Classe 0 representa mais de 70% do dataset.")
        print("   Documente isso no relatorio_dataset.md e avalie oversampling.")

    # Amostras por cenário × ano
    print(f"\n📊 Amostras por cenário e ano:")
    print(df_final.groupby(['cenario', 'ano']).size().unstack(fill_value=0).to_string())

    # Estatísticas descritivas
    print(f"\n📊 Estatísticas descritivas:")
    print(f"   {'Variável':<22} {'Mín':>8} {'Média':>8} {'Máx':>8} {'NaN':>6}")
    print(f"   {'-'*57}")
    for col in ['tensao_solo_kpa', 'chuva_acum_3d_mm', 'tmax_max_3d_c', 'dap']:
        s = df_final[col]
        print(f"   {col:<22} {s.min():>8.2f} {s.mean():>8.2f} "
              f"{s.max():>8.2f} {s.isna().sum():>5d}")

    # --- Salvar CSVs finais ---
    df_almmo = df_final[colunas_almmo].copy()
    df_almmo.to_csv('dataset_cold_start.csv', index=False)
    df_final.to_csv('dataset_cold_start_completo.csv', index=False)

    # --- Gráfico de distribuição de classes ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    fig.suptitle('Distribuição de Classes por Cenário', fontsize=12, fontweight='bold')
    cores = ['#4CAF50', '#2196F3', '#FF9800', '#F44336', '#9C27B0']

    for ax, (cid, nome, net_irr_smt, _desc) in zip(axes, CENARIOS):
        df_c   = df_final[df_final['cenario'] == nome]
        counts = df_c['classe_irrigacao'].value_counts().sort_index()
        ax.bar([str(k) for k in counts.index],
               counts.values,
               color=[cores[k] for k in counts.index])
        ax.set_title(f"Cenário {cid}: {nome.split('_')[-1].capitalize()}")
        ax.set_xlabel('Classe de Irrigação')
        ax.set_ylabel('Nº de dias')
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('distribuicao_classes_cenarios.png', dpi=150, bbox_inches='tight')
    print(f"\n📈 Gráfico salvo: distribuicao_classes_cenarios.png")
    plt.close()

    # --- Gráfico de correlação ---
    features = ['tensao_solo_kpa', 'chuva_acum_3d_mm', 'tmax_max_3d_c', 'dap', 'classe_irrigacao']
    labels   = ['Tensão\n(kPa)', 'Chuva\n3d (mm)', 'Tmax\n3d (°C)', 'DAP', 'Classe\nIrrig.']

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Análise de Correlação — Dataset Cold Start ALMMo-0',
                 fontsize=13, fontweight='bold')

    # --- Subplot 1: matriz de correlação (Pearson) ---
    corr = df_final[features].corr()
    im = axes[0].imshow(corr.values, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_yticks(range(len(labels)))
    axes[0].set_xticklabels(labels, fontsize=9)
    axes[0].set_yticklabels(labels, fontsize=9)
    axes[0].set_title('Matriz de Correlação (Pearson)', fontsize=11)
    for i in range(len(features)):
        for j in range(len(features)):
            val = corr.values[i, j]
            cor_txt = 'white' if abs(val) > 0.6 else 'black'
            axes[0].text(j, i, f'{val:.2f}', ha='center', va='center',
                         fontsize=9, color=cor_txt, fontweight='bold')
    plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    # --- Subplot 2: correlação de cada feature com a classe (por cenário) ---
    cenario_nomes = ['cenario1_otimo', 'cenario2_deficit', 'cenario3_excesso']
    cenario_labels = ['Ótimo', 'Déficit', 'Excesso']
    feat_labels    = ['Tensão\n(kPa)', 'Chuva\n3d (mm)', 'Tmax\n3d (°C)', 'DAP']
    feats          = ['tensao_solo_kpa', 'chuva_acum_3d_mm', 'tmax_max_3d_c', 'dap']

    x      = np.arange(len(feats))
    width  = 0.25
    cores_c = ['#2196F3', '#F44336', '#4CAF50']

    for i, (cn, cl, cor) in enumerate(zip(cenario_nomes, cenario_labels, cores_c)):
        df_c  = df_final[df_final['cenario'] == cn]
        corrs = [df_c[f].corr(df_c['classe_irrigacao']) for f in feats]
        axes[1].bar(x + i * width, corrs, width, label=cl, color=cor, alpha=0.85)

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
    plt.savefig('correlacao_features.png', dpi=150, bbox_inches='tight')
    print(f"📈 Gráfico salvo: correlacao_features.png")
    plt.close()

    # --- Resumo final ---
    print("\n" + "="*60)
    print(f"✅ dataset_cold_start.csv          → {len(df_almmo)} linhas | {len(colunas_almmo)} colunas")
    print(f"✅ dataset_cold_start_completo.csv → {len(df_final)} linhas | com metadados")
    print(f"\nColunas finais: {colunas_almmo}")
    print("\n→ Próximo passo: dataset_cold_start.csv → ALMMo-0 cold_start()")
    print("="*60)


if __name__ == "__main__":
    main()
