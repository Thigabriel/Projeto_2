#!/usr/bin/env python3
"""
============================================================================
Calculo das Distribuicoes de Delta por Classe
============================================================================

Responde a pergunta: "dado que o ensemble decidiu classe C no dia T,
qual delta de tensao e esperado no dia T+1?"

Usa: dataset_cold_start_v11_full.csv  (Imperatriz, 2001-2023)

Os valores calculados aqui sao usados como referencia em evolucao_online.py.
Rode este script sempre que gerar um novo dataset de treino para atualizar
os valores hardcoded em DIST_REFERENCIA.

Execucao:
    python calcular_distribuicoes_delta.py
============================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================================
# CONFIGURACAO
# ============================================================================
DATASET_PATH = "dataset_cold_start_v11_full.csv"
NOMES_CLASSE = {0: "C0 (sem irrigar)", 1: "C1 (manutencao)", 2: "C2 (intensiva)"}

# ============================================================================
# CARREGA DATASET
# ============================================================================
p = Path(DATASET_PATH)
if not p.exists():
    print(f"ERRO: {DATASET_PATH} nao encontrado.")
    print("Execute primeiro: python C_Aquacrop/script_simulacao_v11.py")
    exit(1)

df = pd.read_csv(DATASET_PATH)
print(f"Dataset carregado: {len(df)} amostras | {df['grupo_id'].nunique()} simulacoes")
print(f"Colunas: {df.columns.tolist()}\n")

# ============================================================================
# CALCULA DELTA DO DIA SEGUINTE
# Logica:
#   - O dataset tem multiplas simulacoes identificadas por 'grupo_id'
#   - Dentro de cada simulacao, os dias sao ordenados por 'dap'
#   - O delta do dia T+1 e o delta_tensao_kpa deslocado -1 posicao dentro do grupo
#   - Isso representa: "o que aconteceu com a tensao no dia APOS a decisao de hoje"
# ============================================================================
print("Calculando delta do dia seguinte por simulacao (grupo_id)...")
df_sorted = df.sort_values(['grupo_id', 'dap']).copy()
df_sorted['delta_proximo_dia'] = (
    df_sorted.groupby('grupo_id')['delta_tensao_kpa'].shift(-1)
)

# Remove ultimos dias de cada grupo (nao tem dia seguinte)
df_seq = df_sorted.dropna(subset=['delta_proximo_dia'])
removidos = len(df_sorted) - len(df_seq)
print(f"Amostras com dia seguinte valido: {len(df_seq)} (removidos {removidos} ultimos dias de cada simulacao)\n")

# ============================================================================
# DISTRIBUICOES POR CLASSE
# ============================================================================
sep = "=" * 65
print(sep)
print("DISTRIBUICOES DE DELTA DO DIA SEGUINTE POR CLASSE")
print("(uso em evolucao_online.py — DIST_REFERENCIA)")
print(sep)

dist = {}
for c in sorted(df_seq['classe_irrigacao'].unique()):
    s = df_seq[df_seq['classe_irrigacao'] == c]['delta_proximo_dia']
    dist[int(c)] = {
        'media': round(float(s.mean()), 4),
        'std':   round(float(s.std()),  4),
        'p5':    round(float(s.quantile(0.05)), 4),
        'p25':   round(float(s.quantile(0.25)), 4),
        'p50':   round(float(s.median()), 4),
        'p75':   round(float(s.quantile(0.75)), 4),
        'p95':   round(float(s.quantile(0.95)), 4),
        'p99':   round(float(s.quantile(0.99)), 4),
        'n':     int(len(s)),
    }
    d = dist[int(c)]
    print(f"\n{NOMES_CLASSE.get(c, f'C{c}')} — {d['n']} amostras")
    print(f"  Media:  {d['media']:+.3f} kPa")
    print(f"  Std:    {d['std']:.3f} kPa")
    print(f"  p5:     {d['p5']:+.3f} kPa")
    print(f"  p25:    {d['p25']:+.3f} kPa")
    print(f"  Mediana:{d['p50']:+.3f} kPa")
    print(f"  p75:    {d['p75']:+.3f} kPa")
    print(f"  p95:    {d['p95']:+.3f} kPa   ← limiar anomalia (Z≈1.6)")
    print(f"  p99:    {d['p99']:+.3f} kPa   ← limiar anomalia conservador")

# ============================================================================
# SAIDA FORMATADA PARA COPIAR EM evolucao_online.py
# ============================================================================
print(f"\n{sep}")
print("COPIE O BLOCO ABAIXO PARA evolucao_online.py (DIST_REFERENCIA):")
print(sep)
print("DIST_REFERENCIA = {")
for c, d in dist.items():
    print(f"    {c}: {{'media': {d['media']:+.4f}, 'std': {d['std']:.4f}, "
          f"'p5': {d['p5']:.4f}, 'p95': {d['p95']:.4f}, 'p99': {d['p99']:.4f}, 'n': {d['n']}}},")
print("}")

# ============================================================================
# REFERENCIA — delta NO PROPRIO DIA (entrada do modelo, nao consequencia)
# ============================================================================
print(f"\n{sep}")
print("REFERENCIA — DELTA NO PROPRIO DIA (feature de entrada do modelo)")
print("(para contexto — NAO e usado em evolucao_online.py)")
print(sep)
for c in sorted(df['classe_irrigacao'].unique()):
    s = df[df['classe_irrigacao'] == c]['delta_tensao_kpa']
    print(f"  {NOMES_CLASSE.get(c, f'C{c}')}: media={s.mean():+.3f} | std={s.std():.3f} | "
          f"p5={s.quantile(0.05):.2f} | p95={s.quantile(0.95):.2f}")

# ============================================================================
# INTERPRETACAO
# ============================================================================
print(f"\n{sep}")
print("INTERPRETACAO")
print(sep)
print("""
Apos C0 (sem irrigar):
  O solo seca devagar (+0.83 kPa em media). Se amanha o delta passar de
  +4.43 kPa (p95), o solo esta secando muito mais rapido que o historico
  de dias C0 — sinal de que a decisao C0 foi errada para aquele contexto.

Apos C1 (manutencao leve):
  A irrigacao de manutencao mantém a tensao quase estavel (+0.01 kPa).
  Std muito pequeno (0.50). Qualquer delta acima de +0.71 kPa (p95) e
  anomalo — a irrigacao nao segurou a tensao como deveria.

Apos C2 (irrigacao intensiva):
  Tensao cai -25 kPa em media. O solo umidifica fortemente.
  Delta menos negativo que -15.92 kPa (p95) pode indicar irrigacao
  insuficiente ou solo muito seco — nao retreinamos neste caso por
  excesso de fatores confundidores.
""")
