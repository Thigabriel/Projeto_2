"""
=============================================================================
CONVERSÃO: dados_climaticos_tratados.csv → formato AquaCrop-OSPy (.txt)
Projeto: Sistema de Irrigação com ALMMo-0 | Imperatriz-MA
=============================================================================

Entrada : dados_climaticos_tratados.csv (gerado pelo pipeline INMET)
Saída   : imperatriz_climate.txt

Formato de saída (separado por TAB, sem índice):
  Day  Month  Year  Tmin(C)  Tmax(C)  Prcp(mm)  Et0(mm)

Nota sobre ETo: usamos a coluna eto_mm (Hargreaves-Samani).
A superestimativa conhecida do HS em clima tropical úmido está documentada
no relatorio_dataset.md — aceitável para cold start do ALMMo-0.

Autor: [Seu nome]
Data : 2025
=============================================================================
"""

import pandas as pd
from pathlib import Path

# =============================================================================
# CONFIGURAÇÕES
# =============================================================================

ARQUIVO_ENTRADA = "dados_climaticos_tratados.csv"
ARQUIVO_SAIDA   = "imperatriz_climate.txt"

# =============================================================================
# CONVERSÃO
# =============================================================================

def converter_para_aquacrop(arquivo_entrada: str, arquivo_saida: str) -> None:

    # Carregar CSV tratado
    caminho = Path(arquivo_entrada)
    if not caminho.exists():
        raise FileNotFoundError(
            f"❌ Arquivo não encontrado: '{arquivo_entrada}'\n"
            f"   Execute primeiro o pipeline Tratar_inmet_bdmep.py"
        )

    df = pd.read_csv(arquivo_entrada, sep=';', index_col='data', parse_dates=True)

    print(f"✅ Carregado: {len(df)} dias | {df.index.min().date()} → {df.index.max().date()}")
    print(f"   Anos presentes: {sorted(df.index.year.unique())}")

    # Verificar colunas necessárias
    colunas_necessarias = ['tmin_c', 'tmax_c', 'chuva_mm', 'eto_mm']
    faltando = [c for c in colunas_necessarias if c not in df.columns]
    if faltando:
        raise ValueError(f"❌ Colunas ausentes no CSV: {faltando}")

    # Verificar NaN nas colunas críticas
    for col in colunas_necessarias:
        n_nan = df[col].isna().sum()
        if n_nan > 0:
            print(f"⚠️  {col}: {n_nan} NaN — serão preenchidos com 0 (precipitação) ou média (temperatura/ETo)")

    # Preencher NaN residuais de forma conservadora
    df['chuva_mm']  = df['chuva_mm'].fillna(0.0)
    df['tmax_c']    = df['tmax_c'].fillna(df['tmax_c'].mean())
    df['tmin_c']    = df['tmin_c'].fillna(df['tmin_c'].mean())
    df['eto_mm']    = df['eto_mm'].fillna(df['eto_mm'].mean())

    # Montar DataFrame no formato AquaCrop
    df_aquacrop = pd.DataFrame({
        'Day':      df.index.day,
        'Month':    df.index.month,
        'Year':     df.index.year,
        'Tmin(C)':  df['tmin_c'].round(2),
        'Tmax(C)':  df['tmax_c'].round(2),
        'Prcp(mm)': df['chuva_mm'].round(2),
        'Et0(mm)':  df['eto_mm'].round(6),
    })

    # Salvar como TXT separado por TAB (idêntico ao formato de Córdoba)
    df_aquacrop.to_csv(
        arquivo_saida,
        sep='\t',
        index=False,
        lineterminator='\r\n'  # CRLF — padrão do AquaCrop
    )

    print(f"\n✅ Arquivo gerado: {arquivo_saida}")
    print(f"   {len(df_aquacrop)} linhas | separador: TAB | terminador: CRLF")
    print(f"\n   Primeiras 3 linhas:")
    print(df_aquacrop.head(3).to_string(index=False))
    print(f"\n   Últimas 3 linhas:")
    print(df_aquacrop.tail(3).to_string(index=False))

    # Estatísticas rápidas de validação
    print(f"\n📊 Estatísticas de validação:")
    print(f"   {'Variável':<12} {'Mín':>8} {'Média':>8} {'Máx':>8}")
    print(f"   {'-'*40}")
    for col, label in [('Tmin(C)', 'Tmin'), ('Tmax(C)', 'Tmax'),
                        ('Prcp(mm)', 'Chuva'), ('Et0(mm)', 'ETo')]:
        s = df_aquacrop[col]
        print(f"   {label:<12} {s.min():>8.2f} {s.mean():>8.2f} {s.max():>8.2f}")


if __name__ == "__main__":
    converter_para_aquacrop(ARQUIVO_ENTRADA, ARQUIVO_SAIDA)
