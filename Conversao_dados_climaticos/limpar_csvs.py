import pandas as pd
import numpy as np
import os
import glob

# --- CONFIGURAÇÕES ---
PASTA_ENTRADA = 'Dados__inmet'
ARQUIVO_SAIDA_LIMPO = 'dados_imperatriz_limpos_unificado(2022_2025).csv'

def carregar_e_unificar(pasta):
    """Lê todos os CSVs e remove duplicatas brutas."""
    arquivos = glob.glob(os.path.join(pasta, '*.CSV')) + glob.glob(os.path.join(pasta, '*.csv'))
    print(f"Arquivos encontrados: {len(arquivos)}")
    print(f"Lista: {[os.path.basename(x) for x in arquivos]}")
    
    dfs = []
    for arq in arquivos:
        try:
            df = pd.read_csv(arq, sep=';', decimal=',', skiprows=8, encoding='latin-1', on_bad_lines='skip')
            dfs.append(df)
        except Exception as e:
            print(f"Erro ao ler {arq}: {e}")
    
    if not dfs: return None
    
    df_total = pd.concat(dfs, ignore_index=True)
    
    # --- CORREÇÃO DE DUPLICATAS ---
    # Remove linhas que sejam exatamente iguais (caso tenha arquivos repetidos)
    linhas_antes = len(df_total)
    df_total.drop_duplicates(inplace=True)
    linhas_depois = len(df_total)
    
    if linhas_antes != linhas_depois:
        print(f"AVISO: {linhas_antes - linhas_depois} linhas duplicadas removidas!")
        
    return df_total

def limpar_dados(df):
    """Limpeza, conversão e agregação diária."""
    
    df.columns = df.columns.str.strip()
    mapa_cols = {
        'DATA (YYYY-MM-DD)': 'Data', 'Data': 'Data',
        'HORA (UTC)': 'Hora', 'Hora UTC': 'Hora',
        'PRECIPITAÇÃO TOTAL, HORÁRIO (mm)': 'precip',
        'TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)': 'tmax',
        'TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)': 'tmin',
        'UMIDADE RELATIVA DO AR, HORARIA (%)': 'rh',
        'VENTO, VELOCIDADE HORARIA (m/s)': 'vento',
        'RADIACAO GLOBAL (Kj/m²)': 'rad'
    }
    df.rename(columns=mapa_cols, inplace=True)

    # Conversão de Data/Hora
    df['Hora_Str'] = df['Hora'].astype(str).str.replace(' UTC', '').str.zfill(4)
    df['Hora_Fmt'] = df['Hora_Str'].str[:2] + ':' + df['Hora_Str'].str[2:]
    df['Data'] = df['Data'].astype(str).str.replace('/', '-')
    
    df['Data_Hora'] = pd.to_datetime(df['Data'] + ' ' + df['Hora_Fmt'], dayfirst=True, format='mixed')
    
    cols_num = ['tmin', 'tmax', 'precip', 'rh', 'vento', 'rad']
    for col in cols_num:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df.set_index('Data_Hora', inplace=True)
    
    # --- SEGURANÇA EXTRA CONTRA DUPLICATAS DE HORÁRIO ---
    # Se houver duas linhas para "2022-01-01 12:00", mantemos apenas a primeira.
    # Isso evita somar a chuva duas vezes se o CSV estiver bagunçado.
    df = df[~df.index.duplicated(keep='first')]

    # Agregação Diária
    # Precipitação = SOMA (Fisicamente correto)
    # Temperatura = MÍNIMO / MÁXIMO
    # Outros = MÉDIA
    diario = df.resample('D').agg({
        'tmin': 'min',
        'tmax': 'max',
        'precip': 'sum', 
        'rh': 'mean',
        'vento': 'mean',
        'rad': 'sum'
    })

    # Reindexação para preencher buracos de dias
    idx_completo = pd.date_range(start=diario.index.min(), end=diario.index.max(), freq='D')
    diario = diario.reindex(idx_completo)
    
    # Preenchimento
    diario['precip'].fillna(0, inplace=True) # Dias sem dados de chuva assumem 0
    
    cols_interp = ['tmin', 'tmax', 'rh', 'vento', 'rad']
    diario[cols_interp] = diario[cols_interp].interpolate(method='time')
    diario.fillna(diario.mean(), inplace=True)

    return diario

# --- EXECUÇÃO ---
print("--- LIMPEZA DE DADOS (COM REMOÇÃO DE DUPLICATAS) ---")
df_bruto = carregar_e_unificar(PASTA_ENTRADA)

if df_bruto is not None:
    df_limpo = limpar_dados(df_bruto)
    
    # Verificação rápida de chuva extrema
    max_chuva = df_limpo['precip'].max()
    print(f"Máxima chuva diária encontrada após limpeza: {max_chuva:.2f} mm")
    if max_chuva > 150:
        print("ALERTA: Ainda há valores de chuva muito altos. Verifique se o arquivo CSV original contém erros.")
    
    df_limpo.to_csv(ARQUIVO_SAIDA_LIMPO, sep=';', decimal='.')
    print(f"Arquivo salvo: {ARQUIVO_SAIDA_LIMPO}")