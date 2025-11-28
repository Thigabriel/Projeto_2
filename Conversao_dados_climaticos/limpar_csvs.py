import pandas as pd
import numpy as np
import os
import glob

# --- CONFIGURAÇÕES ---
PASTA_ENTRADA = 'Dados__inmet'
ARQUIVO_SAIDA_LIMPO = 'dados_imperatriz_limpos_unificado.csv'

def carregar_e_unificar(pasta):
    arquivos = glob.glob(os.path.join(pasta, '*.CSV')) + glob.glob(os.path.join(pasta, '*.csv'))
    print(f"Arquivos encontrados: {len(arquivos)}")
    
    dfs = []
    for arq in arquivos:
        try:
            # Codigo para pular o cabeçalho dos arquivos inmet
            df = pd.read_csv(arq, sep=';', decimal=',', skiprows=8, encoding='latin-1', on_bad_lines='skip')
            dfs.append(df)
        except Exception as e:
            print(f"Erro ao ler {arq}: {e}")
    
    if not dfs: return None
    return pd.concat(dfs, ignore_index=True)

def limpar_dados(df):   
    # 1. Renomear Colunas
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

    # 2. Criar Índice de Data e Hora
    df['Hora_Str'] = df['Hora'].astype(str).str.replace(' UTC', '').str.zfill(4)
    df['Hora_Fmt'] = df['Hora_Str'].str[:2] + ':' + df['Hora_Str'].str[2:]
    # Ajusta data "/" para "-"
    df['Data'] = df['Data'].astype(str).str.replace('/', '-')
    
    # Converte para datetime
    df['Data_Hora'] = pd.to_datetime(df['Data'] + ' ' + df['Hora_Fmt'], dayfirst=True, format='mixed')
    
    # Converter colunas numéricas (erros viram NaN)
    cols_num = ['tmin', 'tmax', 'precip', 'rh', 'vento', 'rad']
    for col in cols_num:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df.set_index('Data_Hora', inplace=True)

    # 3. Seleção de Colunas Relevantes
    diario = df.resample('D').agg({
        'tmin': 'min',
        'tmax': 'max',
        'precip': 'sum',
        'rh': 'mean',
        'vento': 'mean',
        'rad': 'sum'
    })

    # 4. Preenchimento de Lacunas
    idx_completo = pd.date_range(start=diario.index.min(), end=diario.index.max(), freq='D')
    diario = diario.reindex(idx_completo)
    

    # Chuva -> Assume 0 se não tem dado
    diario['precip'].fillna(0, inplace=True)
    # Clima (Temp, Vento, Rad) -> Interpolação linear
    cols_interp = ['tmin', 'tmax', 'rh', 'vento', 'rad']
    diario[cols_interp] = diario[cols_interp].interpolate(method='time')
    
    # Se ainda sobrar NaN (ex: no começo ou fim), preenche com a média da coluna
    diario.fillna(diario.mean(), inplace=True)

    return diario

# --- EXECUÇÃO ---
df_bruto = carregar_e_unificar(PASTA_ENTRADA)

if df_bruto is not None:
    df_limpo = limpar_dados(df_bruto)
    
    # Salva o CSV limpo para conferência ou uso futuro
    df_limpo.to_csv(ARQUIVO_SAIDA_LIMPO, sep=';', decimal='.')
    print(f"Sucesso! Arquivo limpo salvo em: {ARQUIVO_SAIDA_LIMPO}")
    print(df_limpo.head())
else:
    print("Nenhum dado encontrado.")