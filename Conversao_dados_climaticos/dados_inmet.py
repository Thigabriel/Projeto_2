import pandas as pd
import numpy as np
import math
import os
import glob

# --- CONFIGURAÇÕES ---
PASTA_DADOS = 'Dados__inmet' 
ARQUIVO_SAIDA = 'imperatriz_climate.txt'

# Constantes geográficas de Imperatriz - MA
LATITUDE = -5.52  
ALTITUDE = 95.0   

def carregar_dados_inmet(pasta):
    """Lê todos os arquivos .CSV da pasta especificada."""
    caminho_busca = os.path.join(pasta, '*.CSV')
    arquivos = glob.glob(caminho_busca)
    if not arquivos:
        arquivos = glob.glob(os.path.join(pasta, '*.csv'))
    
    print(f"Arquivos encontrados em '{pasta}': {len(arquivos)}")
    
    dfs = []
    for arquivo in arquivos:
        try:
            # Pula 8 linhas de cabeçalho, separador ';', decimal ','
            df = pd.read_csv(arquivo, sep=';', decimal=',', skiprows=8, encoding='latin-1', on_bad_lines='skip')
            dfs.append(df)
        except Exception as e:
            print(f"Erro ao ler {arquivo}: {e}")

    if not dfs:
        return None

    return pd.concat(dfs, ignore_index=True)

def limpar_e_padronizar(df):
    """Renomeia colunas e cria o índice de data/hora."""
    # Remove espaços dos nomes das colunas
    df.columns = df.columns.str.strip()
    
    # Mapeamento flexível para nomes de colunas do INMET
    mapa = {
        'DATA (YYYY-MM-DD)': 'Data',
        'Data': 'Data',
        'HORA (UTC)': 'Hora',
        'Hora UTC': 'Hora',
        'PRECIPITAÇÃO TOTAL, HORÁRIO (mm)': 'precip',
        'TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)': 'tmax',
        'TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)': 'tmin',
        'UMIDADE RELATIVA DO AR, HORARIA (%)': 'rh',
        'VENTO, VELOCIDADE HORARIA (m/s)': 'vento',
        'RADIACAO GLOBAL (Kj/m²)': 'rad'
    }
    df.rename(columns=mapa, inplace=True)

    # Criar Data_Hora
    df['Hora_Str'] = df['Hora'].astype(str).str.replace(' UTC', '').str.zfill(4)
    df['Hora_Fmt'] = df['Hora_Str'].str[:2] + ':' + df['Hora_Str'].str[2:]
    df['Data'] = df['Data'].astype(str).str.replace('/', '-')
    
    # format='mixed' resolve o problema de datas DD/MM vs YYYY-MM
    df['Data_Hora'] = pd.to_datetime(df['Data'] + ' ' + df['Hora_Fmt'], dayfirst=True, format='mixed')
    df.set_index('Data_Hora', inplace=True)

    # Converter para numérico
    cols_numericas = ['tmin', 'tmax', 'precip', 'rh', 'vento', 'rad']
    for col in cols_numericas:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    return df

def agregar_diario(df):
    """Agrega dados horários para diários."""
    diario = df.resample('D').agg({
        'tmin': 'min',
        'tmax': 'max',
        'precip': 'sum',
        'rh': 'mean',
        'vento': 'mean',
        'rad': 'sum'
    })
    
    diario.dropna(subset=['tmin', 'tmax'], inplace=True)
    
    # Conversões de Unidade
    diario['rad_mj'] = diario['rad'] / 1000.0 # KJ -> MJ
    diario['vento_2m'] = diario['vento'] * 0.748 # 10m -> 2m
    
    return diario

def calcular_eto(row):
    """Calcula ETo diária (mm/dia) usando Penman-Monteith FAO-56."""
    Tmin, Tmax = row['tmin'], row['tmax']
    Rh, u2, Rs = row['rh'], row['vento_2m'], row['rad_mj']
    
    if any(pd.isna([Tmin, Tmax, Rh, u2, Rs])):
        return np.nan

    Tmean = (Tmin + Tmax) / 2
    lat_rad = math.radians(LATITUDE)
    
    es_Tmin = 0.6108 * np.exp((17.27 * Tmin) / (Tmin + 237.3))
    es_Tmax = 0.6108 * np.exp((17.27 * Tmax) / (Tmax + 237.3))
    es = (es_Tmin + es_Tmax) / 2
    ea = (Rh / 100.0) * es
    
    delta = (4098 * (0.6108 * np.exp((17.27 * Tmean) / (Tmean + 237.3)))) / ((Tmean + 237.3) ** 2)
    P = 101.3 * ((293 - 0.0065 * ALTITUDE) / 293) ** 5.26
    gamma = 0.000665 * P
    
    doy = row.name.dayofyear
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    decl = 0.409 * np.sin((2 * np.pi * doy / 365) - 1.39)
    ws = np.arccos(np.clip(-np.tan(lat_rad) * np.tan(decl), -1, 1))
    Ra = (24 * 60 / np.pi) * 0.0820 * dr * (ws * np.sin(lat_rad) * np.sin(decl) + np.cos(lat_rad) * np.cos(decl) * np.sin(ws))
    
    Rns = (1 - 0.23) * Rs
    sigma = 4.903e-9
    Rnl = sigma * (((Tmax + 273.16)**4 + (Tmin + 273.16)**4) / 2) * (0.34 - 0.14 * np.sqrt(ea)) * (1.35 * (Rs / (0.75 * Ra)) - 0.35)
    Rn = Rns - Rnl
    G = 0
    
    numerador = 0.408 * delta * (Rn - G) + gamma * (900 / (Tmean + 273)) * u2 * (es - ea)
    denominador = delta + gamma * (1 + 0.34 * u2)
    
    return numerador / denominador if denominador != 0 else 0

# --- EXECUÇÃO PRINCIPAL ---

print("--- INICIANDO PROCESSAMENTO ---")
df_bruto = carregar_dados_inmet(PASTA_DADOS)

if df_bruto is not None:
    print("Limpando e agregando...")
    df_limpo = limpar_e_padronizar(df_bruto)
    df_diario = agregar_diario(df_limpo)
    
    print("Calculando ETo...")
    df_diario['ETo'] = df_diario.apply(calcular_eto, axis=1)
    
    # Preencher falhas
    media_eto = df_diario['ETo'].mean()
    df_diario['ETo'].fillna(media_eto, inplace=True)
    
    # --- 5. SALVAR NO FORMATO CORDOBA ---
    # Filtrar colunas necessárias
    df_final = df_diario[['tmin', 'tmax', 'precip', 'ETo']].copy()
    df_final = df_final.round(2)
    
    # Filtrar ano
    df_final = df_final[df_final.index.year >= 2022]
    
    # Criar colunas separadas para Dia, Mês e Ano
    df_final['Day'] = df_final.index.day
    df_final['Month'] = df_final.index.month
    df_final['Year'] = df_final.index.year
    
    # Reordenar para o formato exato: Day Month Year Tmin Tmax Precip ETo
    # Ajustar nomes das colunas para ficarem idênticos ao exemplo (opcional, mas bom para leitura)
    df_export = df_final[['Day', 'Month', 'Year', 'tmin', 'tmax', 'precip', 'ETo']]
    
    print(f"Salvando arquivo '{ARQUIVO_SAIDA}'...")
    
    # Salvar usando tabulação (\t) ou espaços largos para ficar igual ao exemplo
    with open(ARQUIVO_SAIDA, 'w') as f:
        # Cabeçalho exato do arquivo exemplo
        f.write("Day\tMonth\tYear\tTmin(C)\tTmax(C)\tPrcp(mm)\tEt0(mm)\n")
        
        for _, row in df_export.iterrows():
            # Formata: Dia Mês Ano (inteiros) e o resto (floats) separados por tabulação
            f.write(f"{int(row['Day'])}\t{int(row['Month'])}\t{int(row['Year'])}\t"
                    f"{row['tmin']:.2f}\t{row['tmax']:.2f}\t{row['precip']:.2f}\t{row['ETo']:.2f}\n")
            
    print("--- CONCLUÍDO ---")
    print(f"Arquivo gerado no formato correto (Day Month Year...).")
    print(df_export.head())

else:
    print("ERRO: Nenhum dado processado.")