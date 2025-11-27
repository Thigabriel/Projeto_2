import pandas as pd
import numpy as np
import math

# --- CONFIGURAÇÕES ---
ARQUIVO_ENTRADA = 'dados_imperatriz_limpos_unificado(2022_2025).csv'
ARQUIVO_SAIDA_AQUACROP = 'imperatriz_climate.txt'

# Constantes de Imperatriz - MA
LATITUDE = -5.52
ALTITUDE = 95.0

def calcular_eto_pm(row):
    """Calcula ETo (Penman-Monteith FAO-56)."""
    # Lê as colunas do CSV limpo
    Tmin, Tmax = row['tmin'], row['tmax']
    Rh, u2_raw, Rs_raw = row['rh'], row['vento'], row['rad']
    
    # Conversões de Unidade
    u2 = u2_raw * 0.748       # Vento de 10m -> 2m
    Rs = Rs_raw / 1000.0      # Radiação KJ -> MJ

    Tmean = (Tmin + Tmax) / 2
    lat_rad = math.radians(LATITUDE)
    
    # Pressão de vapor
    es_Tmin = 0.6108 * np.exp((17.27 * Tmin) / (Tmin + 237.3))
    es_Tmax = 0.6108 * np.exp((17.27 * Tmax) / (Tmax + 237.3))
    es = (es_Tmin + es_Tmax) / 2
    ea = (Rh / 100.0) * es
    
    delta = (4098 * (0.6108 * np.exp((17.27 * Tmean) / (Tmean + 237.3)))) / ((Tmean + 237.3) ** 2)
    P = 101.3 * ((293 - 0.0065 * ALTITUDE) / 293) ** 5.26
    gamma = 0.000665 * P
    
    # Radiação Extraterrestre (Ra)
    # O índice do DataFrame é DatetimeIndex? Se sim, usamos dayofyear. 
    # Se lemos do CSV, precisamos converter a coluna de data ou usar row.name se definido.
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
    
    num = 0.408 * delta * (Rn - G) + gamma * (900 / (Tmean + 273)) * u2 * (es - ea)
    den = delta + gamma * (1 + 0.34 * u2)
    
    return num / den if den != 0 else 0

# --- EXECUÇÃO ---
print("--- ETAPA 2: GERAÇÃO DO ARQUIVO AQUACROP ---")

try:
    # Carregar dados limpos
    # Precisamos converter a primeira coluna (index) de volta para datetime
    df = pd.read_csv(ARQUIVO_ENTRADA, sep=';', index_col=0, parse_dates=True)
    print(f"Dados limpos carregados: {len(df)} dias.")
    
    # Calcular ETo
    print("Calculando ETo...")
    df['ETo'] = df.apply(calcular_eto_pm, axis=1)
    
    # Formatar para AquaCrop
    # Filtrar colunas e arredondar
    df_final = df[['tmin', 'tmax', 'precip', 'ETo']].copy().round(2)
    
    # Criar colunas de data separadas (Day, Month, Year)
    df_final['Day'] = df_final.index.day
    df_final['Month'] = df_final.index.month
    df_final['Year'] = df_final.index.year
    
    # Selecionar e ordenar
    colunas_saida = ['Day', 'Month', 'Year', 'tmin', 'tmax', 'precip', 'ETo']
    
    print(f"Salvando '{ARQUIVO_SAIDA_AQUACROP}'...")
    with open(ARQUIVO_SAIDA_AQUACROP, 'w') as f:
        # Cabeçalho padrão AquaCrop
        f.write("Day\tMonth\tYear\tTmin(C)\tTmax(C)\tPrcp(mm)\tEt0(mm)\n")
        
        for _, row in df_final.iterrows():
            # Escreve com tabulação
            f.write(f"{int(row['Day'])}\t{int(row['Month'])}\t{int(row['Year'])}\t"
                    f"{row['tmin']:.2f}\t{row['tmax']:.2f}\t{row['precip']:.2f}\t{row['ETo']:.2f}\n")
            
    print("Arquivo de clima pronto para uso!")

except FileNotFoundError:
    print(f"ERRO: Arquivo '{ARQUIVO_ENTRADA}' não encontrado.")
    print("Execute o script 'limpar_csvs.py' primeiro.")
except Exception as e:
    print(f"Ocorreu um erro: {e}")