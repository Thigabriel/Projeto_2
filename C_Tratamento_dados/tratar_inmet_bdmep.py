"""
=============================================================================
TRATAMENTO DE DADOS CLIMÁTICOS — INMET BDMEP (Múltiplos Anos)
Projeto: Sistema de Irrigação com ALMMo-0 | Imperatriz-MA
=============================================================================
v2 — Correções:
  - Parser de datetime robusto (múltiplos formatos de hora do BDMEP)
  - Mapeamento de colunas feito ANTES de converter para minúsculas
  - Radiação e pressão corretamente capturadas
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURAÇÕES
# =============================================================================

PASTA_INMET   = "Dados__inmet"
ARQUIVO_SAIDA = "dados_climaticos_tratados.csv"

LATITUDE_GRAUS  = -5.52
ALTITUDE_METROS = 96.0

ANO_INICIO = 2018
ANO_FIM    = 2025

# Anos com falha crítica da estação INMET — descartados das simulações
# 2024: lacuna de 183 dias (fev–ago), inviabiliza a janela maio–julho do tomate
ANOS_EXCLUIR = [2024]

GERAR_GRAFICOS = True

# =============================================================================
# MAPEAMENTO DE COLUNAS
# IMPORTANTE: as chaves devem estar com a capitalização ORIGINAL do BDMEP,
# pois o rename acontece ANTES de converter tudo para minúsculas.
# =============================================================================

MAPA_COLUNAS = {
    # Data e hora
    'Data':                                                       'data',
    'Hora UTC':                                                   'hora',
    'HORA (UTC)':                                                 'hora',

    # Temperatura
    'TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)':              'temp_bulbo_seco',
    'Temp. Ins. (C)':                                             'temp_bulbo_seco',
    'TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)':                'temp_max_hora',
    'TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)':                'temp_min_hora',
    'Temp. Max. (C)':                                             'temp_max_hora',
    'Temp. Min. (C)':                                             'temp_min_hora',

    # Precipitação
    'PRECIPITAÇÃO TOTAL, HORÁRIO (mm)':                           'precipitacao',
    'Chuva (mm)':                                                 'precipitacao',

    # Umidade
    'UMIDADE RELATIVA DO AR, HORARIA (%)':                        'umidade_relativa',
    'Umi. Ins. (%)':                                              'umidade_relativa',
    'UMIDADE REL. MAX. NA HORA ANT. (AUT) (%)':                  'umidade_max_hora',
    'UMIDADE REL. MIN. NA HORA ANT. (AUT) (%)':                  'umidade_min_hora',

    # Vento
    'VENTO, VELOCIDADE HORARIA (m/s)':                            'vento_velocidade',
    'Vel. Vento (m/s)':                                           'vento_velocidade',
    'VENTO, DIREÇÃO HORARIA (gr) (° (gr))':                      'vento_direcao',
    'VENTO, RAJADA MAXIMA (m/s)':                                 'rajada_vento',
    'Raj. Vento (m/s)':                                           'rajada_vento',

    # Radiação — cobre variações com e sem acento
    'RADIAÇÃO GLOBAL (Kj/m²)':                                    'radiacao_kj_m2',
    'RADIACAO GLOBAL (Kj/m²)':                                    'radiacao_kj_m2',

    # Pressão — cobre variações com e sem acento
    'PRESSÃO ATMOSFÉRICA AO NÍVEL DA ESTAÇÃO, HORÁRIA (mB)':     'pressao_atm',
    'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)':     'pressao_atm',
    'Pressão Ins. (hPa)':                                         'pressao_atm',
    'Pressao Ins. (hPa)':                                         'pressao_atm',

    # Ponto de orvalho (não usado no AquaCrop, mas capturado para referência)
    'TEMPERATURA DO PONTO DE ORVALHO (°C)':                      'temp_orvalho',
}


# =============================================================================
# ETAPA 1 — LEITURA DA PASTA
# =============================================================================

def detectar_linhas_cabecalho(caminho: str) -> int:
    """
    Encontra a linha do cabeçalho real do BDMEP.
    Pula as linhas de metadados iniciais (normalmente 8).
    """
    with open(caminho, 'r', encoding='latin-1') as f:
        for i, linha in enumerate(f):
            lower = linha.strip().lower()
            if 'data' in lower and any(
                p in lower for p in ['hora', 'precipita', 'temperatura', 'temp.']
            ):
                return i
    return 8


def carregar_arquivo_unico(caminho: Path) -> pd.DataFrame | None:
    """Carrega um CSV do BDMEP. Retorna None se falhar."""
    n_skip = detectar_linhas_cabecalho(str(caminho))
    for sep in [';', ',', '\t']:
        try:
            df = pd.read_csv(
                caminho,
                skiprows=n_skip,
                sep=sep,
                encoding='latin-1',
                decimal=',',
                na_values=['-9999', '-9999.0', '', ' ', '//', 'null', 'NULL'],
                low_memory=False
            )
            if len(df.columns) >= 4 and len(df) > 0:
                return df
        except Exception:
            continue
    return None


def carregar_pasta_bdmep(pasta: str) -> pd.DataFrame:
    """Lê todos os CSVs da pasta e concatena."""
    print(f"\n{'='*60}")
    print(f"📂 Lendo pasta: {pasta}")

    pasta_path = Path(pasta)
    if not pasta_path.exists():
        raise FileNotFoundError(
            f"❌ Pasta não encontrada: '{pasta}'\n"
            f"   Ajuste PASTA_INMET no topo do script."
        )

    arquivos = sorted(set(
        list(pasta_path.glob("*.csv")) + list(pasta_path.glob("*.CSV"))
    ))
    if not arquivos:
        raise FileNotFoundError(f"❌ Nenhum .csv encontrado em '{pasta}'")

    print(f"   {len(arquivos)} arquivo(s) encontrado(s):\n")
    dfs = []
    for arq in arquivos:
        df_arq = carregar_arquivo_unico(arq)
        if df_arq is not None:
            print(f"   ✅  {arq.name:<55s} {len(df_arq):>8,} linhas")
            dfs.append(df_arq)
        else:
            print(f"   ❌  {arq.name:<55s} FALHA")

    if not dfs:
        raise ValueError("❌ Nenhum arquivo carregado.")

    df = pd.concat(dfs, ignore_index=True)
    print(f"\n   ✅  Total: {len(df):,} linhas de {len(dfs)} arquivo(s)")
    return df


# =============================================================================
# ETAPA 2 — PADRONIZAÇÃO: rename ANTES de lowercase
# =============================================================================

def padronizar_colunas(df: pd.DataFrame) -> pd.DataFrame:
    """
    1. Strip de espaços
    2. Rename com MAPA_COLUNAS (capitalização original)
    3. Só depois converte tudo para minúsculas
    4. Segunda passagem: mapeia variações que só aparecem em lowercase
    """
    # Passo 1: limpar espaços
    df.columns = df.columns.str.strip()

    # Passo 2: rename com chaves no formato original do BDMEP
    df = df.rename(columns=MAPA_COLUNAS)

    # Passo 3: converter para minúsculas (agora que as colunas importantes já foram renomeadas)
    df.columns = [c.lower().strip() for c in df.columns]

    # Passo 4: segunda passagem para variações que aparecem já em lowercase
    mapa_lowercase = {
        'radiacao global (kj/m²)':                               'radiacao_kj_m2',
        'pressao atmosferica ao nivel da estacao, horaria (mb)': 'pressao_atm',
        'pressão atmosferica max.na hora ant. (aut) (mb)':       'pressao_max',
        'pressão atmosferica min. na hora ant. (aut) (mb)':      'pressao_min',
        'temperatura do ponto de orvalho (°c)':                  'temp_orvalho',
        'temperatura orvalho max. na hora ant. (aut) (°c)':      'temp_orvalho_max',
        'temperatura orvalho min. na hora ant. (aut) (°c)':      'temp_orvalho_min',
        'vento, rajada maxima (m/s)':                             'rajada_vento',
        'unnamed: 19':                                            '_col_vazia',
    }
    df = df.rename(columns=mapa_lowercase)

    # Reportar resultado
    esperadas = {'data', 'hora', 'temp_bulbo_seco', 'temp_max_hora',
                 'temp_min_hora', 'precipitacao', 'umidade_relativa',
                 'vento_velocidade', 'radiacao_kj_m2'}
    presentes    = esperadas & set(df.columns)
    nao_mapeadas = set(df.columns) - esperadas - {
        'umidade_max_hora', 'umidade_min_hora', 'vento_direcao', 'rajada_vento',
        'pressao_atm', 'pressao_max', 'pressao_min', 'temp_orvalho',
        'temp_orvalho_max', 'temp_orvalho_min', '_col_vazia'
    }

    print(f"\n📋 Colunas reconhecidas: {sorted(presentes)}")
    if 'radiacao_kj_m2' in df.columns:
        print("   ✅  Radiação mapeada com sucesso")
    else:
        print("   ⚠️  Radiação NÃO encontrada — ETo usará estimativa")
    if nao_mapeadas:
        print(f"   Ainda não mapeadas: {sorted(nao_mapeadas)}")

    return df


# =============================================================================
# ETAPA 3 — PARSING DE DATETIME (robusto para múltiplos formatos BDMEP)
# =============================================================================

def normalizar_hora(serie: pd.Series) -> pd.Series:
    """
    O BDMEP usa diferentes formatos de hora ao longo dos anos:
      - '0000', '0100', ..., '2300'   (4 dígitos sem ':')
      - '00:00 UTC', '01:00 UTC'      (com sufixo UTC)
      - '0', '100', '1200'            (sem zero à esquerda)
      - '00', '01', ..., '23'         (só hora, 2 dígitos)
    Normaliza tudo para 'HH:MM'.
    """
    s = serie.astype(str).str.strip()

    # Remover sufixo ' UTC' se existir
    s = s.str.replace(r'\s*UTC\s*', '', regex=True).str.strip()

    # Se já tem ':', extrair só HH:MM
    mask_colon = s.str.contains(':', na=False)
    s_colon = s[mask_colon].str[:5]  # pega 'HH:MM' dos primeiros 5 chars

    # Se não tem ':', tratar como número
    s_num = s[~mask_colon].str.replace(r'\D', '', regex=True).str.zfill(4)
    s_num = s_num.str[:2] + ':' + s_num.str[2:4]

    s[mask_colon]  = s_colon
    s[~mask_colon] = s_num

    return s


def parsear_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converte colunas 'data' e 'hora' para índice datetime.
    Cobre todos os formatos conhecidos do BDMEP (2019-2025):
      - Data: AAAA/MM/DD  ou  DD/MM/AAAA
      - Hora: '0000 UTC', '0100 UTC', ..., '2300 UTC'
    """
    # Normalizar hora e montar string datetime completa
    hora_norm = normalizar_hora(df['hora'])

    print(f"\n🔎 Diagnóstico de formato:")
    print(f"   hora bruta  : {df['hora'].dropna().unique()[:3].tolist()}")
    print(f"   hora norm.  : {hora_norm.dropna().unique()[:3].tolist()}")
    print(f"   data bruta  : {df['data'].dropna().unique()[:3].tolist()}")

    dt_str = df['data'].astype(str).str.strip() + ' ' + hora_norm

    # Tentar formatos na ordem de probabilidade para os dados de Imperatriz
    formatos = [
        '%Y/%m/%d %H:%M',   # 2019/01/01 00:00  ← formato atual BDMEP
        '%d/%m/%Y %H:%M',   # 01/01/2019 00:00  ← formato antigo
        '%Y-%m-%d %H:%M',
        '%d-%m-%Y %H:%M',
    ]

    df['datetime'] = pd.NaT
    for fmt in formatos:
        ainda_nulos = df['datetime'].isna()
        if not ainda_nulos.any():
            break
        parsed = pd.to_datetime(dt_str[ainda_nulos], format=fmt, errors='coerce')
        resolvidos = parsed.notna()
        if resolvidos.any():
            df.loc[ainda_nulos[ainda_nulos].index[resolvidos], 'datetime'] = \
                parsed[resolvidos].values
            print(f"   → Formato '{fmt}': {resolvidos.sum():,} linhas resolvidas")

    df['datetime'] = pd.to_datetime(df['datetime'])
    n_inv = df['datetime'].isna().sum()
    if n_inv > 0:
        falhas = df[df['datetime'].isna()][['data', 'hora']].head(3)
        print(f"\n⚠️  {n_inv:,} linhas com datetime inválido removidas")
        print(f"   Exemplos:\n{falhas.to_string()}")
        df = df.dropna(subset=['datetime'])

    df = df.set_index('datetime').sort_index()

    n_dup = df.index.duplicated().sum()
    if n_dup > 0:
        print(f"   ⚠️  {n_dup:,} timestamps duplicados removidos")
        df = df[~df.index.duplicated(keep='first')]

    print(f"\n📅 Série: {df.index.min().date()} → {df.index.max().date()}")
    print(f"   Anos : {sorted(df.index.year.unique())}")
    return df


def converter_numericas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converte colunas meteorológicas para float.
    Remove colunas duplicadas antes de converter (podem surgir da concatenação).
    """
    # Remover colunas duplicadas mantendo a primeira ocorrência
    colunas_dup = df.columns[df.columns.duplicated()].tolist()
    if colunas_dup:
        print(f"\n⚠️  Colunas duplicadas removidas: {colunas_dup}")
        df = df.loc[:, ~df.columns.duplicated(keep='first')]

    colunas = [
        'temp_bulbo_seco', 'temp_max_hora', 'temp_min_hora',
        'precipitacao', 'umidade_relativa', 'umidade_max_hora',
        'umidade_min_hora', 'vento_velocidade', 'radiacao_kj_m2',
        'pressao_atm', 'temp_orvalho'
    ]
    for col in colunas:
        if col not in df.columns:
            continue
        # Garantir que é Series (não DataFrame com colunas duplicadas)
        serie = df[col]
        if isinstance(serie, pd.DataFrame):
            print(f"   ⚠️  Coluna '{col}' ainda duplicada — mantendo primeira")
            serie = serie.iloc[:, 0]
            df = df.drop(columns=col)
            df[col] = serie
        # Converter vírgula decimal e forçar float
        if serie.dtype == object:
            df[col] = df[col].astype(str).str.replace(',', '.', regex=False).str.strip()
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


# =============================================================================
# ETAPA 4 — QA/QC HORÁRIO
# =============================================================================

def qa_qc_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove outliers fisicamente impossíveis para Imperatriz-MA."""
    print("\n🔍 QA/QC — Outliers físicos (série horária):")

    limites = {
        'temp_bulbo_seco': (-5.0, 50.0),
        'temp_max_hora':   (-5.0, 50.0),
        'temp_min_hora':   (-5.0, 50.0),
        'precipitacao':    ( 0.0, 200.0),
        'umidade_relativa':( 1.0, 100.0),
        'vento_velocidade':( 0.0,  60.0),
        'radiacao_kj_m2':  ( 0.0, 5000.0),
    }

    total = 0
    for col, (vmin, vmax) in limites.items():
        if col in df.columns:
            mask = (df[col] < vmin) | (df[col] > vmax)
            n = mask.sum()
            if n:
                df.loc[mask, col] = np.nan
                print(f"   {col:35s}: {n:6,} → NaN")
                total += n

    if 'temp_max_hora' in df.columns and 'temp_min_hora' in df.columns:
        inv = df['temp_max_hora'] < df['temp_min_hora']
        n = inv.sum()
        if n:
            df.loc[inv, ['temp_max_hora', 'temp_min_hora']] = np.nan
            print(f"   {'Tmax < Tmin':35s}: {n:6,} → NaN")
            total += n

    print(f"   Total: {total:,} valores removidos")
    return df


# =============================================================================
# ETAPA 5 — AGREGAÇÃO HORÁRIA → DIÁRIA
# =============================================================================

def agregar_para_diario(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega com função correta por variável."""
    print("\n📊 Agregando horário → diário...")

    cols = set(df.columns)
    agg = {}
    if 'temp_max_hora'    in cols: agg['temp_max_hora']    = 'max'
    if 'temp_min_hora'    in cols: agg['temp_min_hora']    = 'min'
    if 'temp_bulbo_seco'  in cols: agg['temp_bulbo_seco']  = ['max', 'min']
    if 'precipitacao'     in cols: agg['precipitacao']     = 'sum'
    if 'umidade_relativa' in cols: agg['umidade_relativa'] = 'mean'
    if 'vento_velocidade' in cols: agg['vento_velocidade'] = 'mean'
    if 'radiacao_kj_m2'   in cols: agg['radiacao_kj_m2']  = 'sum'
    if 'pressao_atm'      in cols: agg['pressao_atm']      = 'mean'

    df_d = df.resample('D').agg(agg)

    # Achatar MultiIndex
    if isinstance(df_d.columns, pd.MultiIndex):
        df_d.columns = ['_'.join(c).strip('_') for c in df_d.columns]

    # Tmax com prioridade para coluna específica, fallback em bulbo seco
    if 'temp_max_hora_max' in df_d.columns:
        df_d['tmax_c'] = df_d['temp_max_hora_max']
    elif 'temp_bulbo_seco_max' in df_d.columns:
        df_d['tmax_c'] = df_d['temp_bulbo_seco_max']
    else:
        raise ValueError("❌ Nenhuma coluna de temperatura máxima encontrada.")

    if 'temp_min_hora_min' in df_d.columns:
        df_d['tmin_c'] = df_d['temp_min_hora_min']
    elif 'temp_bulbo_seco_min' in df_d.columns:
        df_d['tmin_c'] = df_d['temp_bulbo_seco_min']
    else:
        raise ValueError("❌ Nenhuma coluna de temperatura mínima encontrada.")

    # Preencher NaN cruzado entre as fontes
    for alvo, fonte in [('tmax_c', 'temp_bulbo_seco_max'), ('tmin_c', 'temp_bulbo_seco_min')]:
        if alvo in df_d.columns and fonte in df_d.columns:
            df_d[alvo] = df_d[alvo].fillna(df_d[fonte])

    df_d['tmean_c'] = (df_d['tmax_c'] + df_d['tmin_c']) / 2

    renomear = {
        'precipitacao_sum':      'chuva_mm',
        'precipitacao':          'chuva_mm',
        'umidade_relativa_mean': 'umidade_media',
        'umidade_relativa':      'umidade_media',
        'vento_velocidade_mean': 'vento_medio_ms',
        'vento_velocidade':      'vento_medio_ms',
        'radiacao_kj_m2_sum':    'radiacao_kj_m2_dia',
        'radiacao_kj_m2':        'radiacao_kj_m2_dia',
        'pressao_atm_mean':      'pressao_media_hpa',
        'pressao_atm':           'pressao_media_hpa',
    }
    df_d = df_d.rename(columns={k: v for k, v in renomear.items() if k in df_d.columns})

    print(f"   → {len(df_d):,} dias | {df_d.index.min().date()} a {df_d.index.max().date()}")
    return df_d


# =============================================================================
# ETAPA 6 — QA/QC DIÁRIO
# =============================================================================

def qa_qc_lacunas(df: pd.DataFrame) -> pd.DataFrame:
    """Trata lacunas na série diária."""
    print("\n🔍 QA/QC — Lacunas na série diária:")

    # Reindexar apenas dentro dos anos que têm dados reais
    # Evita criar 366 linhas fantasma para anos descartados (ex: 2024)
    anos_com_dados = set(df.index.year.unique())
    idx_completo = pd.DatetimeIndex([
        d for d in pd.date_range(df.index.min(), df.index.max(), freq='D')
        if d.year in anos_com_dados
    ])
    df = df.reindex(idx_completo)

    colunas = [c for c in ['tmax_c', 'tmin_c', 'tmean_c', 'chuva_mm',
                            'umidade_media', 'vento_medio_ms'] if c in df.columns]

    for col in colunas:
        n_nan = df[col].isna().sum()
        if n_nan == 0:
            print(f"   {col:25s}: sem lacunas ✅")
            continue

        blocos = []
        em_bloco, inicio = False, 0
        for i, v in enumerate(df[col].isna()):
            if v and not em_bloco:
                em_bloco, inicio = True, i
            elif not v and em_bloco:
                blocos.append((inicio, i - 1, i - inicio))
                em_bloco = False
        if em_bloco:
            blocos.append((inicio, len(df) - 1, len(df) - inicio))

        curtas = [b for b in blocos if b[2] <= 3]
        medias = [b for b in blocos if 3 < b[2] <= 15]
        longas = [b for b in blocos if b[2] > 15]

        df[col] = df[col].interpolate(method='linear', limit=3)

        for ini, fim, _ in medias:
            for data in df.index[ini:fim+1]:
                if pd.isna(df.loc[data, col]):
                    mask = (
                        (df.index.month == data.month) &
                        (abs(df.index.day - data.day) <= 7) &
                        (df.index.year != data.year)
                    )
                    media = df.loc[mask, col].mean()
                    if not pd.isna(media):
                        df.loc[data, col] = media

        print(f"   {col:25s}: {n_nan:4d} NaN | "
              f"{len(curtas)}× ≤3d | {len(medias)}× 4-15d | {len(longas)}× >15d ⚠️")

        for ini, fim, tam in longas:
            print(f"     ⚠️  LACUNA CRÍTICA '{col}': "
                  f"{df.index[ini].date()} → {df.index[fim].date()} ({tam} dias)")

    return df


# =============================================================================
# ETAPA 7 — CÁLCULO DE ETo
# =============================================================================

def radiacao_extraterrestre(doy: np.ndarray, lat_rad: float) -> np.ndarray:
    dr      = 1 + 0.033 * np.cos(2 * np.pi / 365 * doy)
    delta   = 0.409 * np.sin(2 * np.pi / 365 * doy - 1.39)
    omega_s = np.arccos(np.clip(-np.tan(lat_rad) * np.tan(delta), -1, 1))
    return (24 * 60 / np.pi) * 0.0820 * dr * (
        omega_s * np.sin(lat_rad) * np.sin(delta) +
        np.cos(lat_rad) * np.cos(delta) * np.sin(omega_s)
    )


def eto_hargreaves_samani(tmax, tmin, tmean, Ra) -> np.ndarray:
    return np.maximum(
        0.0023 * Ra * (tmean + 17.8) * np.sqrt(np.maximum(tmax - tmin, 0.0)), 0.0
    )


def eto_penman_monteith(df: pd.DataFrame, lat_rad: float, altitude_m: float) -> np.ndarray:
    T, Tmax, Tmin = df['tmean_c'].values, df['tmax_c'].values, df['tmin_c'].values
    doy = df.index.dayofyear.values

    es = (0.6108 * np.exp(17.27 * Tmax / (Tmax + 237.3)) +
          0.6108 * np.exp(17.27 * Tmin / (Tmin + 237.3))) / 2

    if 'umidade_media' in df.columns and df['umidade_media'].notna().mean() > 0.5:
        ea = df['umidade_media'].values / 100 * es
    else:
        ea = 0.6108 * np.exp(17.27 * Tmin / (Tmin + 237.3))
        print("   ⚠️  Umidade indisponível → ea estimada pela Tmin")

    vpd   = np.maximum(es - ea, 0.0)
    P     = 101.3 * ((293 - 0.0065 * altitude_m) / 293) ** 5.26
    gamma = 0.000665 * P
    Delta = 4098 * es / (T + 237.3) ** 2
    Ra    = radiacao_extraterrestre(doy, lat_rad)
    Rso   = (0.75 + 2e-5 * altitude_m) * Ra

    if 'radiacao_kj_m2_dia' in df.columns and df['radiacao_kj_m2_dia'].notna().mean() > 0.3:
        Rs = np.clip(df['radiacao_kj_m2_dia'].values / 1000.0, 0.0, Ra)
        print("   ✅  Usando radiação medida para PM")
    else:
        Rs = (0.25 + 0.50 * 0.5) * Ra
        print("   ⚠️  Radiação indisponível → estimada por Ångström-Prescott (n/N=0.5)")

    Rns   = (1 - 0.23) * Rs
    sigma = 4.903e-9
    Rnl = sigma * ((Tmax + 273.16)**4 + (Tmin + 273.16)**4) / 2 * \
          (0.34 - 0.14 * np.sqrt(np.maximum(ea, 0.001))) * \
          (1.35 * np.clip(Rs / np.maximum(Rso, 0.01), 0, 1) - 0.35)
    Rn = np.maximum(Rns - Rnl, 0.0)

    if 'vento_medio_ms' in df.columns and df['vento_medio_ms'].notna().mean() > 0.5:
        u2 = np.maximum(df['vento_medio_ms'].values * (4.87 / np.log(67.8 * 10 - 5.42)), 0.5)
    else:
        u2 = np.full(len(df), 2.0)

    num = 0.408 * Delta * Rn + gamma * (900 / (T + 273)) * u2 * vpd
    den = Delta + gamma * (1 + 0.34 * u2)
    return np.maximum(num / den, 0.0)


def calcular_eto(df: pd.DataFrame, lat_graus: float, altitude_m: float) -> pd.DataFrame:
    print("\n💧 Calculando ETo diária...")
    lat_rad = np.radians(lat_graus)
    Ra      = radiacao_extraterrestre(df.index.dayofyear.values, lat_rad)

    df['eto_hs_mm'] = eto_hargreaves_samani(
        df['tmax_c'].values, df['tmin_c'].values, df['tmean_c'].values, Ra
    )

    # Verificar se o vento medido é plausível para PM
    # Estações automáticas do INMET em regiões tropicais frequentemente reportam
    # velocidades muito baixas (<1 m/s a 10m) que colapsam o termo aerodinâmico do PM.
    # Critério: só usar PM se vento médio diário a 10m > 1.0 m/s E umidade disponível.
    vento_ok   = ('vento_medio_ms' in df.columns and
                  df['vento_medio_ms'].notna().mean() > 0.5 and
                  df['vento_medio_ms'].median() > 1.0)
    umidade_ok = ('umidade_media' in df.columns and
                  df['umidade_media'].notna().mean() > 0.5)

    if vento_ok and umidade_ok:
        print("   → Método: Penman-Monteith FAO-56")
        print(f"     (vento mediano: {df['vento_medio_ms'].median():.2f} m/s — adequado para PM)")
        df['eto_mm']     = eto_penman_monteith(df, lat_rad, altitude_m)
        df['eto_metodo'] = 'Penman-Monteith FAO-56'
    else:
        mediana_vento = df['vento_medio_ms'].median() if 'vento_medio_ms' in df.columns else 0
        print("   → Método: Hargreaves-Samani")
        print(f"     (vento mediano: {mediana_vento:.2f} m/s a 10m — abaixo do limiar para PM confiável)")
        print("     Hargreaves-Samani é mais robusto para estações com vento subestimado.")
        df['eto_mm']     = df['eto_hs_mm']
        df['eto_metodo'] = 'Hargreaves-Samani'

    nan_mask = df['eto_mm'].isna()
    if nan_mask.sum() > 0:
        df.loc[nan_mask, 'eto_mm']     = df.loc[nan_mask, 'eto_hs_mm']
        df.loc[nan_mask, 'eto_metodo'] = 'Hargreaves-Samani (fallback)'

    print(f"\n   ETo (mm/dia): mín={df['eto_mm'].min():.2f} | "
          f"média={df['eto_mm'].mean():.2f} | máx={df['eto_mm'].max():.2f}")
    return df


# =============================================================================
# RELATÓRIO E GRÁFICOS
# =============================================================================

def gerar_relatorio(df: pd.DataFrame) -> None:
    print(f"\n{'='*60}")
    print("📊 RELATÓRIO DE QUALIDADE — DADOS FINAIS")
    print(f"{'='*60}")
    print(f"Período   : {df.index.min().date()} → {df.index.max().date()}")
    print(f"Total dias: {len(df):,}")
    print(f"Anos      : {sorted(df.index.year.unique())}")

    print(f"\n{'Variável':<22} {'Mín':>8} {'Média':>8} {'Máx':>8} {'NaN':>6}")
    print("-" * 57)
    for col in ['tmax_c', 'tmin_c', 'tmean_c', 'chuva_mm', 'eto_mm']:
        if col in df.columns:
            s = df[col]
            print(f"{col:<22} {s.min():>8.2f} {s.mean():>8.2f} "
                  f"{s.max():>8.2f} {s.isna().sum():>5d}")

    if 'eto_metodo' in df.columns:
        print("\nMétodo ETo:")
        for met, cnt in df['eto_metodo'].value_counts().items():
            print(f"   {met}: {cnt} dias ({cnt/len(df)*100:.1f}%)")

    print("\nStatus colunas críticas para AquaCrop:")
    for col in ['tmax_c', 'tmin_c', 'chuva_mm', 'eto_mm']:
        if col in df.columns:
            n = df[col].isna().sum()
            print(f"   {'✅' if n == 0 else '⚠️ '} {col}: {n} NaN")
    print(f"{'='*60}")


def gerar_graficos(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(15, 14), sharex=True)
    fig.suptitle(
        f"Dados Climáticos Tratados — INMET Imperatriz-MA "
        f"({df.index.min().year}–{df.index.max().year})",
        fontsize=13, fontweight='bold'
    )

    ax = axes[0]
    ax.fill_between(df.index, df['tmin_c'], df['tmax_c'],
                    alpha=0.25, color='tomato', label='Amplitude (Tmin–Tmax)')
    ax.plot(df.index, df['tmean_c'], color='tomato', lw=0.8, label='Tmédia')
    ax.set_ylabel('Temperatura (°C)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.bar(df.index, df['chuva_mm'], color='steelblue', width=1, alpha=0.75)
    ax.set_ylabel('Precipitação (mm/dia)')
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(df.index, df['eto_mm'], color='darkorange', lw=0.8, label='ETo final')
    if 'eto_hs_mm' in df.columns:
        ax.plot(df.index, df['eto_hs_mm'], color='gray', lw=0.5,
                alpha=0.5, ls='--', label='ETo HS (ref.)')
    ax.set_ylabel('ETo (mm/dia)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[3]
    bal = df['chuva_mm'] - df['eto_mm']
    ax.fill_between(df.index, bal, 0, where=(bal >= 0),
                    color='steelblue', alpha=0.6, label='Excesso (chuva > ETo)')
    ax.fill_between(df.index, bal, 0, where=(bal < 0),
                    color='salmon',    alpha=0.6, label='Déficit (ETo > chuva)')
    ax.axhline(0, color='black', lw=0.6)
    ax.set_ylabel('Balanço Hídrico (mm/dia)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig('graficos_climaticos_inmet.png', dpi=150, bbox_inches='tight')
    print("\n📈 Gráfico salvo: graficos_climaticos_inmet.png")
    plt.close()


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def processar_inmet(pasta: str, ano_inicio: int, ano_fim: int) -> pd.DataFrame:
    print("\n" + "="*60)
    print("🌿 PIPELINE INMET BDMEP — Múltiplos Anos")
    print("   Projeto: Sistema de Irrigação com ALMMo-0")
    print("="*60)

    df = carregar_pasta_bdmep(pasta)
    df = padronizar_colunas(df)
    df = parsear_datetime(df)
    df = converter_numericas(df)

    df = df[(df.index.year >= ano_inicio) & (df.index.year <= ano_fim)]
    if ANOS_EXCLUIR:
        df = df[~df.index.year.isin(ANOS_EXCLUIR)]
    anos_finais = sorted(df.index.year.unique())
    print(f"\n📅 Após filtro {ano_inicio}–{ano_fim} (excluindo {ANOS_EXCLUIR}): "
          f"{len(df):,} horas | anos: {anos_finais}")

    if len(df) == 0:
        raise ValueError(
            "❌ Nenhuma hora restante após o filtro de período.\n"
            "   Verifique se o parsing de datetime funcionou (ver diagnóstico acima)."
        )

    df = qa_qc_outliers(df)
    df_d = agregar_para_diario(df)

    # Remover dias de anos excluídos que o resample criou como NaN
    # (o resample preenche todos os dias entre min e max, incluindo anos sem dados)
    if ANOS_EXCLUIR:
        df_d = df_d[~df_d.index.year.isin(ANOS_EXCLUIR)]
        print(f"   → Dias de {ANOS_EXCLUIR} removidos do agregado diário")

    df_d = qa_qc_lacunas(df_d)
    df_d = calcular_eto(df_d, LATITUDE_GRAUS, ALTITUDE_METROS)

    colunas_finais = ['tmax_c', 'tmin_c', 'tmean_c', 'chuva_mm',
                      'eto_mm', 'eto_hs_mm', 'eto_metodo',
                      'umidade_media', 'vento_medio_ms']
    df_saida = df_d[[c for c in colunas_finais if c in df_d.columns]].copy()

    gerar_relatorio(df_saida)
    if GERAR_GRAFICOS:
        gerar_graficos(df_saida)

    df_saida.index.name = 'data'
    df_saida.to_csv(ARQUIVO_SAIDA, sep=';', decimal='.', date_format='%Y-%m-%d')
    print(f"\n✅ Salvo: {ARQUIVO_SAIDA}  ({len(df_saida)} dias | {len(df_saida.columns)} colunas)")
    return df_saida


# =============================================================================
# PONTO DE ENTRADA
# =============================================================================

if __name__ == "__main__":
    if not Path(PASTA_INMET).exists():
        print(f"\n❌ Pasta não encontrada: '{PASTA_INMET}'")
        print("   Ajuste PASTA_INMET no topo do script.")
    else:
        df_tratado = processar_inmet(PASTA_INMET, ANO_INICIO, ANO_FIM)
        print("\n" + "="*60)
        print("✅ Pipeline concluído!")
        print("   → Próximo passo: dados_climaticos_tratados.csv → AquaCrop-OSPy")
        print("="*60)
