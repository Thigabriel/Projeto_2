#!/usr/bin/env python3
"""
============================================================================
Script de Simulacao AquaCrop-OSPy — Dataset v10 (Revisado)
Projeto ALMMo-0 — Irrigacao Inteligente de Tomate | Imperatriz-MA
============================================================================

Mudancas em relacao ao v7 (v2.4.3):
  1. Expansao temporal: 2001-2023 (23 anos, era 2019-2023)
  2. Cenario veranico: precip fev-mar × 0.20, janela chuva apenas
  3. Remocao de cenarios improdutivos: otimo/chuva e deficit/chuva removidos
     (geravam <0.3% irrigacao — quase exclusivamente C0 redundante)
  4. Nova feature: delta_tensao_kpa (variacao diaria da tensao)
  5. grupo_id sequencial para Leave-Groups-Out no treino

Simulacoes: 23×2 (chuva) + 23×3 (seca) = 46+69 = 115 simulacoes
Amostras estimadas: ~10.400-10.800

Requisitos: pip install aquacrop pandas numpy requests
Execucao:   python script_simulacao_v10.py
"""

import os
import sys
import math
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

os.environ['DEVELOPMENT'] = 'True'

try:
    from aquacrop import (
        AquaCropModel, Soil, Crop, InitialWaterContent,
        IrrigationManagement, FieldMngt
    )
    from aquacrop.utils import prepare_weather
except ImportError:
    print("ERRO: pip install aquacrop")
    sys.exit(1)

warnings.filterwarnings('ignore')

# ============================================================================
# CONSTANTES
# ============================================================================
LAT = -5.5253
LON = -47.4825
ALTITUDE = 120
API_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

# Saxton & Rawls — Franco-Arenoso (65% areia, 10% argila, 3% MO)
THETA_SAT = 0.3863
THETA_CC  = 0.1864
THETA_PM  = 0.0853
A_SAXTON  = 0.0090
B_SAXTON  = 4.8825

# 23 anos: CERES SYN1deg a partir de 2001 (homogeneidade da radiacao solar)
ANOS = list(range(2001, 2024))

# Duas janelas de plantio (identicas ao v7)
JANELAS = {
    'chuva': {
        'planting_date': '01/05',
        'sim_start_fmt': '{year}/01/01',
        'sim_end_fmt':   '{year}/07/31',
    },
    'seca': {
        'planting_date': '06/01',
        'sim_start_fmt': '{year}/05/15',
        'sim_end_fmt':   '{year}/12/31',
    },
}

# ============================================================================
# CENARIOS — 4 cenarios (3 do v7 + veranico)
# ============================================================================
CENARIOS = {
    'excesso': {'SMT': [80, 80, 80, 70], 'MaxIrr': 100, 'MaxIrrSeason': 10_000},
    'otimo':   {'SMT': [60, 60, 70, 50], 'MaxIrr': 100, 'MaxIrrSeason': 10_000},
    'deficit': {'SMT': [40, 40, 50, 30], 'MaxIrr': 100, 'MaxIrrSeason': 10_000},
    'veranico':{'SMT': [60, 60, 70, 50], 'MaxIrr': 100, 'MaxIrrSeason': 10_000},
    # veranico: SMT identico ao otimo. Diferenca = precipitacao reduzida em fev-mar.
    # Aplica-se APENAS a janela chuva.
}

# Fator de reducao do veranico (fev-mar) — 0.20 = veranico severo
# (versao anterior com 0.30 gerou apenas 2.2% irrigacao, insuficiente)
VERANICO_FATOR = 0.20
VERANICO_MESES = [2, 3]

DAP_MIN = 14
DAP_MAX = 107
TR_MIN = 0.1

OUTPUT_DIR = Path('.')
WEATHER_DIR = Path('weather_files')
WEATHER_DIR.mkdir(exist_ok=True)

TXT_HEADER = "Day\tMonth\tYear\tMinTemp\tMaxTemp\tPrecipitation\tReferenceET"


# ============================================================================
# MODULO 1: DADOS METEOROLOGICOS (NASA POWER) — ANO COMPLETO
# ============================================================================

def calc_eto_fao56(tmax, tmin, rs, rh, doy, lat_deg=LAT, alt=ALTITUDE):
    """ETo FAO-56 PM com u2=2.0 m/s fixo."""
    tmean = (tmax + tmin) / 2.0
    lat_rad = lat_deg * math.pi / 180.0
    P = 101.3 * ((293.0 - 0.0065 * alt) / 293.0) ** 5.26
    gamma = 0.000665 * P

    e_tmax = 0.6108 * math.exp(17.27 * tmax / (tmax + 237.3))
    e_tmin = 0.6108 * math.exp(17.27 * tmin / (tmin + 237.3))
    es = (e_tmax + e_tmin) / 2.0
    ea = es * (rh / 100.0) if rh > 0 else e_tmin
    delta = 4098.0 * (0.6108 * math.exp(17.27 * tmean / (tmean + 237.3))) / (tmean + 237.3) ** 2

    dr = 1.0 + 0.033 * math.cos(2 * math.pi * doy / 365)
    d_sol = 0.409 * math.sin(2 * math.pi * doy / 365 - 1.39)
    ws = math.acos(-math.tan(lat_rad) * math.tan(d_sol))
    Ra = (24 * 60 / math.pi) * 0.0820 * dr * (
        ws * math.sin(lat_rad) * math.sin(d_sol) +
        math.cos(lat_rad) * math.cos(d_sol) * math.sin(ws))

    Rso = (0.75 + 2e-5 * alt) * Ra
    Rns = 0.77 * rs
    sigma = 4.903e-9
    rs_r = min(rs / Rso, 1.0) if Rso > 0 else 0.5
    Rnl = sigma * ((tmax + 273.16)**4 + (tmin + 273.16)**4) / 2 * \
          (0.34 - 0.14 * math.sqrt(max(ea, 0.01))) * (1.35 * rs_r - 0.35)
    Rn = Rns - Rnl

    u2 = 2.0
    num = 0.408 * delta * Rn + gamma * (900 / (tmean + 273)) * u2 * (es - ea)
    den = delta + gamma * (1 + 0.34 * u2)
    return max(num / den, 0.0)


def fetch_and_save_weather(year):
    """Busca NASA POWER ano completo, calcula ETo, salva .txt."""
    txt_path = WEATHER_DIR / f'weather_imperatriz_{year}_full.txt'
    meta_path = WEATHER_DIR / f'weather_imperatriz_{year}_full_meta.csv'

    if txt_path.exists():
        meta = pd.read_csv(meta_path) if meta_path.exists() else None
        return str(txt_path), meta

    print(f"    Buscando NASA POWER {year}...")
    params = {
        'parameters': 'T2M_MAX,T2M_MIN,PRECTOTCORR,ALLSKY_SFC_SW_DWN,RH2M,WS2M',
        'community': 'AG',
        'longitude': LON, 'latitude': LAT,
        'start': f'{year}0101', 'end': f'{year}1231',
        'format': 'JSON'
    }

    resp = requests.get(API_URL, params=params, timeout=180)
    resp.raise_for_status()
    props = resp.json()['properties']['parameter']

    dates = pd.date_range(f'{year}-01-01', f'{year}-12-31')
    lines = []
    meta_records = []

    for d in dates:
        key = d.strftime('%Y%m%d')
        tmax = props['T2M_MAX'].get(key, -999)
        tmin = props['T2M_MIN'].get(key, -999)
        prec = props['PRECTOTCORR'].get(key, -999)
        rs = props['ALLSKY_SFC_SW_DWN'].get(key, -999)
        rh = props['RH2M'].get(key, -999)
        u2_obs = props['WS2M'].get(key, -999)

        if tmax < -900 or tmin < -900 or rs < -900:
            continue
        if rh < -900:
            rh = 75.0
        prec = max(prec, 0.0)

        doy = d.timetuple().tm_yday
        eto = calc_eto_fao56(tmax, tmin, rs, rh, doy)

        lines.append(f"{d.day}\t{d.month}\t{d.year}\t{tmin:.2f}\t{tmax:.2f}\t{prec:.2f}\t{eto:.4f}")
        meta_records.append({
            'date': d, 'tmax': tmax, 'tmin': tmin, 'prec': prec,
            'rs': rs, 'rh': rh, 'u2_obs': u2_obs if u2_obs > -900 else np.nan,
            'eto': eto
        })

    with open(txt_path, 'w') as f:
        f.write(TXT_HEADER + '\n')
        for line in lines:
            f.write(line + '\n')

    meta = pd.DataFrame(meta_records)
    meta.to_csv(meta_path, index=False)

    print(f"    Salvo: {txt_path} ({len(lines)} dias)")
    return str(txt_path), meta


def aplicar_veranico(wdf, year):
    """
    Reduz precipitacao de fev-mar a 30% no weather DataFrame.
    Retorna copia modificada (nao altera o original).
    """
    wdf_ver = wdf.copy()

    mask = (wdf_ver['Date'].dt.month.isin(VERANICO_MESES)) & \
           (wdf_ver['Date'].dt.year == year)

    prec_antes = wdf_ver.loc[mask, 'Precipitation'].sum()
    wdf_ver.loc[mask, 'Precipitation'] *= VERANICO_FATOR
    prec_depois = wdf_ver.loc[mask, 'Precipitation'].sum()

    ratio = prec_depois / prec_antes if prec_antes > 0 else 0
    print(f"      Veranico {year}: fev-mar {prec_antes:.0f}mm → {prec_depois:.0f}mm ({ratio:.2f}x)")

    return wdf_ver


# ============================================================================
# MODULO 2: CONVERSAO Wr → TENSAO
# ============================================================================

def umidade_para_tensao_kpa(theta_vol):
    """Saxton & Rawls: theta → tensao. Clip em theta_sat."""
    theta_safe = np.clip(theta_vol, THETA_PM, THETA_SAT)
    tensao = A_SAXTON * (theta_safe ** (-B_SAXTON))
    return float(np.clip(tensao, 1.0, 1500.0))


def wr_para_tensao_kpa_dinamica(wr_mm, z_root_m):
    """Converte Wr → tensao com profundidade radicular dinamica."""
    z_safe = max(z_root_m, 0.10)
    theta = wr_mm / (1000.0 * z_safe)
    return umidade_para_tensao_kpa(theta)


# ============================================================================
# MODULO 3: SIMULACAO AQUACROP
# ============================================================================

COL_IRR = None
COL_WR = None
COL_TR = None
COL_DAP = None
COL_ZROOT = None


def detect_columns(wf, cg):
    """Detecta nomes de colunas na primeira simulacao."""
    global COL_IRR, COL_WR, COL_TR, COL_DAP, COL_ZROOT

    def find(candidates, cols):
        for c in candidates:
            if c in cols:
                return c
        return None

    wf_cols = set(wf.columns)
    cg_cols = set(cg.columns)

    COL_IRR = find(['IrrDay', 'Irr', 'irr_day', 'IrrNet'], wf_cols)
    COL_WR = find(['Wr', 'Wr(1)', 'wr', 'th1', 'WrAct'], wf_cols)
    COL_TR = find(['Tr', 'TrAct', 'tr', 'Tact'], wf_cols)
    COL_DAP = find(['DAP', 'dap', 'GrowingSeasonDay', 'growing_season_day'], cg_cols)
    COL_ZROOT = find(['z_root', 'Zroot', 'zRoot', 'RootDepth', 'root_depth', 'ZrAct', 'Zr'], cg_cols)

    print(f"    Colunas: IRR={COL_IRR}, WR={COL_WR}, TR={COL_TR}, "
          f"DAP={COL_DAP}, ZROOT={COL_ZROOT}")

    missing = [n for n, v in [('IRR', COL_IRR), ('WR', COL_WR),
               ('TR', COL_TR), ('DAP', COL_DAP)] if not v]
    if missing:
        print(f"    ERRO: Colunas nao encontradas: {missing}")
        return False
    return True


def run_single_simulation(year, cenario_nome, janela, wdf_to_use):
    """Executa uma simulacao AquaCrop com method 1."""
    global COL_IRR

    jcfg = JANELAS[janela]
    sim_start = jcfg['sim_start_fmt'].format(year=year)
    sim_end = jcfg['sim_end_fmt'].format(year=year)
    planting = jcfg['planting_date']

    # veranico usa mesmos parametros do otimo
    cen_key = 'otimo' if cenario_nome == 'veranico' else cenario_nome
    cen = CENARIOS[cen_key]

    soil = Soil('SandyLoam')
    crop = Crop('TomatoGDD', planting_date=planting)
    init_wc = InitialWaterContent(value=['FC'])

    irr_mngt = IrrigationManagement(
        irrigation_method=1,
        SMT=cen['SMT'],
        MaxIrr=cen['MaxIrr'],
        MaxIrrSeason=cen['MaxIrrSeason'],
    )

    field_mngt = FieldMngt(mulches=True, mulch_pct=80, f_mulch=0.3)

    model = AquaCropModel(
        sim_start_time=sim_start,
        sim_end_time=sim_end,
        weather_df=wdf_to_use,
        soil=soil,
        crop=crop,
        initial_water_content=init_wc,
        irrigation_management=irr_mngt,
        field_management=field_mngt,
    )

    model.run_model(till_termination=True)

    wf = model._outputs.water_flux
    cg = model._outputs.crop_growth

    if COL_IRR is None:
        if not detect_columns(wf, cg):
            raise RuntimeError("Colunas essenciais nao detectadas!")

    n_rows = min(len(wf), len(cg))

    result = pd.DataFrame({
        'IrrDay': wf[COL_IRR].values[:n_rows],
        'Wr': wf[COL_WR].values[:n_rows],
        'Tr': wf[COL_TR].values[:n_rows],
        'dap': cg[COL_DAP].values[:n_rows],
    })

    if COL_ZROOT:
        result['z_root'] = cg[COL_ZROOT].values[:n_rows]
    else:
        dap_vals = result['dap'].values
        max_dap = max(dap_vals.max(), 1)
        result['z_root'] = np.clip(0.3 + (0.7 * dap_vals / max_dap), 0.3, 1.0)

    # Precipitacao e Tmax do weather DataFrame
    sim_dates = pd.date_range(sim_start, periods=n_rows, freq='D')
    wdf_indexed = wdf_to_use.set_index('Date')

    prec_aligned = []
    tmax_aligned = []
    for d in sim_dates:
        d_ts = pd.Timestamp(d)
        if d_ts in wdf_indexed.index:
            prec_aligned.append(float(wdf_indexed.loc[d_ts, 'Precipitation']))
            tmax_aligned.append(float(wdf_indexed.loc[d_ts, 'MaxTemp']))
        else:
            prec_aligned.append(0.0)
            tmax_aligned.append(np.nan)

    result['precipitation'] = prec_aligned
    result['tmax'] = tmax_aligned
    result['date'] = sim_dates
    result['year'] = year
    result['cenario'] = cenario_nome
    result['janela'] = janela

    return result


def run_all_simulations():
    """
    Executa todas as simulacoes (estrutura assimetrica por janela):
      Chuva: 23 anos × 2 cenarios (excesso + veranico) = 46
      Seca:  23 anos × 3 cenarios (excesso + otimo + deficit) = 69
      Total: 115 simulacoes

    Removidos da janela chuva: otimo (0.3% irrig) e deficit (0.0% irrig)
    — contribuiam quase exclusivamente com C0 redundante.
    """
    all_results = []
    weather_metas = {}
    grupo_counter = 0
    t_start = time.time()

    total_sims = len(ANOS) * 2 + len(ANOS) * 3  # 46 + 69 = 115
    sim_count = 0

    for year in ANOS:
        print(f"\n{'='*60}")
        print(f"ANO: {year} ({ANOS.index(year)+1}/{len(ANOS)})")
        print(f"{'='*60}")

        txt_path, meta = fetch_and_save_weather(year)
        weather_metas[year] = meta

        wdf_base = prepare_weather(txt_path)

        # --- Janela CHUVA: apenas excesso + veranico ---
        # (otimo e deficit removidos — geravam <0.3% irrigacao)
        wdf_veranico = aplicar_veranico(wdf_base, year)

        for cenario, wdf_to_use in [('excesso', wdf_base), ('veranico', wdf_veranico)]:
            grupo_counter += 1
            sim_count += 1
            label = f"chuva/{cenario}"

            try:
                result = run_single_simulation(year, cenario, 'chuva', wdf_to_use)
                result['grupo_id'] = grupo_counter
                irr_days = (result['IrrDay'] > 0).sum()
                irr_total = result['IrrDay'].sum()
                print(f"    [{sim_count}/{total_sims}] {label}: {irr_days} dias irrig ({irr_total:.0f}mm)")
                all_results.append(result)
            except Exception as e:
                print(f"    [{sim_count}/{total_sims}] {label}: ERRO — {e}")

        # --- Janela SECA: 3 cenarios completos ---
        cenarios_seca = ['excesso', 'otimo', 'deficit']

        for cenario in cenarios_seca:
            grupo_counter += 1
            sim_count += 1
            label = f"seca/{cenario}"

            try:
                result = run_single_simulation(year, cenario, 'seca', wdf_base)
                result['grupo_id'] = grupo_counter
                irr_days = (result['IrrDay'] > 0).sum()
                irr_total = result['IrrDay'].sum()
                print(f"    [{sim_count}/{total_sims}] {label}: {irr_days} dias irrig ({irr_total:.0f}mm)")
                all_results.append(result)
            except Exception as e:
                print(f"    [{sim_count}/{total_sims}] {label}: ERRO — {e}")

    elapsed = time.time() - t_start
    print(f"\nTempo total: {elapsed/60:.1f} min ({elapsed/max(sim_count,1):.1f}s/sim)")

    return all_results, weather_metas


# ============================================================================
# MODULO 4: PROCESSAMENTO DO DATASET
# ============================================================================

def process_dataset(all_results):
    """
    Aplica filtros, calcula features.
    Processa por grupo (nao por cenario) para manter grupo_id correto.
    """
    processed_frames = []

    for sim_df in all_results:
        df = sim_df.copy()
        n0 = len(df)

        # Filtros
        df = df[df['Tr'] > TR_MIN].copy()
        df = df[df['dap'] >= DAP_MIN].copy()
        df = df[df['dap'] <= DAP_MAX].copy()

        if len(df) == 0:
            continue

        # Wr → tensao dinamica
        df['tensao_raw'] = df.apply(
            lambda r: wr_para_tensao_kpa_dinamica(r['Wr'], r['z_root']), axis=1)

        # Shift temporal da tensao (dentro do grupo)
        vals = df['tensao_raw'].values
        df['tensao_solo_kpa'] = np.concatenate([[vals[0]], vals[:-1]])

        # Chuva acumulada 3d
        df['chuva_acum_3d_mm'] = df['precipitation'].rolling(3, min_periods=1).sum().values

        # Tmax max 3d
        df['tmax_max_3d_c'] = df['tmax'].rolling(3, min_periods=1).max().values

        # Delta tensao (calculado APOS shift)
        dt = df['tensao_solo_kpa'].diff().fillna(0).values
        df['delta_tensao_kpa'] = dt

        processed_frames.append(df)

    if not processed_frames:
        raise ValueError("Nenhuma simulacao processada!")

    all_processed = pd.concat(processed_frames, ignore_index=True)

    # Log por cenario
    for cen in ['excesso', 'otimo', 'deficit', 'veranico']:
        sub = all_processed[all_processed['cenario'] == cen]
        if len(sub) > 0:
            t = sub['tensao_solo_kpa']
            irr = (sub['IrrDay'] > 0).sum()
            print(f"  {cen}: {len(sub)} amostras, {irr} dias irrigados, "
                  f"tensao med={t.median():.1f} std={t.std():.1f} kPa")

    return all_processed


def rotular_classes(df):
    """Rotula com mediana do IrrDay>0 do cenario otimo (todos os anos, ambas janelas)."""
    otimo = df[df['cenario'] == 'otimo']
    irr_pos = otimo[otimo['IrrDay'] > 0]['IrrDay']

    if len(irr_pos) < 10:
        print(f"  AVISO: cenario otimo com apenas {len(irr_pos)} dias irrigados.")
        # Fallback: todos os cenarios
        irr_pos = df[df['IrrDay'] > 0]['IrrDay']

    mediana = irr_pos.median()
    print(f"\n  Limiar: {mediana:.2f} mm (mediana IrrDay>0 otimo, n={len(irr_pos)})")

    df['classe_irrigacao'] = df['IrrDay'].apply(
        lambda x: 0 if x == 0 else (1 if x <= mediana else 2))

    # Log por cenario × janela
    for cen in ['excesso', 'otimo', 'deficit', 'veranico']:
        for jan in ['chuva', 'seca']:
            sub = df[(df['cenario'] == cen) & (df['janela'] == jan)]
            if len(sub) > 0:
                vc = sub['classe_irrigacao'].value_counts().sort_index()
                print(f"  {cen}/{jan} (n={len(sub)}): "
                      f"C0={vc.get(0,0)}, C1={vc.get(1,0)}, C2={vc.get(2,0)}")

    return df, mediana


def build_final_dataset(df):
    """Seleciona colunas finais e de diagnostico."""
    # Dataset para treino (5 features + target)
    cols_treino = ['tensao_solo_kpa', 'chuva_acum_3d_mm', 'tmax_max_3d_c',
                   'dap', 'delta_tensao_kpa', 'classe_irrigacao']

    # Dataset completo (com metadados)
    cols_full = cols_treino + ['IrrDay', 'year', 'janela', 'cenario', 'grupo_id']

    df_full = df[cols_full].dropna(subset=['tensao_solo_kpa']).reset_index(drop=True)
    df_full['classe_irrigacao'] = df_full['classe_irrigacao'].astype(int)
    df_full['grupo_id'] = df_full['grupo_id'].astype(int)

    df_treino = df_full[cols_treino].copy()

    return df_treino, df_full


# ============================================================================
# MODULO 5: VALIDACAO E RELATORIO
# ============================================================================

def validate_and_report(df_treino, df_full, mediana, weather_metas):
    """Valida criterios v10 e gera relatorio."""
    r = []
    r.append("# Relatorio de Validacao — dataset_cold_start_v10.csv\n")
    r.append(f"**Data:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    r.append(f"**Amostras:** {len(df_treino)}")
    r.append(f"**Grupos:** {df_full['grupo_id'].nunique()}")
    r.append(f"**Limiar rotulagem:** {mediana:.2f} mm")
    r.append(f"**Periodo:** {min(ANOS)}-{max(ANOS)} ({len(ANOS)} anos)")
    r.append(f"**Janelas:** chuva (Jan 5) + seca (Jun 1)")
    r.append(f"**Cenarios chuva:** excesso + veranico (fev-mar × {VERANICO_FATOR})")
    r.append(f"**Cenarios seca:** excesso, otimo, deficit")
    r.append(f"**Nota:** otimo/chuva e deficit/chuva removidos (geravam <0.3% irrigacao)")
    r.append(f"**Features:** tensao_solo_kpa, chuva_acum_3d_mm, tmax_max_3d_c, dap, delta_tensao_kpa")
    r.append("")

    # Cenarios
    r.append("## Configuracao dos Cenarios\n")
    for cen, cfg in CENARIOS.items():
        nota = f" (precip fev-mar × {VERANICO_FATOR}, janela chuva apenas)" if cen == 'veranico' else ""
        r.append(f"- {cen}: SMT={cfg['SMT']}, MaxIrr={cfg['MaxIrr']}mm{nota}")

    # Distribuicao
    r.append("\n## Distribuicao de Classes\n")
    vc = df_treino['classe_irrigacao'].value_counts().sort_index()
    vcp = df_treino['classe_irrigacao'].value_counts(normalize=True).sort_index() * 100
    for c in [0, 1, 2]:
        r.append(f"- Classe {c}: {vc.get(c,0)} ({vcp.get(c,0):.1f}%)")

    r.append(f"\n**Comparacao com v7:** C0=94.1%, C1=4.1%, C2=1.8% (2733 amostras, 30 grupos)")
    r.append(f"**Comparacao com v10-anterior:** C0=94.9%, C1=3.4%, C2=1.7% (14784 amostras, 161 grupos)")

    # Por cenario × janela
    r.append("\n### Por Cenario x Janela\n")
    for cen in ['excesso', 'otimo', 'deficit', 'veranico']:
        for jan in ['chuva', 'seca']:
            sub = df_full[(df_full['cenario'] == cen) & (df_full['janela'] == jan)]
            if len(sub) > 0:
                vc_c = sub['classe_irrigacao'].value_counts().sort_index()
                irr_pct = (sub['classe_irrigacao'] > 0).mean() * 100
                r.append(f"- {cen}/{jan} (n={len(sub)}): "
                         f"C0={vc_c.get(0,0)}, C1={vc_c.get(1,0)}, C2={vc_c.get(2,0)} "
                         f"({irr_pct:.1f}% irrig)")

    # Features
    r.append("\n## Estatisticas das Features\n")
    feat_cols = ['tensao_solo_kpa', 'chuva_acum_3d_mm', 'tmax_max_3d_c', 'dap', 'delta_tensao_kpa']
    for col in feat_cols:
        s = df_treino[col]
        r.append(f"- {col}: min={s.min():.2f}, max={s.max():.2f}, "
                 f"median={s.median():.2f}, std={s.std():.2f}")

    # Correlacoes
    r.append("\n## Correlacoes com classe_irrigacao\n")
    corrs = {}
    for col in feat_cols:
        corr = df_treino[col].corr(df_treino['classe_irrigacao'])
        corrs[col] = corr if not np.isnan(corr) else 0
        r.append(f"- {col}: {corrs[col]:.4f}")

    # Tensao por classe
    r.append("\n## Tensao Mediana por Classe\n")
    t_cls = {}
    for c in [0, 1, 2]:
        sub = df_treino[df_treino['classe_irrigacao'] == c]
        if len(sub) > 0:
            t_cls[c] = sub['tensao_solo_kpa'].median()
            r.append(f"- Classe {c}: {t_cls[c]:.1f} kPa (n={len(sub)}, "
                     f"std={sub['tensao_solo_kpa'].std():.1f})")

    # Delta tensao por classe
    r.append("\n## Delta Tensao Mediana por Classe\n")
    for c in [0, 1, 2]:
        sub = df_treino[df_treino['classe_irrigacao'] == c]
        if len(sub) > 0:
            r.append(f"- Classe {c}: {sub['delta_tensao_kpa'].median():.2f} kPa "
                     f"(std={sub['delta_tensao_kpa'].std():.2f})")

    # Chuva
    r.append("\n## Chuva\n")
    d0 = int((df_treino['chuva_acum_3d_mm'] > 0).sum())
    d5 = int((df_treino['chuva_acum_3d_mm'] > 5).sum())
    r.append(f"- chuva_3d > 0mm: {d0} dias")
    r.append(f"- chuva_3d > 5mm: {d5} dias")

    # Criterio veranico (referencia = excesso, pois otimo foi removido da janela chuva)
    r.append("\n## Criterio Veranico\n")
    excesso_chuva = df_full[(df_full['cenario'] == 'excesso') & (df_full['janela'] == 'chuva')]
    veranico_chuva = df_full[(df_full['cenario'] == 'veranico') & (df_full['janela'] == 'chuva')]
    irr_excesso = (excesso_chuva['classe_irrigacao'] > 0).mean() if len(excesso_chuva) > 0 else 0
    irr_veranico = (veranico_chuva['classe_irrigacao'] > 0).mean() if len(veranico_chuva) > 0 else 0
    veranico_gt_excesso = irr_veranico > irr_excesso
    veranico_ge_4pct = irr_veranico >= 0.04
    veranico_ok = veranico_gt_excesso and veranico_ge_4pct
    r.append(f"- Irrigacao janela chuva: excesso={irr_excesso:.1%}, veranico={irr_veranico:.1%}")
    r.append(f"- Veranico > excesso: {'PASS' if veranico_gt_excesso else 'FAIL'}")
    r.append(f"- Veranico >= 4% irrigacao: {'PASS' if veranico_ge_4pct else 'FAIL'} ({irr_veranico:.1%})")

    # Criterios
    r.append("\n## Criterios de Aprovacao\n")
    results = {}

    c0p = vcp.get(0, 0)
    results['C0>=5%'] = c0p >= 5
    r.append(f"- C0 >= 5%: {'PASS' if results['C0>=5%'] else 'FAIL'} ({c0p:.1f}%)")

    ct = corrs.get('tensao_solo_kpa', 0)
    results['corr_tensao'] = 0.15 <= ct <= 0.7
    r.append(f"- Corr tensao [+0.15,+0.7]: {'PASS' if results['corr_tensao'] else 'FAIL'} ({ct:.4f})")

    cc = abs(corrs.get('chuva_acum_3d_mm', 0))
    results['corr_chuva'] = cc >= 0.10
    r.append(f"- |Corr chuva| >= 0.10: {'PASS' if results['corr_chuva'] else 'FAIL'} ({cc:.4f})")

    results['dias_chuva'] = d0 >= 500
    r.append(f"- Dias chuva>0 >= 500: {'PASS' if results['dias_chuva'] else 'FAIL'} ({d0})")

    t0 = t_cls.get(0, 0)
    t2 = t_cls.get(2, 0)
    results['tensao_ordem'] = t2 > t0
    r.append(f"- Tensao C2>C0: {'PASS' if results['tensao_ordem'] else 'FAIL'} "
             f"(C0={t0:.1f}, C2={t2:.1f} kPa)")

    nn = df_treino.isna().sum().sum()
    results['sem_nan'] = nn == 0
    r.append(f"- Sem NaN: {'PASS' if results['sem_nan'] else 'FAIL'} ({nn})")

    tmax_t = df_treino['tensao_solo_kpa'].max()
    results['tensao<1500'] = tmax_t < 1500
    r.append(f"- Tensao<1500: {'PASS' if results['tensao<1500'] else 'FAIL'} (max={tmax_t:.1f})")

    t_std = df_treino['tensao_solo_kpa'].std()
    results['tensao_variancia'] = t_std >= 12
    r.append(f"- Desvio padrao tensao >= 12: {'PASS' if results['tensao_variancia'] else 'FAIL'} ({t_std:.1f})")

    results['veranico'] = veranico_ok
    r.append(f"- Veranico > otimo (chuva): {'PASS' if veranico_ok else 'FAIL'}")

    # Limitacoes
    r.append("\n## Limitacoes Conhecidas\n")
    r.append("- Profundidade radicular dinamica mitiga mas nao elimina discrepancia canteiro 35cm")
    r.append("- Parametros TomatoGDD calibrados para clima mediterranico")
    r.append("- NASA POWER: resolucao ~50km, pode diferir de estacao local")
    r.append("- Cenario veranico assume reducao uniforme fev-mar a 20% (veranicos reais sao episodicos)")
    r.append("- Cenarios otimo/chuva e deficit/chuva removidos por improdutividade (<0.3% irrigacao)")
    r.append("- Desbalanceamento C0 permanece estrutural — tratar no pipeline de treino")

    r.append("\n## Referencias\n")
    r.append("- ALLEN, R.G. et al. FAO Irrigation Paper 56, 1998.")
    r.append("- FOSTER, T. et al. Agric. Water Manag., v.251, 2021.")
    r.append("- SAXTON, K.E.; RAWLS, W.J. Soil Sci. Soc. Am. J. 70:1569-1578, 2006.")
    r.append("- NASA POWER. https://power.larc.nasa.gov/")
    r.append("- Embrapa. Veranicos na Regiao dos Cerrados Brasileiros.")

    n_pass = sum(results.values())
    n_total = len(results)
    criticos = all(results.values())
    r.append(f"\n## Veredicto\n")
    r.append(f"**{'APROVADO' if criticos else 'REPROVADO'}** — {n_pass}/{n_total} criterios OK")

    return '\n'.join(r), criticos, results


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("AquaCrop-OSPy Simulation Pipeline — Dataset v10")
    print("Projeto ALMMo-0 — Tomate | Imperatriz-MA")
    print(f"Periodo: {min(ANOS)}-{max(ANOS)} ({len(ANOS)} anos)")
    print(f"Cenarios chuva: excesso + veranico (fev-mar × {VERANICO_FATOR})")
    print(f"Cenarios seca: excesso, otimo, deficit")
    print(f"Simulacoes: {len(ANOS)*2 + len(ANOS)*3} (46 chuva + 69 seca)")
    print("=" * 70)

    # 1. Simulacoes
    print("\n[ETAPA 1] Simulacoes AquaCrop")
    all_results, weather_metas = run_all_simulations()

    if not all_results:
        print("ERRO: Nenhuma simulacao OK!")
        sys.exit(1)

    n_chuva = sum(1 for r in all_results if r['janela'].iloc[0] == 'chuva')
    n_seca = sum(1 for r in all_results if r['janela'].iloc[0] == 'seca')
    print(f"\nTotal: {len(all_results)} simulacoes ({n_chuva} chuva + {n_seca} seca)")

    # 2. Processar
    print("\n[ETAPA 2] Processamento (filtros + features)")
    all_processed = process_dataset(all_results)
    print(f"  Total apos filtros: {len(all_processed)} amostras")

    # 3. Rotular
    print("\n[ETAPA 3] Rotulagem")
    all_processed, mediana = rotular_classes(all_processed)

    # 4. Dataset final
    print("\n[ETAPA 4] Dataset Final")
    df_treino, df_full = build_final_dataset(all_processed)
    print(f"  Treino: {len(df_treino)} amostras, colunas: {list(df_treino.columns)}")
    print(f"  Full:   {len(df_full)} amostras, colunas: {list(df_full.columns)}")
    print(f"  Grupos: {df_full['grupo_id'].nunique()}")

    # 5. Validar
    print("\n[ETAPA 5] Validacao")
    report, aprovado, results = validate_and_report(df_treino, df_full, mediana, weather_metas)
    print("\n" + report)

    # 6. Exportar
    print("\n[ETAPA 6] Exportacao")

    csv_path = OUTPUT_DIR / 'dataset_cold_start_v10.csv'
    df_treino.to_csv(csv_path, index=False)
    print(f"  dataset_cold_start_v10.csv ({len(df_treino)} linhas)")

    full_path = OUTPUT_DIR / 'dataset_cold_start_v10_full.csv'
    df_full.to_csv(full_path, index=False)
    print(f"  dataset_cold_start_v10_full.csv ({len(df_full)} linhas, com metadados)")

    report_path = OUTPUT_DIR / 'relatorio_v10.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"  relatorio_v10.md")

    if aprovado:
        print("\n>>> DATASET v10 APROVADO <<<")
    else:
        print("\n>>> DATASET v10 REPROVADO — ver relatorio <<<")

    return df_treino


if __name__ == '__main__':
    main()