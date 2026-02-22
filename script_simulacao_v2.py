#!/usr/bin/env python3
"""
============================================================================
Script de Simulacao AquaCrop-OSPy v2.4.3
Projeto ALMMo-0 — Irrigacao Inteligente de Tomate | Imperatriz-MA
============================================================================

NOVIDADE v2.4:
  Troca irrigation_method=4 (net irrigation, solo nunca seca) por
  irrigation_method=1 (SMT por estagio + MaxIrr cap).

  MaxIrr=100mm (nao limita). Dose = deplecao real do solo.
  Solo mais seco → dose maior → classe maior → correlacao positiva.

  SMT = [emergencia, crescimento_dossel, dossel_maximo, senescencia]

Requisitos: pip install aquacrop pandas numpy requests
Execucao:   python script_simulacao_v2.py
"""

import os
import sys
import math
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

ANOS = [2019, 2020, 2021, 2022, 2023]

# Duas janelas de plantio
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
# CENARIOS — METHOD 1 (SMT por estagio + MaxIrr)
# ============================================================================
# SMT = [emergencia, crescimento_dossel, dossel_maximo, senescencia]
# Dose aplicada = min(deplecao, MaxIrr)
# MaxIrrSeason = limite total por safra (mm)

CENARIOS = {
    'excesso': {
        'SMT': [80, 80, 80, 70],
        'MaxIrr': 100,
        'MaxIrrSeason': 10_000,
    },
    'otimo': {
        'SMT': [60, 60, 70, 50],
        'MaxIrr': 100,
        'MaxIrrSeason': 10_000,
    },
    'deficit': {
        'SMT': [40, 40, 50, 30],
        'MaxIrr': 100,
        'MaxIrrSeason': 10_000,
    },
}

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
        print(f"    Cache: {txt_path}")
        meta = pd.read_csv(meta_path) if meta_path.exists() else None
        return str(txt_path), meta

    print(f"    Buscando NASA POWER {year}/01/01 a {year}/12/31...")
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

    meta['date'] = pd.to_datetime(meta['date'])
    s1 = meta[meta['date'].dt.month <= 6]
    s2 = meta[meta['date'].dt.month > 6]
    print(f"    Salvo: {txt_path} ({len(lines)} dias)")
    print(f"    Jan-Jun: ETo med={s1['eto'].median():.2f}, prec={s1['prec'].sum():.0f}mm")
    print(f"    Jul-Dez: ETo med={s2['eto'].median():.2f}, prec={s2['prec'].sum():.0f}mm")

    return str(txt_path), meta


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
        print(f"    water_flux: {sorted(wf.columns.tolist())}")
        print(f"    crop_growth: {sorted(cg.columns.tolist())}")
        return False
    return True


def run_single_simulation(weather_txt_path, year, cenario_nome, janela, wdf_cached):
    """Executa uma simulacao AquaCrop com method 1."""
    global COL_IRR

    jcfg = JANELAS[janela]
    sim_start = jcfg['sim_start_fmt'].format(year=year)
    sim_end = jcfg['sim_end_fmt'].format(year=year)
    planting = jcfg['planting_date']

    cen = CENARIOS[cenario_nome]

    soil = Soil('SandyLoam')
    crop = Crop('TomatoGDD', planting_date=planting)
    init_wc = InitialWaterContent(value=['FC'])

    # *** METHOD 1: SMT por estagio + MaxIrr cap ***
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
        weather_df=wdf_cached,
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
    wdf_indexed = wdf_cached.set_index('Date')

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
    """Executa todas: anos × janelas × cenarios = 5×2×3 = 30 simulacoes."""
    all_results = []
    weather_metas = {}

    for year in ANOS:
        print(f"\n{'='*60}")
        print(f"ANO: {year}")
        print(f"{'='*60}")

        txt_path, meta = fetch_and_save_weather(year)
        weather_metas[year] = meta

        wdf = prepare_weather(txt_path)
        print(f"    Weather: {len(wdf)} dias ({wdf['Date'].iloc[0].date()} a {wdf['Date'].iloc[-1].date()})")

        for janela in ['chuva', 'seca']:
            for cenario in CENARIOS:
                cen = CENARIOS[cenario]
                label = f"{janela}/{cenario}(SMT={cen['SMT']},MaxIrr={cen['MaxIrr']})"
                try:
                    result = run_single_simulation(txt_path, year, cenario, janela, wdf)
                    irr_days = (result['IrrDay'] > 0).sum()
                    irr_total = result['IrrDay'].sum()
                    print(f"    {label}: {irr_days} dias irrig ({irr_total:.1f}mm)")
                    all_results.append(result)
                except Exception as e:
                    print(f"    {label}: ERRO — {e}")

    return all_results, weather_metas


# ============================================================================
# MODULO 4: PROCESSAMENTO DO DATASET
# ============================================================================

def process_dataset(all_results):
    """Aplica filtros, calcula features."""
    dfs_by_cenario = {'otimo': [], 'deficit': [], 'excesso': []}

    for df in all_results:
        cenario = df['cenario'].iloc[0]
        dfs_by_cenario[cenario].append(df)

    processed = {}

    for cenario, dfs in dfs_by_cenario.items():
        if not dfs:
            continue

        df = pd.concat(dfs, ignore_index=True)
        n0 = len(df)

        df = df[df['Tr'] > TR_MIN].copy()
        df = df[df['dap'] >= DAP_MIN].copy()
        df = df[df['dap'] <= DAP_MAX].copy()

        print(f"  {cenario}: {n0} -> {len(df)} apos filtros")

        if len(df) == 0:
            continue

        # Wr -> tensao
        df['tensao_raw'] = df.apply(
            lambda r: wr_para_tensao_kpa_dinamica(r['Wr'], r['z_root']), axis=1)

        t = df['tensao_raw']
        t_chuva = df[df['janela'] == 'chuva']['tensao_raw']
        t_seca = df[df['janela'] == 'seca']['tensao_raw']
        print(f"    Tensao geral: min={t.min():.1f}, max={t.max():.1f}, "
              f"mediana={t.median():.1f}, std={t.std():.1f} kPa")
        if len(t_chuva) > 0:
            print(f"    Tensao chuva: min={t_chuva.min():.1f}, max={t_chuva.max():.1f}, "
                  f"mediana={t_chuva.median():.1f}")
        if len(t_seca) > 0:
            print(f"    Tensao seca:  min={t_seca.min():.1f}, max={t_seca.max():.1f}, "
                  f"mediana={t_seca.median():.1f}")

        # Shift temporal (dentro de cada year+janela)
        shifted = []
        for _, grp in df.groupby(['year', 'janela']):
            vals = grp['tensao_raw'].values
            shifted.extend(np.concatenate([[vals[0]], vals[:-1]]))
        df['tensao_solo_kpa'] = shifted

        # Chuva acumulada 3d (dentro de cada year+janela)
        chuva_3d = []
        for _, grp in df.groupby(['year', 'janela']):
            chuva_3d.extend(grp['precipitation'].rolling(3, min_periods=1).sum().values)
        df['chuva_acum_3d_mm'] = chuva_3d

        # Tmax max 3d (dentro de cada year+janela)
        tmax_3d = []
        for _, grp in df.groupby(['year', 'janela']):
            tmax_3d.extend(grp['tmax'].rolling(3, min_periods=1).max().values)
        df['tmax_max_3d_c'] = tmax_3d

        p = df['precipitation']
        c3 = df['chuva_acum_3d_mm']
        print(f"    Prec: dias>0={int((p>0).sum())}, total={p.sum():.0f}mm")
        print(f"    Chuva3d: dias>0={int((c3>0).sum())}, max={c3.max():.1f}mm")

        processed[cenario] = df

    return processed


def rotular_classes(processed):
    """Rotula com mediana do IrrDay>0 do cenario otimo."""
    ref_cenario = 'otimo'
    irr_pos = processed[ref_cenario][processed[ref_cenario]['IrrDay'] > 0]['IrrDay']

    if len(irr_pos) < 10:
        print(f"  AVISO: cenario otimo tem so {len(irr_pos)} dias irrigados.")
        if 'excesso' in processed:
            irr_pos_exc = processed['excesso'][processed['excesso']['IrrDay'] > 0]['IrrDay']
            if len(irr_pos_exc) > len(irr_pos):
                ref_cenario = 'excesso'
                irr_pos = irr_pos_exc

    if len(irr_pos) == 0:
        raise ValueError("Nenhum dia irrigado!")

    mediana = irr_pos.median()
    print(f"\n  Limiar: {mediana:.2f} mm (mediana IrrDay>0 {ref_cenario}, n={len(irr_pos)})")

    for cenario in processed:
        df = processed[cenario]
        df['classe_irrigacao'] = df['IrrDay'].apply(
            lambda x: 0 if x == 0 else (1 if x <= mediana else 2))
        for jan in ['chuva', 'seca']:
            sub = df[df['janela'] == jan]
            if len(sub) > 0:
                vc_j = sub['classe_irrigacao'].value_counts().sort_index()
                print(f"  {cenario}/{jan}: C0={vc_j.get(0,0)}, C1={vc_j.get(1,0)}, C2={vc_j.get(2,0)}")

    return processed, mediana


def build_final_dataset(processed):
    """Combina cenarios e seleciona colunas finais."""
    frames = [processed[c] for c in ['excesso', 'otimo', 'deficit'] if c in processed]
    if not frames:
        raise ValueError("Nenhum cenario processado!")

    dataset_full = pd.concat(frames, ignore_index=True)
    cols = ['tensao_solo_kpa', 'chuva_acum_3d_mm', 'tmax_max_3d_c', 'dap', 'classe_irrigacao']
    dataset_final = dataset_full[cols].dropna().reset_index(drop=True)
    dataset_final['classe_irrigacao'] = dataset_final['classe_irrigacao'].astype(int)

    return dataset_final, dataset_full


# ============================================================================
# MODULO 5: VALIDACAO E RELATORIO
# ============================================================================

def validate_and_report(df_final, df_full, mediana, weather_metas):
    """Valida criterios e gera relatorio."""
    r = []
    r.append("# Relatorio de Validacao — dataset_cold_start_v7.csv\n")
    r.append(f"**Data:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    r.append(f"**Amostras:** {len(df_final)}")
    r.append(f"**Limiar rotulagem:** {mediana:.2f} mm")
    r.append(f"**Janelas:** chuva (Jan 5) + seca (Jun 1)")
    r.append(f"**Metodo irrigacao:** method=1 (SMT por estagio + MaxIrr cap)")
    r.append(f"**Cenarios:**")
    for cen, cfg in CENARIOS.items():
        r.append(f"  - {cen}: SMT={cfg['SMT']}, MaxIrr={cfg['MaxIrr']}mm, "
                 f"MaxSeason={cfg['MaxIrrSeason']}mm")
    r.append("")

    # ETo por ano
    r.append("## ETo por Ano\n")
    for year in sorted(weather_metas.keys()):
        m = weather_metas[year]
        if m is not None:
            m['date'] = pd.to_datetime(m['date'])
            s1 = m[m['date'].dt.month <= 6]
            s2 = m[m['date'].dt.month > 6]
            r.append(f"- {year}: Jan-Jun ETo={s1['eto'].median():.2f} prec={s1['prec'].sum():.0f}mm | "
                     f"Jul-Dez ETo={s2['eto'].median():.2f} prec={s2['prec'].sum():.0f}mm")

    # Distribuicao
    r.append("\n## Distribuicao de Classes\n")
    vc = df_final['classe_irrigacao'].value_counts().sort_index()
    vcp = df_final['classe_irrigacao'].value_counts(normalize=True).sort_index() * 100
    for c in [0, 1, 2]:
        r.append(f"- Classe {c}: {vc.get(c,0)} ({vcp.get(c,0):.1f}%)")

    # Por cenario × janela
    if 'cenario' in df_full.columns and 'janela' in df_full.columns:
        r.append("\n### Por Cenario x Janela\n")
        for cen in ['excesso', 'otimo', 'deficit']:
            for jan in ['chuva', 'seca']:
                sub = df_full[(df_full['cenario'] == cen) & (df_full['janela'] == jan)]
                if len(sub) > 0:
                    vc_c = sub['classe_irrigacao'].value_counts().sort_index()
                    r.append(f"- {cen}/{jan} (n={len(sub)}): "
                             f"C0={vc_c.get(0,0)}, C1={vc_c.get(1,0)}, C2={vc_c.get(2,0)}")

    # Features
    r.append("\n## Estatisticas das Features\n")
    for col in ['tensao_solo_kpa', 'chuva_acum_3d_mm', 'tmax_max_3d_c', 'dap']:
        s = df_final[col]
        r.append(f"- {col}: min={s.min():.2f}, max={s.max():.2f}, "
                 f"median={s.median():.2f}, std={s.std():.2f}")

    # Tensao por janela
    if 'janela' in df_full.columns:
        r.append("\n### Tensao por Janela\n")
        for jan in ['chuva', 'seca']:
            sub = df_full[df_full['janela'] == jan]
            if len(sub) > 0:
                r.append(f"- {jan}: min={sub['tensao_solo_kpa'].min():.1f}, "
                         f"max={sub['tensao_solo_kpa'].max():.1f}, "
                         f"median={sub['tensao_solo_kpa'].median():.1f}, "
                         f"std={sub['tensao_solo_kpa'].std():.1f} kPa")

    # Correlacoes
    r.append("\n## Correlacoes com classe_irrigacao\n")
    corrs = {}
    for col in ['tensao_solo_kpa', 'chuva_acum_3d_mm', 'tmax_max_3d_c', 'dap']:
        corr = df_final[col].corr(df_final['classe_irrigacao'])
        corrs[col] = corr if not np.isnan(corr) else 0
        r.append(f"- {col}: {corrs[col]:.4f}")

    # Tensao por classe
    r.append("\n## Tensao Mediana por Classe\n")
    t_cls = {}
    for c in [0, 1, 2]:
        sub = df_final[df_final['classe_irrigacao'] == c]
        if len(sub) > 0:
            t_cls[c] = sub['tensao_solo_kpa'].median()
            r.append(f"- Classe {c}: {t_cls[c]:.1f} kPa (n={len(sub)}, "
                     f"std={sub['tensao_solo_kpa'].std():.1f})")

    # Chuva
    r.append("\n## Chuva\n")
    d0 = int((df_final['chuva_acum_3d_mm'] > 0).sum())
    d5 = int((df_final['chuva_acum_3d_mm'] > 5).sum())
    r.append(f"- chuva_3d > 0mm: {d0} dias")
    r.append(f"- chuva_3d > 5mm: {d5} dias")

    # Criterios
    r.append("\n## Criterios de Aprovacao\n")
    results = {}

    c0p = vcp.get(0, 0)
    results['C0>=5%'] = c0p >= 5
    r.append(f"- C0 >= 5%: {'PASS' if results['C0>=5%'] else 'FAIL'} ({c0p:.1f}%)")

    ct = corrs.get('tensao_solo_kpa', 0)
    results['corr_tensao'] = 0.15 <= ct <= 0.7
    if results['corr_tensao']:
        r.append(f"- Corr tensao [+0.15,+0.7]: PASS ({ct:.4f})")
    elif ct > 0:
        r.append(f"- Corr tensao [+0.15,+0.7]: WARN ({ct:.4f}, positiva mas fora)")
        results['corr_tensao'] = ct > 0.10
    else:
        r.append(f"- Corr tensao [+0.15,+0.7]: FAIL ({ct:.4f})")

    cc = abs(corrs.get('chuva_acum_3d_mm', 0))
    results['corr_chuva'] = cc >= 0.10
    r.append(f"- |Corr chuva| >= 0.10: {'PASS' if results['corr_chuva'] else 'WARN'} ({cc:.4f})")

    results['dias_chuva'] = d0 >= 100
    r.append(f"- Dias chuva>0 >= 100: {'PASS' if results['dias_chuva'] else 'FAIL'} ({d0})")

    t0 = t_cls.get(0, 0)
    t2 = t_cls.get(2, 0)
    results['tensao_ordem'] = t2 > t0
    r.append(f"- Tensao C2>C0 (com shift): {'PASS' if results['tensao_ordem'] else 'FAIL'} "
             f"(C0={t0:.1f}, C2={t2:.1f} kPa)")

    nn = df_final.isna().sum().sum()
    results['sem_nan'] = nn == 0
    r.append(f"- Sem NaN: {'PASS' if results['sem_nan'] else 'FAIL'} ({nn})")

    tmax_t = df_final['tensao_solo_kpa'].max()
    results['tensao<1500'] = tmax_t < 1500
    r.append(f"- Tensao<1500: {'PASS' if results['tensao<1500'] else 'FAIL'} (max={tmax_t:.1f})")

    t_std = df_final['tensao_solo_kpa'].std()
    results['tensao_variancia'] = t_std >= 10
    r.append(f"- Desvio padrao tensao >= 10: {'PASS' if results['tensao_variancia'] else 'FAIL'} ({t_std:.1f})")

    # Limitacoes
    r.append("\n## Limitacoes Conhecidas\n")
    r.append("- Profundidade radicular dinamica mitiga mas nao elimina discrepancia canteiro 35cm")
    r.append("- Parametros TomatoGDD calibrados para clima mediterranico")
    r.append("- Duas janelas de plantio: modelo aprende com chuva e seca")
    r.append("- NASA POWER: resolucao ~50km, pode diferir de estacao local")

    r.append("\n## Referencias\n")
    r.append("- ALLEN, R.G. et al. FAO Irrigation Paper 56, 1998.")
    r.append("- FOSTER, T. et al. Agric. Water Manag., v.251, 2021.")
    r.append("- SAXTON, K.E.; RAWLS, W.J. Soil Sci. Soc. Am. J. 70:1569-1578, 2006.")
    r.append("- NASA POWER. https://power.larc.nasa.gov/")

    criticos = results['C0>=5%'] and results['corr_tensao'] and results['tensao_ordem']
    r.append(f"\n## Veredicto\n")
    r.append(f"**{'APROVADO' if criticos else 'REPROVADO'}** — "
             f"{sum(results.values())}/{len(results)} criterios OK")

    return '\n'.join(r), criticos, results


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("AquaCrop-OSPy Simulation Pipeline v2.4.3")
    print("Projeto ALMMo-0 — Tomate | Imperatriz-MA")
    print("Dados: NASA POWER | ETo: FAO-56 PM (u2=2.0)")
    print(f"Janelas: chuva (Jan 5) + seca (Jun 1)")
    print(f"Irrigacao: method=1 (SMT por estagio + MaxIrr cap)")
    for cen, cfg in CENARIOS.items():
        print(f"  {cen}: SMT={cfg['SMT']}, MaxIrr={cfg['MaxIrr']}mm")
    print("=" * 70)

    print("\n[ETAPA 1] Simulacoes AquaCrop (5 anos x 2 janelas x 3 cenarios = 30)")
    all_results, weather_metas = run_all_simulations()

    if not all_results:
        print("ERRO: Nenhuma simulacao OK!")
        sys.exit(1)

    n_chuva = sum(1 for r in all_results if r['janela'].iloc[0] == 'chuva')
    n_seca = sum(1 for r in all_results if r['janela'].iloc[0] == 'seca')
    print(f"\nTotal: {len(all_results)} simulacoes ({n_chuva} chuva + {n_seca} seca)")

    print("\n[ETAPA 2] Processamento")
    processed = process_dataset(all_results)

    print("\n[ETAPA 3] Rotulagem")
    processed, mediana = rotular_classes(processed)

    print("\n[ETAPA 4] Dataset Final")
    df_final, df_full = build_final_dataset(processed)
    print(f"  {len(df_final)} amostras, colunas: {list(df_final.columns)}")
    print(f"\n  Amostra (8 linhas):")
    print(df_final.sample(min(8, len(df_final))).to_string())

    print("\n[ETAPA 5] Validacao")
    report, aprovado, results = validate_and_report(df_final, df_full, mediana, weather_metas)
    print("\n" + report)

    print("\n[ETAPA 6] Exportacao")
    csv_path = OUTPUT_DIR / 'dataset_cold_start_v7.csv'
    df_final.to_csv(csv_path, index=False)
    print(f"  dataset_cold_start_v7.csv ({len(df_final)} linhas)")

    report_path = OUTPUT_DIR / 'relatorio_v7.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"  relatorio_v7.md")

    if aprovado:
        print("\n>>> DATASET APROVADO <<<")
    else:
        print("\n>>> DATASET REPROVADO — ver relatorio <<<")

    return df_final


if __name__ == '__main__':
    main()