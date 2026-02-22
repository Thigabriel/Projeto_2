#!/usr/bin/env python3
"""
============================================================================
Script de Diagnóstico v3.1 — AquaCrop-OSPy + NASA POWER
============================================================================
Correcao: adiciona cabecalho ao .txt para prepare_weather() nao consumir
a primeira linha de dados.
"""

import sys
import os
import math
from datetime import datetime
from pathlib import Path

print("=" * 60)
print("DIAGNOSTICO v3.1 — AquaCrop-OSPy + NASA POWER")
print("Imperatriz-MA (lat=-5.5253, lon=-47.4825)")
print("=" * 60)

# ---------------------------------------------------------------
# TESTE 1: Imports
# ---------------------------------------------------------------
print("\n[1/7] Testando imports...")

try:
    import numpy as np
    import pandas as pd
    print(f"  numpy {np.__version__}, pandas {pd.__version__} OK")
except ImportError as e:
    print(f"  {e}")
    sys.exit(1)

try:
    import requests
    print(f"  requests {requests.__version__} OK")
except ImportError:
    print("  requests nao instalado. Execute: pip install requests")
    sys.exit(1)

try:
    os.environ['DEVELOPMENT'] = 'True'
    from aquacrop import AquaCropModel, Soil, Crop, InitialWaterContent, IrrigationManagement, FieldMngt
    from aquacrop.utils import prepare_weather
    print("  aquacrop OK")
except ImportError as e:
    print(f"  aquacrop: {e}")
    sys.exit(1)

# ---------------------------------------------------------------
# TESTE 2: NASA POWER API
# ---------------------------------------------------------------
print("\n[2/7] Testando NASA POWER API...")

LAT = -5.5253
LON = -47.4825
ALTITUDE = 120
API_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

try:
    params = {
        'parameters': 'T2M_MAX,T2M_MIN,PRECTOTCORR,ALLSKY_SFC_SW_DWN,RH2M,WS2M',
        'community': 'AG',
        'longitude': LON, 'latitude': LAT,
        'start': '20230101', 'end': '20230110',
        'format': 'JSON'
    }
    resp = requests.get(API_URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    props = data['properties']['parameter']
    tmax_s = list(props['T2M_MAX'].values())[:3]
    print(f"  API OK. Tmax(3d)={tmax_s}")
except Exception as e:
    print(f"  Erro: {e}")
    sys.exit(1)

# ---------------------------------------------------------------
# TESTE 3: Crop TomatoGDD
# ---------------------------------------------------------------
print("\n[3/7] Testando Crop TomatoGDD...")

crop_name = None
for name in ['TomatoGDD', 'Tomato']:
    try:
        c = Crop(name, planting_date='01/05')
        crop_name = name
        print(f"  Crop('{name}') OK")
        for attr in ['Maturity', 'Tbase', 'Tupp', 'Zmin', 'Zmax', 'CCx', 'HI0']:
            if hasattr(c, attr):
                print(f"    {attr} = {getattr(c, attr)}")
        break
    except Exception as e:
        print(f"  Crop('{name}'): {e}")

if crop_name is None:
    print("  Nenhum crop Tomato encontrado!")
    sys.exit(1)

# ---------------------------------------------------------------
# TESTE 4: Buscar dados + calcular ETo + salvar .txt COM CABECALHO
# ---------------------------------------------------------------
print("\n[4/7] Buscando NASA POWER Jan-Jul 2023...")

def calc_eto_fao56(tmax, tmin, rs, rh, doy, lat_deg=LAT, alt=ALTITUDE):
    """ETo FAO-56 PM com u2=2.0 fixo."""
    tmean = (tmax + tmin) / 2.0
    lat_rad = lat_deg * math.pi / 180.0
    P = 101.3 * ((293.0 - 0.0065 * alt) / 293.0) ** 5.26
    gamma = 0.000665 * P
    e_tmax = 0.6108 * math.exp(17.27 * tmax / (tmax + 237.3))
    e_tmin = 0.6108 * math.exp(17.27 * tmin / (tmin + 237.3))
    es = (e_tmax + e_tmin) / 2.0
    ea = es * (rh / 100.0) if rh > 0 else e_tmin
    delta = 4098.0 * (0.6108 * math.exp(17.27 * tmean / (tmean + 237.3))) / (tmean + 237.3)**2
    dr = 1.0 + 0.033 * math.cos(2 * math.pi * doy / 365)
    d_sol = 0.409 * math.sin(2 * math.pi * doy / 365 - 1.39)
    ws = math.acos(-math.tan(lat_rad) * math.tan(d_sol))
    Ra = (24*60/math.pi) * 0.0820 * dr * (
        ws * math.sin(lat_rad) * math.sin(d_sol) +
        math.cos(lat_rad) * math.cos(d_sol) * math.sin(ws))
    Rso = (0.75 + 2e-5 * alt) * Ra
    Rns = 0.77 * rs
    sigma = 4.903e-9
    rs_r = min(rs / Rso, 1.0) if Rso > 0 else 0.5
    Rnl = sigma * ((tmax+273.16)**4 + (tmin+273.16)**4)/2 * \
          (0.34 - 0.14*math.sqrt(max(ea, 0.01))) * (1.35*rs_r - 0.35)
    Rn = Rns - Rnl
    u2 = 2.0
    num = 0.408*delta*Rn + gamma*(900/(tmean+273))*u2*(es - ea)
    den = delta + gamma*(1 + 0.34*u2)
    return max(num/den, 0.0)

try:
    params_full = {
        'parameters': 'T2M_MAX,T2M_MIN,PRECTOTCORR,ALLSKY_SFC_SW_DWN,RH2M,WS2M',
        'community': 'AG',
        'longitude': LON, 'latitude': LAT,
        'start': '20230101', 'end': '20230731',
        'format': 'JSON'
    }
    resp = requests.get(API_URL, params=params_full, timeout=120)
    resp.raise_for_status()
    data_full = resp.json()['properties']['parameter']

    dates = pd.date_range('2023-01-01', '2023-07-31')
    lines = []
    eto_vals = []

    for d in dates:
        key = d.strftime('%Y%m%d')
        tmax = data_full['T2M_MAX'].get(key, -999)
        tmin = data_full['T2M_MIN'].get(key, -999)
        prec = data_full['PRECTOTCORR'].get(key, -999)
        rs = data_full['ALLSKY_SFC_SW_DWN'].get(key, -999)
        rh = data_full['RH2M'].get(key, -999)
        if tmax < -900 or tmin < -900 or rs < -900:
            continue
        if rh < -900:
            rh = 75.0
        prec = max(prec, 0.0)
        doy = d.timetuple().tm_yday
        eto = calc_eto_fao56(tmax, tmin, rs, rh, doy)
        eto_vals.append(eto)
        lines.append(f"{d.day}\t{d.month}\t{d.year}\t{tmin:.2f}\t{tmax:.2f}\t{prec:.2f}\t{eto:.4f}")

    # *** CORRECAO v3.1: Adicionar cabecalho ***
    txt_path = Path('weather_imperatriz_2023_test.txt')
    with open(txt_path, 'w') as f:
        f.write("Day\tMonth\tYear\tMinTemp\tMaxTemp\tPrecipitation\tReferenceET\n")
        for line in lines:
            f.write(line + '\n')

    print(f"  {len(lines)} dias salvos em {txt_path} (com cabecalho)")
    print(f"  ETo: mediana={np.median(eto_vals):.2f}, max={max(eto_vals):.2f} mm/dia")

    # GDD
    tbase = 7.0
    gdd_list = []
    for d in dates:
        key = d.strftime('%Y%m%d')
        tmax = data_full['T2M_MAX'].get(key, -999)
        tmin = data_full['T2M_MIN'].get(key, -999)
        if tmax < -900: continue
        gdd_list.append(max((tmax + tmin)/2 - tbase, 0))
    gdd_cum = np.cumsum(gdd_list)
    print(f"  GDD acumulado: {gdd_cum[-1]:.0f}")
    idx = np.argmax(gdd_cum >= 1933)
    if gdd_cum[idx] >= 1933:
        print(f"  Maturidade (1933 GDD) ~dia {idx+1} (~{dates[idx].strftime('%Y-%m-%d')})")

except Exception as e:
    print(f"  Erro: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ---------------------------------------------------------------
# TESTE 5: Simulação com prepare_weather()
# ---------------------------------------------------------------
print(f"\n[5/7] Simulacao AquaCrop via prepare_weather('{txt_path}')...")

try:
    wdf = prepare_weather(str(txt_path))
    print(f"  prepare_weather OK: {len(wdf)} linhas")
    print(f"  Colunas: {list(wdf.columns)}")
    print(f"  Primeira data: {wdf['Date'].iloc[0]}")
    print(f"  Ultima data:   {wdf['Date'].iloc[-1]}")
    print(f"  Primeiras 3 linhas:\n{wdf.head(3)}")

    soil = Soil('SandyLoam')
    crop = Crop(crop_name, planting_date='01/05')
    init_wc = InitialWaterContent(value=['FC'])
    irr = IrrigationManagement(irrigation_method=4, NetIrrSMT=79)
    field = FieldMngt(mulches=True, mulch_pct=80, f_mulch=0.3)

    model = AquaCropModel(
        sim_start_time='2023/01/01',
        sim_end_time='2023/07/31',
        weather_df=wdf,
        soil=soil,
        crop=crop,
        initial_water_content=init_wc,
        irrigation_management=irr,
        field_management=field,
    )

    model.run_model(till_termination=True)

    wf = model._outputs.water_flux
    cg = model._outputs.crop_growth
    fs = model._outputs.final_stats

    print(f"\n  SIMULACAO COMPLETADA!")

    print(f"\n  --- water_flux ({wf.shape[0]} x {wf.shape[1]}) ---")
    for col in wf.columns:
        v = wf[col].dropna()
        if len(v) > 0 and v.dtype in ['float64', 'int64', 'float32']:
            print(f"    {col:20s}: min={v.min():8.3f}  max={v.max():8.3f}  median={v.median():8.3f}")
        else:
            print(f"    {col:20s}: {v.dtype}")

    print(f"\n  --- crop_growth ({cg.shape[0]} x {cg.shape[1]}) ---")
    for col in cg.columns:
        v = cg[col].dropna()
        if len(v) > 0 and v.dtype in ['float64', 'int64', 'float32']:
            print(f"    {col:20s}: min={v.min():8.3f}  max={v.max():8.3f}  median={v.median():8.3f}")
        else:
            print(f"    {col:20s}: {v.dtype}")

    if fs is not None and len(fs) > 0:
        print(f"\n  --- final_stats ---")
        print(f"  {fs.to_string()}")

    # ========================
    # RESUMO PARA SCRIPT PRINCIPAL
    # ========================
    print(f"\n  {'='*55}")
    print(f"  RESUMO PARA O SCRIPT PRINCIPAL (COPIE ISTO):")
    print(f"  {'='*55}")

    for c in ['IrrDay', 'Irr', 'irr_day', 'IrrNet']:
        if c in wf.columns:
            print(f"  COL_IRR = '{c}'  # {(wf[c]>0).sum()} dias irrig, total={wf[c].sum():.1f}mm")
            break

    for c in ['Wr', 'Wr(1)', 'wr', 'th1', 'WrAct']:
        if c in wf.columns:
            print(f"  COL_WR = '{c}'  # min={wf[c].min():.1f}, max={wf[c].max():.1f} mm")
            break

    for c in ['Tr', 'TrAct', 'tr', 'Tact']:
        if c in wf.columns:
            print(f"  COL_TR = '{c}'  # dias Tr>0.1: {(wf[c]>0.1).sum()}")
            break

    for c in ['Precipitation', 'Prec', 'Rain', 'precipitation', 'P']:
        if c in wf.columns:
            print(f"  COL_PREC = '{c}'")
            break

    for c in ['DAP', 'dap', 'GrowingSeasonDay', 'growing_season_day']:
        if c in cg.columns:
            print(f"  COL_DAP = '{c}'  # max={cg[c].max():.0f}")
            break

    found_zroot = False
    for c in ['z_root', 'Zroot', 'zRoot', 'RootDepth', 'root_depth', 'ZrAct', 'Zr']:
        if c in cg.columns:
            print(f"  COL_ZROOT = '{c}'  # min={cg[c].min():.3f}, max={cg[c].max():.3f} m")
            found_zroot = True
            break
    if not found_zroot:
        print(f"  COL_ZROOT = None  # NAO ENCONTRADA!")
        print(f"  crop_growth colunas: {list(cg.columns)}")

    print(f"  CROP_NAME = '{crop_name}'")
    print(f"  SIM_END = '07/31'  # GDD OK")
    print(f"  {'='*55}")

except Exception as e:
    print(f"\n  ERRO: {e}")
    import traceback; traceback.print_exc()
    print(f"\n  Se o erro mencionar 'Timestamp' ou 'float':")
    print(f"  -> Tente: pip install pandas==2.0.3")
    print(f"  -> Ou:    pip install numpy==1.26.4")

# ---------------------------------------------------------------
# TESTE 6: ETo
# ---------------------------------------------------------------
print(f"\n[6/7] ETo: mediana={np.median(eto_vals):.2f} mm/dia ", end="")
print("OK" if 4.0 <= np.median(eto_vals) <= 6.0 else "FORA DO ESPERADO!")

# ---------------------------------------------------------------
# TESTE 7: Versões
# ---------------------------------------------------------------
print(f"\n[7/7] Versoes: numpy={np.__version__}, pandas={pd.__version__}, requests={requests.__version__}")
try:
    print(f"  aquacrop={__import__('aquacrop').__version__}")
except:
    print(f"  aquacrop=desconhecida")

print(f"\n{'='*60}")
print("DIAGNOSTICO COMPLETO — Cole o RESUMO na conversa com Claude")
print(f"{'='*60}")