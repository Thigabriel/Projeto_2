#!/usr/bin/env python3
"""
============================================================================
Demo de Evolucao Online — Acailandia-MA | 3 Anos (2020-2021-2022)
============================================================================

Simula um ciclo de campo real de 3 anos completos:
  - AquaCrop gera o "ground truth" de irrigacao (via NASA POWER)
  - O ensemble toma decisoes dia a dia
  - O modulo de evolucao observa as consequencias e retreina quando detecta
    anomalia no delta do dia seguinte
  - Tudo ocorre simultaneamente, na ordem correta:

      Dia T:
        1. Observa tensao_hoje (consequencia da decisao de ontem)
        2. Modulo de evolucao calcula z-score → retreina se anomalo
        3. Ensemble (possivelmente evoluido) decide a classe de hoje
        4. Registra decisao para avaliacao amanha

Saida:
  - decisoes_evolucao_acailandia_3anos.csv  (linha por linha, dia a dia)

Pre-requisito:
  - memoria_cold_start_v12_ensemble.pkl

Execucao:
  python demo_evolucao_acailandia.py

Obs: a primeira execucao busca dados NASA POWER (~3 chamadas de API).
     As seguintes usam os arquivos em cache em weather_files/.
============================================================================
"""

import os, sys, math, time, warnings, threading, pickle, csv
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import requests

os.environ['DEVELOPMENT'] = 'True'
try:
    from aquacrop import (AquaCropModel, Soil, Crop, InitialWaterContent,
                          IrrigationManagement, FieldMngt)
    from aquacrop.utils import prepare_weather
except ImportError:
    print("ERRO: pip install aquacrop"); sys.exit(1)

from evolucao_online import ALMMo0

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACAO
# ============================================================================
LAT, LON, ALTITUDE = -4.9484, -47.5025, 120   # Acailandia-MA
ANOS               = [2020, 2021, 2022]         # 3 anos de campo simulado
CENARIO_NOME       = 'smt_otimo'
JANELA_NOME        = 'seca'
PKL_PATH           = 'memoria_cold_start_v12_ensemble.pkl'
OUTPUT_CSV         = 'decisoes_evolucao_acailandia_3anos.csv'
WEATHER_DIR        = Path('weather_files')
WEATHER_DIR.mkdir(exist_ok=True)

API_URL      = "https://power.larc.nasa.gov/api/temporal/daily/point"
TXT_HEADER   = "Day\tMonth\tYear\tMinTemp\tMaxTemp\tPrecipitation\tReferenceET"
MAX_SIM_S    = 120

# Solo e cultura (idem ao treino)
THETA_SAT, THETA_CC, THETA_PM = 0.3863, 0.1864, 0.0853
A_SAXTON, B_SAXTON             = 0.0090, 4.8825
DAP_MIN, DAP_MAX, TR_MIN       = 14, 107, 0.1
IRR_MIN_MM                     = 2.0
CLASSE_C1_MAX                  = 10.0

# Janela seca: plantio 01/jun, sim de 15/mai a 31/dez
JANELA_CFG = {
    'planting_date': '06/01',
    'sim_start_tpl': '{ano}/05/15',
    'sim_end_tpl':   '{ano}/12/31',
}

CENARIO_CFG = {
    'method': 1, 'SMT': [60,60,70,50], 'MaxIrr': 100, 'MaxIrrSeason': 10000
}

FEATURE_COLS = [
    'tensao_solo_kpa', 'chuva_acum_3d_mm',
    'tmax_max_3d_c',   'dap',
    'delta_tensao_kpa'
]

NOMES_CLASSE = {0: 'C0-SemIrrig', 1: 'C1-Manutencao', 2: 'C2-Intensiva'}

# ============================================================================
# MODULO 1 — METEOROLOGIA
# ============================================================================
def _eto(tmax, tmin, rs, rh, doy):
    tmean   = (tmax + tmin) / 2.0
    lat_r   = LAT * math.pi / 180.0
    P       = 101.3 * ((293.0 - 0.0065*ALTITUDE) / 293.0)**5.26
    gam     = 0.000665 * P
    etmax   = 0.6108 * math.exp(17.27*tmax  / (tmax  + 237.3))
    etmin   = 0.6108 * math.exp(17.27*tmin  / (tmin  + 237.3))
    es      = (etmax + etmin) / 2.0
    ea      = es * (rh/100.0) if rh > 0 else etmin
    delta   = 4098.0 * (0.6108*math.exp(17.27*tmean/(tmean+237.3))) / (tmean+237.3)**2
    dr      = 1.0 + 0.033*math.cos(2*math.pi*doy/365)
    d_sol   = 0.409*math.sin(2*math.pi*doy/365 - 1.39)
    ws      = math.acos(-math.tan(lat_r)*math.tan(d_sol))
    Ra      = (24*60/math.pi)*0.0820*dr*(ws*math.sin(lat_r)*math.sin(d_sol)
              + math.cos(lat_r)*math.cos(d_sol)*math.sin(ws))
    Rso     = (0.75 + 2e-5*ALTITUDE)*Ra
    Rns     = 0.77 * rs
    sig     = 4.903e-9
    rs_r    = min(rs/Rso, 1.0) if Rso > 0 else 0.5
    Rnl     = sig*((tmax+273.16)**4+(tmin+273.16)**4)/2*(0.34-0.14*math.sqrt(max(ea,0.01)))*(1.35*rs_r-0.35)
    Rn      = Rns - Rnl
    u2      = 2.0
    num     = 0.408*delta*Rn + gam*(900/(tmean+273))*u2*(es-ea)
    den     = delta + gam*(1 + 0.34*u2)
    return max(num/den, 0.0)


def fetch_weather(ano):
    txt  = WEATHER_DIR / f'weather_acailandia_{ano}_full.txt'
    meta = WEATHER_DIR / f'weather_acailandia_{ano}_full_meta.csv'
    if txt.exists():
        print(f"    Cache: {txt}")
        return str(txt)
    print(f"    NASA POWER → Acailandia {ano} ...")
    params = {
        'parameters': 'T2M_MAX,T2M_MIN,PRECTOTCORR,ALLSKY_SFC_SW_DWN,RH2M,WS2M',
        'community': 'AG', 'longitude': LON, 'latitude': LAT,
        'start': f'{ano}0101', 'end': f'{ano}1231', 'format': 'JSON'
    }
    r = requests.get(API_URL, params=params, timeout=180); r.raise_for_status()
    p = r.json()['properties']['parameter']
    dates = pd.date_range(f'{ano}-01-01', f'{ano}-12-31')
    lines, mrows = [], []
    for d in dates:
        k    = d.strftime('%Y%m%d')
        tmax = p['T2M_MAX'].get(k, -999); tmin = p['T2M_MIN'].get(k, -999)
        prec = p['PRECTOTCORR'].get(k, -999); rs = p['ALLSKY_SFC_SW_DWN'].get(k, -999)
        rh   = p['RH2M'].get(k, -999)
        if tmax < -900 or tmin < -900 or rs < -900: continue
        if rh < -900: rh = 75.0
        prec = max(prec, 0.0)
        eto  = _eto(tmax, tmin, rs, rh, d.timetuple().tm_yday)
        lines.append(f"{d.day}\t{d.month}\t{d.year}\t{tmin:.2f}\t{tmax:.2f}\t{prec:.2f}\t{eto:.4f}")
        mrows.append({'date': d, 'tmax': tmax, 'tmin': tmin, 'prec': prec})
    with open(txt, 'w') as f:
        f.write(TXT_HEADER + '\n')
        for l in lines: f.write(l + '\n')
    pd.DataFrame(mrows).to_csv(meta, index=False)
    print(f"    Salvo: {len(lines)} dias")
    return str(txt)

# ============================================================================
# MODULO 2 — SIMULACAO AQUACROP
# ============================================================================
_COL = {}

def _detect(wf, cg):
    def f(cs, cols): return next((x for x in cs if x in cols), None)
    _COL['irr']   = f(['IrrDay','Irr','irr_day','IrrNet'],    set(wf.columns))
    _COL['wr']    = f(['Wr','Wr(1)','wr','th1','WrAct'],      set(wf.columns))
    _COL['tr']    = f(['Tr','TrAct','tr','Tact'],              set(wf.columns))
    _COL['dap']   = f(['DAP','dap','GrowingSeasonDay'],        set(cg.columns))
    _COL['zroot'] = f(['z_root','Zroot','zRoot','RootDepth','ZrAct','Zr'], set(cg.columns))
    miss = [n for n,v in [('irr',_COL['irr']),('wr',_COL['wr']),
                           ('tr',_COL['tr']),('dap',_COL['dap'])] if not v]
    if miss: raise RuntimeError(f"Colunas faltando: {miss}")


def _wr_tensao(wr_mm, z_m):
    th = np.clip(wr_mm / (1000.0 * max(z_m, 0.10)), THETA_PM, THETA_SAT)
    return float(np.clip(A_SAXTON * th**(-B_SAXTON), 1.0, 1500.0))


def simular_ano(ano, wdf):
    ss = JANELA_CFG['sim_start_tpl'].format(ano=ano)
    se = JANELA_CFG['sim_end_tpl'].format(ano=ano)
    irr_m = IrrigationManagement(
        irrigation_method=CENARIO_CFG['method'],
        SMT=CENARIO_CFG['SMT'],
        MaxIrr=CENARIO_CFG['MaxIrr'],
        MaxIrrSeason=CENARIO_CFG['MaxIrrSeason']
    )
    model = AquaCropModel(
        sim_start_time=ss, sim_end_time=se, weather_df=wdf,
        soil=Soil('SandyLoam'),
        crop=Crop('TomatoGDD', planting_date=JANELA_CFG['planting_date']),
        initial_water_content=InitialWaterContent(value=['FC']),
        irrigation_management=irr_m,
        field_management=FieldMngt(mulches=True, mulch_pct=80, f_mulch=0.3)
    )
    err = [None]
    def _run():
        try: model.run_model(till_termination=True)
        except Exception as e: err[0] = e
    t = threading.Thread(target=_run, daemon=True)
    t.start(); t.join(timeout=MAX_SIM_S)
    if t.is_alive(): raise TimeoutError("Timeout AquaCrop")
    if err[0]: raise err[0]

    wf, cg = model._outputs.water_flux, model._outputs.crop_growth
    if wf is None or len(wf) == 0: raise RuntimeError("Sem output")
    if not _COL: _detect(wf, cg)
    if not _COL.get('irr'): _detect(wf, cg)

    nr  = min(len(wf), len(cg))
    res = pd.DataFrame({
        'IrrDay': wf[_COL['irr']].values[:nr],
        'Wr':     wf[_COL['wr']].values[:nr],
        'Tr':     wf[_COL['tr']].values[:nr],
        'dap':    cg[_COL['dap']].values[:nr],
    })
    if _COL.get('zroot'):
        res['z_root'] = cg[_COL['zroot']].values[:nr]
    else:
        dv = res['dap'].values
        res['z_root'] = np.clip(0.3 + 0.7*dv/max(dv.max(), 1), 0.3, 1.0)

    sd  = pd.date_range(ss, periods=nr, freq='D')
    wi  = wdf.set_index('Date')
    res['precipitation'] = [
        float(wi.loc[pd.Timestamp(d), 'Precipitation'])
        if pd.Timestamp(d) in wi.index else 0.0 for d in sd
    ]
    res['tmax'] = [
        float(wi.loc[pd.Timestamp(d), 'MaxTemp'])
        if pd.Timestamp(d) in wi.index else np.nan for d in sd
    ]
    res['date'] = sd
    res['ano']  = ano
    return res


def processar_features(sim_df):
    df = sim_df.copy()
    df = df[(df['Tr'] > TR_MIN) & (df['dap'] >= DAP_MIN) & (df['dap'] <= DAP_MAX)].copy()
    if len(df) == 0: return df
    vals = df.apply(lambda r: _wr_tensao(r['Wr'], r['z_root']), axis=1).values
    # NOTA: o shift abaixo existe APENAS porque estamos usando AquaCrop,
    # que calcula o estado do solo no fim do dia (pos-transpiracao e irrigacao).
    # No campo real com sensor Arduino, a leitura e instantanea — NAO aplicar shift.
    df['tensao_raw']        = vals                                        # leitura bruta (sem shift) — usada pelo modulo de evolucao
    df['tensao_solo_kpa']   = np.concatenate([[vals[0]], vals[:-1]])      # shiftada (idem ao treino com AquaCrop) — usada pelo ensemble
    df['chuva_acum_3d_mm']  = df['precipitation'].rolling(3, min_periods=1).sum().values
    df['tmax_max_3d_c']     = df['tmax'].rolling(3, min_periods=1).max().values
    df['delta_tensao_kpa']  = df['tensao_solo_kpa'].diff().fillna(0).values
    def cls3(v):
        if v < IRR_MIN_MM:    return 0
        elif v < CLASSE_C1_MAX: return 1
        else:                   return 2
    df['classe_aquacrop'] = df['IrrDay'].apply(cls3)
    return df.reset_index(drop=True)

# ============================================================================
# MODULO 3 — LOOP PRINCIPAL (predicao pura, sem aprendizado online)
# ============================================================================
def loop_campo(modelos, df_todos):
    """
    Prediz a classe de irrigacao para cada dia usando o ensemble estatico.
    Sem retreino — modelo congelado como saiu do cold start.
    """
    from collections import Counter
    registros = []

    for _, row in df_todos.iterrows():
        features = [float(row[c]) for c in FEATURE_COLS]
        x        = np.array(features, dtype=float)

        votos_brutos = Counter(m.predict(x)[0] for m in modelos)
        classe_ens   = votos_brutos.most_common(1)[0][0]
        consenso_pct = votos_brutos[classe_ens] / len(modelos) * 100

        classe_acq = int(row['classe_aquacrop'])
        registros.append({
            'date':             str(row['date'])[:10],
            'ano':              int(row['ano']),
            'dap':              int(row['dap']),
            'tensao_solo_kpa':  round(float(row['tensao_solo_kpa']), 2),
            'delta_tensao_kpa': round(float(row['delta_tensao_kpa']), 3),
            'chuva_acum_3d_mm': round(float(row['chuva_acum_3d_mm']), 2),
            'tmax_max_3d_c':    round(float(row['tmax_max_3d_c']), 1),
            'IrrDay_mm':        round(float(row['IrrDay']), 2),
            'classe_aquacrop':  classe_acq,
            'nome_aquacrop':    NOMES_CLASSE[classe_acq],
            'classe_ensemble':  classe_ens,
            'nome_ensemble':    NOMES_CLASSE[classe_ens],
            'acertou':          int(classe_ens == classe_acq),
            'consenso_pct':     round(consenso_pct, 1),
            'votos_C0':         votos_brutos.get(0, 0),
            'votos_C1':         votos_brutos.get(1, 0),
            'votos_C2':         votos_brutos.get(2, 0),
        })

    return registros

# ============================================================================
# MODULO 4 — RELATORIO FINAL
# ============================================================================
def imprimir_relatorio(registros):
    sep = '=' * 65
    df  = pd.DataFrame(registros)

    print(f"\n{sep}")
    print("RELATORIO — Ensemble ALMMo-0 v12 | Acailandia 3 Anos (modelo estatico)")
    print(sep)

    # Acuracia geral e por ano
    print(f"\nACURACIA GERAL: {df['acertou'].mean():.3f} ({df['acertou'].sum()}/{len(df)} dias)")
    for ano in sorted(df['ano'].unique()):
        sub = df[df['ano'] == ano]
        print(f"  {ano}: {sub['acertou'].mean():.3f}  ({sub['acertou'].sum()}/{len(sub)} dias)")

    # Por classe
    print(f"\n{'─'*65}")
    print("DESEMPENHO POR CLASSE (AquaCrop = ground truth):")
    for c in [0, 1, 2]:
        sub = df[df['classe_aquacrop'] == c]
        if len(sub) == 0:
            print(f"  {NOMES_CLASSE[c]}: sem dias reais neste cenario")
            continue
        acertos = (sub['classe_ensemble'] == c).sum()
        print(f"  {NOMES_CLASSE[c]}: {acertos}/{len(sub)} acertos ({acertos/len(sub)*100:.1f}%)")

    # Matriz de confusao simples
    print(f"\n{'─'*65}")
    print("MATRIZ DE CONFUSAO (linhas=AquaCrop, colunas=Ensemble):")
    print(f"  {'':20s} {'C0':>8} {'C1':>8} {'C2':>8}")
    for c_real in [0, 1, 2]:
        sub = df[df['classe_aquacrop'] == c_real]
        if len(sub) == 0: continue
        row_str = f"  {NOMES_CLASSE[c_real]:20s}"
        for c_pred in [0, 1, 2]:
            n = (sub['classe_ensemble'] == c_pred).sum()
            row_str += f" {n:>8}"
        print(row_str)

    # Por faixa de DAP
    print(f"\n{'─'*65}")
    print("ACURACIA POR FAIXA DE DAP (todos os anos):")
    for d0, d1 in [(14, 30), (31, 55), (56, 80), (81, 107)]:
        sub = df[(df['dap'] >= d0) & (df['dap'] <= d1)]
        if len(sub) == 0: continue
        irr = (sub['classe_aquacrop'] > 0).sum()
        print(f"  DAP {d0:3d}-{d1:3d}: {len(sub):4d} dias | "
              f"acuracia={sub['acertou'].mean():.3f} | irrig_dias={irr}")

    # Analise dos dias de irrigacao
    print(f"\n{'─'*65}")
    print("DIAS COM IRRIGACAO REAL (AquaCrop C2):")
    df_c2 = df[df['classe_aquacrop'] == 2].copy()
    if len(df_c2) > 0:
        acertos_c2 = (df_c2['classe_ensemble'] == 2).sum()
        print(f"  Total: {len(df_c2)} dias | Detectados: {acertos_c2} | Perdidos: {len(df_c2)-acertos_c2}")
        print(f"\n  {'Data':<12} {'DAP':>4} {'Tensao':>8} {'Delta':>7} "
              f"{'IrrDay':>8} {'AquaCrop':>14} {'Ensemble':>14} {'OK':>4}")
        print(f"  {'-'*12} {'-'*4} {'-'*8} {'-'*7} {'-'*8} {'-'*14} {'-'*14} {'-'*4}")
        for _, r in df_c2.iterrows():
            ok = 'OK' if r['acertou'] else 'ERRO'
            print(f"  {r['date']:<12} {int(r['dap']):>4} {r['tensao_solo_kpa']:>8.1f} "
                  f"{r['delta_tensao_kpa']:>+7.2f} {r['IrrDay_mm']:>8.1f} "
                  f"{r['nome_aquacrop']:>14} {r['nome_ensemble']:>14} {ok:>4}")

    # Falsos negativos criticos
    fn = df[(df['classe_aquacrop'] == 2) & (df['classe_ensemble'] == 0)]
    print(f"\n{'─'*65}")
    print(f"FALSOS NEGATIVOS CRITICOS (AquaCrop=C2, Ensemble=C0): {len(fn)} dias")
    if len(fn) > 0:
        for _, r in fn.iterrows():
            print(f"  {r['date']} DAP={r['dap']} tensao={r['tensao_solo_kpa']:.1f} "
                  f"delta={r['delta_tensao_kpa']:+.2f}")

    print(f"\n{sep}")

# ============================================================================
# MAIN
# ============================================================================
def main():
    sep = '=' * 65
    t_total = time.time()
    print(sep)
    print(f"DEMO EVOLUCAO ONLINE — Acailandia-MA | Anos: {ANOS}")
    print(sep)

    # --- Carrega ensemble ---
    print(f"\n[1] Carregando ensemble: {PKL_PATH}")
    if not Path(PKL_PATH).exists():
        print(f"ERRO: {PKL_PATH} nao encontrado."); sys.exit(1)
    with open(PKL_PATH, 'rb') as f:
        raw = pickle.load(f)
    modelos = [ALMMo0.from_dict(d) for d in raw['models']]
    print(f"    {len(modelos)} modelos | regras totais iniciais: {sum(len(m.rules) for m in modelos)}")

    # --- Simula cada ano ---
    print(f"\n[2] Simulacoes AquaCrop — Acailandia ({CENARIO_NOME}/{JANELA_NOME})")
    frames = []
    for ano in ANOS:
        print(f"\n  Ano {ano}:")
        try:
            txt  = fetch_weather(ano)
            wdf  = prepare_weather(txt)
            sim  = simular_ano(ano, wdf)
            feat = processar_features(sim)
            if len(feat) == 0:
                print(f"    AVISO: nenhuma amostra valida para {ano}"); continue
            irr  = (feat['IrrDay'] >= IRR_MIN_MM).sum()
            print(f"    {len(feat)} dias validos | irrig_dias={irr} | "
                  f"C0={( feat['classe_aquacrop']==0).sum()} "
                  f"C1={(feat['classe_aquacrop']==1).sum()} "
                  f"C2={(feat['classe_aquacrop']==2).sum()}")
            frames.append(feat)
        except Exception as e:
            print(f"    ERRO {ano}: {e}")

    if not frames:
        print("ERRO: nenhuma simulacao bem-sucedida."); sys.exit(1)

    df_todos = pd.concat(frames, ignore_index=True)
    print(f"\n  Total concatenado: {len(df_todos)} dias ({len(frames)} anos)")

    # --- Loop de predicao ---
    print(f"\n[3] Predicoes do ensemble (modelo estatico, sem retreino)")
    t0 = time.time()
    registros = loop_campo(modelos, df_todos)
    print(f"    {len(registros)} dias processados em {time.time()-t0:.1f}s")

    # --- Salva CSV ---
    print(f"\n[4] Exportando CSV: {OUTPUT_CSV}")
    df_out = pd.DataFrame(registros)
    df_out.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
    print(f"    {len(df_out)} linhas | {len(df_out.columns)} colunas")
    print(f"    Colunas: {df_out.columns.tolist()}")

    # --- Relatorio ---
    imprimir_relatorio(registros)

    print(f"\nTempo total: {time.time()-t_total:.1f}s")
    print(f">>> CSV salvo: {OUTPUT_CSV} <<<")


if __name__ == '__main__':
    main()
