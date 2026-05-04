#!/usr/bin/env python3
"""
============================================================================
Validacao Externa — Ensemble ALMMo-0 v12 vs AquaCrop-OSPy
Cidade: Acailandia-MA  |  Ano: 2022  |  Cenario: smt_otimo / seca
============================================================================

Objetivo:
  Usar dados meteorologicos reais da NASA POWER para Acailandia (cidade proxima
  a Imperatriz, fora do conjunto de treino) para gerar uma simulacao AquaCrop
  "ground truth" e comparar contra as predicoes do ensemble ALMMo-0.

  Mapeamento de classes AquaCrop (4) -> Ensemble (3):
    AquaCrop C0 (IrrDay < 2mm)     -> Ensemble C0 (sem irrigacao)
    AquaCrop C1 (2-10mm)           -> Ensemble C1 (manutencao)
    AquaCrop C2 (10-30mm)          -> Ensemble C2 (intensiva)
    AquaCrop C3 (>= 30mm)          -> Ensemble C2 (intensiva — agrupado)

Saidas:
  - validacao_acailandia_2022.csv : comparacao dia a dia
  - validacao_acailandia_2022_report.txt : metricas de desempenho

Execucao:
    python validacao_acailandia.py

Requisitos: pip install aquacrop pandas numpy requests scikit-learn
"""

import os, sys, math, time, warnings, threading
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import requests
import pickle
from collections import Counter

os.environ['DEVELOPMENT'] = 'True'
try:
    from aquacrop import (AquaCropModel, Soil, Crop, InitialWaterContent,
                          IrrigationManagement, FieldMngt)
    from aquacrop.utils import prepare_weather
except ImportError:
    print("ERRO: pip install aquacrop"); sys.exit(1)

try:
    from sklearn.metrics import (classification_report, confusion_matrix,
                                  f1_score, accuracy_score)
    HAS_SKL = True
except ImportError:
    HAS_SKL = False
    print("AVISO: scikit-learn nao encontrado. Metricas basicas serao usadas.")

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACAO — Acailandia-MA
# ============================================================================
LAT, LON, ALTITUDE = -4.9484, -47.5025, 120   # Acailandia (~64km ao norte de Imperatriz)
ANO_VALIDACAO     = 2022                        # Fora do conjunto de treino (2001-2023 Imperatriz)
CENARIO_NOME      = 'smt_otimo'
JANELA_NOME       = 'seca'

API_URL     = "https://power.larc.nasa.gov/api/temporal/daily/point"
PKL_PATH    = "memoria_cold_start_v12_ensemble.pkl"
OUTPUT_DIR  = Path('.')
WEATHER_DIR = Path('weather_files')
WEATHER_DIR.mkdir(exist_ok=True)

# Parametros solo (SandyLoam — idem ao treino)
THETA_SAT, THETA_CC, THETA_PM = 0.3863, 0.1864, 0.0853
A_SAXTON, B_SAXTON             = 0.0090, 4.8825

# Janela de simulacao — plantio na seca (junho)
JANELA_CFG = {
    'planting_date': '06/01',
    'sim_start': f'{ANO_VALIDACAO}/05/15',
    'sim_end':   f'{ANO_VALIDACAO}/12/31',
}

# Cenario de irrigacao — SMT otimo (method 1)
CENARIO_CFG = {
    'method': 1, 'SMT': [60,60,70,50], 'MaxIrr': 100, 'MaxIrrSeason': 10000
}

# Filtros de DAP (idem ao treino)
DAP_MIN, DAP_MAX, TR_MIN = 14, 107, 0.1
MAX_SIM_SECONDS = 120

# Limiares de classe (AquaCrop 4 classes -> mapeado para 3 no ensemble)
IRR_MIN_MM    = 2.0
CLASSE_C1_MAX = 10.0
CLASSE_C2_MAX = 30.0

# Voto simples — modelo cru sem pesos assimetricos
# (cada modelo contribui 1 voto, classe com mais votos ganha)

NOMES_ENS = {0: 'C0-SemIrrig', 1: 'C1-Manutencao', 2: 'C2-Intensiva'}
NOMES_ACQ = {0: 'C0-Sem', 1: 'C1-Manut(2-10mm)', 2: 'C2-Supl(10-30mm)', 3: 'C3-Int(>=30mm)'}
TXT_HEADER = "Day\tMonth\tYear\tMinTemp\tMaxTemp\tPrecipitation\tReferenceET"

# ============================================================================
# MODULO 1: ALMMo-0 (reconstrucao do pkl)
# ============================================================================
class ALMMo0:
    def __init__(self, n_inputs=5, r_threshold=0.5, max_rules=50,
                 age_limit=100, epsilon=1e-8, n_classes=3,
                 min_rules_per_class=3):
        self.n_inputs=n_inputs; self.r_threshold=r_threshold
        self.max_rules=max_rules; self.age_limit=age_limit
        self.epsilon=epsilon; self.n_classes=n_classes
        self.min_rules_per_class=min_rules_per_class
        self.rules=[]; self.input_mean=np.zeros(n_inputs)
        self.input_std=np.ones(n_inputs); self.n_samples_seen=0

    def normalize(self, x):
        return (x - self.input_mean) / self.input_std

    def predict(self, x):
        x_n = self.normalize(x)
        d = np.array([np.sqrt(np.sum((x_n - np.array(r['center']))**2))
                      for r in self.rules])
        w = 1.0 / (d**2 + self.epsilon)
        v = np.zeros(self.n_classes)
        for i, r in enumerate(self.rules):
            v[r['consequent']] += w[i]
        return int(np.argmax(v)), v / v.sum()

    @classmethod
    def from_dict(cls, d):
        m = cls(n_inputs=d['n_inputs'], r_threshold=d['r_threshold'],
                max_rules=d['max_rules'], age_limit=d['age_limit'],
                n_classes=d.get('n_classes', 3),
                min_rules_per_class=d.get('min_rules_per_class', 3))
        m.rules         = d['rules']
        m.input_mean    = np.array(d['input_mean'])
        m.input_std     = np.array(d['input_std'])
        m.n_samples_seen = d.get('n_samples_seen', 0)
        return m


def carregar_ensemble(pkl_path):
    print(f"  Carregando ensemble: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        raw = pickle.load(f)
    modelos = [ALMMo0.from_dict(d) for d in raw['models']]
    print(f"  {len(modelos)} modelos carregados | Votacao: simples (sem pesos)")
    return modelos


def predizer_ensemble(modelos, x):
    """Voto simples — modelo cru, sem pesos."""
    votos_brutos = Counter(m.predict(x)[0] for m in modelos)
    classe_final = votos_brutos.most_common(1)[0][0]
    consenso_pct = votos_brutos[classe_final] / len(modelos) * 100
    return classe_final, dict(votos_brutos), consenso_pct

# ============================================================================
# MODULO 2: METEOROLOGIA
# ============================================================================
def calc_eto_fao56(tmax, tmin, rs, rh, doy, lat_deg=LAT, alt=ALTITUDE):
    tmean   = (tmax + tmin) / 2.0
    lat_rad = lat_deg * math.pi / 180.0
    P       = 101.3 * ((293.0 - 0.0065*alt) / 293.0)**5.26
    gamma   = 0.000665 * P
    e_tmax  = 0.6108 * math.exp(17.27*tmax  / (tmax  + 237.3))
    e_tmin  = 0.6108 * math.exp(17.27*tmin  / (tmin  + 237.3))
    es      = (e_tmax + e_tmin) / 2.0
    ea      = es * (rh / 100.0) if rh > 0 else e_tmin
    delta   = 4098.0 * (0.6108*math.exp(17.27*tmean/(tmean+237.3))) / (tmean+237.3)**2
    dr      = 1.0 + 0.033*math.cos(2*math.pi*doy/365)
    d_sol   = 0.409*math.sin(2*math.pi*doy/365 - 1.39)
    ws      = math.acos(-math.tan(lat_rad)*math.tan(d_sol))
    Ra      = (24*60/math.pi)*0.0820*dr*(ws*math.sin(lat_rad)*math.sin(d_sol)
              + math.cos(lat_rad)*math.cos(d_sol)*math.sin(ws))
    Rso     = (0.75 + 2e-5*alt)*Ra
    Rns     = 0.77 * rs
    sigma   = 4.903e-9
    rs_r    = min(rs/Rso, 1.0) if Rso > 0 else 0.5
    Rnl     = (sigma * ((tmax+273.16)**4 + (tmin+273.16)**4) / 2
               * (0.34 - 0.14*math.sqrt(max(ea, 0.01)))
               * (1.35*rs_r - 0.35))
    Rn      = Rns - Rnl
    u2      = 2.0
    num     = 0.408*delta*Rn + gamma*(900/(tmean+273))*u2*(es-ea)
    den     = delta + gamma*(1 + 0.34*u2)
    return max(num/den, 0.0)


def fetch_weather(year):
    txt_path  = WEATHER_DIR / f'weather_acailandia_{year}_full.txt'
    meta_path = WEATHER_DIR / f'weather_acailandia_{year}_full_meta.csv'

    if txt_path.exists():
        print(f"  Arquivo ja existe: {txt_path}")
        meta = pd.read_csv(meta_path) if meta_path.exists() else None
        return str(txt_path), meta

    print(f"  Buscando NASA POWER para Acailandia ({year})...")
    params = {
        'parameters': 'T2M_MAX,T2M_MIN,PRECTOTCORR,ALLSKY_SFC_SW_DWN,RH2M,WS2M',
        'community': 'AG',
        'longitude': LON, 'latitude': LAT,
        'start': f'{year}0101', 'end': f'{year}1231',
        'format': 'JSON'
    }
    resp = requests.get(API_URL, params=params, timeout=180)
    resp.raise_for_status()
    props  = resp.json()['properties']['parameter']
    dates  = pd.date_range(f'{year}-01-01', f'{year}-12-31')
    lines, meta_records = [], []

    for d in dates:
        key  = d.strftime('%Y%m%d')
        tmax = props['T2M_MAX'].get(key, -999)
        tmin = props['T2M_MIN'].get(key, -999)
        prec = props['PRECTOTCORR'].get(key, -999)
        rs   = props['ALLSKY_SFC_SW_DWN'].get(key, -999)
        rh   = props['RH2M'].get(key, -999)
        u2   = props['WS2M'].get(key, -999)

        if tmax < -900 or tmin < -900 or rs < -900:
            continue
        if rh < -900:
            rh = 75.0
        prec = max(prec, 0.0)

        doy = d.timetuple().tm_yday
        eto = calc_eto_fao56(tmax, tmin, rs, rh, doy)
        lines.append(f"{d.day}\t{d.month}\t{d.year}\t{tmin:.2f}\t{tmax:.2f}\t{prec:.2f}\t{eto:.4f}")
        meta_records.append({'date': d, 'tmax': tmax, 'tmin': tmin,
                              'prec': prec, 'rs': rs, 'rh': rh,
                              'u2_obs': u2 if u2 > -900 else np.nan, 'eto': eto})

    with open(txt_path, 'w') as f:
        f.write(TXT_HEADER + '\n')
        for l in lines:
            f.write(l + '\n')

    meta = pd.DataFrame(meta_records)
    meta.to_csv(meta_path, index=False)
    print(f"  Salvo: {txt_path} ({len(lines)} dias)")
    return str(txt_path), meta

# ============================================================================
# MODULO 3: CONVERSAO Wr -> TENSAO
# ============================================================================
def umidade_para_tensao_kpa(theta_vol):
    ts = np.clip(theta_vol, THETA_PM, THETA_SAT)
    return float(np.clip(A_SAXTON * (ts**(-B_SAXTON)), 1.0, 1500.0))


def wr_para_tensao_kpa(wr_mm, z_root_m):
    return umidade_para_tensao_kpa(wr_mm / (1000.0 * max(z_root_m, 0.10)))

# ============================================================================
# MODULO 4: SIMULACAO AQUACROP
# ============================================================================
COL_IRR = COL_WR = COL_TR = COL_DAP = COL_ZROOT = None


def detect_columns(wf, cg):
    global COL_IRR, COL_WR, COL_TR, COL_DAP, COL_ZROOT

    def find(candidates, cols):
        return next((x for x in candidates if x in cols), None)

    wc, gc = set(wf.columns), set(cg.columns)
    COL_IRR   = find(['IrrDay','Irr','irr_day','IrrNet'], wc)
    COL_WR    = find(['Wr','Wr(1)','wr','th1','WrAct'],  wc)
    COL_TR    = find(['Tr','TrAct','tr','Tact'],          wc)
    COL_DAP   = find(['DAP','dap','GrowingSeasonDay'],    gc)
    COL_ZROOT = find(['z_root','Zroot','zRoot','RootDepth','ZrAct','Zr'], gc)
    print(f"  Colunas detectadas: IRR={COL_IRR} WR={COL_WR} TR={COL_TR} DAP={COL_DAP} ZROOT={COL_ZROOT}")
    miss = [n for n, v in [('IRR',COL_IRR),('WR',COL_WR),('TR',COL_TR),('DAP',COL_DAP)] if not v]
    if miss:
        print(f"  ERRO: colunas faltando {miss}")
        return False
    return True


def run_simulation(wdf):
    global COL_IRR
    irr_mgmt = IrrigationManagement(
        irrigation_method=CENARIO_CFG['method'],
        SMT=CENARIO_CFG['SMT'],
        MaxIrr=CENARIO_CFG['MaxIrr'],
        MaxIrrSeason=CENARIO_CFG['MaxIrrSeason']
    )
    model = AquaCropModel(
        sim_start_time=JANELA_CFG['sim_start'],
        sim_end_time=JANELA_CFG['sim_end'],
        weather_df=wdf,
        soil=Soil('SandyLoam'),
        crop=Crop('TomatoGDD', planting_date=JANELA_CFG['planting_date']),
        initial_water_content=InitialWaterContent(value=['FC']),
        irrigation_management=irr_mgmt,
        field_management=FieldMngt(mulches=True, mulch_pct=80, f_mulch=0.3)
    )

    sim_error = [None]
    def _run():
        try:
            model.run_model(till_termination=True)
        except Exception as e:
            sim_error[0] = e

    t = threading.Thread(target=_run, daemon=True)
    t.start(); t.join(timeout=MAX_SIM_SECONDS)
    if t.is_alive():
        raise TimeoutError("Simulacao demorou mais de 120s")
    if sim_error[0]:
        raise sim_error[0]

    wf, cg = model._outputs.water_flux, model._outputs.crop_growth
    if wf is None or cg is None or len(wf) == 0:
        raise RuntimeError("AquaCrop nao retornou resultados")

    if COL_IRR is None:
        if not detect_columns(wf, cg):
            raise RuntimeError("Colunas invalidas")

    nr     = min(len(wf), len(cg))
    result = pd.DataFrame({
        'IrrDay': wf[COL_IRR].values[:nr],
        'Wr':     wf[COL_WR].values[:nr],
        'Tr':     wf[COL_TR].values[:nr],
        'dap':    cg[COL_DAP].values[:nr],
    })
    if COL_ZROOT:
        result['z_root'] = cg[COL_ZROOT].values[:nr]
    else:
        dv = result['dap'].values
        result['z_root'] = np.clip(0.3 + (0.7 * dv / max(dv.max(), 1)), 0.3, 1.0)

    sd = pd.date_range(JANELA_CFG['sim_start'], periods=nr, freq='D')
    wi = wdf.set_index('Date')
    result['precipitation'] = [
        float(wi.loc[pd.Timestamp(d), 'Precipitation'])
        if pd.Timestamp(d) in wi.index else 0.0
        for d in sd
    ]
    result['tmax'] = [
        float(wi.loc[pd.Timestamp(d), 'MaxTemp'])
        if pd.Timestamp(d) in wi.index else np.nan
        for d in sd
    ]
    result['date'] = sd
    return result

# ============================================================================
# MODULO 5: PROCESSAMENTO DE FEATURES
# ============================================================================
def processar_features(sim_df):
    df = sim_df.copy()
    # Filtros de qualidade (idem ao treino)
    df = df[(df['Tr'] > TR_MIN) & (df['dap'] >= DAP_MIN) & (df['dap'] <= DAP_MAX)].copy()

    if len(df) == 0:
        raise ValueError("Nenhuma amostra valida apos filtros DAP/Tr")

    # Tensao solo kPa a partir de Wr (agua no solo em mm)
    df['tensao_raw'] = df.apply(
        lambda r: wr_para_tensao_kpa(r['Wr'], r['z_root']), axis=1
    )

    # Tensao do DIA ANTERIOR (idem ao process_dataset do v11)
    vals = df['tensao_raw'].values
    df['tensao_solo_kpa'] = np.concatenate([[vals[0]], vals[:-1]])

    # Features temporais
    df['chuva_acum_3d_mm'] = df['precipitation'].rolling(3, min_periods=1).sum().values
    df['tmax_max_3d_c']    = df['tmax'].rolling(3, min_periods=1).max().values
    df['delta_tensao_kpa'] = df['tensao_solo_kpa'].diff().fillna(0).values

    return df.reset_index(drop=True)


def rotular_aquacrop(df):
    """
    Rotulagem AquaCrop (4 classes) e mapeamento para 3 classes do ensemble.

    AquaCrop:    C0 < 2mm | C1 [2,10) | C2 [10,30) | C3 >= 30mm
    Ensemble:    C0       | C1         | C2 (C2+C3 agrupados)
    """
    def cls4(v):
        if v < IRR_MIN_MM:    return 0
        elif v < CLASSE_C1_MAX: return 1
        elif v < CLASSE_C2_MAX: return 2
        else:                   return 3

    def cls3(v):
        if v < IRR_MIN_MM:    return 0
        elif v < CLASSE_C1_MAX: return 1
        else:                   return 2   # C2 + C3 -> Intensiva

    df['classe_aquacrop4'] = df['IrrDay'].apply(cls4)
    df['classe_aquacrop3'] = df['IrrDay'].apply(cls3)
    return df

# ============================================================================
# MODULO 6: PREDICAO ENSEMBLE
# ============================================================================
def predizer_todos(modelos, df):
    feature_cols = ['tensao_solo_kpa', 'chuva_acum_3d_mm', 'tmax_max_3d_c',
                    'dap', 'delta_tensao_kpa']
    classes_ens, consensos, votos_list = [], [], []

    for _, row in df.iterrows():
        x = np.array([row[c] for c in feature_cols], dtype=float)
        # Verifica NaN
        if np.any(np.isnan(x)):
            classes_ens.append(-1)
            consensos.append(0.0)
            votos_list.append({})
            continue
        cls, votos, consenso = predizer_ensemble(modelos, x)
        classes_ens.append(cls)
        consensos.append(round(consenso, 1))
        votos_list.append(votos)

    df = df.copy()
    df['classe_ensemble']  = classes_ens
    df['consenso_pct']     = consensos
    df['votos_ensemble']   = [str(v) for v in votos_list]
    return df

# ============================================================================
# MODULO 7: METRICAS E RELATORIO
# ============================================================================
def calcular_metricas_basicas(y_true, y_pred, classes):
    """Metricas sem scikit-learn."""
    acc = sum(yt == yp for yt, yp in zip(y_true, y_pred)) / len(y_true)
    report_lines = [f"Acuracia: {acc:.4f} ({acc*100:.1f}%)"]

    for c in classes:
        tp = sum((yt == c and yp == c) for yt, yp in zip(y_true, y_pred))
        fp = sum((yt != c and yp == c) for yt, yp in zip(y_true, y_pred))
        fn = sum((yt == c and yp != c) for yt, yp in zip(y_true, y_pred))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0.0
        sup  = sum(yt == c for yt in y_true)
        report_lines.append(f"  C{c} ({NOMES_ENS[c]}): Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f} sup={sup}")

    macro_f1 = np.mean([
        2*(tp/(tp+fp+1e-9))*(tp/(tp+fn+1e-9)) / ((tp/(tp+fp+1e-9))+(tp/(tp+fn+1e-9))+1e-9)
        for c in classes
        for tp, fp, fn in [(
            sum((yt==c and yp==c) for yt,yp in zip(y_true,y_pred)),
            sum((yt!=c and yp==c) for yt,yp in zip(y_true,y_pred)),
            sum((yt==c and yp!=c) for yt,yp in zip(y_true,y_pred))
        )]
    ])
    report_lines.append(f"F1 macro: {macro_f1:.4f}")
    return acc, '\n'.join(report_lines)


def gerar_relatorio(df_val, ano=ANO_VALIDACAO):
    # Remove linhas com predicao invalida (-1)
    df_ok = df_val[df_val['classe_ensemble'] >= 0].copy()

    y_true = df_ok['classe_aquacrop3'].values
    y_pred = df_ok['classe_ensemble'].values
    classes = sorted(set(y_true) | set(y_pred))

    linhas = []
    sep = '=' * 70

    linhas.append(sep)
    linhas.append(f"RELATORIO DE VALIDACAO — Acailandia-MA {ano}")
    linhas.append(f"Cenario: {CENARIO_NOME} / {JANELA_NOME}")
    linhas.append(f"Coordenadas: LAT={LAT} LON={LON}")
    linhas.append(f"Data de geracao: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    linhas.append(sep)

    # Distribuicao de classes AquaCrop
    linhas.append("\n[1] DISTRIBUICAO — AquaCrop (4 classes originais)")
    vc4 = pd.Series(df_ok['classe_aquacrop4'].values).value_counts().sort_index()
    for c in range(4):
        n = vc4.get(c, 0)
        pct = n / len(df_ok) * 100
        linhas.append(f"  {NOMES_ACQ.get(c,'C'+str(c))}: {n} dias ({pct:.1f}%)")

    linhas.append("\n[2] DISTRIBUICAO — AquaCrop (3 classes, mapeado para ensemble)")
    vc3 = pd.Series(y_true).value_counts().sort_index()
    for c in range(3):
        n = vc3.get(c, 0)
        pct = n / len(df_ok) * 100
        linhas.append(f"  {NOMES_ENS[c]}: {n} dias ({pct:.1f}%)")

    linhas.append("\n[3] DISTRIBUICAO — Ensemble (predicoes)")
    vce = pd.Series(y_pred).value_counts().sort_index()
    for c in range(3):
        n = vce.get(c, 0)
        pct = n / len(df_ok) * 100
        linhas.append(f"  {NOMES_ENS[c]}: {n} dias ({pct:.1f}%)")

    # Metricas
    linhas.append("\n[4] METRICAS DE DESEMPENHO")
    if HAS_SKL:
        acc   = accuracy_score(y_true, y_pred)
        f1mac = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1wei = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        linhas.append(f"  Acuracia:   {acc:.4f} ({acc*100:.1f}%)")
        linhas.append(f"  F1 macro:   {f1mac:.4f}")
        linhas.append(f"  F1 pesado:  {f1wei:.4f}")
        linhas.append("\n  Relatorio por classe:")
        cr = classification_report(y_true, y_pred,
                                    target_names=[NOMES_ENS[c] for c in sorted(NOMES_ENS)],
                                    zero_division=0)
        for l in cr.strip().split('\n'):
            linhas.append('    ' + l)

        linhas.append("\n[5] MATRIZ DE CONFUSAO (linhas=AquaCrop, colunas=Ensemble)")
        cm = confusion_matrix(y_true, y_pred, labels=list(range(3)))
        header = '         ' + '  '.join(f'{NOMES_ENS[c]:>14}' for c in range(3))
        linhas.append(header)
        for i, row in enumerate(cm):
            linhas.append(f"  {NOMES_ENS[i]:>12}  " + '  '.join(f'{v:>14}' for v in row))
    else:
        acc, metric_str = calcular_metricas_basicas(y_true, y_pred, classes)
        linhas.append(metric_str)

    # Analise por faixa de DAP
    linhas.append("\n[6] ACURACIA POR FAIXA DE DAP")
    for dap_ini, dap_fim in [(14,30),(31,55),(56,80),(81,107)]:
        sub = df_ok[(df_ok['dap'] >= dap_ini) & (df_ok['dap'] <= dap_fim)]
        if len(sub) > 0:
            a = (sub['classe_aquacrop3'] == sub['classe_ensemble']).mean()
            irr_dias = (sub['classe_aquacrop3'] > 0).sum()
            linhas.append(f"  DAP {dap_ini:3d}-{dap_fim:3d}: {len(sub):4d} dias | acuracia={a:.3f} | irrig_dias={irr_dias}")

    # Analise de falsos negativos (C2 real -> C0 predito = perigoso)
    fn_perigosos = df_ok[(df_ok['classe_aquacrop3'] == 2) & (df_ok['classe_ensemble'] == 0)]
    linhas.append(f"\n[7] FALSOS NEGATIVOS CRITICOS (AquaCrop=C2, Ensemble=C0)")
    linhas.append(f"  Total: {len(fn_perigosos)} dias ({len(fn_perigosos)/len(df_ok)*100:.1f}% do dataset)")
    if len(fn_perigosos) > 0:
        linhas.append(f"  Tensao media: {fn_perigosos['tensao_solo_kpa'].mean():.1f} kPa")
        linhas.append(f"  Delta medio:  {fn_perigosos['delta_tensao_kpa'].mean():.2f} kPa/dia")
        linhas.append(f"  IrrDay medio: {fn_perigosos['IrrDay'].mean():.1f} mm")
        linhas.append(f"  DAPs:         {sorted(fn_perigosos['dap'].unique().tolist())}")

    # Top 5 dias com maior irrigacao e predicao errada
    erros = df_ok[df_ok['classe_aquacrop3'] != df_ok['classe_ensemble']].copy()
    erros_irr = erros[erros['IrrDay'] >= IRR_MIN_MM].sort_values('IrrDay', ascending=False)
    linhas.append(f"\n[8] TOP ERROS EM DIAS DE IRRIGACAO REAL (IrrDay >= {IRR_MIN_MM}mm)")
    if len(erros_irr) > 0:
        linhas.append(f"  {'DAP':>4} {'IrrDay':>7} {'Tensao':>8} {'Delta':>7} {'AquaCrop':>15} {'Ensemble':>15}")
        for _, r in erros_irr.head(10).iterrows():
            linhas.append(f"  {int(r['dap']):>4} {r['IrrDay']:>7.1f} {r['tensao_solo_kpa']:>8.1f} "
                          f"{r['delta_tensao_kpa']:>7.2f} {NOMES_ENS[int(r['classe_aquacrop3'])]:>15} "
                          f"{NOMES_ENS[int(r['classe_ensemble'])]:>15}")
    else:
        linhas.append("  Nenhum erro em dias de irrigacao real!")

    linhas.append(f"\n{sep}")
    # Veredicto
    if HAS_SKL:
        f1_val = f1_score(y_true, y_pred, average='macro', zero_division=0)
        acc_val = accuracy_score(y_true, y_pred)
    else:
        acc_val, _ = calcular_metricas_basicas(y_true, y_pred, classes)
        f1_val = 0.0

    if acc_val >= 0.80 and f1_val >= 0.60:
        veredicto = "ENSEMBLE VALIDADO — desempenho satisfatorio em dados externos"
    elif acc_val >= 0.70:
        veredicto = "ENSEMBLE PARCIALMENTE VALIDADO — acuracia razoavel, F1 pode melhorar"
    else:
        veredicto = "ENSEMBLE REQUER REVISAO — desempenho insatisfatorio em dados externos"

    linhas.append(f"VEREDICTO: {veredicto}")
    linhas.append(sep)

    return '\n'.join(linhas), df_ok

# ============================================================================
# MAIN
# ============================================================================
def main():
    sep = '=' * 70
    print(sep)
    print(f"VALIDACAO EXTERNA — Ensemble ALMMo-0 v12")
    print(f"Cidade: Acailandia-MA (LAT={LAT}, LON={LON})")
    print(f"Ano: {ANO_VALIDACAO} | Cenario: {CENARIO_NOME} | Janela: {JANELA_NOME}")
    print(sep)

    # ------------------------------------------------------------------
    print(f"\n[ETAPA 1] Carregando ensemble")
    if not Path(PKL_PATH).exists():
        print(f"ERRO: {PKL_PATH} nao encontrado. Execute o script do projeto principal primeiro.")
        sys.exit(1)
    modelos = carregar_ensemble(PKL_PATH)

    # ------------------------------------------------------------------
    print(f"\n[ETAPA 2] Buscando meteorologia — Acailandia {ANO_VALIDACAO}")
    txt_path, meta = fetch_weather(ANO_VALIDACAO)

    # ------------------------------------------------------------------
    print(f"\n[ETAPA 3] Preparando dados para AquaCrop")
    wdf = prepare_weather(txt_path)
    print(f"  Weather DataFrame: {len(wdf)} linhas")

    # ------------------------------------------------------------------
    print(f"\n[ETAPA 4] Rodando simulacao AquaCrop ({CENARIO_NOME}/{JANELA_NOME} {ANO_VALIDACAO})")
    t0 = time.time()
    sim_df = run_simulation(wdf)
    dt = time.time() - t0
    print(f"  Simulacao concluida em {dt:.1f}s | {len(sim_df)} linhas brutas")
    irr_dias = (sim_df['IrrDay'] >= IRR_MIN_MM).sum()
    print(f"  Dias com irrigacao >= {IRR_MIN_MM}mm: {irr_dias} | Total irrigado: {sim_df['IrrDay'].sum():.0f}mm")

    # ------------------------------------------------------------------
    print(f"\n[ETAPA 5] Processando features")
    df_feat = processar_features(sim_df)
    print(f"  Amostras pos-filtro (DAP {DAP_MIN}-{DAP_MAX}, Tr>{TR_MIN}): {len(df_feat)}")

    # Estatisticas basicas
    for col in ['tensao_solo_kpa','delta_tensao_kpa','chuva_acum_3d_mm','tmax_max_3d_c']:
        s = df_feat[col]
        print(f"  {col}: min={s.min():.2f} max={s.max():.2f} med={s.median():.2f} std={s.std():.2f}")

    # ------------------------------------------------------------------
    print(f"\n[ETAPA 6] Rotulando classes AquaCrop")
    df_rot = rotular_aquacrop(df_feat)
    vc3 = df_rot['classe_aquacrop3'].value_counts().sort_index()
    for c, nome in NOMES_ENS.items():
        n = vc3.get(c, 0)
        print(f"  {nome}: {n} dias ({n/len(df_rot)*100:.1f}%)")

    # ------------------------------------------------------------------
    print(f"\n[ETAPA 7] Predicoes do ensemble ({len(modelos)} modelos)")
    t0 = time.time()
    df_val = predizer_todos(modelos, df_rot)
    dt = time.time() - t0
    print(f"  {len(df_val)} predicoes em {dt:.1f}s")

    vce = df_val['classe_ensemble'].value_counts().sort_index()
    for c, nome in NOMES_ENS.items():
        n = vce.get(c, 0)
        print(f"  {nome}: {n} predicoes ({n/len(df_val)*100:.1f}%)")

    # ------------------------------------------------------------------
    print(f"\n[ETAPA 8] Gerando relatorio e exportando")
    relatorio, df_final = gerar_relatorio(df_val, ANO_VALIDACAO)
    print(relatorio)

    # CSV dia a dia
    cols_csv = [
        'date', 'dap',
        'tensao_solo_kpa', 'delta_tensao_kpa', 'chuva_acum_3d_mm', 'tmax_max_3d_c',
        'IrrDay', 'Wr', 'z_root',
        'classe_aquacrop4', 'classe_aquacrop3', 'classe_ensemble',
        'consenso_pct', 'votos_ensemble'
    ]
    cols_existentes = [c for c in cols_csv if c in df_final.columns]
    df_final[cols_existentes].to_csv(
        OUTPUT_DIR / 'validacao_acailandia_2022.csv', index=False
    )
    print(f"\n  validacao_acailandia_2022.csv ({len(df_final)} linhas)")

    # Relatorio texto
    with open(OUTPUT_DIR / 'validacao_acailandia_2022_report.txt', 'w', encoding='utf-8') as f:
        f.write(relatorio)
    print(f"  validacao_acailandia_2022_report.txt")

    print(f"\n>>> Validacao concluida. <<<")


if __name__ == '__main__':
    main()
