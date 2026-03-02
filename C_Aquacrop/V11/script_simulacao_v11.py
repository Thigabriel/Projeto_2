#!/usr/bin/env python3
"""
============================================================================
Script de Simulacao AquaCrop-OSPy — Dataset v11 (Revisado)
Projeto ALMMo-0 — Irrigacao Inteligente de Tomate | Imperatriz-MA
============================================================================

Mudancas v11-revisado vs v11-original:
  1. Limiar minimo de irrigacao: IrrDay < 2mm = ruido numerico -> C0
  2. Rotulagem em 4 classes agronomicas (era 3 com mediana):
     C0: sem irrigacao (< 2mm)
     C1: manutencao [2, 10) mm
     C2: suplementar [10, 30) mm
     C3: intensiva >= 30mm
  3. Intervalo method 2: 3 -> 5 dias (tensao std era 2.6 com 3)
  4. Geracao de 6 graficos diagnosticos (matplotlib)

Simulacoes: 23x4 (chuva) + 23x3 (seca) = 161

Requisitos: pip install aquacrop pandas numpy requests matplotlib
"""

import os, sys, math, time, warnings, threading
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import requests

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("AVISO: matplotlib nao encontrado. Graficos pulados.")

os.environ['DEVELOPMENT'] = 'True'
try:
    from aquacrop import (AquaCropModel, Soil, Crop, InitialWaterContent,
                          IrrigationManagement, FieldMngt)
    from aquacrop.utils import prepare_weather
except ImportError:
    print("ERRO: pip install aquacrop"); sys.exit(1)

warnings.filterwarnings('ignore')

# ============================================================================
# CONSTANTES
# ============================================================================
LAT, LON, ALTITUDE = -5.5253, -47.4825, 120
API_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
THETA_SAT, THETA_CC, THETA_PM = 0.3863, 0.1864, 0.0853
A_SAXTON, B_SAXTON = 0.0090, 4.8825
ANOS = list(range(2001, 2024))

JANELAS = {
    'chuva': {'planting_date':'01/05','sim_start_fmt':'{year}/01/01','sim_end_fmt':'{year}/07/31'},
    'seca':  {'planting_date':'06/01','sim_start_fmt':'{year}/05/15','sim_end_fmt':'{year}/12/31'},
}

CENARIOS = {
    'smt_otimo':  {'method':1,'SMT':[60,60,70,50],'MaxIrr':100,'MaxIrrSeason':10000,
                   'descricao':'SMT otimo (method 1) — irriga sob demanda'},
    'manutencao': {'method':4,'NetIrrSMT':70,
                   'descricao':'Manutencao (method 4) — net irrigation diaria, alvo 70% TAW'},
    'intervalo':  {'method':2,'IrrInterval':5,
                   'descricao':'Intervalo fixo (method 2) — a cada 5 dias'},
    'veranico':   {'method':1,'SMT':[60,60,70,50],'MaxIrr':100,'MaxIrrSeason':10000,
                   'descricao':'Veranico (method 1 + precip fev-mar x 0.20)'},
}
CENARIOS_CHUVA = ['smt_otimo','manutencao','intervalo','veranico']
CENARIOS_SECA  = ['smt_otimo','manutencao','intervalo']

VERANICO_FATOR, VERANICO_MESES = 0.20, [2, 3]
DAP_MIN, DAP_MAX, TR_MIN = 14, 107, 0.1
MAX_SIM_SECONDS = 120

# Limiar minimo de irrigacao real (mm)
IRR_MIN_MM = 2.0
# Limiares 4 classes agronomicas
CLASSE_C1_MAX = 10.0
CLASSE_C2_MAX = 30.0
NOMES_CLASSES = {0:'Sem irrigacao',1:'Manutencao',2:'Suplementar',3:'Intensiva'}

OUTPUT_DIR = Path('.')
WEATHER_DIR = Path('weather_files')
WEATHER_DIR.mkdir(exist_ok=True)
TXT_HEADER = "Day\tMonth\tYear\tMinTemp\tMaxTemp\tPrecipitation\tReferenceET"

# ============================================================================
# MODULO 1: METEOROLOGIA
# ============================================================================
def calc_eto_fao56(tmax, tmin, rs, rh, doy, lat_deg=LAT, alt=ALTITUDE):
    tmean = (tmax+tmin)/2.0
    lat_rad = lat_deg*math.pi/180.0
    P = 101.3*((293.0-0.0065*alt)/293.0)**5.26
    gamma = 0.000665*P
    e_tmax = 0.6108*math.exp(17.27*tmax/(tmax+237.3))
    e_tmin = 0.6108*math.exp(17.27*tmin/(tmin+237.3))
    es = (e_tmax+e_tmin)/2.0
    ea = es*(rh/100.0) if rh > 0 else e_tmin
    delta = 4098.0*(0.6108*math.exp(17.27*tmean/(tmean+237.3)))/(tmean+237.3)**2
    dr = 1.0+0.033*math.cos(2*math.pi*doy/365)
    d_sol = 0.409*math.sin(2*math.pi*doy/365-1.39)
    ws = math.acos(-math.tan(lat_rad)*math.tan(d_sol))
    Ra = (24*60/math.pi)*0.0820*dr*(ws*math.sin(lat_rad)*math.sin(d_sol)+
         math.cos(lat_rad)*math.cos(d_sol)*math.sin(ws))
    Rso = (0.75+2e-5*alt)*Ra
    Rns = 0.77*rs
    sigma = 4.903e-9
    rs_r = min(rs/Rso,1.0) if Rso > 0 else 0.5
    Rnl = sigma*((tmax+273.16)**4+(tmin+273.16)**4)/2*(0.34-0.14*math.sqrt(max(ea,0.01)))*(1.35*rs_r-0.35)
    Rn = Rns-Rnl
    u2 = 2.0
    num = 0.408*delta*Rn+gamma*(900/(tmean+273))*u2*(es-ea)
    den = delta+gamma*(1+0.34*u2)
    return max(num/den, 0.0)

def fetch_and_save_weather(year):
    txt_path = WEATHER_DIR/f'weather_imperatriz_{year}_full.txt'
    meta_path = WEATHER_DIR/f'weather_imperatriz_{year}_full_meta.csv'
    if txt_path.exists():
        meta = pd.read_csv(meta_path) if meta_path.exists() else None
        return str(txt_path), meta
    print(f"    Buscando NASA POWER {year}...")
    params = {'parameters':'T2M_MAX,T2M_MIN,PRECTOTCORR,ALLSKY_SFC_SW_DWN,RH2M,WS2M',
              'community':'AG','longitude':LON,'latitude':LAT,
              'start':f'{year}0101','end':f'{year}1231','format':'JSON'}
    resp = requests.get(API_URL, params=params, timeout=180); resp.raise_for_status()
    props = resp.json()['properties']['parameter']
    dates = pd.date_range(f'{year}-01-01',f'{year}-12-31')
    lines, meta_records = [], []
    for d in dates:
        key = d.strftime('%Y%m%d')
        tmax=props['T2M_MAX'].get(key,-999); tmin=props['T2M_MIN'].get(key,-999)
        prec=props['PRECTOTCORR'].get(key,-999); rs=props['ALLSKY_SFC_SW_DWN'].get(key,-999)
        rh=props['RH2M'].get(key,-999); u2_obs=props['WS2M'].get(key,-999)
        if tmax<-900 or tmin<-900 or rs<-900: continue
        if rh<-900: rh=75.0
        prec=max(prec,0.0)
        doy=d.timetuple().tm_yday; eto=calc_eto_fao56(tmax,tmin,rs,rh,doy)
        lines.append(f"{d.day}\t{d.month}\t{d.year}\t{tmin:.2f}\t{tmax:.2f}\t{prec:.2f}\t{eto:.4f}")
        meta_records.append({'date':d,'tmax':tmax,'tmin':tmin,'prec':prec,'rs':rs,'rh':rh,
                             'u2_obs':u2_obs if u2_obs>-900 else np.nan,'eto':eto})
    with open(txt_path,'w') as f:
        f.write(TXT_HEADER+'\n')
        for l in lines: f.write(l+'\n')
    meta = pd.DataFrame(meta_records); meta.to_csv(meta_path, index=False)
    print(f"    Salvo: {txt_path} ({len(lines)} dias)")
    return str(txt_path), meta

def aplicar_veranico(wdf, year):
    wdf_ver = wdf.copy()
    mask = (wdf_ver['Date'].dt.month.isin(VERANICO_MESES))&(wdf_ver['Date'].dt.year==year)
    pa = wdf_ver.loc[mask,'Precipitation'].sum()
    wdf_ver.loc[mask,'Precipitation'] *= VERANICO_FATOR
    pd2 = wdf_ver.loc[mask,'Precipitation'].sum()
    print(f"      Veranico {year}: fev-mar {pa:.0f}mm -> {pd2:.0f}mm ({pd2/pa:.2f}x)" if pa>0 else f"      Veranico {year}: sem chuva fev-mar")
    return wdf_ver

# ============================================================================
# MODULO 2: CONVERSAO Wr -> TENSAO
# ============================================================================
def umidade_para_tensao_kpa(theta_vol):
    ts = np.clip(theta_vol, THETA_PM, THETA_SAT)
    return float(np.clip(A_SAXTON*(ts**(-B_SAXTON)), 1.0, 1500.0))

def wr_para_tensao_kpa_dinamica(wr_mm, z_root_m):
    return umidade_para_tensao_kpa(wr_mm/(1000.0*max(z_root_m, 0.10)))

# ============================================================================
# MODULO 3: SIMULACAO AQUACROP
# ============================================================================
COL_IRR=COL_WR=COL_TR=COL_DAP=COL_ZROOT=None

def detect_columns(wf, cg):
    global COL_IRR,COL_WR,COL_TR,COL_DAP,COL_ZROOT
    def find(c,cols): return next((x for x in c if x in cols),None)
    wc,gc = set(wf.columns),set(cg.columns)
    COL_IRR=find(['IrrDay','Irr','irr_day','IrrNet'],wc)
    COL_WR=find(['Wr','Wr(1)','wr','th1','WrAct'],wc)
    COL_TR=find(['Tr','TrAct','tr','Tact'],wc)
    COL_DAP=find(['DAP','dap','GrowingSeasonDay'],gc)
    COL_ZROOT=find(['z_root','Zroot','zRoot','RootDepth','ZrAct','Zr'],gc)
    print(f"    Colunas: IRR={COL_IRR}, WR={COL_WR}, TR={COL_TR}, DAP={COL_DAP}, ZROOT={COL_ZROOT}")
    miss=[n for n,v in [('IRR',COL_IRR),('WR',COL_WR),('TR',COL_TR),('DAP',COL_DAP)] if not v]
    if miss: print(f"    ERRO: {miss}"); return False
    return True

def build_irr_management(cn):
    c=CENARIOS[cn]; m=c['method']
    if m==1: return IrrigationManagement(irrigation_method=1,SMT=c['SMT'],MaxIrr=c['MaxIrr'],MaxIrrSeason=c['MaxIrrSeason'])
    if m==4: return IrrigationManagement(irrigation_method=4,NetIrrSMT=c['NetIrrSMT'])
    if m==2: return IrrigationManagement(irrigation_method=2,IrrInterval=c['IrrInterval'])
    raise ValueError(f"Method {m}")

def run_single_simulation(year, cenario_nome, janela, wdf_to_use):
    global COL_IRR
    jcfg=JANELAS[janela]
    ss=jcfg['sim_start_fmt'].format(year=year); se=jcfg['sim_end_fmt'].format(year=year)
    model = AquaCropModel(sim_start_time=ss, sim_end_time=se, weather_df=wdf_to_use,
        soil=Soil('SandyLoam'), crop=Crop('TomatoGDD',planting_date=jcfg['planting_date']),
        initial_water_content=InitialWaterContent(value=['FC']),
        irrigation_management=build_irr_management(cenario_nome),
        field_management=FieldMngt(mulches=True, mulch_pct=80, f_mulch=0.3))

    sim_error=[None]
    def _run():
        try: model.run_model(till_termination=True)
        except Exception as e: sim_error[0]=e
    t=threading.Thread(target=_run,daemon=True); t.start(); t.join(timeout=MAX_SIM_SECONDS)
    if t.is_alive(): raise TimeoutError(f"Timeout method {CENARIOS[cenario_nome]['method']}")
    if sim_error[0]: raise sim_error[0]

    wf,cg = model._outputs.water_flux, model._outputs.crop_growth
    if wf is None or cg is None or len(wf)==0: raise RuntimeError("Sem resultados")
    if COL_IRR is None:
        if not detect_columns(wf,cg): raise RuntimeError("Colunas!")

    nr=min(len(wf),len(cg))
    result = pd.DataFrame({'IrrDay':wf[COL_IRR].values[:nr],'Wr':wf[COL_WR].values[:nr],
                           'Tr':wf[COL_TR].values[:nr],'dap':cg[COL_DAP].values[:nr]})
    if COL_ZROOT: result['z_root']=cg[COL_ZROOT].values[:nr]
    else:
        dv=result['dap'].values; result['z_root']=np.clip(0.3+(0.7*dv/max(dv.max(),1)),0.3,1.0)

    sd=pd.date_range(ss,periods=nr,freq='D'); wi=wdf_to_use.set_index('Date')
    result['precipitation']=[float(wi.loc[pd.Timestamp(d),'Precipitation']) if pd.Timestamp(d) in wi.index else 0.0 for d in sd]
    result['tmax']=[float(wi.loc[pd.Timestamp(d),'MaxTemp']) if pd.Timestamp(d) in wi.index else np.nan for d in sd]
    result['date']=sd; result['year']=year; result['cenario']=cenario_nome; result['janela']=janela
    return result

def run_all_simulations():
    all_results,weather_metas,failed=[],{},[]
    gc,sc=0,0; t0=time.time()
    total=len(ANOS)*(len(CENARIOS_CHUVA)+len(CENARIOS_SECA))
    for year in ANOS:
        print(f"\n{'='*60}\nANO: {year} ({ANOS.index(year)+1}/{len(ANOS)})\n{'='*60}")
        tp,meta=fetch_and_save_weather(year); weather_metas[year]=meta
        wb=prepare_weather(tp); wv=aplicar_veranico(wb,year)
        for janela,clist in [('chuva',CENARIOS_CHUVA),('seca',CENARIOS_SECA)]:
            for cen in clist:
                wdf=wv if (cen=='veranico' and janela=='chuva') else wb
                gc+=1; sc+=1; m=CENARIOS[cen]['method']; lb=f"{janela}/{cen}(m{m})"
                try:
                    r=run_single_simulation(year,cen,janela,wdf); r['grupo_id']=gc
                    ir=(r['IrrDay']>=IRR_MIN_MM).sum(); it=r['IrrDay'].sum()
                    im=r.loc[r['IrrDay']>=IRR_MIN_MM,'IrrDay'].mean() if ir>0 else 0
                    print(f"    [{sc:3d}/{total}] {lb}: {ir} dias, {it:.0f}mm, {im:.1f}mm/ev")
                    all_results.append(r)
                except Exception as e:
                    failed.append(f"{year}/{janela}/{cen}: {e}")
                    print(f"    [{sc:3d}/{total}] {lb}: ERRO - {e}")
    el=time.time()-t0
    print(f"\nTempo: {el/60:.1f}min | OK: {len(all_results)}/{total} | Falhas: {len(failed)}")
    for f in failed: print(f"  - {f}")
    return all_results, weather_metas

# ============================================================================
# MODULO 4: PROCESSAMENTO
# ============================================================================
def process_dataset(all_results):
    frames=[]
    for sim_df in all_results:
        df=sim_df.copy()
        df=df[(df['Tr']>TR_MIN)&(df['dap']>=DAP_MIN)&(df['dap']<=DAP_MAX)].copy()
        if len(df)==0: continue
        df['tensao_raw']=df.apply(lambda r: wr_para_tensao_kpa_dinamica(r['Wr'],r['z_root']),axis=1)
        vals=df['tensao_raw'].values
        df['tensao_solo_kpa']=np.concatenate([[vals[0]],vals[:-1]])
        df['chuva_acum_3d_mm']=df['precipitation'].rolling(3,min_periods=1).sum().values
        df['tmax_max_3d_c']=df['tmax'].rolling(3,min_periods=1).max().values
        df['delta_tensao_kpa']=df['tensao_solo_kpa'].diff().fillna(0).values
        frames.append(df)
    if not frames: raise ValueError("Nenhuma simulacao!")
    out=pd.concat(frames,ignore_index=True)
    print("\n  Resumo por cenario:")
    for cen in CENARIOS:
        s=out[out['cenario']==cen]
        if len(s)>0:
            ir=(s['IrrDay']>=IRR_MIN_MM).sum()
            print(f"  {cen:15s}: {len(s):5d} amostras, {ir:4d} irrig reais, tensao std={s['tensao_solo_kpa'].std():.1f}")
    return out

def rotular_classes(df):
    def cls(v):
        if v<IRR_MIN_MM: return 0
        elif v<CLASSE_C1_MAX: return 1
        elif v<CLASSE_C2_MAX: return 2
        else: return 3
    df['classe_irrigacao']=df['IrrDay'].apply(cls)
    ruido=((df['IrrDay']>0)&(df['IrrDay']<IRR_MIN_MM)).sum()
    print(f"\n  Limiar: {IRR_MIN_MM}mm | Ruido->C0: {ruido} amostras")
    print(f"  C0: <{IRR_MIN_MM}mm | C1: [{IRR_MIN_MM},{CLASSE_C1_MAX}) | C2: [{CLASSE_C1_MAX},{CLASSE_C2_MAX}) | C3: >={CLASSE_C2_MAX}mm")
    for cen in CENARIOS:
        for jan in ['chuva','seca']:
            s=df[(df['cenario']==cen)&(df['janela']==jan)]
            if len(s)>0:
                vc=s['classe_irrigacao'].value_counts().sort_index()
                print(f"  {cen}/{jan} (n={len(s)}): "+', '.join(f"C{c}={vc.get(c,0)}" for c in range(4)))
    return df

def build_final_dataset(df):
    ct=['tensao_solo_kpa','chuva_acum_3d_mm','tmax_max_3d_c','dap','delta_tensao_kpa','classe_irrigacao']
    cf=ct+['IrrDay','year','janela','cenario','grupo_id']
    df_full=df[cf].dropna(subset=['tensao_solo_kpa']).reset_index(drop=True)
    df_full['classe_irrigacao']=df_full['classe_irrigacao'].astype(int)
    df_full['grupo_id']=df_full['grupo_id'].astype(int)
    return df_full[ct].copy(), df_full

# ============================================================================
# MODULO 5: GRAFICOS
# ============================================================================
CC={0:'#2196F3',1:'#4CAF50',2:'#FF9800',3:'#F44336'}
LC={0:'C0: Sem irrigacao',1:'C1: Manutencao\n(2-10mm)',2:'C2: Suplementar\n(10-30mm)',3:'C3: Intensiva\n(>30mm)'}

def gerar_graficos(df_treino, df_full):
    if not HAS_MPL: print("  matplotlib indisponivel."); return
    print("  Gerando graficos...")
    plt.rcParams.update({'figure.facecolor':'white','axes.facecolor':'#FAFAFA',
                         'axes.grid':True,'grid.alpha':0.3,'font.size':10})
    cls=[0,1,2,3]
    vc=df_treino['classe_irrigacao'].value_counts().sort_index()
    vcp=df_treino['classe_irrigacao'].value_counts(normalize=True).sort_index()*100
    vals=[vc.get(c,0) for c in cls]; pcts=[vcp.get(c,0) for c in cls]

    # FIG 1: Distribuicao + evolucao
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    ax=axes[0]
    bars=ax.bar([f'C{c}' for c in cls],vals,color=[CC[c] for c in cls],edgecolor='white',linewidth=1.5)
    for b,p,v in zip(bars,pcts,vals):
        ax.text(b.get_x()+b.get_width()/2,b.get_height()+max(vals)*0.02,
                f'{p:.1f}%\n({v})',ha='center',va='bottom',fontsize=9,fontweight='bold')
    ax.set_title('Distribuicao de Classes — v11',fontsize=13,fontweight='bold'); ax.set_ylabel('Amostras')

    ax2=axes[1]; x=np.arange(4); w=0.25
    ax2.bar(x-w,[94.1,4.1,1.8,0],w,label='v7',color='#90CAF9',edgecolor='white')
    ax2.bar(x,[94.9,3.4,1.7,0],w,label='v10',color='#A5D6A7',edgecolor='white')
    ax2.bar(x+w,pcts,w,label='v11',color='#FFCC80',edgecolor='white')
    ax2.set_xticks(x); ax2.set_xticklabels(['C0','C1','C2','C3'])
    ax2.set_ylabel('%'); ax2.set_title('Evolucao v7 vs v10 vs v11',fontsize=13,fontweight='bold'); ax2.legend()
    plt.tight_layout(); fig.savefig(OUTPUT_DIR/'fig1_distribuicao_classes.png',dpi=150,bbox_inches='tight'); plt.close()
    print("    fig1_distribuicao_classes.png")

    # FIG 2: Contribuicao por metodo
    fig,ax=plt.subplots(figsize=(12,6))
    co=['smt_otimo','manutencao','intervalo','veranico']
    bottom=np.zeros(4)
    for c in cls:
        vc2=[(df_full[df_full['cenario']==cn]['classe_irrigacao']==c).sum() for cn in co]
        ax.bar(range(4),vc2,bottom=bottom,label=LC[c],color=CC[c],edgecolor='white',linewidth=0.5)
        for i,(v,b) in enumerate(zip(vc2,bottom)):
            if v>50:
                tot=len(df_full[df_full['cenario']==co[i]])
                ax.text(i,b+v/2,f'{v/tot*100:.0f}%',ha='center',va='center',fontsize=8,color='white',fontweight='bold')
        bottom+=vc2
    ax.set_xticks(range(4)); ax.set_xticklabels([f"{c}\n(method {CENARIOS[c]['method']})" for c in co],fontsize=9)
    ax.set_ylabel('Amostras'); ax.set_title('Contribuicao por Metodo',fontsize=13,fontweight='bold'); ax.legend(loc='upper right',fontsize=9)
    plt.tight_layout(); fig.savefig(OUTPUT_DIR/'fig2_metodos_por_classe.png',dpi=150,bbox_inches='tight'); plt.close()
    print("    fig2_metodos_por_classe.png")

    # FIG 3: Boxplots
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    for idx,(col,tit) in enumerate([('tensao_solo_kpa','Tensao do Solo por Classe'),
                                     ('delta_tensao_kpa','Delta Tensao por Classe')]):
        ax=axes[idx]
        data=[df_treino[df_treino['classe_irrigacao']==c][col].values for c in cls]
        bp=ax.boxplot(data,labels=[f'C{c}' for c in cls],patch_artist=True,showfliers=False,
                      medianprops=dict(color='black',linewidth=2))
        for p,c in zip(bp['boxes'],cls): p.set_facecolor(CC[c]); p.set_alpha(0.7)
        ax.set_ylabel(col); ax.set_title(tit,fontsize=13,fontweight='bold')
        if 'delta' in col: ax.axhline(y=0,color='gray',linestyle='--',alpha=0.5)
    plt.tight_layout(); fig.savefig(OUTPUT_DIR/'fig3_tensao_por_classe.png',dpi=150,bbox_inches='tight'); plt.close()
    print("    fig3_tensao_por_classe.png")

    # FIG 4: Histograma doses
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    ax=axes[0]
    ip=df_full[df_full['IrrDay']>=IRR_MIN_MM]['IrrDay']
    ax.hist(ip,bins=50,color='#5C6BC0',edgecolor='white',alpha=0.8)
    for lim,cor,lb in [(IRR_MIN_MM,'red',f'Min={IRR_MIN_MM}mm'),(CLASSE_C1_MAX,'green',f'C1/C2={CLASSE_C1_MAX}mm'),
                        (CLASSE_C2_MAX,'orange',f'C2/C3={CLASSE_C2_MAX}mm')]:
        ax.axvline(x=lim,color=cor,linestyle='--',linewidth=1.5,label=lb)
    ax.set_xlabel('Dose (mm)'); ax.set_ylabel('Freq'); ax.set_title('Doses >= 2mm',fontsize=13,fontweight='bold'); ax.legend(fontsize=9)
    ax2=axes[1]
    cm={'smt_otimo':'#E53935','manutencao':'#43A047','intervalo':'#1E88E5','veranico':'#FDD835'}
    for cn in co:
        s=df_full[(df_full['cenario']==cn)&(df_full['IrrDay']>=IRR_MIN_MM)]
        if len(s)>0: ax2.hist(s['IrrDay'],bins=30,alpha=0.5,label=cn,color=cm[cn],edgecolor='none')
    ax2.set_xlabel('Dose (mm)'); ax2.set_ylabel('Freq'); ax2.set_title('Doses por Metodo',fontsize=13,fontweight='bold'); ax2.legend(fontsize=9)
    plt.tight_layout(); fig.savefig(OUTPUT_DIR/'fig4_histograma_doses.png',dpi=150,bbox_inches='tight'); plt.close()
    print("    fig4_histograma_doses.png")

    # FIG 5: Scatter tensao vs chuva
    fig,ax=plt.subplots(figsize=(10,7))
    for c in cls:
        s=df_treino[df_treino['classe_irrigacao']==c]
        a,sz=(0.1,5) if c==0 else (0.6,20)
        ax.scatter(s['tensao_solo_kpa'],s['chuva_acum_3d_mm'],c=CC[c],s=sz,alpha=a,label=LC[c],edgecolors='none')
    ax.set_xlabel('Tensao (kPa)',fontsize=12); ax.set_ylabel('Chuva 3d (mm)',fontsize=12)
    ax.set_title('Tensao vs Chuva por Classe',fontsize=13,fontweight='bold'); ax.legend(fontsize=9,markerscale=3)
    plt.tight_layout(); fig.savefig(OUTPUT_DIR/'fig5_scatter_tensao_chuva.png',dpi=150,bbox_inches='tight'); plt.close()
    print("    fig5_scatter_tensao_chuva.png")

    # FIG 6: Perfil temporal
    fig,axes=plt.subplots(3,1,figsize=(14,10),sharex=True)
    ano_ex=2005
    for idx,cen in enumerate(['smt_otimo','manutencao','intervalo']):
        ax=axes[idx]
        sub=df_full[(df_full['cenario']==cen)&(df_full['janela']=='seca')&(df_full['year']==ano_ex)]
        if len(sub)==0:
            ad=df_full[(df_full['cenario']==cen)&(df_full['janela']=='seca')]['year'].unique()
            if len(ad)>0: sub=df_full[(df_full['cenario']==cen)&(df_full['janela']=='seca')&(df_full['year']==int(ad[len(ad)//2]))]
        if len(sub)==0: continue
        daps,tens,irr=sub['dap'].values,sub['tensao_solo_kpa'].values,sub['IrrDay'].values
        ax.plot(daps,tens,color='#5C6BC0',linewidth=1.5)
        ax.set_ylabel('Tensao (kPa)',color='#5C6BC0')
        ax2=ax.twinx()
        cb=[('#E0E0E0' if v<IRR_MIN_MM else CC[1] if v<CLASSE_C1_MAX else CC[2] if v<CLASSE_C2_MAX else CC[3]) for v in irr]
        ax2.bar(daps,irr,color=cb,alpha=0.7,width=0.8); ax2.set_ylabel('Irrigacao (mm)',color='#F44336')
        ax.set_title(f'{cen} (method {CENARIOS[cen]["method"]}) — seca {ano_ex}',fontsize=11,fontweight='bold')
    axes[-1].set_xlabel('DAP')
    plt.tight_layout(); fig.savefig(OUTPUT_DIR/'fig6_perfil_temporal.png',dpi=150,bbox_inches='tight'); plt.close()
    print("    fig6_perfil_temporal.png")
    print("  6 graficos gerados.")

# ============================================================================
# MODULO 6: VALIDACAO E RELATORIO
# ============================================================================
def validate_and_report(df_treino, df_full, weather_metas):
    r=[]
    r.append("# Relatorio de Validacao — dataset_cold_start_v11.csv\n")
    r.append(f"**Data:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    r.append(f"**Amostras:** {len(df_treino)} | **Grupos:** {df_full['grupo_id'].nunique()}")
    r.append(f"**Periodo:** {min(ANOS)}-{max(ANOS)} ({len(ANOS)} anos)")
    r.append(f"**Limiar ruido:** {IRR_MIN_MM}mm | **Classes:** C0/<{IRR_MIN_MM}, C1/[{IRR_MIN_MM},{CLASSE_C1_MAX}), C2/[{CLASSE_C1_MAX},{CLASSE_C2_MAX}), C3/>={CLASSE_C2_MAX}mm")
    r.append(f"**Features:** tensao_solo_kpa, chuva_acum_3d_mm, tmax_max_3d_c, dap, delta_tensao_kpa\n")

    r.append("## Cenarios\n")
    for cn,cfg in CENARIOS.items(): r.append(f"- **{cn}** (method {cfg['method']}): {cfg['descricao']}")

    r.append("\n## Distribuicao de Classes\n")
    vc=df_treino['classe_irrigacao'].value_counts().sort_index()
    vcp=df_treino['classe_irrigacao'].value_counts(normalize=True).sort_index()*100
    for c in range(4): r.append(f"- {NOMES_CLASSES[c]} (C{c}): {vc.get(c,0)} ({vcp.get(c,0):.1f}%)")
    r.append(f"\n**v7:** C0=94.1%, C1=4.1%, C2=1.8% | **v10:** C0=94.9%, C1=3.4%, C2=1.7%")

    r.append("\n### Por Cenario x Janela\n")
    for cn in CENARIOS:
        for jan in ['chuva','seca']:
            s=df_full[(df_full['cenario']==cn)&(df_full['janela']==jan)]
            if len(s)>0:
                vc2=s['classe_irrigacao'].value_counts().sort_index()
                im=s.loc[s['IrrDay']>=IRR_MIN_MM,'IrrDay'].mean() if (s['IrrDay']>=IRR_MIN_MM).any() else 0
                r.append(f"- {cn}/{jan} (n={len(s)}): "+', '.join(f"C{c}={vc2.get(c,0)}" for c in range(4))+f" | dose={im:.1f}mm")

    r.append("\n### Por Metodo\n")
    for cn in CENARIOS:
        s=df_full[df_full['cenario']==cn]
        if len(s)>0:
            parts=[f"C{c}={(s['classe_irrigacao']==c).mean()*100:.1f}%" for c in range(4)]
            r.append(f"- {cn} (m{CENARIOS[cn]['method']}): {', '.join(parts)} | tensao std={s['tensao_solo_kpa'].std():.1f}")

    fc=['tensao_solo_kpa','chuva_acum_3d_mm','tmax_max_3d_c','dap','delta_tensao_kpa']
    r.append("\n## Features\n")
    for col in fc:
        s=df_treino[col]; r.append(f"- {col}: min={s.min():.2f}, max={s.max():.2f}, med={s.median():.2f}, std={s.std():.2f}")

    r.append("\n## Correlacoes\n")
    corrs={}
    for col in fc:
        co=df_treino[col].corr(df_treino['classe_irrigacao'])
        corrs[col]=co if not np.isnan(co) else 0; r.append(f"- {col}: {corrs[col]:.4f}")

    r.append("\n## Tensao por Classe\n")
    tcls={}
    for c in range(4):
        s=df_treino[df_treino['classe_irrigacao']==c]
        if len(s)>0:
            tcls[c]=s['tensao_solo_kpa'].median()
            r.append(f"- C{c} ({NOMES_CLASSES[c]}): {tcls[c]:.1f} kPa (n={len(s)}, std={s['tensao_solo_kpa'].std():.1f})")

    r.append("\n## Criterios\n")
    res={}
    c0p=vcp.get(0,0); res['C0>=5%']=c0p>=5; r.append(f"- C0>=5%: {'PASS' if res['C0>=5%'] else 'FAIL'} ({c0p:.1f}%)")
    c3p=vcp.get(3,0); res['C3>=0.5%']=c3p>=0.5; r.append(f"- C3>=0.5%: {'PASS' if res['C3>=0.5%'] else 'FAIL'} ({c3p:.1f}%)")
    ct=corrs.get('tensao_solo_kpa',0); res['corr_t']=0.10<=ct<=0.7; r.append(f"- Corr tensao [0.10,0.70]: {'PASS' if res['corr_t'] else 'FAIL'} ({ct:.4f})")
    cc=abs(corrs.get('chuva_acum_3d_mm',0)); res['corr_c']=cc>=0.05; r.append(f"- |Corr chuva|>=0.05: {'PASS' if res['corr_c'] else 'FAIL'} ({cc:.4f})")
    d0=int((df_treino['chuva_acum_3d_mm']>0).sum()); res['chuva']=d0>=500; r.append(f"- Dias chuva>0>=500: {'PASS' if res['chuva'] else 'FAIL'} ({d0})")
    to=all(tcls.get(c,0)<tcls.get(c+1,999) for c in range(3) if c+1 in tcls)
    res['t_ord']=to; r.append(f"- Tensao C0<C1<C2<C3: {'PASS' if to else 'FAIL'} ({', '.join(f'C{c}={tcls.get(c,0):.1f}' for c in range(4) if c in tcls)})")
    nn=df_treino.isna().sum().sum(); res['nan']=nn==0; r.append(f"- Sem NaN: {'PASS' if res['nan'] else 'FAIL'} ({nn})")
    tm=df_treino['tensao_solo_kpa'].max(); res['t<1500']=tm<1500; r.append(f"- Tensao<1500: {'PASS' if res['t<1500'] else 'FAIL'} ({tm:.1f})")
    ts=df_treino['tensao_solo_kpa'].std(); res['t_std']=ts>=8; r.append(f"- Std tensao>=8: {'PASS' if res['t_std'] else 'FAIL'} ({ts:.1f})")
    sc2=df_full[(df_full['cenario']=='smt_otimo')&(df_full['janela']=='chuva')]
    vc2=df_full[(df_full['cenario']=='veranico')&(df_full['janela']=='chuva')]
    vok=(vc2['classe_irrigacao']>0).mean()>(sc2['classe_irrigacao']>0).mean() if len(sc2)>0 and len(vc2)>0 else False
    res['ver']=vok; r.append(f"- Veranico>smt: {'PASS' if vok else 'FAIL'}")
    mi=(df_full[df_full['cenario']=='manutencao']['classe_irrigacao']>0).mean()
    si=(df_full[df_full['cenario']=='smt_otimo']['classe_irrigacao']>0).mean()
    dok=mi>si; res['div']=dok; r.append(f"- Diversidade: {'PASS' if dok else 'FAIL'} (manut={mi:.1%} vs smt={si:.1%})")

    np2=sum(res.values()); nt=len(res)
    r.append(f"\n## Veredicto\n\n**{'APROVADO' if all(res.values()) else 'REPROVADO'}** — {np2}/{nt} criterios OK")
    return '\n'.join(r), all(res.values()), res

# ============================================================================
# MAIN
# ============================================================================
def main():
    total=len(ANOS)*(len(CENARIOS_CHUVA)+len(CENARIOS_SECA))
    print("="*70)
    print("AquaCrop-OSPy — Dataset v11 (Revisado)")
    print(f"Periodo: {min(ANOS)}-{max(ANOS)} | {total} simulacoes")
    print(f"Metodos: 1(SMT), 4(net irr), 2(intervalo 5d)")
    print(f"Classes: C0/<{IRR_MIN_MM}mm, C1/[{IRR_MIN_MM},{CLASSE_C1_MAX}), C2/[{CLASSE_C1_MAX},{CLASSE_C2_MAX}), C3/>={CLASSE_C2_MAX}mm")
    print("="*70)

    print("\n[ETAPA 1] Simulacoes")
    all_results, weather_metas = run_all_simulations()
    if not all_results: print("ERRO!"); sys.exit(1)
    nc=sum(1 for r in all_results if r['janela'].iloc[0]=='chuva')
    ns=sum(1 for r in all_results if r['janela'].iloc[0]=='seca')
    print(f"\nOK: {len(all_results)} ({nc} chuva + {ns} seca)")

    print("\n[ETAPA 2] Processamento")
    processed = process_dataset(all_results)
    print(f"  Total: {len(processed)} amostras")

    print("\n[ETAPA 3] Rotulagem (4 classes)")
    processed = rotular_classes(processed)

    print("\n[ETAPA 4] Dataset Final")
    df_treino, df_full = build_final_dataset(processed)
    print(f"  Treino: {len(df_treino)} | Full: {len(df_full)} | Grupos: {df_full['grupo_id'].nunique()}")

    print("\n[ETAPA 5] Graficos")
    gerar_graficos(df_treino, df_full)

    print("\n[ETAPA 6] Validacao")
    report, aprovado, results = validate_and_report(df_treino, df_full, weather_metas)
    print("\n" + report)

    print("\n[ETAPA 7] Exportacao")
    df_treino.to_csv(OUTPUT_DIR/'dataset_cold_start_v11.csv', index=False)
    print(f"  dataset_cold_start_v11.csv ({len(df_treino)} linhas)")
    df_full.to_csv(OUTPUT_DIR/'dataset_cold_start_v11_full.csv', index=False)
    print(f"  dataset_cold_start_v11_full.csv ({len(df_full)} linhas)")
    with open(OUTPUT_DIR/'relatorio_v11.md','w',encoding='utf-8') as f: f.write(report)
    print(f"  relatorio_v11.md")

    print(f"\n>>> DATASET v11 {'APROVADO' if aprovado else 'REPROVADO'} <<<")
    return df_treino

if __name__ == '__main__':
    main()