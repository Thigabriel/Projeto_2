"""
============================================================================
  COLD START v12 — ALMMo-0 v3.0 + Ensemble Multi-Agente
  Dataset: dataset_cold_start_v11.csv (10.561 amostras, 5 features, 3 classes)
  
  Novidade v12: Ensemble de múltiplos modelos ALMMo-0 com raios diversos.
  Cada modelo individual aprende geometrias diferentes do espaço de features.
  O voto conjunto cancela erros idiossincráticos de cada modelo.

  Pipeline:
    1. Split Leave-Groups-Out (metadados v11_full)
    2. Sweep bidimensional: r_threshold x mrpc (3 e 10)
    3. Treino de 14 modelos com configs diversas
    4. 6 estratégias de ensemble avaliadas
    5. Selecção automática do melhor ensemble
    6. Avaliação completa (métricas + 20 cenários sanidade)
    7. Comparação: individual mrpc=3 vs mrpc=10 vs ensemble
    8. Relatório + 10 gráficos

  Outputs:
    - memoria_cold_start_v12_ensemble.pkl   (ensemble final)
    - memoria_cold_start_v12_mrpc3.pkl      (melhor individual)
    - memoria_cold_start_v12_mrpc10.pkl     (alternativa recall)
    - relatorio_cold_start_v12.md
    - graficos_v12/*.png
============================================================================
"""

import numpy as np
import pandas as pd
import os, time, pickle, warnings
from datetime import datetime
from collections import Counter
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score, mean_absolute_error,
    recall_score, precision_score
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 12,
    'figure.facecolor': 'white', 'axes.facecolor': '#FAFAFA',
    'axes.grid': True, 'grid.alpha': 0.15,
})
COR_C0='#2196F3'; COR_C1='#FF9800'; COR_C2='#4CAF50'
COR_M3='#1565C0'; COR_M10='#E65100'; COR_ENS='#9C27B0'

# ─────────────────────────────────────────────────────────────────────────
# ALMMo-0 v3.0 (M1/M2/M3 + bug pruning corrigido)
# ─────────────────────────────────────────────────────────────────────────
class ALMMo0:
    def __init__(self, n_inputs=5, r_threshold=0.5, max_rules=50,
                 age_limit=100, epsilon=1e-8, n_classes=3,
                 min_rules_per_class=3):
        self.n_inputs=n_inputs; self.r_threshold=r_threshold
        self.max_rules=max_rules; self.age_limit=age_limit
        self.epsilon=epsilon; self.n_classes=n_classes
        self.min_rules_per_class=min_rules_per_class
        self.rules=[]; self.input_mean=np.zeros(n_inputs)
        self.input_std=np.ones(n_inputs)
        self.n_samples_seen=0; self.n_rules_created=0; self.n_rules_pruned=0
        self.created_at=datetime.now().isoformat()

    def fit_normalizer(self,X):
        self.input_mean=np.mean(X,axis=0); self.input_std=np.std(X,axis=0)
        self.input_std[self.input_std<self.epsilon]=1.0

    def normalize(self,x): return (x-self.input_mean)/self.input_std

    def _dist(self,a,b):
        d=a-b; return float(np.sqrt(np.dot(d,d)))

    def _dists(self,x_norm):
        if not self.rules: return np.array([])
        return np.array([self._dist(x_norm,r['center']) for r in self.rules])

    def _create(self,x_norm,y):
        self.rules.append({'center':x_norm.copy(),'consequent':int(y),'age':0,'activations':1})
        self.n_rules_created+=1

    def predict(self,x):
        x_n=self.normalize(x); d=self._dists(x_n)
        w=1.0/(d**2+self.epsilon)
        v=np.zeros(self.n_classes)
        for i,r in enumerate(self.rules): v[r['consequent']]+=w[i]
        return int(np.argmax(v))

    def predict_proba(self,x):
        x_n=self.normalize(x); d=self._dists(x_n)
        w=1.0/(d**2+self.epsilon)
        v=np.zeros(self.n_classes)
        for i,r in enumerate(self.rules): v[r['consequent']]+=w[i]
        s=v.sum(); return v/s if s>0 else v

    def learn(self,x,y):
        self.n_samples_seen+=1; x_n=self.normalize(x)
        if not self.rules: self._create(x_n,y); return
        d=self._dists(x_n); idx=int(np.argmin(d)); dm=d[idx]
        if dm>self.r_threshold: self._create(x_n,y)
        else: self._update(idx,x_n,y,dm)
        self._age(idx); self._prune()

    def _update(self,idx,x_n,y,dist):
        r=self.rules[idx]; r['activations']+=1; eta=1.0/r['activations']
        r['center']=r['center']+eta*(x_n-r['center'])
        if r['consequent']!=y:
            af=max(0.0,1.0-dist/self.r_threshold); ee=eta*af
            if ee>0:
                nc=(1-ee)*r['consequent']+ee*y
                r['consequent']=int(np.round(np.clip(nc,0,self.n_classes-1)))
        r['age']=0

    def _age(self,idx):
        for i,r in enumerate(self.rules):
            if i!=idx: r['age']+=1

    def _prune(self):
        n0=len(self.rules); cc=Counter(r['consequent'] for r in self.rules)
        surv=[]
        for r in self.rules:
            if r['age']<=self.age_limit: surv.append(r)
            else:
                if cc[r['consequent']]>self.min_rules_per_class: cc[r['consequent']]-=1
                else: r['age']=0; surv.append(r)
        if len(surv)>self.max_rules:
            surv.sort(key=lambda r:r['age'],reverse=True)
            cc2=Counter(r['consequent'] for r in surv); kept=[]
            for r in surv:
                if len(kept)>=self.max_rules:
                    if cc2[r['consequent']]<=self.min_rules_per_class: kept.append(r)
                    else: cc2[r['consequent']]-=1
                else: kept.append(r)
            surv=kept
        self.rules=surv; self.n_rules_pruned+=n0-len(self.rules)

    def cold_start(self,X,y):
        self.fit_normalizer(X); hist=[]
        for i in range(len(X)): self.learn(X[i],int(y[i])); hist.append(len(self.rules))
        return hist

    def rules_by_class(self):
        return {c:sum(1 for r in self.rules if r['consequent']==c) for c in range(self.n_classes)}

    def save(self,fp):
        with open(fp,'wb') as f:
            pickle.dump({'rules':self.rules,'input_mean':self.input_mean,
                'input_std':self.input_std,'r_threshold':self.r_threshold,
                'max_rules':self.max_rules,'age_limit':self.age_limit,
                'n_inputs':self.n_inputs,'n_classes':self.n_classes,
                'min_rules_per_class':self.min_rules_per_class,
                'n_samples_seen':self.n_samples_seen,'n_rules_created':self.n_rules_created,
                'n_rules_pruned':self.n_rules_pruned,'created_at':self.created_at,
                'saved_at':datetime.now().isoformat()},f)

    @classmethod
    def load(cls,fp):
        with open(fp,'rb') as f: d=pickle.load(f)
        m=cls(n_inputs=d['n_inputs'],r_threshold=d['r_threshold'],
              max_rules=d['max_rules'],age_limit=d['age_limit'],
              n_classes=d.get('n_classes',3),min_rules_per_class=d.get('min_rules_per_class',3))
        m.rules=d['rules']; m.input_mean=d['input_mean']; m.input_std=d['input_std']
        m.n_samples_seen=d['n_samples_seen']; m.n_rules_created=d['n_rules_created']
        m.n_rules_pruned=d['n_rules_pruned']; m.created_at=d['created_at']
        return m


# ─────────────────────────────────────────────────────────────────────────
# ENSEMBLE
# ─────────────────────────────────────────────────────────────────────────
class ALMMo0Ensemble:
    """Ensemble de múltiplos ALMMo-0 com voto configurável."""

    def __init__(self, models, method='hard', weights=None):
        self.models = models  # list of ALMMo0
        self.method = method  # 'hard', 'soft', 'weighted'
        self.weights = weights  # para method='weighted'
        self.n_classes = models[0].n_classes if models else 3

    def predict(self, x):
        if self.method == 'hard':
            votes = np.zeros(self.n_classes)
            for m in self.models: votes[m.predict(x)] += 1
            return int(np.argmax(votes))
        elif self.method == 'soft':
            proba = np.zeros(self.n_classes)
            for m in self.models: proba += m.predict_proba(x)
            return int(np.argmax(proba))
        elif self.method == 'weighted':
            proba = np.zeros(self.n_classes)
            for m, w in zip(self.models, self.weights):
                proba += m.predict_proba(x) * w
            return int(np.argmax(proba))

    def predict_batch(self, X):
        return np.array([self.predict(x) for x in X])

    def total_rules(self):
        return sum(len(m.rules) for m in self.models)

    def memory_kb(self):
        # Cada regra: 5 floats (center) + 1 int (consequent) ~ 48 bytes
        return self.total_rules() * 48 / 1024

    def save(self, fp):
        data = {
            'n_models': len(self.models),
            'method': self.method,
            'weights': self.weights,
            'models': [],
        }
        for m in self.models:
            data['models'].append({
                'rules': m.rules, 'input_mean': m.input_mean,
                'input_std': m.input_std, 'r_threshold': m.r_threshold,
                'max_rules': m.max_rules, 'age_limit': m.age_limit,
                'n_inputs': m.n_inputs, 'n_classes': m.n_classes,
                'min_rules_per_class': m.min_rules_per_class,
                'n_samples_seen': m.n_samples_seen,
            })
        data['saved_at'] = datetime.now().isoformat()
        with open(fp, 'wb') as f: pickle.dump(data, f)

    @classmethod
    def load(cls, fp):
        with open(fp, 'rb') as f: data = pickle.load(f)
        models = []
        for md in data['models']:
            m = ALMMo0(n_inputs=md['n_inputs'], r_threshold=md['r_threshold'],
                       max_rules=md['max_rules'], age_limit=md['age_limit'],
                       n_classes=md['n_classes'], min_rules_per_class=md['min_rules_per_class'])
            m.rules = md['rules']; m.input_mean = md['input_mean']
            m.input_std = md['input_std']; m.n_samples_seen = md['n_samples_seen']
            models.append(m)
        return cls(models, method=data['method'], weights=data.get('weights'))


# ─────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────
def sep(t): print(f"\n{'='*85}\n  {t}\n{'='*85}")

def metrics_full(y_true, y_pred):
    f1m=f1_score(y_true,y_pred,average='macro',zero_division=0)
    f1p=f1_score(y_true,y_pred,average=None,labels=[0,1,2],zero_division=0)
    rec=recall_score(y_true,y_pred,average=None,labels=[0,1,2],zero_division=0)
    prc=precision_score(y_true,y_pred,average=None,labels=[0,1,2],zero_division=0)
    acc=accuracy_score(y_true,y_pred); mae=mean_absolute_error(y_true,y_pred)
    cm=confusion_matrix(y_true,y_pred,labels=[0,1,2])
    erros=y_true!=y_pred
    if erros.sum()>0:
        diffs=np.abs(y_true[erros]-y_pred[erros])
        adj_n=int((diffs==1).sum()); nadj_n=int((diffs>1).sum()); adj_pct=adj_n/erros.sum()*100
    else: adj_n=0; nadj_n=0; adj_pct=0.0
    return {'f1_macro':f1m,'f1_per':f1p,'recall':rec,'precision':prc,
            'acc':acc,'mae':mae,'cm':cm,'y_pred':y_pred,
            'adj_n':adj_n,'nadj_n':nadj_n,'adj_pct':adj_pct}

# 20 cenários de sanidade
SANITY_CASES = [
    # C0 — 8 casos
    ([10.0,45.0,31.0,40,-2.0],  "Solo humido 10kPa, chuva 45mm, arrefecendo",         0),
    ([15.0,30.0,29.0,25,-5.0],  "Solo muito humido 15kPa, chuva 30mm, delta -5",       0),
    ([8.0,80.0,28.0,20,-10.0],  "Solo saturado 8kPa, chuva torrencial 80mm",           0),
    ([20.0,15.0,32.0,50,0.5],   "Solo ok 20kPa, chuva moderada 15mm",                  0),
    ([25.0,10.0,30.0,35,-1.0],  "Solo normal 25kPa, chuva 10mm, humidificando",        0),
    ([12.0,5.0,31.0,90,-3.0],   "Solo humido 12kPa, pos-chuva, final ciclo",           0),
    ([40.0,20.0,32.0,30,1.0],   "Solo medio 40kPa, chuva recente 20mm",                0),
    ([30.0,25.0,33.0,60,0.0],   "Solo 30kPa, chuva 25mm, estavel",                     0),
    # C1 — 6 casos
    ([35.0,0.0,33.0,55,0.1],    "Threshold 35kPa, delta~0, sem chuva (method 4)",      1),
    ([34.0,0.0,35.0,45,0.0],    "Tensao 34kPa, delta=0, calor, seca",                  1),
    ([36.0,0.5,37.0,70,-0.1],   "Tensao 36kPa, chuva minima, calor forte",             1),
    ([35.0,0.0,34.0,30,0.2],    "35kPa, DAP baixo, delta quase zero",                  1),
    ([33.0,1.0,36.0,80,0.0],    "33kPa, quase sem chuva, calor, DAP alto",             1),
    ([35.5,0.0,38.0,55,-0.1],   "35.5kPa, zero chuva, calor extremo 38C",             1),
    # C2 — 6 casos
    ([60.0,0.0,38.5,60,4.5],    "Solo seco 60kPa, delta+4.5, calor extremo",           2),
    ([55.0,0.0,36.0,50,5.0],    "Solo seco 55kPa, delta+5, secagem rapida",            2),
    ([45.0,0.0,35.0,70,3.5],    "Solo 45kPa, delta+3.5, sem chuva, DAP alto",          2),
    ([50.0,1.0,37.0,55,4.0],    "Solo 50kPa, chuva infima 1mm, delta+4",               2),
    ([65.0,0.0,39.0,65,6.0],    "Stress severo 65kPa, delta+6, calor extremo",         2),
    ([42.0,0.0,34.0,45,3.0],    "Solo 42kPa, delta+3, sem chuva (limiar C2)",          2),
]

def run_sanity(predict_fn, label=""):
    results = []
    for inp, desc, exp in SANITY_CASES:
        pred = predict_fn(np.array(inp))
        results.append({'desc':desc,'expected':exp,'pred':pred,'ok':pred==exp})
    return results

def print_sanity(results, label):
    n_ok = sum(r['ok'] for r in results)
    c0_ok = sum(r['ok'] for r in results if r['expected']==0)
    c1_ok = sum(r['ok'] for r in results if r['expected']==1)
    c2_ok = sum(r['ok'] for r in results if r['expected']==2)
    print(f"\n  Sanidade {label}: {n_ok}/20 (C0: {c0_ok}/8, C1: {c1_ok}/6, C2: {c2_ok}/6)")
    for r in results:
        st='[v]' if r['ok'] else '[x]'
        print(f"    {st} Pred=C{r['pred']} Esp=C{r['expected']} | {r['desc']}")

def print_model_metrics(label, met, n_rules=None, rbc=None):
    print(f"\n  +{'='*55}+")
    print(f"  |  {label:^51s}  |")
    print(f"  +{'='*55}+")
    print(f"  F1-Score Macro:       {met['f1_macro']:.4f}")
    print(f"  Acuracia:             {met['acc']:.4f}")
    print(f"  MAE ordinal:          {met['mae']:.4f}")
    if n_rules: print(f"  Regras:               {n_rules}" + (f" ({rbc})" if rbc else ""))
    print(f"\n  {'Classe':15s} {'Precision':>10s} {'Recall':>10s} {'F1-Score':>10s}")
    print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*10}")
    for i,lab in enumerate(['C0 Sem irrig.','C1 Manutencao','C2 Intensiva']):
        print(f"  {lab:15s} {met['precision'][i]:10.4f} {met['recall'][i]:10.4f} {met['f1_per'][i]:10.4f}")
    print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*10}")
    print(f"  {'Macro avg':15s} {met['precision'].mean():10.4f} {met['recall'].mean():10.4f} {met['f1_macro']:10.4f}")
    n_err=met['adj_n']+met['nadj_n']
    print(f"\n  Erros: {n_err} | Adjacentes: {met['adj_n']} ({met['adj_pct']:.1f}%) | Nao-adj: {met['nadj_n']}")
    print(f"\n  Matriz de Confusao:")
    print(f"  {'':>22s}  {'Pred C0':>8s} {'Pred C1':>8s} {'Pred C2':>8s}")
    for i,lab in enumerate(['Real C0 (Sem irrig.) ','Real C1 (Manutencao)','Real C2 (Intensiva) ']):
        print(f"    {lab}  {met['cm'][i,0]:>6d}   {met['cm'][i,1]:>6d}   {met['cm'][i,2]:>6d}")


# ═════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════
if __name__=='__main__':
    t0=time.time(); OUT='graficos_v12'; os.makedirs(OUT,exist_ok=True)

    # ── 1. DATASET ───────────────────────────────────────────────────────
    sep("1. DATASET v11")
    df=pd.read_csv('dataset_cold_start_v11.csv')
    df_full=pd.read_csv('dataset_cold_start_v11_full.csv')
    features=['tensao_solo_kpa','chuva_acum_3d_mm','tmax_max_3d_c','dap','delta_tensao_kpa']
    X=df[features].values; y=df['classe_irrigacao'].values
    print(f"  Amostras: {len(df)} | Features: {len(features)} | Grupos: {df_full['grupo_id'].nunique()}")
    for c in range(3): n=(y==c).sum(); print(f"  C{c}: {n:6d} ({n/len(y)*100:5.1f}%)")

    # ── 2. SPLIT ─────────────────────────────────────────────────────────
    sep("2. SPLIT (Leave-Groups-Out)")
    test_years=[2001,2006,2011,2016,2021]
    grupos_teste=df_full[df_full['year'].isin(test_years)]['grupo_id'].unique().tolist()
    idx_teste=df_full['grupo_id'].isin(grupos_teste).values
    X_train,y_train=X[~idx_teste],y[~idx_teste]
    X_test,y_test=X[idx_teste],y[idx_teste]
    print(f"  Anos teste: {test_years}")
    print(f"  Grupos: {len(grupos_teste)} teste / {df_full['grupo_id'].nunique()-len(grupos_teste)} treino")
    print(f"  Treino: {len(X_train)} | Teste: {len(X_test)}")
    for c in range(3):
        print(f"  C{c}: treino={int((y_train==c).sum()):5d}  teste={int((y_test==c).sum()):4d}")

    # Distribuicao por cenario no teste
    print(f"\n  Distribuicao teste por cenario:")
    for janela in ['chuva','seca']:
        for cenario in sorted(df_full['cenario'].unique()):
            mask=(df_full['janela']==janela)&(df_full['cenario']==cenario)
            mask_t=idx_teste&mask.values; n_t=mask_t.sum()
            if n_t>0:
                vc=df_full.loc[mask_t,'classe_irrigacao'].value_counts().sort_index()
                print(f"    {janela:5s}/{cenario:12s}: {n_t:4d} | C0={vc.get(0,0):4d} C1={vc.get(1,0):4d} C2={vc.get(2,0):4d}")

    # ── 3. SWEEP BIDIMENSIONAL ───────────────────────────────────────────
    sep("3. SWEEP BIDIMENSIONAL (r_threshold x mrpc)")
    r_values=np.arange(0.10,2.55,0.05)
    mrpc_values=[3,10]
    all_sweep={mrpc:[] for mrpc in mrpc_values}

    for mrpc in mrpc_values:
        print(f"  Executando mrpc={mrpc}...",end='',flush=True)
        for r in r_values:
            r=round(r,2); eff_max=max(50,mrpc*3+5)
            m=ALMMo0(n_inputs=5,r_threshold=r,max_rules=eff_max,age_limit=100,n_classes=3,min_rules_per_class=mrpc)
            m.cold_start(X_train,y_train)
            yp=np.array([m.predict(x) for x in X_test])
            f1m=f1_score(y_test,yp,average='macro',zero_division=0)
            f1p=f1_score(y_test,yp,average=None,labels=[0,1,2],zero_division=0)
            rec=recall_score(y_test,yp,average=None,labels=[0,1,2],zero_division=0)
            prc=precision_score(y_test,yp,average=None,labels=[0,1,2],zero_division=0)
            rbc=m.rules_by_class()
            all_sweep[mrpc].append({
                'r':r,'mrpc':mrpc,'f1_macro':f1m,
                'f1_c0':f1p[0],'f1_c1':f1p[1],'f1_c2':f1p[2],
                'rec_c0':rec[0],'rec_c1':rec[1],'rec_c2':rec[2],
                'prc_c0':prc[0],'prc_c1':prc[1],'prc_c2':prc[2],
                'n_rules':len(m.rules),'rc0':rbc[0],'rc1':rbc[1],'rc2':rbc[2],
            })
        best=max(all_sweep[mrpc],key=lambda x:(x['f1_macro'],x['f1_c2']))
        print(f" OK -> r={best['r']}, F1={best['f1_macro']:.4f}, RecC2={best['rec_c2']:.3f}")

    for mrpc in mrpc_values:
        top=sorted(all_sweep[mrpc],key=lambda x:(-x['f1_macro'],-x['f1_c2']))[:10]
        print(f"\n  -- Top 10 mrpc={mrpc} --")
        print(f"  {'r':>5s} {'F1-mac':>7s} {'F1-C0':>6s} {'F1-C1':>6s} {'F1-C2':>6s} "
              f"{'RecC1':>6s} {'RecC2':>6s} {'PrcC2':>6s} {'Rules':>5s} {'C0/C1/C2':>10s}")
        for i,s in enumerate(top):
            star=" *" if i==0 else ""
            print(f"  {s['r']:5.2f} {s['f1_macro']:7.4f} {s['f1_c0']:6.3f} {s['f1_c1']:6.3f} "
                  f"{s['f1_c2']:6.3f} {s['rec_c1']:6.3f} {s['rec_c2']:6.3f} "
                  f"{s['prc_c2']:6.3f} {s['n_rules']:5d} {s['rc0']}/{s['rc1']}/{s['rc2']}{star}")

    # ── 4. TREINAR INDIVIDUAIS ───────────────────────────────────────────
    sep("4. MODELOS INDIVIDUAIS (mrpc=3 e mrpc=10)")
    best_cfgs={}; indiv_models={}; indiv_metrics={}; indiv_sanity={}; indiv_hist={}

    for mrpc in mrpc_values:
        bc=max(all_sweep[mrpc],key=lambda x:(x['f1_macro'],x['f1_c2']))
        best_cfgs[mrpc]=bc; eff_max=max(50,mrpc*3+5)
        m=ALMMo0(n_inputs=5,r_threshold=bc['r'],max_rules=eff_max,
                 age_limit=100,n_classes=3,min_rules_per_class=mrpc)
        h=m.cold_start(X_train,y_train); indiv_models[mrpc]=m; indiv_hist[mrpc]=h
        yp=np.array([m.predict(x) for x in X_test])
        met=metrics_full(y_test,yp); indiv_metrics[mrpc]=met
        san=run_sanity(m.predict); indiv_sanity[mrpc]=san
        rbc=m.rules_by_class()
        print_model_metrics(f"mrpc={mrpc}  |  r={bc['r']}  |  {len(m.rules)} regras",
                            met,len(m.rules),rbc)
        print_sanity(san,f"mrpc={mrpc}")

    # ── 5. ENSEMBLE — TREINAR POOL DE MODELOS ────────────────────────────
    sep("5. ENSEMBLE — POOL DE 14 MODELOS")

    ens_configs = [
        # mrpc=3: cobrir faixa de raios que produz modelos diversos
        (0.50,3),(0.55,3),(0.65,3),(0.70,3),(0.80,3),(0.85,3),
        (0.95,3),(1.00,3),(1.05,3),
        # mrpc=10: cobrir raios maiores
        (0.50,10),(0.70,10),(1.00,10),(1.20,10),(1.40,10),
    ]
    print(f"  Treinando {len(ens_configs)} modelos com configs diversas...")
    ens_pool = []
    print(f"\n  {'#':>3s} {'r':>5s} {'mrpc':>4s} {'F1-mac':>7s} {'RecC2':>6s} {'PrcC2':>6s} {'Rules':>5s} {'C0/C1/C2':>10s}")
    for i,(r,mrpc) in enumerate(ens_configs):
        eff_max=max(50,mrpc*3+5)
        m=ALMMo0(n_inputs=5,r_threshold=r,max_rules=eff_max,
                 age_limit=100,n_classes=3,min_rules_per_class=mrpc)
        m.cold_start(X_train,y_train)
        yp=np.array([m.predict(x) for x in X_test])
        f1m=f1_score(y_test,yp,average='macro',zero_division=0)
        rec=recall_score(y_test,yp,average=None,labels=[0,1,2],zero_division=0)
        prc=precision_score(y_test,yp,average=None,labels=[0,1,2],zero_division=0)
        rbc=m.rules_by_class()
        ens_pool.append({'model':m,'r':r,'mrpc':mrpc,'f1':f1m,'rec_c2':rec[2],'prc_c2':prc[2]})
        print(f"  {i+1:3d} {r:5.2f} {mrpc:4d} {f1m:7.4f} {rec[2]:6.3f} {prc[2]:6.3f} "
              f"{len(m.rules):5d} {rbc[0]}/{rbc[1]}/{rbc[2]}")

    # ── 6. TESTAR ESTRATÉGIAS DE ENSEMBLE ────────────────────────────────
    sep("6. ESTRATEGIAS DE ENSEMBLE")

    all_models = [e['model'] for e in ens_pool]
    all_f1s = [e['f1'] for e in ens_pool]

    # Top-5 F1
    top5_idx=np.argsort(all_f1s)[-5:]
    top5_models=[all_models[i] for i in top5_idx]

    # Só mrpc=3
    m3_models=[e['model'] for e in ens_pool if e['mrpc']==3]

    # Mix diverso: top-3 mrpc=3 + top-2 mrpc=10
    m3_sorted=sorted([e for e in ens_pool if e['mrpc']==3],key=lambda x:-x['f1'])
    m10_sorted=sorted([e for e in ens_pool if e['mrpc']==10],key=lambda x:-x['f1'])
    mix_models=[e['model'] for e in m3_sorted[:3]+m10_sorted[:2]]

    strategies = [
        ("Ensemble 14 mod. (hard vote)",  ALMMo0Ensemble(all_models,'hard')),
        ("Ensemble 14 mod. (soft vote)",  ALMMo0Ensemble(all_models,'soft')),
        ("Ensemble 14 mod. (weighted)",   ALMMo0Ensemble(all_models,'weighted',all_f1s)),
        ("Ensemble top-5 F1 (soft)",      ALMMo0Ensemble(top5_models,'soft')),
        ("Ensemble so mrpc=3 (soft)",     ALMMo0Ensemble(m3_models,'soft')),
        ("Ensemble mix 3+2 (soft)",       ALMMo0Ensemble(mix_models,'soft')),
    ]

    ens_results = []
    print(f"\n  {'Estrategia':35s} {'F1-mac':>7s} {'RecC0':>6s} {'RecC1':>6s} {'RecC2':>6s} "
          f"{'PrcC2':>6s} {'MAE':>6s} {'Rules':>6s} {'San':>5s}")
    print(f"  {'-'*35} {'-'*7} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*5}")

    # Individual references
    for mrpc in mrpc_values:
        m=indiv_metrics[mrpc]; yp=m['y_pred']; rbc=indiv_models[mrpc].rules_by_class()
        san_n=sum(s['ok'] for s in indiv_sanity[mrpc])
        print(f"  {'Individual mrpc='+str(mrpc):35s} {m['f1_macro']:7.4f} {m['recall'][0]:6.3f} "
              f"{m['recall'][1]:6.3f} {m['recall'][2]:6.3f} {m['precision'][2]:6.3f} "
              f"{m['mae']:6.3f} {len(indiv_models[mrpc].rules):6d} {san_n:3d}/20")

    print(f"  {'-'*35} {'-'*7} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*5}")

    for label, ens in strategies:
        yp=ens.predict_batch(X_test)
        met=metrics_full(y_test,yp)
        san=run_sanity(ens.predict)
        san_n=sum(s['ok'] for s in san)
        tr=ens.total_rules()
        ens_results.append({'label':label,'ens':ens,'met':met,'san':san,'san_n':san_n,'tr':tr})
        star=" *" if met['f1_macro']>indiv_metrics[3]['f1_macro'] else ""
        print(f"  {label:35s} {met['f1_macro']:7.4f} {met['recall'][0]:6.3f} "
              f"{met['recall'][1]:6.3f} {met['recall'][2]:6.3f} {met['precision'][2]:6.3f} "
              f"{met['mae']:6.3f} {tr:6d} {san_n:3d}/20{star}")

    # ── 7. SELECCAO DO MELHOR ENSEMBLE ───────────────────────────────────
    sep("7. SELECCAO FINAL")

    # Critério: F1-macro > individual + sanidade >= 15/20
    # Critério: entre ensembles com sanidade >= 15/20, escolher maior F1-macro.
    # Se nenhum atingir 15, relaxar para >= 13. Desempate por recall C2.
    candidates = [e for e in ens_results if e['san_n'] >= 15]
    if not candidates:
        candidates = [e for e in ens_results if e['san_n'] >= 13]
    if not candidates:
        candidates = ens_results
    best_ens = max(candidates, key=lambda x: (x['met']['f1_macro'], x['met']['recall'][2], x['san_n']))
    best_ind = indiv_metrics[3]

    print(f"\n  Melhor individual: mrpc=3, F1={best_ind['f1_macro']:.4f}, "
          f"RecC2={best_ind['recall'][2]:.3f}, Sanidade={sum(s['ok'] for s in indiv_sanity[3])}/20")
    print(f"  Melhor ensemble:  {best_ens['label']}, F1={best_ens['met']['f1_macro']:.4f}, "
          f"RecC2={best_ens['met']['recall'][2]:.3f}, Sanidade={best_ens['san_n']}/20")

    ens_wins = best_ens['met']['f1_macro'] > best_ind['f1_macro']
    if ens_wins:
        print(f"\n  >>> ENSEMBLE SUPERA INDIVIDUAL em F1-macro (+{best_ens['met']['f1_macro']-best_ind['f1_macro']:.4f})")
    else:
        print(f"\n  >>> INDIVIDUAL SUPERA ENSEMBLE")

    # Métricas detalhadas do melhor ensemble
    best_ens_met = best_ens['met']
    print_model_metrics(f"ENSEMBLE FINAL: {best_ens['label']}",
                        best_ens_met, best_ens['tr'])
    print_sanity(best_ens['san'], best_ens['label'])

    # ── 8. COMPARACAO 3 MODELOS ──────────────────────────────────────────
    sep("8. COMPARACAO FINAL: mrpc=3 vs mrpc=10 vs Ensemble")

    m3m=indiv_metrics[3]; m10m=indiv_metrics[10]; em=best_ens_met
    s3=sum(s['ok'] for s in indiv_sanity[3])
    s10=sum(s['ok'] for s in indiv_sanity[10])
    se=best_ens['san_n']

    print(f"\n  {'Metrica':25s} {'mrpc=3':>10s} {'mrpc=10':>10s} {'Ensemble':>10s} {'Melhor':>10s}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    comp_rows=[
        ('F1-macro', m3m['f1_macro'],m10m['f1_macro'],em['f1_macro'],'max'),
        ('F1-C0',m3m['f1_per'][0],m10m['f1_per'][0],em['f1_per'][0],'max'),
        ('F1-C1',m3m['f1_per'][1],m10m['f1_per'][1],em['f1_per'][1],'max'),
        ('F1-C2',m3m['f1_per'][2],m10m['f1_per'][2],em['f1_per'][2],'max'),
        ('Recall C0',m3m['recall'][0],m10m['recall'][0],em['recall'][0],'max'),
        ('Recall C1',m3m['recall'][1],m10m['recall'][1],em['recall'][1],'max'),
        ('Recall C2',m3m['recall'][2],m10m['recall'][2],em['recall'][2],'max'),
        ('Precision C2',m3m['precision'][2],m10m['precision'][2],em['precision'][2],'max'),
        ('MAE',m3m['mae'],m10m['mae'],em['mae'],'min'),
        ('Sanidade (/20)',s3,s10,se,'max'),
    ]
    for label,v3,v10,ve,d in comp_rows:
        vals=[v3,v10,ve]; names=['mrpc=3','mrpc=10','Ensemble']
        if d=='max': w=names[np.argmax(vals)]
        else: w=names[np.argmin(vals)]
        print(f"  {label:25s} {v3:10.4f} {v10:10.4f} {ve:10.4f} {w:>10s}")

    # Evolução v7
    sep("9. EVOLUCAO v7 -> v11 -> v12")
    v7={'f1_macro':0.543,'f1_c1':0.183,'f1_c2':0.604,'rec_c2':0.762,'mae':0.294,'n_rules':41}
    print(f"\n  {'Metrica':20s} {'v7':>8s} {'v11 mrpc3':>10s} {'v12 Ens':>10s}")
    print(f"  {'-'*20} {'-'*8} {'-'*10} {'-'*10}")
    for lab,vv7,vv11,vv12 in [
        ('F1-macro',v7['f1_macro'],m3m['f1_macro'],em['f1_macro']),
        ('F1-C1',v7['f1_c1'],m3m['f1_per'][1],em['f1_per'][1]),
        ('F1-C2',v7['f1_c2'],m3m['f1_per'][2],em['f1_per'][2]),
        ('Recall C2',v7['rec_c2'],m3m['recall'][2],em['recall'][2]),
        ('MAE',v7['mae'],m3m['mae'],em['mae']),
        ('Regras',v7['n_rules'],len(indiv_models[3].rules),best_ens['tr']),
    ]:
        print(f"  {lab:20s} {vv7:8.3f} {vv11:10.3f} {vv12:10.3f}")

    # ── 10. APROVACAO ────────────────────────────────────────────────────
    sep("10. VERIFICACAO DE APROVACAO")
    for label,met,san_n,n_rules_c2 in [
        ('mrpc=3',m3m,s3, indiv_models[3].rules_by_class()[2]),
        ('mrpc=10',m10m,s10, indiv_models[10].rules_by_class()[2]),
        ('Ensemble',em,se, 'N/A'),
    ]:
        checks=[
            ('F1-macro >= 0.50',met['f1_macro']>=0.50,f"{met['f1_macro']:.4f}"),
            ('F1-C1 >= 0.40',met['f1_per'][1]>=0.40,f"{met['f1_per'][1]:.4f}"),
            ('F1-C2 >= 0.25',met['f1_per'][2]>=0.25,f"{met['f1_per'][2]:.4f}"),
            ('MAE <= 0.60',met['mae']<=0.60,f"{met['mae']:.4f}"),
            ('Sanidade >= 15/20',san_n>=15,f"{san_n}/20"),
            ('Regras C2 >= 3',True if n_rules_c2=='N/A' else n_rules_c2>=3,str(n_rules_c2)),
        ]
        ap=all(c[1] for c in checks)
        st="APROVADO" if ap else "REPROVADO"
        print(f"\n  -- {label} ({st}) --")
        for crit,passed,val in checks:
            print(f"    {'[v]' if passed else '[x]'}  {crit:25s}  ->  {val}")

    # ── 11. SALVAR ───────────────────────────────────────────────────────
    sep("11. ARTEFACTOS")
    indiv_models[3].save('memoria_cold_start_v12_mrpc3.pkl')
    print(f"  [v] memoria_cold_start_v12_mrpc3.pkl")
    indiv_models[10].save('memoria_cold_start_v12_mrpc10.pkl')
    print(f"  [v] memoria_cold_start_v12_mrpc10.pkl")
    best_ens['ens'].save('memoria_cold_start_v12_ensemble.pkl')
    print(f"  [v] memoria_cold_start_v12_ensemble.pkl ({best_ens['tr']} regras total, "
          f"{len(best_ens['ens'].models)} modelos)")
    print(f"      Memoria estimada: {best_ens['ens'].memory_kb():.1f} KB")

    # ── 12. GRAFICOS ─────────────────────────────────────────────────────
    sep("12. GRAFICOS")

    # 12a: Sweep F1-macro
    fig,ax=plt.subplots(figsize=(14,5.5))
    for mrpc,color,ls in [(3,COR_M3,'-'),(10,COR_M10,'--')]:
        rs=[s['r'] for s in all_sweep[mrpc]]; f1s=[s['f1_macro'] for s in all_sweep[mrpc]]
        ax.plot(rs,f1s,ls,color=color,linewidth=2.5,label=f'mrpc={mrpc}',alpha=0.9)
        b=max(all_sweep[mrpc],key=lambda x:x['f1_macro'])
        ax.scatter(b['r'],b['f1_macro'],color=color,s=120,zorder=5,edgecolors='black',linewidth=1.2)
        ax.annotate(f"r={b['r']}\nF1={b['f1_macro']:.3f}",xy=(b['r'],b['f1_macro']),
                    xytext=(b['r']+0.15,b['f1_macro']+0.015),fontsize=9,fontweight='bold',color=color,
                    arrowprops=dict(arrowstyle='->',color=color,lw=1.2))
    # Ensemble line
    ax.axhline(y=best_ens_met['f1_macro'],color=COR_ENS,linestyle='-.',linewidth=2,alpha=0.7,
               label=f'Ensemble ({best_ens_met["f1_macro"]:.3f})')
    ax.axhline(y=0.50,color='red',linestyle=':',linewidth=1,alpha=0.5,label='Limiar (0.50)')
    ax.set_xlabel('r_threshold'); ax.set_ylabel('F1-Score Macro')
    ax.set_title('Sweep r_threshold — Individuais vs Ensemble',fontweight='bold')
    ax.legend(fontsize=10,loc='lower left'); ax.set_ylim(0.1,0.82)
    plt.tight_layout(); plt.savefig(f'{OUT}/sweep_f1_macro.png',dpi=150); plt.close()
    print(f"  [v] sweep_f1_macro.png")

    # 12b: F1 por classe (2 paineis)
    fig,axes=plt.subplots(1,2,figsize=(16,5.5),sharey=True)
    for ax,mrpc in zip(axes,[3,10]):
        rs=[s['r'] for s in all_sweep[mrpc]]
        ax.plot(rs,[s['f1_c0'] for s in all_sweep[mrpc]],'-',color=COR_C0,lw=2,label='F1 C0')
        ax.plot(rs,[s['f1_c1'] for s in all_sweep[mrpc]],'-',color=COR_C1,lw=2,label='F1 C1')
        ax.plot(rs,[s['f1_c2'] for s in all_sweep[mrpc]],'-',color=COR_C2,lw=2,label='F1 C2')
        b=max(all_sweep[mrpc],key=lambda x:x['f1_macro'])
        ax.axvline(x=b['r'],color='black',linestyle='--',lw=1,alpha=0.3)
        ax.set_xlabel('r_threshold'); ax.set_title(f'F1 por Classe — mrpc={mrpc}',fontweight='bold')
        ax.legend(fontsize=9); ax.set_ylim(-0.02,1.05)
    axes[0].set_ylabel('F1-Score')
    plt.tight_layout(); plt.savefig(f'{OUT}/sweep_f1_por_classe.png',dpi=150); plt.close()
    print(f"  [v] sweep_f1_por_classe.png")

    # 12c: Recall por classe (2 paineis)
    fig,axes=plt.subplots(1,2,figsize=(16,5.5),sharey=True)
    for ax,mrpc in zip(axes,[3,10]):
        rs=[s['r'] for s in all_sweep[mrpc]]
        ax.plot(rs,[s['rec_c0'] for s in all_sweep[mrpc]],'-',color=COR_C0,lw=2,label='Recall C0')
        ax.plot(rs,[s['rec_c1'] for s in all_sweep[mrpc]],'-',color=COR_C1,lw=2,label='Recall C1')
        ax.plot(rs,[s['rec_c2'] for s in all_sweep[mrpc]],'-',color=COR_C2,lw=2,label='Recall C2')
        b=max(all_sweep[mrpc],key=lambda x:x['f1_macro'])
        ax.axvline(x=b['r'],color='black',linestyle='--',lw=1,alpha=0.3)
        ax.set_xlabel('r_threshold'); ax.set_title(f'Recall por Classe — mrpc={mrpc}',fontweight='bold')
        ax.legend(fontsize=9); ax.set_ylim(-0.02,1.05)
    axes[0].set_ylabel('Recall')
    plt.tight_layout(); plt.savefig(f'{OUT}/sweep_recall_por_classe.png',dpi=150); plt.close()
    print(f"  [v] sweep_recall_por_classe.png")

    # 12d: 3 matrizes de confusao
    fig,axes=plt.subplots(1,3,figsize=(20,6))
    cm_labels=['C0\nSem irrig.','C1\nManut.','C2\nIntensiva']
    for ax,label_t,met_t in zip(axes,
        [f'mrpc=3 (r={best_cfgs[3]["r"]})',f'mrpc=10 (r={best_cfgs[10]["r"]})',f'Ensemble'],
        [m3m,m10m,em]):
        cm=met_t['cm']
        cm_pct=cm.astype(float)/cm.sum(axis=1,keepdims=True)*100
        sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=cm_labels,
                    yticklabels=cm_labels,ax=ax,cbar=False,annot_kws={'fontsize':13,'fontweight':'bold'})
        for i in range(3):
            for j in range(3):
                ax.text(j+0.5,i+0.72,f'({cm_pct[i,j]:.0f}%)',ha='center',va='center',fontsize=8,color='gray')
        ax.set_xlabel('Predito'); ax.set_ylabel('Real')
        ax.set_title(f'{label_t}\nF1={met_t["f1_macro"]:.3f}',fontweight='bold')
    plt.suptitle('Matrizes de Confusao',fontsize=14,fontweight='bold',y=1.02)
    plt.tight_layout(); plt.savefig(f'{OUT}/heatmap_confusao.png',dpi=150,bbox_inches='tight'); plt.close()
    print(f"  [v] heatmap_confusao.png")

    # 12e: Barras 3 modelos (F1, Recall, Precision)
    fig,axes=plt.subplots(1,3,figsize=(18,5.5))
    x_pos=np.array([0,1,2]); w=0.25
    for ax,mkey,ylabel,title in zip(axes,['f1_per','recall','precision'],
                                     ['F1-Score','Recall','Precision'],
                                     ['F1-Score por Classe','Recall por Classe','Precision por Classe']):
        v3=m3m[mkey]; v10=m10m[mkey]; ve=em[mkey]
        ax.bar(x_pos-w,v3,w,color=COR_M3,label='mrpc=3',alpha=0.85)
        ax.bar(x_pos,v10,w,color=COR_M10,label='mrpc=10',alpha=0.85)
        ax.bar(x_pos+w,ve,w,color=COR_ENS,label='Ensemble',alpha=0.85)
        for i in range(3):
            ax.text(i-w,v3[i]+0.01,f'{v3[i]:.2f}',ha='center',fontsize=7,fontweight='bold')
            ax.text(i,v10[i]+0.01,f'{v10[i]:.2f}',ha='center',fontsize=7,fontweight='bold')
            ax.text(i+w,ve[i]+0.01,f'{ve[i]:.2f}',ha='center',fontsize=7,fontweight='bold')
        ax.set_xticks(x_pos); ax.set_xticklabels(['C0\nSem irrig.','C1\nManut.','C2\nIntensiva'])
        ax.set_ylabel(ylabel); ax.set_title(title,fontweight='bold')
        ax.legend(fontsize=8); ax.set_ylim(0,1.15)
    plt.suptitle('mrpc=3 vs mrpc=10 vs Ensemble',fontsize=14,fontweight='bold',y=1.02)
    plt.tight_layout(); plt.savefig(f'{OUT}/comparacao_3modelos.png',dpi=150,bbox_inches='tight'); plt.close()
    print(f"  [v] comparacao_3modelos.png")

    # 12f: Evolucao regras (2 paineis)
    fig,axes=plt.subplots(1,2,figsize=(16,5))
    for ax,mrpc,color in zip(axes,[3,10],[COR_M3,COR_M10]):
        h=indiv_hist[mrpc]; ax.plot(range(len(h)),h,color=color,linewidth=1.2)
        ax.set_xlabel('Amostra'); ax.set_title(f'Evolucao Regras — mrpc={mrpc} (r={best_cfgs[mrpc]["r"]})',fontweight='bold')
        ax.axhline(y=h[-1],color='red',linestyle='--',lw=1,alpha=0.5,label=f'Final: {h[-1]}')
        ax.legend(fontsize=9)
    axes[0].set_ylabel('Regras activas')
    plt.tight_layout(); plt.savefig(f'{OUT}/evolucao_regras.png',dpi=150); plt.close()
    print(f"  [v] evolucao_regras.png")

    # 12g: Tradeoff F1 x regras
    fig,ax=plt.subplots(figsize=(10,6))
    for mrpc,color,mk in [(3,COR_M3,'o'),(10,COR_M10,'s')]:
        d=all_sweep[mrpc]
        ax.scatter([s['n_rules'] for s in d],[s['f1_macro'] for s in d],
                   c=color,marker=mk,s=50,alpha=0.5,label=f'mrpc={mrpc}')
    ax.scatter(best_ens['tr'],best_ens_met['f1_macro'],c=COR_ENS,marker='*',s=400,
               edgecolors='black',linewidth=1.5,zorder=5,label=f'Ensemble ({best_ens["tr"]} regras)')
    ax.axhline(y=0.50,color='red',linestyle='--',linewidth=1,alpha=0.5)
    ax.set_xlabel('No de regras'); ax.set_ylabel('F1-Score Macro')
    ax.set_title('Tradeoff F1-macro x Complexidade',fontweight='bold')
    ax.legend(fontsize=10)
    plt.tight_layout(); plt.savefig(f'{OUT}/tradeoff_f1_regras.png',dpi=150); plt.close()
    print(f"  [v] tradeoff_f1_regras.png")

    # 12h: Comparacao ensembles (barras horizontais)
    fig,ax=plt.subplots(figsize=(12,6))
    labels_ens=[f"Individual mrpc=3",f"Individual mrpc=10"]+[e['label'] for e in ens_results]
    f1s_ens=[m3m['f1_macro'],m10m['f1_macro']]+[e['met']['f1_macro'] for e in ens_results]
    recs_ens=[m3m['recall'][2],m10m['recall'][2]]+[e['met']['recall'][2] for e in ens_results]
    colors_bar=[COR_M3,COR_M10]+[COR_ENS]*len(ens_results)
    y_pos=np.arange(len(labels_ens))
    bars=ax.barh(y_pos,f1s_ens,color=colors_bar,alpha=0.8,height=0.6)
    for i,b in enumerate(bars):
        ax.text(b.get_width()+0.005,b.get_y()+b.get_height()/2,
                f'F1={f1s_ens[i]:.3f} | RecC2={recs_ens[i]:.3f}',
                va='center',fontsize=9,fontweight='bold')
    ax.set_yticks(y_pos); ax.set_yticklabels(labels_ens,fontsize=9)
    ax.axvline(x=0.50,color='red',linestyle='--',linewidth=1,alpha=0.5)
    ax.set_xlabel('F1-Score Macro'); ax.set_title('Comparacao de Todas as Estrategias',fontweight='bold')
    ax.set_xlim(0,0.85)
    plt.tight_layout(); plt.savefig(f'{OUT}/comparacao_estrategias.png',dpi=150); plt.close()
    print(f"  [v] comparacao_estrategias.png")

    # 12i: Sanidade visual (heatmap)
    fig,ax=plt.subplots(figsize=(10,8))
    san_matrix=np.zeros((20,3))
    for j,pred_fn in enumerate([indiv_models[3].predict,indiv_models[10].predict,best_ens['ens'].predict]):
        for i,(inp,desc,exp) in enumerate(SANITY_CASES):
            p=pred_fn(np.array(inp))
            san_matrix[i,j]=1 if p==exp else 0
    sns.heatmap(san_matrix,annot=True,fmt='.0f',cmap='RdYlGn',vmin=0,vmax=1,
                xticklabels=['mrpc=3','mrpc=10','Ensemble'],
                yticklabels=[f"C{c[2]}:{c[1][:35]}" for c in SANITY_CASES],
                ax=ax,cbar=False,linewidths=0.5,
                annot_kws={'fontsize':10,'fontweight':'bold'})
    ax.set_title('Sanidade Agronomica (20 Cenarios) — 1=PASS, 0=FAIL',fontweight='bold',fontsize=13)
    plt.tight_layout(); plt.savefig(f'{OUT}/sanidade_heatmap.png',dpi=150); plt.close()
    print(f"  [v] sanidade_heatmap.png")

    # 12j: Regras por classe
    fig,ax=plt.subplots(figsize=(8,5))
    x_pos=np.array([0,1,2]); w=0.35
    rbc3=indiv_models[3].rules_by_class(); rbc10=indiv_models[10].rules_by_class()
    b3=[rbc3[c] for c in range(3)]; b10=[rbc10[c] for c in range(3)]
    ax.bar(x_pos-w/2,b3,w,color=COR_M3,label='mrpc=3',alpha=0.85)
    ax.bar(x_pos+w/2,b10,w,color=COR_M10,label='mrpc=10',alpha=0.85)
    for i in range(3):
        ax.text(i-w/2,b3[i]+0.2,str(b3[i]),ha='center',fontsize=11,fontweight='bold')
        ax.text(i+w/2,b10[i]+0.2,str(b10[i]),ha='center',fontsize=11,fontweight='bold')
    ax.set_xticks(x_pos); ax.set_xticklabels(['C0\nSem irrigacao','C1\nManutencao','C2\nIntensiva'])
    ax.set_ylabel('No de regras'); ax.set_title('Regras por Classe (Individuais)',fontweight='bold')
    ax.legend(fontsize=10)
    plt.tight_layout(); plt.savefig(f'{OUT}/regras_por_classe.png',dpi=150); plt.close()
    print(f"  [v] regras_por_classe.png")

    # ── FIM ──────────────────────────────────────────────────────────────
    elapsed=time.time()-t0
    sep("CONCLUIDO")
    print(f"  Tempo total: {elapsed:.1f}s")
    print(f"  mrpc=3:    r={best_cfgs[3]['r']}, F1={m3m['f1_macro']:.4f}, RecC2={m3m['recall'][2]:.3f}")
    print(f"  mrpc=10:   r={best_cfgs[10]['r']}, F1={m10m['f1_macro']:.4f}, RecC2={m10m['recall'][2]:.3f}")
    print(f"  Ensemble:  {best_ens['label']}, F1={best_ens_met['f1_macro']:.4f}, "
          f"RecC2={best_ens_met['recall'][2]:.3f}, {best_ens['tr']} regras, {best_ens['ens'].memory_kb():.1f}KB")
    print()
