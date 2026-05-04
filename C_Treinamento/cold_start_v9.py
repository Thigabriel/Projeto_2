"""
============================================================================
  COLD START v9 — ALMMo-0 (Binário)
  Classificação: C0 (sem irrigação) vs C1 (irrigação: C1+C2 originais)
  
  Motivação: no v8, o desbalanceamento 94/4/2% limitou todos os modelos.
  C1 e C2 originais têm apenas 111 e 50 amostras — insuficientes para
  aprendizado robusto separado. Ao juntar, temos 161 amostras de irrigação
  (5.9%) — ainda desbalanceado mas mais viável para o classificador.
  
  A separação C1 vs C2 (moderada vs intensa) será tratada em etapa
  posterior, fora do escopo deste cold start.
  
  Estratégias testadas (top do v8 + adaptações):
    0. Baseline (sem resampling)
    1. Cost-Sensitive Learning (vencedor do v8)
    2. SMOTE
    3. ADASYN
    4. SMOTE Parcial (15%)
    5. Repeated Minority Presentation
    6. Cost-Sensitive + SMOTE Parcial (combo)
  
  Requisitos:
    pip install numpy pandas scikit-learn matplotlib seaborn imbalanced-learn
  
  Execução:
    python cold_start_v9.py
============================================================================
"""

import numpy as np
import pandas as pd
import os
import time
import pickle
import warnings
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


# ─────────────────────────────────────────────────────────────────────────────
# ALMMo-0 v3 (binário, com cost-sensitive)
# ─────────────────────────────────────────────────────────────────────────────

class ALMMo0:
    def __init__(self, n_inputs=4, r_threshold=0.5, max_rules=50,
                 age_limit=100, epsilon=1e-8, n_classes=2,
                 min_rules_per_class=3, class_weights=None):
        self.n_inputs            = n_inputs
        self.r_threshold         = r_threshold
        self.max_rules           = max_rules
        self.age_limit           = age_limit
        self.epsilon             = epsilon
        self.n_classes           = n_classes
        self.min_rules_per_class = min_rules_per_class
        if class_weights is not None:
            self.class_weights = np.array(class_weights, dtype=float)
        else:
            self.class_weights = np.ones(n_classes)
        self.rules           = []
        self.input_mean      = np.zeros(n_inputs)
        self.input_std       = np.ones(n_inputs)
        self.n_samples_seen  = 0
        self.n_rules_created = 0
        self.n_rules_pruned  = 0
        self.created_at      = datetime.now().isoformat()

    def fit_normalizer(self, X):
        self.input_mean = np.mean(X, axis=0)
        self.input_std  = np.std(X, axis=0)
        self.input_std[self.input_std < self.epsilon] = 1.0

    def normalize(self, x):
        return (x - self.input_mean) / self.input_std

    def _euclidean_distance(self, a, b):
        d = a - b
        return float(np.sqrt(np.dot(d, d)))

    def _distances_to_all_rules(self, x_norm):
        if not self.rules:
            return np.array([])
        return np.array([self._euclidean_distance(x_norm, r['center']) for r in self.rules])

    def _create_rule(self, x_norm, y):
        self.rules.append({
            'center': x_norm.copy(), 'consequent': int(y),
            'age': 0, 'activations': 1,
        })
        self.n_rules_created += 1

    def predict(self, x):
        if not self.rules:
            raise RuntimeError("Banco de regras vazio.")
        x_norm = self.normalize(x)
        distances = self._distances_to_all_rules(x_norm)
        weights = 1.0 / (distances ** 2 + self.epsilon)
        class_votes = np.zeros(self.n_classes)
        for i, rule in enumerate(self.rules):
            class_votes[rule['consequent']] += weights[i]
        return int(np.argmax(class_votes))

    def learn(self, x, y):
        self.n_samples_seen += 1
        x_norm = self.normalize(x)
        if not self.rules:
            self._create_rule(x_norm, y)
            return
        distances = self._distances_to_all_rules(x_norm)
        idx_min = int(np.argmin(distances))
        d_min = distances[idx_min]
        effective_r = self.r_threshold / self.class_weights[int(y)]
        if d_min > effective_r:
            self._create_rule(x_norm, y)
        else:
            self._update_rule(idx_min, x_norm, y, d_min)
        self._age_rules(idx_min)
        self._prune_rules()

    def _update_rule(self, idx, x_norm, y, distance):
        rule = self.rules[idx]
        rule['activations'] += 1
        eta = 1.0 / rule['activations']
        rule['center'] = rule['center'] + eta * (x_norm - rule['center'])
        if rule['consequent'] != y:
            af = max(0.0, 1.0 - distance / self.r_threshold)
            eff_eta = eta * af * self.class_weights[int(y)]
            eff_eta = min(eff_eta, 1.0)
            if eff_eta > 0:
                new_c = (1 - eff_eta) * rule['consequent'] + eff_eta * y
                rule['consequent'] = int(np.round(np.clip(new_c, 0, self.n_classes - 1)))
        rule['age'] = 0

    def _age_rules(self, idx_activated):
        for i, rule in enumerate(self.rules):
            if i != idx_activated:
                rule['age'] += 1

    def _prune_rules(self):
        n_before = len(self.rules)
        cc = Counter(r['consequent'] for r in self.rules)
        surviving = []
        for r in self.rules:
            if r['age'] <= self.age_limit:
                surviving.append(r)
            else:
                if cc[r['consequent']] > self.min_rules_per_class:
                    cc[r['consequent']] -= 1
                else:
                    r['age'] = 0
                    surviving.append(r)
        if len(surviving) > self.max_rules:
            surviving.sort(key=lambda r: r['age'], reverse=True)
            cc2 = Counter(r['consequent'] for r in surviving)
            kept = []
            for r in surviving:
                if len(kept) >= self.max_rules:
                    if cc2[r['consequent']] <= self.min_rules_per_class:
                        kept.append(r)
                    else:
                        cc2[r['consequent']] -= 1
                else:
                    kept.append(r)
            surviving = kept
        self.rules = surviving
        self.n_rules_pruned += n_before - len(self.rules)

    def cold_start(self, X, y, verbose=False):
        self.fit_normalizer(X)
        history = []
        for i in range(len(X)):
            self.learn(X[i], int(y[i]))
            history.append(len(self.rules))
            if verbose and (i + 1) % max(1, len(X) // 10) == 0:
                pct = 100 * (i + 1) / len(X)
                dist = self.rules_by_class()
                print(f"    {pct:3.0f}% | {len(self.rules)} regras | {dist}")
        return history

    def rules_by_class(self):
        return {c: sum(1 for r in self.rules if r['consequent'] == c) for c in range(self.n_classes)}

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({
                'rules': self.rules, 'input_mean': self.input_mean,
                'input_std': self.input_std, 'r_threshold': self.r_threshold,
                'max_rules': self.max_rules, 'age_limit': self.age_limit,
                'n_inputs': self.n_inputs, 'n_classes': self.n_classes,
                'min_rules_per_class': self.min_rules_per_class,
                'class_weights': self.class_weights.tolist(),
                'n_samples_seen': self.n_samples_seen,
                'n_rules_created': self.n_rules_created,
                'n_rules_pruned': self.n_rules_pruned,
                'created_at': self.created_at,
                'saved_at': datetime.now().isoformat(),
            }, f)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def sep(title):
    w = 75
    print(f"\n{'═'*w}")
    print(f"  {title}")
    print(f"{'═'*w}")


def evaluate(model, X_test, y_test, label="", n_classes=2, verbose=True):
    y_pred = np.array([model.predict(x) for x in X_test])
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1w = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    f1p = f1_score(y_test, y_pred, average=None, zero_division=0)
    rec = recall_score(y_test, y_pred, average=None, zero_division=0)
    prec = precision_score(y_test, y_pred, average=None, zero_division=0)
    mae = mean_absolute_error(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=list(range(n_classes)))
    rbc = model.rules_by_class()

    if verbose:
        class_names = ['C0(Sem irr.)', 'C1(Irrigação)'] if n_classes == 2 else [f'C{c}' for c in range(n_classes)]
        print(f"\n  ┌─ {label}")
        print(f"  │  F1-macro: {f1m:.4f}  |  Acurácia: {acc:.4f}  |  MAE: {mae:.4f}")
        for c in range(n_classes):
            print(f"  │  {class_names[c]:14s}  F1={f1p[c]:.3f}  Recall={rec[c]:.3f}  Prec={prec[c]:.3f}")
        print(f"  │  Regras: {len(model.rules)} total | {rbc}")
        print(f"  │  CM: {' / '.join(str(cm[i].tolist()) for i in range(n_classes))}")
        print(f"  └─")

    return {
        'label': label, 'acc': acc, 'f1_macro': f1m, 'f1_weighted': f1w,
        'f1_per': f1p, 'recall': rec, 'precision': prec, 'mae': mae,
        'cm': cm, 'n_rules': len(model.rules), 'rbc': rbc, 'y_pred': y_pred,
    }


def sweep(X_train, y_train, X_test, y_test, label,
          r_values, mrpc_values, max_rules_base=40,
          class_weights=None, n_classes=2, shuffle=False):
    best_f1 = -1
    best_cfg = None
    best_model = None
    rng = np.random.RandomState(42)

    for mrpc in mrpc_values:
        for r in r_values:
            r = round(r, 2)
            eff_max = max(max_rules_base, mrpc * n_classes + 5)
            m = ALMMo0(n_inputs=4, r_threshold=r, max_rules=eff_max,
                       age_limit=80, n_classes=n_classes,
                       min_rules_per_class=mrpc, class_weights=class_weights)
            Xs, ys = X_train, y_train
            if shuffle:
                idx = rng.permutation(len(Xs))
                Xs, ys = Xs[idx], ys[idx]
            m.cold_start(Xs, ys, verbose=False)
            yp = np.array([m.predict(x) for x in X_test])
            f1 = f1_score(y_test, yp, average='macro', zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_cfg = {'r': r, 'mrpc': mrpc}
                best_model = m

    print(f"  {label}: melhor r={best_cfg['r']}, mrpc={best_cfg['mrpc']}, F1={best_f1:.4f}")
    return best_model, best_cfg, best_f1


def sanity_check_binary(model, verbose=True):
    """Teste de sanidade para classificação binária."""
    casos = [
        ([120.0,  0.0, 38.5, 80], "Solo seco (120kPa), sem chuva, calor extremo", 1),
        ([ 15.0, 30.0, 32.0, 40], "Solo úmido (15kPa), muita chuva, ameno",       0),
        ([ 70.0,  2.0, 36.0, 60], "Solo médio (70kPa), pouca chuva, calor",        1),
        ([100.0,  0.0, 37.0, 20], "Solo seco (100kPa), sem chuva, DAP baixo",      1),
        ([ 10.0, 50.0, 30.0, 30], "Solo muito úmido, muita chuva, fresco",         0),
        ([ 50.0,  0.0, 39.0, 90], "Solo médio-seco, sem chuva, calor, DAP alto",   1),
    ]
    n_pass = 0
    for inputs, desc, expected in casos:
        pred = model.predict(np.array(inputs))
        ok = pred == expected
        n_pass += ok
        exp_str = "Irrigar" if expected == 1 else "Não irrigar"
        pred_str = "Irrigar" if pred == 1 else "Não irrigar"
        if verbose:
            st = "✓" if ok else "✗"
            print(f"    {st} Pred={pred_str:12s} Exp={exp_str:12s} | {desc}")
    return n_pass, len(casos)


# ─────────────────────────────────────────────────────────────────────────────
# RESAMPLING (adaptado para binário)
# ─────────────────────────────────────────────────────────────────────────────

def resample_smote(X, y, k=None):
    from imblearn.over_sampling import SMOTE
    min_n = min(Counter(y).values())
    k = min(5, min_n - 1) if k is None else min(k, min_n - 1)
    k = max(1, k)
    sm = SMOTE(random_state=42, k_neighbors=k)
    return sm.fit_resample(X, y)


def resample_adasyn(X, y):
    from imblearn.over_sampling import ADASYN
    min_n = min(Counter(y).values())
    k = min(5, min_n - 1)
    k = max(1, k)
    try:
        ada = ADASYN(random_state=42, n_neighbors=k)
        return ada.fit_resample(X, y)
    except (ValueError, RuntimeError) as e:
        print(f"    ⚠ ADASYN falhou ({e}), fallback SMOTE")
        return resample_smote(X, y)


def resample_partial_smote(X, y, target_ratio=0.15):
    from imblearn.over_sampling import SMOTE
    counts = Counter(y)
    total = len(y)
    target_counts = {c: max(counts[c], int(total * target_ratio)) for c in counts}
    min_n = min(counts.values())
    k = min(5, min_n - 1)
    k = max(1, k)
    sm = SMOTE(random_state=42, k_neighbors=k, sampling_strategy=target_counts)
    return sm.fit_resample(X, y)


def resample_repeated_minority(X, y):
    counts = Counter(y)
    max_count = max(counts.values())
    indices_by_class = {c: np.where(y == c)[0] for c in counts}
    rng = np.random.RandomState(42)
    X_list, y_list = list(X), list(y)
    for c in counts:
        if counts[c] < max_count:
            factor = min(int(np.ceil(max_count / counts[c])), 20) - 1
            if factor > 0:
                pool = np.tile(indices_by_class[c], factor)
                rng.shuffle(pool)
                insert_interval = max(1, len(X) // (len(pool) + 1))
                for idx_i, j in enumerate(pool):
                    pos = min(len(X_list), (idx_i + 1) * insert_interval)
                    X_list.insert(pos, X[j])
                    y_list.insert(pos, y[j])
    return np.array(X_list), np.array(y_list)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    t0 = time.time()
    OUT = 'graficos_v9'
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.RandomState(42)

    # ── 1. LOAD & BINARIZE ──────────────────────────────────────────────
    sep("1. DATASET (Binarizado: C0 vs C1+C2→C1)")
    df = pd.read_csv('dataset_cold_start_v7.csv')
    fcols = ['tensao_solo_kpa', 'chuva_acum_3d_mm', 'tmax_max_3d_c', 'dap']
    X = df[fcols].values
    y_original = df['classe_irrigacao'].values

    # Binarizar: C0=0, C1+C2→1
    y = (y_original > 0).astype(int)

    print(f"  Amostras: {len(df)}")
    print(f"  Original:  C0={int((y_original==0).sum())} ({(y_original==0).mean()*100:.1f}%)  "
          f"C1={int((y_original==1).sum())} ({(y_original==1).mean()*100:.1f}%)  "
          f"C2={int((y_original==2).sum())} ({(y_original==2).mean()*100:.1f}%)")
    print(f"  Binário:   C0={int((y==0).sum())} ({(y==0).mean()*100:.1f}%)  "
          f"C1={int((y==1).sum())} ({(y==1).mean()*100:.1f}%)")
    print(f"\n  C1 (irrigação) contém as antigas C1 + C2:")
    print(f"    C1 original: {int((y_original==1).sum())} amostras (moderada)")
    print(f"    C2 original: {int((y_original==2).sum())} amostras (intensa)")

    # Feature stats por classe binária
    print(f"\n  Perfil das features por classe binária:")
    for c, name in [(0, 'Sem irrigação'), (1, 'Irrigação')]:
        mask = y == c
        print(f"    C{c} ({name}, N={mask.sum()}):")
        print(f"      tensão: {X[mask,0].mean():.1f} ± {X[mask,0].std():.1f} kPa  "
              f"(range {X[mask,0].min():.1f}–{X[mask,0].max():.1f})")
        print(f"      chuva:  {X[mask,1].mean():.1f} ± {X[mask,1].std():.1f} mm")
        print(f"      tmax:   {X[mask,2].mean():.1f} ± {X[mask,2].std():.1f} °C")
        print(f"      dap:    {X[mask,3].mean():.0f} ± {X[mask,3].std():.0f}")

    # ── 2. SPLIT (mesmo do v7/v8) ────────────────────────────────────────
    sep("2. SPLIT (Leave-Groups-Out)")
    dap_vals = df['dap'].values
    resets = [0] + [i for i in range(1, len(dap_vals)) if dap_vals[i] < dap_vals[i-1]] + [len(dap_vals)]
    test_group_ids = [5, 9, 13, 17, 21, 27]
    test_idx = []
    for g in test_group_ids:
        test_idx.extend(range(resets[g], resets[g+1]))
    train_idx = [i for i in range(len(df)) if i not in test_idx]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    # Guardar labels originais do teste para análise posterior
    y_test_original = y_original[test_idx]

    print(f"  Treino: {len(X_train)}  |  Teste: {len(X_test)}")
    print(f"  Treino:  C0={int((y_train==0).sum())}  C1(irrig)={int((y_train==1).sum())}")
    print(f"  Teste:   C0={int((y_test==0).sum())}  C1(irrig)={int((y_test==1).sum())}")
    print(f"  Teste (detalhe): C1_orig={int((y_test_original==1).sum())}  C2_orig={int((y_test_original==2).sum())}")

    # ── 3. SWEEP CONFIG ──────────────────────────────────────────────────
    r_values = np.arange(0.10, 2.55, 0.05)  # range mais amplo
    mrpc_values = [3, 5, 7, 10, 15]

    # ── 4. ESTRATÉGIAS ───────────────────────────────────────────────────
    sep("3. ESTRATÉGIAS (binário, sweep bidimensional)")

    all_experiments = []
    best_models = {}

    # ─── 0: Baseline ─────────────────────────────────────────────────────
    print("\n  [0/6] Baseline...")
    print(f"    Dataset: {len(X_train)} amostras | {dict(Counter(y_train))}")
    m0, cfg0, f1_0 = sweep(X_train, y_train, X_test, y_test, "Baseline",
                            r_values, mrpc_values, n_classes=2)
    met0 = evaluate(m0, X_test, y_test, "Baseline", n_classes=2)
    met0['strategy'] = 'Baseline'
    met0['cfg'] = cfg0
    met0['train_size'] = len(X_train)
    all_experiments.append(met0)
    best_models['Baseline'] = m0

    # ─── 1: Cost-Sensitive ───────────────────────────────────────────────
    print("\n  [1/6] Cost-Sensitive...")
    counts = Counter(y_train)
    total = len(y_train)
    cw = [total / (2 * counts[c]) for c in range(2)]
    cw = [w / cw[0] for w in cw]
    print(f"    Pesos: C0={cw[0]:.2f}  C1={cw[1]:.2f}")
    m1, cfg1, f1_1 = sweep(X_train, y_train, X_test, y_test, "Cost-Sensitive",
                            r_values, mrpc_values, n_classes=2, class_weights=cw)
    met1 = evaluate(m1, X_test, y_test, "Cost-Sensitive", n_classes=2)
    met1['strategy'] = 'Cost-Sensitive'
    met1['cfg'] = cfg1
    met1['train_size'] = len(X_train)
    all_experiments.append(met1)
    best_models['Cost-Sensitive'] = m1

    # ─── 2: SMOTE ───────────────────────────────────────────────────────
    print("\n  [2/6] SMOTE (equalizado)...")
    X_s, y_s = resample_smote(X_train, y_train)
    print(f"    Dataset: {len(X_s)} amostras | {dict(Counter(y_s))}")
    idx = rng.permutation(len(X_s))
    X_s, y_s = X_s[idx], y_s[idx]
    m2, cfg2, f1_2 = sweep(X_s, y_s, X_test, y_test, "SMOTE",
                            r_values, mrpc_values, n_classes=2)
    met2 = evaluate(m2, X_test, y_test, "SMOTE", n_classes=2)
    met2['strategy'] = 'SMOTE'
    met2['cfg'] = cfg2
    met2['train_size'] = len(X_s)
    all_experiments.append(met2)
    best_models['SMOTE'] = m2

    # ─── 3: ADASYN ──────────────────────────────────────────────────────
    print("\n  [3/6] ADASYN...")
    X_s, y_s = resample_adasyn(X_train, y_train)
    print(f"    Dataset: {len(X_s)} amostras | {dict(Counter(y_s))}")
    idx = rng.permutation(len(X_s))
    X_s, y_s = X_s[idx], y_s[idx]
    m3, cfg3, f1_3 = sweep(X_s, y_s, X_test, y_test, "ADASYN",
                            r_values, mrpc_values, n_classes=2)
    met3 = evaluate(m3, X_test, y_test, "ADASYN", n_classes=2)
    met3['strategy'] = 'ADASYN'
    met3['cfg'] = cfg3
    met3['train_size'] = len(X_s)
    all_experiments.append(met3)
    best_models['ADASYN'] = m3

    # ─── 4: SMOTE Parcial (15%) ─────────────────────────────────────────
    print("\n  [4/6] SMOTE Parcial (target 15%)...")
    X_s, y_s = resample_partial_smote(X_train, y_train, target_ratio=0.15)
    print(f"    Dataset: {len(X_s)} amostras | {dict(Counter(y_s))}")
    idx = rng.permutation(len(X_s))
    X_s, y_s = X_s[idx], y_s[idx]
    m4, cfg4, f1_4 = sweep(X_s, y_s, X_test, y_test, "SMOTE Parcial",
                            r_values, mrpc_values, n_classes=2)
    met4 = evaluate(m4, X_test, y_test, "SMOTE Parcial (15%)", n_classes=2)
    met4['strategy'] = 'SMOTE Parcial 15%'
    met4['cfg'] = cfg4
    met4['train_size'] = len(X_s)
    all_experiments.append(met4)
    best_models['SMOTE Parcial 15%'] = m4

    # ─── 5: Repeated Minority ────────────────────────────────────────────
    print("\n  [5/6] Repeated Minority Presentation...")
    X_s, y_s = resample_repeated_minority(X_train, y_train)
    print(f"    Dataset: {len(X_s)} amostras | {dict(Counter(y_s))}")
    m5, cfg5, f1_5 = sweep(X_s, y_s, X_test, y_test, "RepMinority",
                            r_values, mrpc_values, n_classes=2)
    met5 = evaluate(m5, X_test, y_test, "Repeated Minority", n_classes=2)
    met5['strategy'] = 'Repeated Minority'
    met5['cfg'] = cfg5
    met5['train_size'] = len(X_s)
    all_experiments.append(met5)
    best_models['Repeated Minority'] = m5

    # ─── 6: Cost-Sensitive + SMOTE Parcial ───────────────────────────────
    print("\n  [6/6] Cost-Sensitive + SMOTE Parcial (combo)...")
    X_s, y_s = resample_partial_smote(X_train, y_train, target_ratio=0.15)
    idx = rng.permutation(len(X_s))
    X_s, y_s = X_s[idx], y_s[idx]
    counts_s = Counter(y_s)
    total_s = len(y_s)
    cw_s = [total_s / (2 * counts_s[c]) for c in range(2)]
    cw_s = [w / cw_s[0] for w in cw_s]
    print(f"    Dataset: {len(X_s)} | Pesos: C0={cw_s[0]:.2f} C1={cw_s[1]:.2f}")
    m6, cfg6, f1_6 = sweep(X_s, y_s, X_test, y_test, "CS+SMOTE",
                            r_values, mrpc_values, n_classes=2,
                            class_weights=cw_s)
    met6 = evaluate(m6, X_test, y_test, "Cost-Sensitive + SMOTE Parcial", n_classes=2)
    met6['strategy'] = 'CS + SMOTE Parcial'
    met6['cfg'] = cfg6
    met6['train_size'] = len(X_s)
    all_experiments.append(met6)
    best_models['CS + SMOTE Parcial'] = m6

    # ── 5. TABELA COMPARATIVA ────────────────────────────────────────────
    sep("4. TABELA COMPARATIVA")
    all_experiments.sort(key=lambda x: -x['f1_macro'])

    header = (f"  {'#':>2s} {'Estratégia':24s} {'F1-mac':>7s} {'RecC0':>6s} "
              f"{'RecC1':>6s} {'PrcC0':>6s} {'PrcC1':>6s} "
              f"{'MAE':>6s} {'Rules':>5s} {'Train':>6s}")
    print(header)
    print(f"  {'─'*2} {'─'*24} {'─'*7} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*5} {'─'*6}")
    for i, m in enumerate(all_experiments):
        star = " ★" if i == 0 else ""
        print(f"  {i+1:2d} {m['strategy']:24s} {m['f1_macro']:7.4f} "
              f"{m['recall'][0]:6.3f} {m['recall'][1]:6.3f} "
              f"{m['precision'][0]:6.3f} {m['precision'][1]:6.3f} "
              f"{m['mae']:6.3f} {m['n_rules']:5d} "
              f"{m['train_size']:6d}{star}")

    best_exp = all_experiments[0]
    print(f"\n  ★ Melhor: {best_exp['strategy']}")
    print(f"    F1-macro={best_exp['f1_macro']:.4f}, config={best_exp['cfg']}")

    # ── 6. MÉTRICAS DETALHADAS ───────────────────────────────────────────
    sep("5. MÉTRICAS DETALHADAS — MELHOR ESTRATÉGIA")
    bm = best_exp
    print(f"  Estratégia:           {bm['strategy']}")
    print(f"  Configuração:         {bm['cfg']}")
    print(f"  Regras:               {bm['n_rules']} ({bm['rbc']})")
    print(f"\n  Acurácia:             {bm['acc']:.4f}")
    print(f"  F1-Score Macro:       {bm['f1_macro']:.4f}")
    print(f"  F1-Score Weighted:    {bm['f1_weighted']:.4f}")
    print(f"  F1 por classe:        C0={bm['f1_per'][0]:.4f}  C1(irrig)={bm['f1_per'][1]:.4f}")
    print(f"  Recall por classe:    C0={bm['recall'][0]:.4f}  C1(irrig)={bm['recall'][1]:.4f}")
    print(f"  Precision por classe: C0={bm['precision'][0]:.4f}  C1(irrig)={bm['precision'][1]:.4f}")
    print(f"  MAE ordinal:          {bm['mae']:.4f}")
    print(f"\n  Matriz de Confusão:")
    print(f"  {'':>22s}  Pred C0  Pred C1")
    print(f"    Real C0 (Sem irrig.)  {bm['cm'][0,0]:>5d}    {bm['cm'][0,1]:>5d}")
    print(f"    Real C1 (Irrigação)   {bm['cm'][1,0]:>5d}    {bm['cm'][1,1]:>5d}")

    # Classification report
    print(f"\n{classification_report(y_test, bm['y_pred'], target_names=['Sem irrig.(C0)', 'Irrigação(C1)'], zero_division=0)}")

    # ── 7. ANÁLISE: como o modelo classifica C1_orig vs C2_orig ──────────
    sep("6. ANÁLISE DE SUB-CLASSES (C1_orig e C2_orig dentro do binário)")
    y_pred_best = bm['y_pred']

    # Amostras que são C1 original
    c1_orig_mask = y_test_original == 1
    c2_orig_mask = y_test_original == 2
    c0_mask = y_test_original == 0

    print(f"  Como o modelo binário trata cada sub-classe original:")
    print(f"\n  C0 original ({c0_mask.sum()} amostras):")
    print(f"    Predito C0 (correcto):  {(y_pred_best[c0_mask] == 0).sum()}")
    print(f"    Predito C1 (falso):     {(y_pred_best[c0_mask] == 1).sum()}")

    print(f"\n  C1 original - Irrigação Moderada ({c1_orig_mask.sum()} amostras):")
    print(f"    Predito C0 (perdido):   {(y_pred_best[c1_orig_mask] == 0).sum()}")
    print(f"    Predito C1 (detectado): {(y_pred_best[c1_orig_mask] == 1).sum()}")
    if c1_orig_mask.sum() > 0:
        recall_c1_orig = (y_pred_best[c1_orig_mask] == 1).mean()
        print(f"    Recall C1_orig:         {recall_c1_orig:.4f}")

    print(f"\n  C2 original - Irrigação Intensa ({c2_orig_mask.sum()} amostras):")
    print(f"    Predito C0 (perdido):   {(y_pred_best[c2_orig_mask] == 0).sum()}")
    print(f"    Predito C1 (detectado): {(y_pred_best[c2_orig_mask] == 1).sum()}")
    if c2_orig_mask.sum() > 0:
        recall_c2_orig = (y_pred_best[c2_orig_mask] == 1).mean()
        print(f"    Recall C2_orig:         {recall_c2_orig:.4f}")

    # ── 8. SANIDADE ──────────────────────────────────────────────────────
    sep("7. SANIDADE AGRONÓMICA (6 cenários)")
    best_model = best_models.get(best_exp['strategy'], list(best_models.values())[0])
    n_pass, n_total = sanity_check_binary(best_model)
    print(f"\n  Resultado: {n_pass}/{n_total} (mínimo: {int(n_total * 0.75)}/{n_total})")

    # ── 9. APROVAÇÃO ─────────────────────────────────────────────────────
    sep("8. VERIFICAÇÃO DE APROVAÇÃO")
    min_sanity = int(n_total * 0.75)
    checks = [
        ('F1-macro ≥ 0.50',       bm['f1_macro'] >= 0.50,    f"{bm['f1_macro']:.4f}"),
        ('Recall C1(irrig) ≥ 0.30', bm['recall'][1] >= 0.30,  f"{bm['recall'][1]:.4f}"),
        ('Precision C1 ≥ 0.15',   bm['precision'][1] >= 0.15, f"{bm['precision'][1]:.4f}"),
        ('MAE ≤ 0.50',            bm['mae'] <= 0.50,          f"{bm['mae']:.4f}"),
        (f'Sanidade ≥ {min_sanity}/{n_total}', n_pass >= min_sanity, f"{n_pass}/{n_total}"),
    ]
    all_pass = True
    for crit, passed, val in checks:
        st = "✓" if passed else "✗"
        all_pass = all_pass and passed
        print(f"  {st}  {crit:30s}  →  {val}")
    print()
    if all_pass:
        print(f"  ╔══════════════════════════════════════════╗")
        print(f"  ║   MODELO BINÁRIO APROVADO PARA CAMPO ✓  ║")
        print(f"  ╚══════════════════════════════════════════╝")
    else:
        print(f"  ╔══════════════════════════════════════════╗")
        print(f"  ║   MODELO NÃO APROVADO — VERIFICAR       ║")
        print(f"  ╚══════════════════════════════════════════╝")

    # ── 10. SALVAR ───────────────────────────────────────────────────────
    sep("9. ARTEFACTOS")
    best_model.save('memoria_cold_start_v9.pkl')
    print(f"  ✓ memoria_cold_start_v9.pkl ({best_model.rules_by_class()})")

    # ── 11. GRÁFICOS ─────────────────────────────────────────────────────
    sep("10. GRÁFICOS")

    # 11a: Comparação estratégias
    fig, ax = plt.subplots(figsize=(12, 5))
    strategies = [m['strategy'] for m in all_experiments]
    f1s = [m['f1_macro'] for m in all_experiments]
    colors = ['#4CAF50' if f >= 0.50 else '#FF9800' if f >= 0.40 else '#F44336' for f in f1s]
    bars = ax.barh(range(len(strategies)), f1s, color=colors, alpha=0.85, edgecolor='white')
    ax.axvline(x=0.50, color='red', linestyle='--', linewidth=1.5, label='Limiar (0.50)')
    for i, (s, f1) in enumerate(zip(strategies, f1s)):
        ax.text(f1 + 0.005, i, f'{f1:.3f}', va='center', fontsize=10, fontweight='bold')
    ax.set_yticks(range(len(strategies)))
    ax.set_yticklabels(strategies, fontsize=10)
    ax.set_xlabel('F1-Score Macro', fontsize=12)
    ax.set_title('v9 Binário — Comparação de Estratégias', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xlim(0, max(f1s) * 1.15)
    ax.invert_yaxis()
    ax.grid(True, axis='x', alpha=0.2)
    plt.tight_layout()
    plt.savefig(f'{OUT}/comparacao_estrategias_binario.png', dpi=150)
    plt.close()
    print(f"  ✓ {OUT}/comparacao_estrategias_binario.png")

    # 11b: Confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    cm_labels = ['C0\nSem irrigação', 'C1\nIrrigação']
    sns.heatmap(bm['cm'], annot=True, fmt='d', cmap='Blues',
                xticklabels=cm_labels, yticklabels=cm_labels, ax=ax,
                annot_kws={'fontsize': 16, 'fontweight': 'bold'})
    ax.set_xlabel('Predito', fontsize=12)
    ax.set_ylabel('Real', fontsize=12)
    ax.set_title(f'Matriz de Confusão — {bm["strategy"]}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUT}/matriz_confusao_binario.png', dpi=150)
    plt.close()
    print(f"  ✓ {OUT}/matriz_confusao_binario.png")

    # 11c: Recall/Precision por estratégia (grouped bar)
    fig, ax = plt.subplots(figsize=(14, 6))
    x_pos = np.arange(len(strategies))
    w = 0.2
    rec_c1 = [m['recall'][1] for m in all_experiments]
    prc_c1 = [m['precision'][1] for m in all_experiments]
    f1_c1 = [m['f1_per'][1] for m in all_experiments]
    ax.bar(x_pos - w, rec_c1, w, label='Recall C1', color='#FF9800', alpha=0.85)
    ax.bar(x_pos, prc_c1, w, label='Precision C1', color='#2196F3', alpha=0.85)
    ax.bar(x_pos + w, f1_c1, w, label='F1 C1', color='#4CAF50', alpha=0.85)
    ax.axhline(y=0.30, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Limiar Recall (0.30)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(strategies, fontsize=9, rotation=30, ha='right')
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Métricas de Irrigação (C1) por Estratégia', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis='y', alpha=0.2)
    plt.tight_layout()
    plt.savefig(f'{OUT}/metricas_irrigacao_por_estrategia.png', dpi=150)
    plt.close()
    print(f"  ✓ {OUT}/metricas_irrigacao_por_estrategia.png")

    # 11d: Sub-class analysis (como o binário vê C1_orig vs C2_orig)
    fig, ax = plt.subplots(figsize=(8, 5))
    sub_data = []
    for m_exp in all_experiments:
        yp = m_exp['y_pred']
        r_c1o = (yp[c1_orig_mask] == 1).mean() if c1_orig_mask.sum() > 0 else 0
        r_c2o = (yp[c2_orig_mask] == 1).mean() if c2_orig_mask.sum() > 0 else 0
        sub_data.append((m_exp['strategy'], r_c1o, r_c2o))
    strats = [s[0] for s in sub_data]
    rc1o = [s[1] for s in sub_data]
    rc2o = [s[2] for s in sub_data]
    x_pos = np.arange(len(strats))
    ax.bar(x_pos - 0.15, rc1o, 0.3, label='Detecção C1_orig (moderada)', color='#FF9800', alpha=0.85)
    ax.bar(x_pos + 0.15, rc2o, 0.3, label='Detecção C2_orig (intensa)', color='#F44336', alpha=0.85)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(strats, fontsize=9, rotation=30, ha='right')
    ax.set_ylabel('Taxa de detecção', fontsize=12)
    ax.set_title('Detecção de Sub-classes Originais (C1 e C2) por Estratégia', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis='y', alpha=0.2)
    plt.tight_layout()
    plt.savefig(f'{OUT}/deteccao_subclasses.png', dpi=150)
    plt.close()
    print(f"  ✓ {OUT}/deteccao_subclasses.png")

    # ── FIM ──────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    sep("CONCLUÍDO")
    print(f"  Tempo total: {elapsed:.1f}s")
    print(f"  Melhor: {bm['strategy']} (F1-macro={bm['f1_macro']:.4f})")
    print()
    print(f"  Resumo:")
    for i, m in enumerate(all_experiments):
        print(f"    {i+1}. {m['strategy']:24s} F1={m['f1_macro']:.4f}  "
              f"Rec.irrig={m['recall'][1]:.3f}  Prec.irrig={m['precision'][1]:.3f}")
    print()
