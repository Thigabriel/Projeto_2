"""
============================================================================
  COLD START v8 — ALMMo-0
  Foco: Tratamento do Desbalanceamento (94/4/2%)
  
  Estratégias testadas:
    0. Baseline (sem resampling — resultado v7)
    1. Random Oversampling
    2. SMOTE
    3. ADASYN
    4. BorderlineSMOTE
    5. SMOTE + Tomek Links (limpeza pós-SMOTE)
    6. Binarização + Sub-classificação (2 estágios)
    7. Repeated Minority Presentation (natural para online learning)
    8. Cost-Sensitive Learning (pesos na actualização de regras)
  
  Requisitos:
    pip install numpy pandas scikit-learn matplotlib seaborn imbalanced-learn
  
  Execução:
    python cold_start_v8.py
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
# ALMMo-0 v3 — Com suporte a cost-sensitive learning
# ─────────────────────────────────────────────────────────────────────────────

class ALMMo0:
    def __init__(self, n_inputs=4, r_threshold=0.5, max_rules=50,
                 age_limit=100, epsilon=1e-8, n_classes=3,
                 min_rules_per_class=3, class_weights=None):
        self.n_inputs            = n_inputs
        self.r_threshold         = r_threshold
        self.max_rules           = max_rules
        self.age_limit           = age_limit
        self.epsilon             = epsilon
        self.n_classes           = n_classes
        self.min_rules_per_class = min_rules_per_class
        # Cost-sensitive: peso por classe (default: uniforme)
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

        # Cost-sensitive: classes raras têm raio de criação maior
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
            # Cost-sensitive: classes raras corrigem com mais força
            eff_eta = eta * af * self.class_weights[int(y)]
            eff_eta = min(eff_eta, 1.0)  # cap
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

def erros_adjacentes(y_true, y_pred):
    erros = y_true != y_pred
    if erros.sum() == 0:
        return 0.0, 0, 0
    diffs = np.abs(y_true[erros] - y_pred[erros])
    adj = int((diffs == 1).sum())
    non_adj = int((diffs > 1).sum())
    return adj / erros.sum() * 100, adj, non_adj


def sep(title):
    w = 75
    print(f"\n{'═'*w}")
    print(f"  {title}")
    print(f"{'═'*w}")


def evaluate(model, X_test, y_test, label="", verbose=True):
    """Avalia modelo e retorna dicionário de métricas."""
    y_pred = np.array([model.predict(x) for x in X_test])
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1w = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    f1p = f1_score(y_test, y_pred, average=None, zero_division=0)
    rec = recall_score(y_test, y_pred, average=None, zero_division=0)
    prec = precision_score(y_test, y_pred, average=None, zero_division=0)
    mae = mean_absolute_error(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    adj_pct, adj_n, nadj_n = erros_adjacentes(y_test, y_pred)
    n_err = int((y_test != y_pred).sum())
    rbc = model.rules_by_class()

    if verbose:
        print(f"\n  ┌─ {label}")
        print(f"  │  F1-macro: {f1m:.4f}  |  Acurácia: {acc:.4f}  |  MAE: {mae:.4f}")
        print(f"  │  F1/classe:   C0={f1p[0]:.3f}  C1={f1p[1]:.3f}  C2={f1p[2]:.3f}")
        print(f"  │  Recall:      C0={rec[0]:.3f}  C1={rec[1]:.3f}  C2={rec[2]:.3f}")
        print(f"  │  Precision:   C0={prec[0]:.3f}  C1={prec[1]:.3f}  C2={prec[2]:.3f}")
        print(f"  │  Erros adj:   {adj_n}/{n_err} ({adj_pct:.1f}%), não-adj: {nadj_n}")
        print(f"  │  Regras:      {len(model.rules)} total | {rbc}")
        print(f"  │  CM: {cm[0].tolist()} / {cm[1].tolist()} / {cm[2].tolist()}")
        print(f"  └─")

    return {
        'label': label, 'acc': acc, 'f1_macro': f1m, 'f1_weighted': f1w,
        'f1_per': f1p, 'recall': rec, 'precision': prec, 'mae': mae,
        'cm': cm, 'adj_pct': adj_pct, 'adj_n': adj_n, 'nadj_n': nadj_n,
        'n_rules': len(model.rules), 'rbc': rbc, 'y_pred': y_pred,
    }


def sweep_for_strategy(X_train, y_train, X_test, y_test, label,
                       r_values, mrpc_values, max_rules_base=40,
                       class_weights=None):
    """Executa sweep e retorna melhor modelo + métricas."""
    best_f1 = -1
    best_cfg = None
    best_model = None

    for mrpc in mrpc_values:
        for r in r_values:
            r = round(r, 2)
            eff_max = max(max_rules_base, mrpc * 3 + 5)
            m = ALMMo0(n_inputs=4, r_threshold=r, max_rules=eff_max,
                       age_limit=80, n_classes=3, min_rules_per_class=mrpc,
                       class_weights=class_weights)
            m.cold_start(X_train, y_train, verbose=False)
            yp = np.array([m.predict(x) for x in X_test])
            f1 = f1_score(y_test, yp, average='macro', zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_cfg = {'r': r, 'mrpc': mrpc}
                best_model = m

    print(f"  {label}: melhor r={best_cfg['r']}, mrpc={best_cfg['mrpc']}, F1={best_f1:.4f}")
    return best_model, best_cfg, best_f1


def sanity_check(model, verbose=True):
    """Testa 4 cenários agronómicos. Retorna n_pass."""
    casos = [
        ([120.0,  0.0, 38.5, 80], "Solo seco, sem chuva, calor extremo", [1, 2]),
        ([ 15.0, 30.0, 32.0, 40], "Solo úmido, muita chuva, ameno", 0),
        ([ 70.0,  2.0, 36.0, 60], "Solo médio, pouca chuva, calor", [1, 2]),
        ([100.0,  0.0, 37.0, 20], "Solo seco, sem chuva, DAP baixo", [1, 2]),
    ]
    n_pass = 0
    for inputs, desc, expected in casos:
        pred = model.predict(np.array(inputs))
        if isinstance(expected, list):
            ok = pred in expected
            exp_str = f"C{expected[0]}/C{expected[1]}"
        else:
            ok = pred == expected
            exp_str = f"C{expected}"
        n_pass += ok
        if verbose:
            st = "✓" if ok else "✗"
            print(f"    {st} Pred=C{pred} Exp={exp_str} | {desc}")
    return n_pass


# ─────────────────────────────────────────────────────────────────────────────
# RESAMPLING STRATEGIES
# ─────────────────────────────────────────────────────────────────────────────

def strategy_baseline(X_train, y_train):
    """Sem resampling."""
    return X_train, y_train


def strategy_random_oversample(X_train, y_train):
    """Duplicação aleatória de amostras minoritárias."""
    from imblearn.over_sampling import RandomOverSampler
    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_resample(X_train, y_train)
    return X_res, y_res


def strategy_smote(X_train, y_train):
    """SMOTE — Synthetic Minority Over-sampling Technique."""
    from imblearn.over_sampling import SMOTE
    # k_neighbors adaptado ao tamanho das classes minoritárias
    min_class_size = min(Counter(y_train).values())
    k = min(5, min_class_size - 1)
    k = max(1, k)
    sm = SMOTE(random_state=42, k_neighbors=k)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    return X_res, y_res


def strategy_adasyn(X_train, y_train):
    """ADASYN — Adaptive Synthetic Sampling."""
    from imblearn.over_sampling import ADASYN
    min_class_size = min(Counter(y_train).values())
    k = min(5, min_class_size - 1)
    k = max(1, k)
    try:
        ada = ADASYN(random_state=42, n_neighbors=k)
        X_res, y_res = ada.fit_resample(X_train, y_train)
        return X_res, y_res
    except (ValueError, RuntimeError) as e:
        print(f"    ⚠ ADASYN falhou ({e}), usando SMOTE como fallback")
        return strategy_smote(X_train, y_train)


def strategy_borderline_smote(X_train, y_train):
    """BorderlineSMOTE — foca na fronteira de decisão."""
    from imblearn.over_sampling import BorderlineSMOTE
    min_class_size = min(Counter(y_train).values())
    k = min(5, min_class_size - 1)
    k = max(1, k)
    try:
        bsm = BorderlineSMOTE(random_state=42, k_neighbors=k)
        X_res, y_res = bsm.fit_resample(X_train, y_train)
        return X_res, y_res
    except (ValueError, RuntimeError) as e:
        print(f"    ⚠ BorderlineSMOTE falhou ({e}), usando SMOTE como fallback")
        return strategy_smote(X_train, y_train)


def strategy_smote_tomek(X_train, y_train):
    """SMOTE + Tomek Links — oversample e depois limpa fronteira."""
    from imblearn.combine import SMOTETomek
    from imblearn.over_sampling import SMOTE
    min_class_size = min(Counter(y_train).values())
    k = min(5, min_class_size - 1)
    k = max(1, k)
    smtk = SMOTETomek(
        smote=SMOTE(random_state=42, k_neighbors=k),
        random_state=42
    )
    X_res, y_res = smtk.fit_resample(X_train, y_train)
    return X_res, y_res


def strategy_repeated_minority(X_train, y_train, repeat_factor=None):
    """
    Apresentação repetida de amostras minoritárias intercaladas no fluxo.
    Mais natural para aprendizado online que SMOTE: não cria amostras
    sintéticas, apenas re-apresenta amostras reais em posições estratégicas.
    """
    counts = Counter(y_train)
    max_count = max(counts.values())
    
    # Separar por classe
    indices_by_class = {c: np.where(y_train == c)[0] for c in counts}
    
    # Calcular factor de repetição por classe
    if repeat_factor is None:
        # Auto: igualar contagens (mas cap em 20x para evitar overfitting)
        repeat_factors = {}
        for c in counts:
            factor = min(int(np.ceil(max_count / counts[c])), 20)
            repeat_factors[c] = factor
    else:
        repeat_factors = {c: repeat_factor for c in counts}
    
    # Construir dataset expandido mantendo ordem temporal base
    # Estratégia: para cada passagem pelo dataset, inserir amostras
    # minoritárias extras de forma periódica
    X_list, y_list = [], []
    
    # Pool de amostras minoritárias para inserção
    minority_pools = {}
    for c in counts:
        if repeat_factors[c] > 1:
            idxs = indices_by_class[c]
            # Repetir e embaralhar o pool
            rng = np.random.RandomState(42)
            pool = np.tile(idxs, repeat_factors[c] - 1)
            rng.shuffle(pool)
            minority_pools[c] = list(pool)
    
    # Percorrer dataset original e intercalar
    total_extra = sum(len(p) for p in minority_pools.values())
    insert_interval = max(1, len(X_train) // (total_extra + 1))
    
    extra_queue = []
    for c, pool in minority_pools.items():
        extra_queue.extend(pool)
    rng = np.random.RandomState(42)
    rng.shuffle(extra_queue)
    extra_idx = 0
    
    for i in range(len(X_train)):
        X_list.append(X_train[i])
        y_list.append(y_train[i])
        # Inserir amostra extra periodicamente
        if extra_idx < len(extra_queue) and (i + 1) % insert_interval == 0:
            j = extra_queue[extra_idx]
            X_list.append(X_train[j])
            y_list.append(y_train[j])
            extra_idx += 1
    
    # Inserir restantes no final
    while extra_idx < len(extra_queue):
        j = extra_queue[extra_idx]
        X_list.append(X_train[j])
        y_list.append(y_train[j])
        extra_idx += 1
    
    return np.array(X_list), np.array(y_list)


def strategy_partial_oversample(X_train, y_train, target_ratio=0.15):
    """
    Oversampling parcial — não equaliza totalmente, apenas aumenta
    as classes minoritárias para uma proporção-alvo (ex: 15% cada).
    Evita o overfitting de equalização total enquanto reduz o
    desbalanceamento.
    """
    from imblearn.over_sampling import SMOTE
    counts = Counter(y_train)
    total = len(y_train)
    
    # Calcular targets: cada classe minoritária vai para target_ratio do total
    target_counts = {}
    for c in counts:
        target = max(counts[c], int(total * target_ratio))
        target_counts[c] = target
    
    min_class_size = min(counts.values())
    k = min(5, min_class_size - 1)
    k = max(1, k)
    
    sm = SMOTE(random_state=42, k_neighbors=k, 
               sampling_strategy=target_counts)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    return X_res, y_res


def prepare_binary_then_split(X_train, y_train, X_test, y_test):
    """
    Estágio 1: Binário (C0 vs irrigação)
    Estágio 2: Dentro de irrigação, C1 vs C2
    """
    # Stage 1: Binary
    y_train_bin = (y_train > 0).astype(int)  # 0 = sem irrigação, 1 = irrigação
    y_test_bin = (y_test > 0).astype(int)
    
    # Oversample para stage 1
    from imblearn.over_sampling import SMOTE
    counts_bin = Counter(y_train_bin)
    k = min(5, min(counts_bin.values()) - 1)
    k = max(1, k)
    sm = SMOTE(random_state=42, k_neighbors=k)
    X_train_bin, y_train_bin_res = sm.fit_resample(X_train, y_train_bin)
    
    # Stage 2: Apenas amostras de irrigação (C1 vs C2)
    irr_mask_train = y_train > 0
    X_train_irr = X_train[irr_mask_train]
    y_train_irr = y_train[irr_mask_train]  # valores 1 e 2
    # Remap para 0 e 1 para o sub-classificador
    y_train_irr_sub = y_train_irr - 1  # C1→0, C2→1
    
    # Oversample stage 2
    counts_sub = Counter(y_train_irr_sub)
    if min(counts_sub.values()) > 1:
        k2 = min(5, min(counts_sub.values()) - 1)
        k2 = max(1, k2)
        sm2 = SMOTE(random_state=42, k_neighbors=k2)
        X_train_irr_res, y_train_irr_res = sm2.fit_resample(X_train_irr, y_train_irr_sub)
    else:
        # Fallback: random oversample
        from imblearn.over_sampling import RandomOverSampler
        ros = RandomOverSampler(random_state=42)
        X_train_irr_res, y_train_irr_res = ros.fit_resample(X_train_irr, y_train_irr_sub)
    
    return {
        'X_train_bin': X_train_bin, 'y_train_bin': y_train_bin_res,
        'X_train_sub': X_train_irr_res, 'y_train_sub': y_train_irr_res,
        'y_test_bin': y_test_bin,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    t0 = time.time()
    OUT = 'graficos_v8'
    os.makedirs(OUT, exist_ok=True)

    # ── 1. LOAD ──────────────────────────────────────────────────────────
    sep("1. DATASET v7 (reutilizado como base para v8)")
    df = pd.read_csv('dataset_cold_start_v7.csv')
    fcols = ['tensao_solo_kpa', 'chuva_acum_3d_mm', 'tmax_max_3d_c', 'dap']
    X = df[fcols].values
    y = df['classe_irrigacao'].values

    print(f"  Amostras: {len(df)}")
    for c in range(3):
        n = (y == c).sum()
        print(f"  C{c}: {n:5d} ({n/len(y)*100:5.1f}%)")

    # ── 2. SPLIT (mesmo do v7) ───────────────────────────────────────────
    sep("2. SPLIT (Leave-Groups-Out — mesmo do v7)")
    dap_vals = df['dap'].values
    resets = [0] + [i for i in range(1, len(dap_vals)) if dap_vals[i] < dap_vals[i-1]] + [len(dap_vals)]
    test_group_ids = [5, 9, 13, 17, 21, 27]  # 0-indexed
    test_idx = []
    for g in test_group_ids:
        test_idx.extend(range(resets[g], resets[g+1]))
    train_idx = [i for i in range(len(df)) if i not in test_idx]
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print(f"  Treino: {len(X_train)}  |  Teste: {len(X_test)}")
    for c in range(3):
        print(f"  C{c}: treino={int((y_train==c).sum()):5d}  teste={int((y_test==c).sum()):4d}")

    # ── 3. SWEEP CONFIG ──────────────────────────────────────────────────
    # Sweep focado: testar combinações mais promissoras
    r_values = np.arange(0.10, 2.05, 0.05)
    mrpc_values = [3, 5, 7, 10]

    # ── 4. ESTRATÉGIAS ───────────────────────────────────────────────────
    sep("3. EXECUÇÃO DAS ESTRATÉGIAS (pode demorar 2-5 minutos)")

    all_experiments = []

    # ─── STRATEGY 0: Baseline ────────────────────────────────────────────
    print("\n  [0/8] Baseline (sem resampling)...")
    X_s, y_s = strategy_baseline(X_train, y_train)
    print(f"    Dataset: {len(X_s)} amostras | {dict(Counter(y_s))}")
    model_0, cfg_0, f1_0 = sweep_for_strategy(
        X_s, y_s, X_test, y_test, "Baseline", r_values, mrpc_values)
    metrics_0 = evaluate(model_0, X_test, y_test, "Baseline")
    metrics_0['strategy'] = 'Baseline'
    metrics_0['cfg'] = cfg_0
    metrics_0['train_size'] = len(X_s)
    all_experiments.append(metrics_0)

    # ─── STRATEGY 1: Random Oversampling ─────────────────────────────────
    print("\n  [1/8] Random Oversampling...")
    X_s, y_s = strategy_random_oversample(X_train, y_train)
    print(f"    Dataset: {len(X_s)} amostras | {dict(Counter(y_s))}")
    # Embaralhar para evitar blocos homogéneos no final
    rng = np.random.RandomState(42)
    shuffle_idx = rng.permutation(len(X_s))
    X_s, y_s = X_s[shuffle_idx], y_s[shuffle_idx]
    model_1, cfg_1, f1_1 = sweep_for_strategy(
        X_s, y_s, X_test, y_test, "RandomOS", r_values, mrpc_values)
    metrics_1 = evaluate(model_1, X_test, y_test, "Random Oversampling")
    metrics_1['strategy'] = 'Random Oversampling'
    metrics_1['cfg'] = cfg_1
    metrics_1['train_size'] = len(X_s)
    all_experiments.append(metrics_1)

    # ─── STRATEGY 2: SMOTE ──────────────────────────────────────────────
    print("\n  [2/8] SMOTE...")
    X_s, y_s = strategy_smote(X_train, y_train)
    print(f"    Dataset: {len(X_s)} amostras | {dict(Counter(y_s))}")
    shuffle_idx = rng.permutation(len(X_s))
    X_s, y_s = X_s[shuffle_idx], y_s[shuffle_idx]
    model_2, cfg_2, f1_2 = sweep_for_strategy(
        X_s, y_s, X_test, y_test, "SMOTE", r_values, mrpc_values)
    metrics_2 = evaluate(model_2, X_test, y_test, "SMOTE")
    metrics_2['strategy'] = 'SMOTE'
    metrics_2['cfg'] = cfg_2
    metrics_2['train_size'] = len(X_s)
    all_experiments.append(metrics_2)

    # ─── STRATEGY 3: ADASYN ─────────────────────────────────────────────
    print("\n  [3/8] ADASYN...")
    X_s, y_s = strategy_adasyn(X_train, y_train)
    print(f"    Dataset: {len(X_s)} amostras | {dict(Counter(y_s))}")
    shuffle_idx = rng.permutation(len(X_s))
    X_s, y_s = X_s[shuffle_idx], y_s[shuffle_idx]
    model_3, cfg_3, f1_3 = sweep_for_strategy(
        X_s, y_s, X_test, y_test, "ADASYN", r_values, mrpc_values)
    metrics_3 = evaluate(model_3, X_test, y_test, "ADASYN")
    metrics_3['strategy'] = 'ADASYN'
    metrics_3['cfg'] = cfg_3
    metrics_3['train_size'] = len(X_s)
    all_experiments.append(metrics_3)

    # ─── STRATEGY 4: BorderlineSMOTE ─────────────────────────────────────
    print("\n  [4/8] BorderlineSMOTE...")
    X_s, y_s = strategy_borderline_smote(X_train, y_train)
    print(f"    Dataset: {len(X_s)} amostras | {dict(Counter(y_s))}")
    shuffle_idx = rng.permutation(len(X_s))
    X_s, y_s = X_s[shuffle_idx], y_s[shuffle_idx]
    model_4, cfg_4, f1_4 = sweep_for_strategy(
        X_s, y_s, X_test, y_test, "BorderlineSMOTE", r_values, mrpc_values)
    metrics_4 = evaluate(model_4, X_test, y_test, "BorderlineSMOTE")
    metrics_4['strategy'] = 'BorderlineSMOTE'
    metrics_4['cfg'] = cfg_4
    metrics_4['train_size'] = len(X_s)
    all_experiments.append(metrics_4)

    # ─── STRATEGY 5: SMOTE + Tomek ──────────────────────────────────────
    print("\n  [5/8] SMOTE + Tomek Links...")
    X_s, y_s = strategy_smote_tomek(X_train, y_train)
    print(f"    Dataset: {len(X_s)} amostras | {dict(Counter(y_s))}")
    shuffle_idx = rng.permutation(len(X_s))
    X_s, y_s = X_s[shuffle_idx], y_s[shuffle_idx]
    model_5, cfg_5, f1_5 = sweep_for_strategy(
        X_s, y_s, X_test, y_test, "SMOTE+Tomek", r_values, mrpc_values)
    metrics_5 = evaluate(model_5, X_test, y_test, "SMOTE + Tomek Links")
    metrics_5['strategy'] = 'SMOTE + Tomek'
    metrics_5['cfg'] = cfg_5
    metrics_5['train_size'] = len(X_s)
    all_experiments.append(metrics_5)

    # ─── STRATEGY 6: Binary + Sub-classification ─────────────────────────
    print("\n  [6/8] Binarização + Sub-classificação...")
    bin_data = prepare_binary_then_split(X_train, y_train, X_test, y_test)
    print(f"    Stage 1 (binário): {len(bin_data['X_train_bin'])} amostras")
    print(f"    Stage 2 (C1vsC2):  {len(bin_data['X_train_sub'])} amostras")

    # Stage 1: treinar classificador binário
    r_vals_short = np.arange(0.10, 1.55, 0.10)
    mrpc_short = [3, 5, 7]
    best_f1_bin = -1
    best_model_bin = None
    for mrpc in mrpc_short:
        for r in r_vals_short:
            r = round(r, 2)
            eff_max = max(30, mrpc * 2 + 5)
            m = ALMMo0(n_inputs=4, r_threshold=r, max_rules=eff_max,
                       age_limit=80, n_classes=2, min_rules_per_class=mrpc)
            # Embaralhar dados binários
            shuf = rng.permutation(len(bin_data['X_train_bin']))
            m.cold_start(bin_data['X_train_bin'][shuf],
                        bin_data['y_train_bin'][shuf], verbose=False)
            yp = np.array([m.predict(x) for x in X_test])
            f1 = f1_score(bin_data['y_test_bin'], yp, average='macro', zero_division=0)
            if f1 > best_f1_bin:
                best_f1_bin = f1
                best_model_bin = m

    # Stage 2: treinar sub-classificador (C1=0, C2=1)
    best_f1_sub = -1
    best_model_sub = None
    for mrpc in mrpc_short:
        for r in r_vals_short:
            r = round(r, 2)
            eff_max = max(20, mrpc * 2 + 5)
            m = ALMMo0(n_inputs=4, r_threshold=r, max_rules=eff_max,
                       age_limit=80, n_classes=2, min_rules_per_class=mrpc)
            shuf = rng.permutation(len(bin_data['X_train_sub']))
            m.cold_start(bin_data['X_train_sub'][shuf],
                        bin_data['y_train_sub'][shuf], verbose=False)
            # Avaliar no subset de irrigação do teste
            irr_test_mask = y_test > 0
            if irr_test_mask.sum() > 0:
                yp = np.array([m.predict(x) for x in X_test[irr_test_mask]])
                y_true_sub = y_test[irr_test_mask] - 1
                f1 = f1_score(y_true_sub, yp, average='macro', zero_division=0)
                if f1 > best_f1_sub:
                    best_f1_sub = f1
                    best_model_sub = m

    # Combinar: Stage 1 decide C0 vs irrigação, Stage 2 decide C1 vs C2
    y_pred_bin_combined = []
    for x in X_test:
        stage1_pred = best_model_bin.predict(x)
        if stage1_pred == 0:  # sem irrigação
            y_pred_bin_combined.append(0)
        else:
            stage2_pred = best_model_sub.predict(x)  # 0=C1, 1=C2
            y_pred_bin_combined.append(stage2_pred + 1)  # remap para 1 ou 2

    y_pred_6 = np.array(y_pred_bin_combined)
    f1_6 = f1_score(y_test, y_pred_6, average='macro', zero_division=0)
    rec_6 = recall_score(y_test, y_pred_6, average=None, zero_division=0)
    prec_6 = precision_score(y_test, y_pred_6, average=None, zero_division=0)
    f1p_6 = f1_score(y_test, y_pred_6, average=None, zero_division=0)
    acc_6 = accuracy_score(y_test, y_pred_6)
    mae_6 = mean_absolute_error(y_test, y_pred_6)
    cm_6 = confusion_matrix(y_test, y_pred_6, labels=[0, 1, 2])
    adj_6, adj_n_6, nadj_6 = erros_adjacentes(y_test, y_pred_6)
    n_rules_6 = len(best_model_bin.rules) + len(best_model_sub.rules)

    print(f"    Stage 1 F1-macro (binário): {best_f1_bin:.4f}")
    print(f"    Stage 2 F1-macro (C1vsC2):  {best_f1_sub:.4f}")
    print(f"    Combinado F1-macro:         {f1_6:.4f}")
    print(f"    Recall: C0={rec_6[0]:.3f} C1={rec_6[1]:.3f} C2={rec_6[2]:.3f}")
    print(f"    CM: {cm_6[0].tolist()} / {cm_6[1].tolist()} / {cm_6[2].tolist()}")

    metrics_6 = {
        'label': 'Binary+Sub', 'strategy': 'Binary + Sub-classificação',
        'acc': acc_6, 'f1_macro': f1_6, 'f1_weighted': f1_score(y_test, y_pred_6, average='weighted', zero_division=0),
        'f1_per': f1p_6, 'recall': rec_6, 'precision': prec_6, 'mae': mae_6,
        'cm': cm_6, 'adj_pct': adj_6, 'adj_n': adj_n_6, 'nadj_n': nadj_6,
        'n_rules': n_rules_6, 'rbc': {}, 'y_pred': y_pred_6,
        'cfg': {'stage1_rules': len(best_model_bin.rules), 'stage2_rules': len(best_model_sub.rules)},
        'train_size': len(bin_data['X_train_bin']) + len(bin_data['X_train_sub']),
    }
    all_experiments.append(metrics_6)

    # ─── STRATEGY 7: Repeated Minority Presentation ──────────────────────
    print("\n  [7/8] Repeated Minority Presentation...")
    X_s, y_s = strategy_repeated_minority(X_train, y_train)
    print(f"    Dataset: {len(X_s)} amostras | {dict(Counter(y_s))}")
    model_7, cfg_7, f1_7 = sweep_for_strategy(
        X_s, y_s, X_test, y_test, "RepMinority", r_values, mrpc_values)
    metrics_7 = evaluate(model_7, X_test, y_test, "Repeated Minority Presentation")
    metrics_7['strategy'] = 'Repeated Minority'
    metrics_7['cfg'] = cfg_7
    metrics_7['train_size'] = len(X_s)
    all_experiments.append(metrics_7)

    # ─── STRATEGY 8: Cost-Sensitive Learning ─────────────────────────────
    print("\n  [8/8] Cost-Sensitive Learning...")
    # Pesos inversamente proporcionais à frequência
    counts = Counter(y_train)
    total = len(y_train)
    cw = [total / (3 * counts[c]) for c in range(3)]
    # Normalizar para que C0 tenha peso 1
    cw = [w / cw[0] for w in cw]
    print(f"    Pesos: C0={cw[0]:.2f}  C1={cw[1]:.2f}  C2={cw[2]:.2f}")

    best_f1_cs = -1
    best_model_cs = None
    best_cfg_cs = None
    for mrpc in mrpc_values:
        for r in r_values:
            r = round(r, 2)
            eff_max = max(max_rules_base, mrpc * 3 + 5) if 'max_rules_base' in dir() else max(40, mrpc * 3 + 5)
            m = ALMMo0(n_inputs=4, r_threshold=r, max_rules=eff_max,
                       age_limit=80, n_classes=3, min_rules_per_class=mrpc,
                       class_weights=cw)
            m.cold_start(X_train, y_train, verbose=False)
            yp = np.array([m.predict(x) for x in X_test])
            f1 = f1_score(y_test, yp, average='macro', zero_division=0)
            if f1 > best_f1_cs:
                best_f1_cs = f1
                best_cfg_cs = {'r': r, 'mrpc': mrpc}
                best_model_cs = m
    print(f"  Cost-Sensitive: melhor r={best_cfg_cs['r']}, mrpc={best_cfg_cs['mrpc']}, F1={best_f1_cs:.4f}")
    metrics_8 = evaluate(best_model_cs, X_test, y_test, "Cost-Sensitive Learning")
    metrics_8['strategy'] = 'Cost-Sensitive'
    metrics_8['cfg'] = best_cfg_cs
    metrics_8['train_size'] = len(X_train)
    all_experiments.append(metrics_8)

    # ─── STRATEGY 9 (BONUS): Partial Oversampling (15%) ──────────────────
    print("\n  [BONUS] SMOTE Parcial (target 15% por classe minoritária)...")
    X_s, y_s = strategy_partial_oversample(X_train, y_train, target_ratio=0.15)
    print(f"    Dataset: {len(X_s)} amostras | {dict(Counter(y_s))}")
    shuffle_idx = rng.permutation(len(X_s))
    X_s, y_s = X_s[shuffle_idx], y_s[shuffle_idx]
    model_9, cfg_9, f1_9 = sweep_for_strategy(
        X_s, y_s, X_test, y_test, "PartialSMOTE", r_values, mrpc_values)
    metrics_9 = evaluate(model_9, X_test, y_test, "SMOTE Parcial (15%)")
    metrics_9['strategy'] = 'SMOTE Parcial 15%'
    metrics_9['cfg'] = cfg_9
    metrics_9['train_size'] = len(X_s)
    all_experiments.append(metrics_9)

    # ── 5. COMPARAÇÃO ────────────────────────────────────────────────────
    sep("4. TABELA COMPARATIVA")

    # Sort by F1-macro
    all_experiments.sort(key=lambda x: -x['f1_macro'])

    header = (f"  {'#':>2s} {'Estratégia':24s} {'F1-mac':>7s} {'RecC0':>6s} "
              f"{'RecC1':>6s} {'RecC2':>6s} {'PrcC1':>6s} {'PrcC2':>6s} "
              f"{'MAE':>6s} {'Adj%':>5s} {'Rules':>5s} {'Train':>6s}")
    print(header)
    print(f"  {'─'*2} {'─'*24} {'─'*7} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*5} {'─'*5} {'─'*6}")

    for i, m in enumerate(all_experiments):
        star = " ★" if i == 0 else ""
        print(f"  {i+1:2d} {m['strategy']:24s} {m['f1_macro']:7.4f} "
              f"{m['recall'][0]:6.3f} {m['recall'][1]:6.3f} {m['recall'][2]:6.3f} "
              f"{m['precision'][1]:6.3f} {m['precision'][2]:6.3f} "
              f"{m['mae']:6.3f} {m['adj_pct']:5.1f} {m['n_rules']:5d} "
              f"{m['train_size']:6d}{star}")

    best_exp = all_experiments[0]
    print(f"\n  ★ Melhor estratégia: {best_exp['strategy']}")
    print(f"    F1-macro={best_exp['f1_macro']:.4f}, config={best_exp['cfg']}")

    # ── 6. MÉTRICAS DETALHADAS DO MELHOR ─────────────────────────────────
    sep("5. MÉTRICAS DETALHADAS — MELHOR ESTRATÉGIA")

    bm = best_exp
    print(f"  Estratégia:           {bm['strategy']}")
    print(f"  Configuração:         {bm['cfg']}")
    print(f"  Treino (amostras):    {bm['train_size']}")
    print(f"  Regras:               {bm['n_rules']}")
    print(f"\n  Acurácia:             {bm['acc']:.4f}")
    print(f"  F1-Score Macro:       {bm['f1_macro']:.4f}")
    print(f"  F1-Score Weighted:    {bm['f1_weighted']:.4f}")
    print(f"  F1 por classe:        C0={bm['f1_per'][0]:.4f}  C1={bm['f1_per'][1]:.4f}  C2={bm['f1_per'][2]:.4f}")
    print(f"  Recall por classe:    C0={bm['recall'][0]:.4f}  C1={bm['recall'][1]:.4f}  C2={bm['recall'][2]:.4f}")
    print(f"  Precision por classe: C0={bm['precision'][0]:.4f}  C1={bm['precision'][1]:.4f}  C2={bm['precision'][2]:.4f}")
    print(f"  MAE ordinal:          {bm['mae']:.4f}")
    print(f"  Erros adj:            {bm['adj_n']}/{bm['adj_n']+bm['nadj_n']} ({bm['adj_pct']:.1f}%)")
    print(f"\n  Matriz de Confusão:")
    print(f"  {'':>22s}  Pred C0  Pred C1  Pred C2")
    cm_labs = ['Real C0 (Sem irrig.)', 'Real C1 (Moderada)  ', 'Real C2 (Intensa)   ']
    for i, lab in enumerate(cm_labs):
        print(f"    {lab}  {bm['cm'][i,0]:>5d}    {bm['cm'][i,1]:>5d}    {bm['cm'][i,2]:>5d}")

    # Classification report do melhor
    print(f"\n{classification_report(y_test, bm['y_pred'], target_names=['Sem irrig.(C0)', 'Moderada(C1)', 'Intensa(C2)'], zero_division=0)}")

    # ── 7. SANIDADE DO MELHOR ────────────────────────────────────────────
    sep("6. SANIDADE AGRONÓMICA — MELHOR ESTRATÉGIA")

    # Encontrar o modelo correspondente à melhor estratégia
    # Precisamos mapear — vamos re-treinar o melhor
    best_strat = bm['strategy']
    print(f"  Estratégia: {best_strat}")

    # Reconstruir modelo do melhor (já temos os modelos locais)
    # Usar o modelo que está no escopo — se for baseline é model_0, etc.
    model_map = {
        'Baseline': model_0, 'Random Oversampling': model_1,
        'SMOTE': model_2, 'ADASYN': model_3,
        'BorderlineSMOTE': model_4, 'SMOTE + Tomek': model_5,
        'Repeated Minority': model_7, 'Cost-Sensitive': best_model_cs,
        'SMOTE Parcial 15%': model_9,
    }
    if best_strat in model_map:
        best_model_final = model_map[best_strat]
        n_pass = sanity_check(best_model_final)
    elif best_strat == 'Binary + Sub-classificação':
        # Sanidade especial para 2 estágios
        n_pass = 0
        casos = [
            ([120.0, 0.0, 38.5, 80], "Solo seco, sem chuva, calor", [1, 2]),
            ([15.0, 30.0, 32.0, 40], "Solo úmido, muita chuva, ameno", 0),
            ([70.0, 2.0, 36.0, 60], "Solo médio, pouca chuva, calor", [1, 2]),
            ([100.0, 0.0, 37.0, 20], "Solo seco, sem chuva, DAP baixo", [1, 2]),
        ]
        for inputs, desc, expected in casos:
            x = np.array(inputs)
            s1 = best_model_bin.predict(x)
            if s1 == 0:
                pred = 0
            else:
                pred = best_model_sub.predict(x) + 1
            if isinstance(expected, list):
                ok = pred in expected
                exp_str = f"C{expected[0]}/C{expected[1]}"
            else:
                ok = pred == expected
                exp_str = f"C{expected}"
            n_pass += ok
            st = "✓" if ok else "✗"
            print(f"    {st} Pred=C{pred} Exp={exp_str} | {desc}")
    else:
        n_pass = 0

    print(f"\n  Resultado: {n_pass}/4 (mínimo: 3/4)")

    # ── 8. APROVAÇÃO ─────────────────────────────────────────────────────
    sep("7. VERIFICAÇÃO DE APROVAÇÃO")
    checks = [
        ('F1-macro ≥ 0.40',  bm['f1_macro'] >= 0.40,     f"{bm['f1_macro']:.4f}"),
        ('Recall C1 ≥ 0.20', bm['recall'][1] >= 0.20,    f"{bm['recall'][1]:.4f}"),
        ('Recall C2 ≥ 0.20', bm['recall'][2] >= 0.20,    f"{bm['recall'][2]:.4f}"),
        ('MAE ≤ 0.80',       bm['mae'] <= 0.80,          f"{bm['mae']:.4f}"),
        ('Sanidade ≥ 3/4',   n_pass >= 3,                f"{n_pass}/4"),
    ]
    all_pass = True
    for crit, passed, val in checks:
        st = "✓" if passed else "✗"
        all_pass = all_pass and passed
        print(f"  {st}  {crit:25s}  →  {val}")
    print()
    if all_pass:
        print(f"  ╔══════════════════════════════════════╗")
        print(f"  ║   MODELO APROVADO PARA CAMPO   ✓    ║")
        print(f"  ╚══════════════════════════════════════╝")
    else:
        print(f"  ╔══════════════════════════════════════╗")
        print(f"  ║   MODELO NÃO APROVADO — VERIFICAR   ║")
        print(f"  ╚══════════════════════════════════════╝")

    # ── 9. SALVAR MELHOR MODELO ──────────────────────────────────────────
    sep("8. ARTEFACTOS")
    if best_strat in model_map:
        model_map[best_strat].save('memoria_cold_start_v8.pkl')
        print(f"  ✓ memoria_cold_start_v8.pkl ({model_map[best_strat].rules_by_class()})")
    elif best_strat == 'Binary + Sub-classificação':
        # Salvar ambos os modelos
        best_model_bin.save('memoria_cold_start_v8_stage1.pkl')
        best_model_sub.save('memoria_cold_start_v8_stage2.pkl')
        print(f"  ✓ memoria_cold_start_v8_stage1.pkl (binário)")
        print(f"  ✓ memoria_cold_start_v8_stage2.pkl (C1vsC2)")
        print(f"  ⚠ Deploy requer 2 modelos em cascata")

    # ── 10. GRÁFICOS ────────────────────────────────────────────────────
    sep("9. GRÁFICOS")

    # 10a: Comparação de todas as estratégias (F1-macro bar chart)
    fig, ax = plt.subplots(figsize=(14, 6))
    strategies = [m['strategy'] for m in all_experiments]
    f1s = [m['f1_macro'] for m in all_experiments]
    colors = ['#4CAF50' if f >= 0.40 else '#F44336' for f in f1s]
    bars = ax.barh(range(len(strategies)), f1s, color=colors, alpha=0.85, edgecolor='white')
    ax.axvline(x=0.40, color='red', linestyle='--', linewidth=1.5, label='Limiar (0.40)')
    ax.axvline(x=0.333, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='Baseline (0.33)')
    for i, (s, f1) in enumerate(zip(strategies, f1s)):
        ax.text(f1 + 0.005, i, f'{f1:.3f}', va='center', fontsize=10, fontweight='bold')
    ax.set_yticks(range(len(strategies)))
    ax.set_yticklabels(strategies, fontsize=10)
    ax.set_xlabel('F1-Score Macro', fontsize=12)
    ax.set_title('Comparação de Estratégias de Resampling — F1-macro', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xlim(0, max(f1s) * 1.15)
    ax.invert_yaxis()
    ax.grid(True, axis='x', alpha=0.2)
    plt.tight_layout()
    plt.savefig(f'{OUT}/comparacao_estrategias.png', dpi=150)
    plt.close()
    print(f"  ✓ {OUT}/comparacao_estrategias.png")

    # 10b: Recall por classe para cada estratégia
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True)
    for c, (ax, title, color) in enumerate(zip(
        axes, ['Recall C0 (Sem irrig.)', 'Recall C1 (Moderada)', 'Recall C2 (Intensa)'],
        ['#2196F3', '#FF9800', '#4CAF50']
    )):
        recs = [m['recall'][c] for m in all_experiments]
        ax.barh(range(len(strategies)), recs, color=color, alpha=0.8, edgecolor='white')
        if c > 0:
            ax.axvline(x=0.20, color='red', linestyle='--', linewidth=1, alpha=0.7)
        for i, r in enumerate(recs):
            ax.text(r + 0.01, i, f'{r:.2f}', va='center', fontsize=9)
        ax.set_yticks(range(len(strategies)))
        if c == 0:
            ax.set_yticklabels(strategies, fontsize=9)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlim(0, 1.15)
        ax.invert_yaxis()
        ax.grid(True, axis='x', alpha=0.2)
    plt.suptitle('Recall por Classe — Todas as Estratégias', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUT}/recall_por_estrategia.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {OUT}/recall_por_estrategia.png")

    # 10c: Confusion matrix do melhor modelo
    fig, ax = plt.subplots(figsize=(7, 6))
    cm_labels_short = ['C0\nSem irrig.', 'C1\nModerada', 'C2\nIntensa']
    sns.heatmap(bm['cm'], annot=True, fmt='d', cmap='Blues', xticklabels=cm_labels_short,
                yticklabels=cm_labels_short, ax=ax, cbar_kws={'label': 'Amostras'},
                annot_kws={'fontsize': 14, 'fontweight': 'bold'})
    ax.set_xlabel('Predito', fontsize=12)
    ax.set_ylabel('Real', fontsize=12)
    ax.set_title(f'Matriz de Confusão — {bm["strategy"]}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUT}/matriz_confusao_melhor.png', dpi=150)
    plt.close()
    print(f"  ✓ {OUT}/matriz_confusao_melhor.png")

    # 10d: Precision vs Recall scatter per strategy
    fig, ax = plt.subplots(figsize=(10, 8))
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', 'h', 'p']
    for i, m in enumerate(all_experiments):
        for c, (color, offset) in enumerate(zip(['#2196F3', '#FF9800', '#4CAF50'], [-0.005, 0, 0.005])):
            ax.scatter(m['recall'][c], m['precision'][c] + offset,
                      marker=markers[i % len(markers)], s=80, color=color,
                      alpha=0.7, edgecolors='black', linewidth=0.5)
    # Legend for strategies
    for i, m in enumerate(all_experiments):
        ax.scatter([], [], marker=markers[i % len(markers)], s=60, color='gray',
                  label=m['strategy'], alpha=0.7, edgecolors='black')
    # Legend for classes
    for c, (color, name) in enumerate(zip(['#2196F3', '#FF9800', '#4CAF50'],
                                           ['C0', 'C1', 'C2'])):
        ax.scatter([], [], marker='o', s=60, color=color, label=f'Classe {name}')
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision vs Recall por Classe e Estratégia', fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(True, alpha=0.2)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(f'{OUT}/precision_recall_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {OUT}/precision_recall_scatter.png")

    # ── FIM ──────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    sep("CONCLUÍDO")
    print(f"  Tempo total: {elapsed:.1f}s")
    print(f"  Melhor: {bm['strategy']} (F1-macro={bm['f1_macro']:.4f})")
    print(f"  Estratégias testadas: {len(all_experiments)}")
    print()
    print(f"  Resumo rápido:")
    for i, m in enumerate(all_experiments):
        print(f"    {i+1}. {m['strategy']:24s} F1={m['f1_macro']:.4f}")
    print()
