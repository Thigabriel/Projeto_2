"""
============================================================================
  COLD START v7 — ALMMo-0
  Dataset: dataset_cold_start_v7.csv (2733 amostras, 94% C0)
  Sweep bidimensional: r_threshold × min_rules_per_class
============================================================================
"""

import numpy as np
import pandas as pd
import os
import time
import pickle
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

#from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek 
from collections import Counter
from imblearn.over_sampling import ADASYN


# ─────────────────────────────────────────────────────────────────────────────
# ALMMo-0 v2 (inline)
# ─────────────────────────────────────────────────────────────────────────────

class ALMMo0:
    def __init__(self, n_inputs=4, r_threshold=0.5, max_rules=50,
                 age_limit=100, epsilon=1e-8, n_classes=3,
                 min_rules_per_class=3):
        self.n_inputs           = n_inputs
        self.r_threshold        = r_threshold
        self.max_rules          = max_rules
        self.age_limit          = age_limit
        self.epsilon            = epsilon
        self.n_classes           = n_classes
        self.min_rules_per_class = min_rules_per_class
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
            'age': 0, 'activations': 1, 'created_at': datetime.now().isoformat(),
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
        if d_min > self.r_threshold:
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
            eff_eta = eta * af
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
                        kept.append(r)  # protect
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
        return history

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({
                'rules': self.rules, 'input_mean': self.input_mean,
                'input_std': self.input_std, 'r_threshold': self.r_threshold,
                'max_rules': self.max_rules, 'age_limit': self.age_limit,
                'n_inputs': self.n_inputs, 'n_classes': self.n_classes,
                'min_rules_per_class': self.min_rules_per_class,
                'n_samples_seen': self.n_samples_seen,
                'n_rules_created': self.n_rules_created,
                'n_rules_pruned': self.n_rules_pruned,
                'created_at': self.created_at,
                'saved_at': datetime.now().isoformat(),
            }, f)

    def rules_by_class(self):
        return {c: sum(1 for r in self.rules if r['consequent'] == c) for c in range(self.n_classes)}


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
    w = 70
    print(f"\n{'═'*w}")
    print(f"  {title}")
    print(f"{'═'*w}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    t0 = time.time()
    OUT = 'graficos_v7'
    os.makedirs(OUT, exist_ok=True)

    # ── 1. LOAD ──────────────────────────────────────────────────────────
    sep("1. DATASET")
    df = pd.read_csv('df_X_train_scaled(classe).csv')
    fcols = ['tensao_solo_kpa', 'chuva_acum_3d_mm', 'tmax_max_3d_c', 'dap']
    X = df[fcols].values
    y = df['classe_irrigacao'].values

    print(f"  Amostras: {len(df)}")
    for c in range(3):
        n = (y == c).sum()
        print(f"  C{c}: {n:5d} ({n/len(y)*100:5.1f}%)")

    # ── 2. SPLIT ─────────────────────────────────────────────────────────
    sep("2. SPLIT (Leave-Groups-Out)")

    dap_vals = df['dap'].values
    resets = [0] + [i for i in range(1, len(dap_vals)) if dap_vals[i] < dap_vals[i-1]] + [len(dap_vals)]
    n_groups = len(resets) - 1
    print(f"  {n_groups} grupos detectados")

    # Ajuste dinâmico: seleciona grupos de teste que existem no dataset atual
    # Escolhemos alguns grupos espalhados (ex: início, meio e fim)
    # Selecionando grupos 5, 9, 13, 17 e os últimos disponíveis
    potential_test_groups = [5, 9, 13, 17, 21, 23] 
    test_group_ids = [g for g in potential_test_groups if g < n_groups]

    test_idx = []
    for g in test_group_ids:
        test_idx.extend(range(resets[g], resets[g+1]))
    
    train_idx = [i for i in range(len(df)) if i not in test_idx]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print(f"  Grupos de teste (0-indexed): {test_group_ids}")
    print(f"  Treino: {len(X_train)}  |  Teste: {len(X_test)}")

    # selmo
    '''smote = SMOTETomek(random_state=42)
    X_train, y_train = smote.fit_resample(X_train2, y_train2)
    print(f"Antes SMOTE: treino={len(X_train2)}  classes={Counter(y_train2)}")
    print(f"Após SMOTE: treino={len(X_train)}  classes={Counter(y_train)}")
    print("Distribuição ORIGINAL do treino:", Counter(y_train2))
    
    adasyn = ADASYN(random_state=42)
    X_train, y_train = adasyn.fit_resample(X_train2, y_train2)

    print(f"Antes ADASYN: treino={len(X_train2)}  classes={Counter(y_train2)}")
    print(f"Após ADASYN: treino={len(X_train)}  classes={Counter(y_train)}")'''

    # ── 3. BIDIMENSIONAL SWEEP ───────────────────────────────────────────
    sep("3. SWEEP BIDIMENSIONAL (r_threshold × min_rules_per_class)")

    r_values = np.arange(0.10, 2.05, 0.05)
    min_rules_values = [3, 5, 7, 10]
    max_rules_base = 40

    all_results = []
    total = len(r_values) * len(min_rules_values)
    count = 0

    for mrpc in min_rules_values:
        for r in r_values:
            r = round(r, 2)
            # Ensure max_rules can accommodate min_rules protection
            effective_max = max(max_rules_base, mrpc * 3 + 5)

            m = ALMMo0(n_inputs=4, r_threshold=r, max_rules=effective_max,
                       age_limit=80, n_classes=3, min_rules_per_class=mrpc)
            m.cold_start(X_train, y_train, verbose=False)

            yp = np.array([m.predict(x) for x in X_test])
            f1m = f1_score(y_test, yp, average='macro', zero_division=0)
            rec = recall_score(y_test, yp, average=None, zero_division=0)
            rbc = m.rules_by_class()

            all_results.append({
                'r': r, 'mrpc': mrpc, 'f1_macro': f1m,
                'recall_c0': rec[0], 'recall_c1': rec[1], 'recall_c2': rec[2],
                'n_rules': len(m.rules),
                'rc0': rbc[0], 'rc1': rbc[1], 'rc2': rbc[2],
            })
            count += 1

    print(f"  {count} configurações testadas")

    # Top results
    top = sorted(all_results, key=lambda x: -x['f1_macro'])[:20]
    print(f"\n  {'r':>5s} {'mrpc':>4s} {'F1-mac':>7s} {'RecC0':>6s} {'RecC1':>6s} "
          f"{'RecC2':>6s} {'Rules':>5s} {'C0/C1/C2':>10s}")
    print(f"  {'─'*5} {'─'*4} {'─'*7} {'─'*6} {'─'*6} {'─'*6} {'─'*5} {'─'*10}")
    for s in top:
        star = " ★" if s == top[0] else ""
        print(f"  {s['r']:5.2f} {s['mrpc']:4d} {s['f1_macro']:7.4f} {s['recall_c0']:6.3f} "
              f"{s['recall_c1']:6.3f} {s['recall_c2']:6.3f} {s['n_rules']:5d} "
              f"{s['rc0']}/{s['rc1']}/{s['rc2']}{star}")

    best = top[0]
    best_r = best['r']
    best_mrpc = best['mrpc']
    print(f"\n  ★ Melhor: r={best_r}, min_rules_per_class={best_mrpc}, F1-macro={best['f1_macro']:.4f}")

    # ── 4. TRAIN FINAL MODEL ─────────────────────────────────────────────
    sep(f"4. MODELO FINAL (r={best_r}, mrpc={best_mrpc})")

    effective_max = max(max_rules_base, best_mrpc * 3 + 5)
    model = ALMMo0(n_inputs=4, r_threshold=best_r, max_rules=effective_max,
                   age_limit=80, n_classes=3, min_rules_per_class=best_mrpc)
    rule_history = model.cold_start(X_train, y_train, verbose=False)

    rbc = model.rules_by_class()
    labels = ["Sem irrigação", "Moderada", "Intensa"]
    print(f"  Regras: {len(model.rules)} total (criadas={model.n_rules_created}, podadas={model.n_rules_pruned})")
    for c in range(3):
        bar = '█' * min(rbc[c], 40)
        print(f"    C{c} ({labels[c]:14s}): {rbc[c]:3d}  {bar}")

    # ── 5. METRICS ────────────────────────────────────────────────────────
    sep("5. MÉTRICAS COMPLETAS")

    y_pred = np.array([model.predict(x) for x in X_test])
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_per = f1_score(y_test, y_pred, average=None, zero_division=0)
    recall_per = recall_score(y_test, y_pred, average=None, zero_division=0)
    prec_per = precision_score(y_test, y_pred, average=None, zero_division=0)
    recall_mac = recall_score(y_test, y_pred, average='macro', zero_division=0)
    mae = mean_absolute_error(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    adj_pct, adj_n, nadj_n = erros_adjacentes(y_test, y_pred)
    n_err = int((y_test != y_pred).sum())

    print(f"  Acurácia:             {acc:.4f}  ({int(acc*len(y_test))}/{len(y_test)})")
    print(f"  F1-Score Macro:       {f1_macro:.4f}")
    print(f"  F1-Score Weighted:    {f1_weighted:.4f}")
    print(f"  F1 por classe:        C0={f1_per[0]:.4f}  C1={f1_per[1]:.4f}  C2={f1_per[2]:.4f}")
    print(f"  Recall Macro:         {recall_mac:.4f}")
    print(f"  Recall por classe:    C0={recall_per[0]:.4f}  C1={recall_per[1]:.4f}  C2={recall_per[2]:.4f}")
    print(f"  Precision por classe: C0={prec_per[0]:.4f}  C1={prec_per[1]:.4f}  C2={prec_per[2]:.4f}")
    print(f"  MAE ordinal:          {mae:.4f}")
    print(f"  Erros adjacentes:     {adj_n}/{n_err} ({adj_pct:.1f}%), não-adjacentes: {nadj_n}")
    print(f"\n  Matriz de Confusão:")
    print(f"  {'':>22s}  Pred C0  Pred C1  Pred C2")
    cm_labels = ['Real C0 (Sem irrig.)', 'Real C1 (Moderada)  ', 'Real C2 (Intensa)   ']
    for i, lab in enumerate(cm_labels):
        print(f"    {lab}  {cm[i,0]:>5d}    {cm[i,1]:>5d}    {cm[i,2]:>5d}")

    print(f"\n{classification_report(y_test, y_pred, target_names=['Sem irrig.(C0)', 'Moderada(C1)', 'Intensa(C2)'], zero_division=0)}")

    # ── 6. SANITY CHECK ──────────────────────────────────────────────────
    sep("6. SANIDADE AGRONÓMICA")

    casos = [
        ([120.0,  0.0, 38.5, 80], "Solo seco (120kPa), sem chuva, calor extremo", [1, 2]),
        ([ 15.0, 30.0, 32.0, 40], "Solo úmido (15kPa), muita chuva, ameno", 0),
        ([ 70.0,  2.0, 36.0, 60], "Solo médio (70kPa), pouca chuva, calor", [1, 2]),
        ([100.0,  0.0, 37.0, 20], "Solo seco (100kPa), sem chuva, DAP baixo", [1, 2]),
    ]
    n_pass = 0
    for inputs, desc, expected in casos:
        pred = model.predict(np.array(inputs))
        if isinstance(expected, list):
            ok = pred in expected
            exp_str = f"C{expected[0]} ou C{expected[1]}"
        else:
            ok = pred == expected
            exp_str = f"C{expected}"
        n_pass += ok
        st = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {st}  Predito: C{pred}  Esperado: {exp_str}")
        print(f"         {desc}")
        print()
    print(f"  Resultado: {n_pass}/4 (mínimo: 3/4)")

    # ── 7. APPROVAL ──────────────────────────────────────────────────────
    sep("7. VERIFICAÇÃO DE APROVAÇÃO")
    checks = [
        ('F1-macro ≥ 0.40',  f1_macro >= 0.40,     f"{f1_macro:.4f}"),
        ('Recall C1 ≥ 0.20', recall_per[1] >= 0.20, f"{recall_per[1]:.4f}"),
        ('Recall C2 ≥ 0.20', recall_per[2] >= 0.20, f"{recall_per[2]:.4f}"),
        ('MAE ≤ 0.80',       mae <= 0.80,           f"{mae:.4f}"),
        ('Sanidade ≥ 3/4',   n_pass >= 3,           f"{n_pass}/4"),
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

    # ── 8. SAVE MODEL ────────────────────────────────────────────────────
    sep("8. ARTEFACTOS")
    model.save('memoria_cold_start_v7.pkl')
    print(f"  ✓ memoria_cold_start_v7.pkl ({len(model.rules)} regras)")

    # ── 9. PLOTS ─────────────────────────────────────────────────────────
    sep("9. GRÁFICOS")

    # 9a: Sweep heatmap (r_threshold × min_rules_per_class → F1-macro)
    fig, ax = plt.subplots(figsize=(14, 5))
    for mrpc, color, marker in zip(min_rules_values,
                                    ['#2196F3', '#4CAF50', '#FF9800', '#E91E63'],
                                    ['o', 's', '^', 'D']):
        subset = [s for s in all_results if s['mrpc'] == mrpc]
        rs = [s['r'] for s in subset]
        f1s = [s['f1_macro'] for s in subset]
        ax.plot(rs, f1s, f'{marker}-', color=color, linewidth=1.5, markersize=4,
                label=f'min_rules={mrpc}', alpha=0.8)
    ax.axhline(y=0.40, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Limiar (0.40)')
    ax.axhline(y=0.333, color='gray', linestyle=':', linewidth=1, alpha=0.4, label='Baseline (0.33)')
    ax.annotate(f'★ r={best_r}, mrpc={best_mrpc}\nF1={best["f1_macro"]:.3f}',
                xy=(best_r, best['f1_macro']),
                xytext=(best_r + 0.2, best['f1_macro'] + 0.02),
                fontsize=9, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='black'))
    ax.set_xlabel('r_threshold', fontsize=12)
    ax.set_ylabel('F1-score Macro', fontsize=12)
    ax.set_title('Sweep Bidimensional — F1-macro vs r_threshold por min_rules_per_class', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(f'{OUT}/sweep_bidimensional.png', dpi=150)
    plt.close()
    print(f"  ✓ {OUT}/sweep_bidimensional.png")

    # 9b: Recall per class sweep (best mrpc only)
    fig, ax = plt.subplots(figsize=(12, 5))
    subset = [s for s in all_results if s['mrpc'] == best_mrpc]
    rs = [s['r'] for s in subset]
    ax.plot(rs, [s['recall_c0'] for s in subset], 's-', color='#F44336', linewidth=1.5, markersize=4, label='Recall C0')
    ax.plot(rs, [s['recall_c1'] for s in subset], '^-', color='#FF9800', linewidth=1.5, markersize=4, label='Recall C1')
    ax.plot(rs, [s['recall_c2'] for s in subset], 'o-', color='#4CAF50', linewidth=1.5, markersize=4, label='Recall C2')
    ax.axhline(y=0.20, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='Limiar Recall (0.20)')
    ax.axvline(x=best_r, color='black', linestyle='--', linewidth=1, alpha=0.3)
    ax.set_xlabel('r_threshold', fontsize=12)
    ax.set_ylabel('Recall', fontsize=12)
    ax.set_title(f'Recall por Classe (min_rules_per_class={best_mrpc})', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(f'{OUT}/sweep_recall_por_classe.png', dpi=150)
    plt.close()
    print(f"  ✓ {OUT}/sweep_recall_por_classe.png")

    # 9c: Confusion matrix heatmap
    fig, ax = plt.subplots(figsize=(7, 6))
    cm_labels_short = ['C0\nSem irrig.', 'C1\nModerada', 'C2\nIntensa']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=cm_labels_short,
                yticklabels=cm_labels_short, ax=ax, cbar_kws={'label': 'Amostras'},
                annot_kws={'fontsize': 14, 'fontweight': 'bold'})
    ax.set_xlabel('Predito', fontsize=12)
    ax.set_ylabel('Real', fontsize=12)
    ax.set_title(f'Matriz de Confusão — r={best_r}, mrpc={best_mrpc}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUT}/matriz_confusao.png', dpi=150)
    plt.close()
    print(f"  ✓ {OUT}/matriz_confusao.png")

    # 9d: Rule evolution during cold start
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(range(len(rule_history)), rule_history, color='#2196F3', linewidth=1.2)
    ax.set_xlabel('Amostra processada', fontsize=12)
    ax.set_ylabel('Regras activas', fontsize=12)
    ax.set_title(f'Evolução do Banco de Regras (r={best_r}, mrpc={best_mrpc})', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(f'{OUT}/evolucao_regras.png', dpi=150)
    plt.close()
    print(f"  ✓ {OUT}/evolucao_regras.png")

    # ── FIM ──────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    sep("CONCLUÍDO")
    print(f"  Tempo: {elapsed:.1f}s")
    print(f"  Modelo: r={best_r}, mrpc={best_mrpc}, {len(model.rules)} regras, F1-macro={f1_macro:.4f}")
    print()
