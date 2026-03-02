# ── 1. LOAD ──────────────────────────────────────────────────────────
    
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
    
    
    
    
    

df = pd.read_csv('dataset_cold_start_v7.csv')
fcols = ['tensao_solo_kpa', 'chuva_acum_3d_mm', 'tmax_max_3d_c', 'dap']
X = df[fcols].values
y = df['classe_irrigacao'].values

print(f"  Amostras: {len(df)}")
for c in range(3):
    n = (y == c).sum()
    print(f"  C{c}: {n:5d} ({n/len(y)*100:5.1f}%)")

    # ── 2. SPLIT ─────────────────────────────────────────────────────────


dap_vals = df['dap'].values
resets = [0] + [i for i in range(1, len(dap_vals)) if dap_vals[i] < dap_vals[i-1]] + [len(dap_vals)]
n_groups = len(resets) - 1
print(f"  {n_groups} grupos detectados")

    # Select test groups: pick groups that have C1 or C2, spread across dataset
    # Groups with C1+C2: 6,8,12,14,16,18,20 — pick 4 spread out
    # Groups with only C2: 22,24,26,28,30 — pick 2
    # This gives diverse coverage
test_group_ids = [5, 9, 13, 17, 21, 27]  # 0-indexed (groups 6,10,14,18,22,28 in 1-indexed)

test_idx = []
for g in test_group_ids:
    test_idx.extend(range(resets[g], resets[g+1]))
train_idx = [i for i in range(len(df)) if i not in test_idx]

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

print(f"  Grupos de teste (1-indexed): {[g+1 for g in test_group_ids]}")
print(f"  Treino: {len(X_train)}  |  Teste: {len(X_test)}")
for c in range(3):
    n_tr = (y_train == c).sum()
    n_te = (y_test == c).sum()
    print(f"  C{c}: treino={n_tr:5d}  teste={n_te:4d}")

    # Check test has C1 and C2
if (y_test == 1).sum() == 0 or (y_test == 2).sum() == 0:
    print("  ⚠ ALERTA: teste sem C1 ou C2 — ajustar grupos!")

#####

df_X_train = pd.DataFrame(X_train, columns=fcols)
df_y_train = pd.Series(y_train, name='classe_irrigacao')

df_treino_completo = pd.concat([df_X_train, df_y_train], axis=1)

print(df_X_train.head())

df_X_train.hist(figsize=(10, 8))
plt.show()