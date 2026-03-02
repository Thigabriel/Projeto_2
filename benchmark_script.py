#!/usr/bin/env python3
"""
==========================================================================
BENCHMARK COMPARATIVO: ALMMo-0 vs. Algoritmos Clássicos de ML
==========================================================================
Projeto: Sistema de Irrigação com ALMMo-0
Etapa:   Benchmark ALMMo-0 vs. Algoritmos Clássicos de ML
TCC:     Classificação de irrigação — tomate, Imperatriz-MA (AquaCrop-OSPy)

Uso:
    pip install -r requirements.txt
    python benchmark_script.py

Tempo estimado: < 5 minutos (dependendo do hardware)
==========================================================================
"""

import warnings
warnings.filterwarnings('ignore')

import os
import time
import numpy as np
import pandas as pd
from collections import Counter

# Scikit-learn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, recall_score, precision_score,
    confusion_matrix, classification_report
)
from sklearn.utils.class_weight import compute_sample_weight

# XGBoost
from xgboost import XGBClassifier

# Imbalanced-learn
from imblearn.over_sampling import SMOTE, ADASYN

# Visualização
import matplotlib
matplotlib.use('Agg')  # Backend não-interativo (funciona sem display)
import matplotlib.pyplot as plt
import seaborn as sns

# ========================================================================
# CONFIGURAÇÃO
# ========================================================================

# Caminho do dataset — único arquivo necessário
DATASET_V7 = "df_X_train_scaled(classe).csv"
# O cenário binário (v7_bin) é gerado automaticamente a partir do v7,
# fundindo classes 1 e 2 → classe 1 (irrigação necessária).

# Features e target
FEATURES = ['tensao_solo_kpa', 'chuva_acum_3d_mm', 'tmax_max_3d_c', 'dap']
TARGET = 'classe_irrigacao'

# Grupos de teste (idênticos ao cold start do ALMMo-0)
GRUPOS_TESTE = [6, 10, 14, 18, 22, 28]

# Tamanhos dos 30 grupos (5 anos × 2 janelas × 3 cenários)
TAMANHOS_GRUPOS = [
    92, 90, 90, 92, 92, 89, 92, 94, 92, 88,
    92, 90, 90, 92, 92, 89, 92, 94, 92, 88,
    92, 90, 90, 92, 92, 89, 92, 94, 92, 88
]

# Resultados de referência do ALMMo-0
ALMMO_REF = {
    'v7': {
        'A': {'f1_macro': 0.543, 'recall_c0': 0.771, 'recall_c1': 0.341,
               'recall_c2': 0.762, 'precision_c1': 0.125, 'mae': 0.294,
               'erros_adj_pct': 87.4},
        'B': {'f1_macro': 0.598},
    },
    'v7_bin': {
        'B': {'f1_macro': 0.644},  # ALMMo-0 binário Cost-Sensitive
    },
}

# Seed global
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# ========================================================================
# FUNÇÕES AUXILIARES
# ========================================================================

def criar_grupos(df):
    """Recria coluna de grupo baseada nas linhas do dataset."""
    grupos = []
    for i, t in enumerate(TAMANHOS_GRUPOS, 1):
        grupos.extend([i] * t)
    # Truncar ou estender para o tamanho real do dataframe
    if len(grupos) >= len(df):
        return grupos[:len(df)]
    else:
        # Se o dataset for maior (v8 com oversampling no dataset),
        # os grupos extras não existem — usar -1 como marcador
        grupos.extend([-1] * (len(df) - len(grupos)))
        return grupos


def split_leave_groups_out(df, features, target, grupos_teste):
    """
    Split Leave-Groups-Out idêntico ao cold start do ALMMo-0.
    Retorna X_train, X_test, y_train, y_test.
    """
    df = df.copy()

    # Verificar se coluna de grupo já existe
    if 'grupo' not in df.columns:
        df['grupo'] = criar_grupos(df)

    mask_teste = df['grupo'].isin(grupos_teste)

    X_train = df[~mask_teste][features].values
    y_train = df[~mask_teste][target].values
    X_test  = df[mask_teste][features].values
    y_test  = df[mask_teste][target].values

    return X_train, X_test, y_train, y_test


def normalizar(X_train, X_test):
    """Normaliza com StandardScaler fitado no treino."""
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm  = scaler.transform(X_test)
    return X_train_norm, X_test_norm


# ========================================================================
# MÉTODOS DE TRATAMENTO DE DESBALANCEAMENTO
# ========================================================================

def smote_parcial(X_train, y_train, target_ratio=0.15):
    """
    Método C: Oversampla classes minoritárias até target_ratio do total.
    Não modifica a classe majoritária.
    """
    counts = Counter(y_train)
    n_total = len(y_train)
    n_total_alvo = int(n_total / (1 - target_ratio))

    sampling_strategy = {}
    for classe, n in counts.items():
        alvo = max(n, int(n_total_alvo * target_ratio))
        if alvo > n:
            sampling_strategy[classe] = alvo

    if sampling_strategy:
        # Ajustar k_neighbors se alguma classe minoritária for muito pequena
        min_samples = min(counts[c] for c in sampling_strategy)
        k = min(5, min_samples - 1) if min_samples > 1 else 1
        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            random_state=RANDOM_STATE,
            k_neighbors=k
        )
        return smote.fit_resample(X_train, y_train)
    return X_train, y_train


def smote_integral(X_train, y_train):
    """
    Método D: Oversampla TODAS as classes minoritárias até igualar a majoritária.
    Balanceamento completo.
    """
    counts = Counter(y_train)
    min_samples = min(counts.values())
    k = min(5, min_samples - 1) if min_samples > 1 else 1
    smote = SMOTE(
        sampling_strategy='auto',
        random_state=RANDOM_STATE,
        k_neighbors=k
    )
    return smote.fit_resample(X_train, y_train)


def adasyn_resample(X_train, y_train):
    """
    Método E: ADASYN — oversampling adaptativo.
    Fallback para SMOTE integral se ADASYN falhar.
    """
    try:
        counts = Counter(y_train)
        min_samples = min(counts.values())
        k = min(5, min_samples - 1) if min_samples > 1 else 1
        ada = ADASYN(
            sampling_strategy='auto',
            random_state=RANDOM_STATE,
            n_neighbors=k
        )
        return ada.fit_resample(X_train, y_train), False  # False = sem fallback
    except ValueError as e:
        print(f"  [ADASYN] Falhou ({e}). Usando SMOTE integral como fallback.")
        return smote_integral(X_train, y_train), True  # True = fallback acionado


# ========================================================================
# DEFINIÇÃO DOS ALGORITMOS
# ========================================================================

def get_algoritmos(metodo, n_classes):
    """
    Retorna dicionário de algoritmos configurados para o método de tratamento.

    metodo: 'A' (baseline), 'B' (cost-sensitive), 'C'/'D'/'E' (resampling)
    n_classes: 2 ou 3
    """
    algos = {}

    if metodo == 'B':
        # Cost-Sensitive
        algos['LogReg'] = LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, class_weight='balanced'
        )
        algos['SVC'] = SVC(
            kernel='rbf', probability=True, random_state=RANDOM_STATE,
            class_weight='balanced'
        )
        algos['Random Forest'] = RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, class_weight='balanced'
        )
        # KNN não tem class_weight — não incluir no método B
        # XGBoost: para multiclasse, usar sample_weight no fit()
        algos['XGBoost'] = XGBClassifier(
            n_estimators=100, random_state=RANDOM_STATE,
            eval_metric='mlogloss' if n_classes > 2 else 'logloss',
            use_label_encoder=False
        )
    else:
        # Métodos A, C, D, E — algoritmos com configuração padrão
        algos['LogReg'] = LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        )
        algos['SVC'] = SVC(
            kernel='rbf', probability=True, random_state=RANDOM_STATE
        )
        algos['Random Forest'] = RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE
        )
        algos['KNN'] = KNeighborsClassifier(n_neighbors=5)
        algos['XGBoost'] = XGBClassifier(
            n_estimators=100, random_state=RANDOM_STATE,
            eval_metric='mlogloss' if n_classes > 2 else 'logloss',
            use_label_encoder=False
        )

    return algos


# ========================================================================
# FUNÇÃO DE AVALIAÇÃO
# ========================================================================

def avaliar(y_test, y_pred, nome, n_classes):
    """
    Calcula todas as métricas obrigatórias.
    n_classes: 2 ou 3
    """
    f1_macro   = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_weight  = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    recall     = recall_score(y_test, y_pred, average=None, zero_division=0)
    precision  = precision_score(y_test, y_pred, average=None, zero_division=0)
    cm         = confusion_matrix(y_test, y_pred)

    resultado = {
        'nome'        : nome,
        'f1_macro'    : round(f1_macro, 4),
        'f1_weighted' : round(f1_weight, 4),
        'recall_c0'   : round(recall[0], 4) if len(recall) > 0 else None,
        'recall_c1'   : round(recall[1], 4) if len(recall) > 1 else None,
        'precision_c1': round(precision[1], 4) if len(precision) > 1 else None,
        'confusion_matrix': cm.tolist(),
    }

    if n_classes == 3:
        resultado['recall_c2'] = round(recall[2], 4) if len(recall) > 2 else None
        # MAE ordinal
        mae = np.mean(np.abs(y_test - y_pred))
        resultado['mae_ordinal'] = round(mae, 4)
        # Proporção de erros adjacentes
        total_erros = np.sum(y_test != y_pred)
        if total_erros > 0:
            erros_adj = (
                np.sum((y_test == 0) & (y_pred == 1)) +
                np.sum((y_test == 1) & (y_pred == 0)) +
                np.sum((y_test == 1) & (y_pred == 2)) +
                np.sum((y_test == 2) & (y_pred == 1))
            )
            pct_adj = erros_adj / total_erros
        else:
            pct_adj = 1.0
        resultado['erros_adj_pct'] = round(pct_adj * 100, 1)
    else:
        resultado['recall_c2'] = None
        resultado['mae_ordinal'] = None
        resultado['erros_adj_pct'] = None

    return resultado


# ========================================================================
# PIPELINE PRINCIPAL
# ========================================================================

def executar_benchmark_dataset(df, dataset_nome, n_classes):
    """
    Executa todos os algoritmos × todos os métodos para um dataset.
    Retorna lista de dicionários com resultados.
    """
    print(f"\n{'='*70}")
    print(f"DATASET: {dataset_nome} ({n_classes} classes)")
    print(f"{'='*70}")

    # Split
    X_train, X_test, y_train, y_test = split_leave_groups_out(
        df, FEATURES, TARGET, GRUPOS_TESTE
    )

    print(f"  Treino: {len(X_train)} amostras | Teste: {len(X_test)} amostras")
    print(f"  Distribuição treino: {dict(Counter(y_train))}")
    print(f"  Distribuição teste:  {dict(Counter(y_test))}")

    # Normalizar
    X_train_norm, X_test_norm = normalizar(X_train, X_test)

    resultados = []
    metodos = ['A', 'B', 'C', 'D', 'E']
    metodo_nomes = {
        'A': 'Baseline',
        'B': 'Cost-Sensitive',
        'C': 'SMOTE Parcial 15%',
        'D': 'SMOTE Integral',
        'E': 'ADASYN'
    }

    adasyn_fallbacks = {}

    for metodo in metodos:
        print(f"\n  --- Método {metodo}: {metodo_nomes[metodo]} ---")

        # Preparar dados de treino conforme o método
        if metodo in ['A', 'B']:
            X_tr, y_tr = X_train_norm.copy(), y_train.copy()
            dist_pos_resample = None
        elif metodo == 'C':
            X_tr, y_tr = smote_parcial(X_train_norm, y_train)
            dist_pos_resample = dict(Counter(y_tr))
            print(f"    Distribuição pós-SMOTE parcial: {dist_pos_resample}")
        elif metodo == 'D':
            X_tr, y_tr = smote_integral(X_train_norm, y_train)
            dist_pos_resample = dict(Counter(y_tr))
            print(f"    Distribuição pós-SMOTE integral: {dist_pos_resample}")
        elif metodo == 'E':
            (X_tr, y_tr), fallback = adasyn_resample(X_train_norm, y_train)
            dist_pos_resample = dict(Counter(y_tr))
            adasyn_fallbacks[dataset_nome] = fallback
            print(f"    Distribuição pós-ADASYN: {dist_pos_resample}")
            if fallback:
                print(f"    ⚠ ADASYN usou fallback para SMOTE integral")

        # Obter algoritmos para este método
        algos = get_algoritmos(metodo, n_classes)

        for algo_nome, modelo in algos.items():
            t0 = time.time()
            nome_completo = f"{algo_nome}_{metodo}_{dataset_nome}"

            try:
                # Fit com sample_weight para XGBoost Cost-Sensitive
                if metodo == 'B' and algo_nome == 'XGBoost':
                    sw = compute_sample_weight('balanced', y_tr)
                    modelo.fit(X_tr, y_tr, sample_weight=sw)
                else:
                    modelo.fit(X_tr, y_tr)

                y_pred = modelo.predict(X_test_norm)
                resultado = avaliar(y_test, y_pred, nome_completo, n_classes)
                resultado['algoritmo'] = algo_nome
                resultado['metodo'] = metodo
                resultado['metodo_nome'] = metodo_nomes[metodo]
                resultado['dataset'] = dataset_nome
                resultado['n_classes'] = n_classes
                resultado['dist_pos_resample'] = str(dist_pos_resample) if dist_pos_resample else 'N/A'
                resultado['adasyn_fallback'] = adasyn_fallbacks.get(dataset_nome, False) if metodo == 'E' else False
                resultado['tempo_s'] = round(time.time() - t0, 2)

                delta_ref = 0.543 if dataset_nome == 'v7' else 0.644  # ref por cenário
                delta_almmo = resultado['f1_macro'] - delta_ref
                resultado['delta_almmo_baseline'] = round(delta_almmo, 4)

                print(f"    {algo_nome:15s} | F1-macro={resultado['f1_macro']:.4f} "
                      f"| Δ ALMMo-0={delta_almmo:+.4f} | {resultado['tempo_s']:.1f}s")

                resultados.append(resultado)

            except Exception as e:
                print(f"    {algo_nome:15s} | ERRO: {e}")
                resultados.append({
                    'nome': nome_completo,
                    'algoritmo': algo_nome,
                    'metodo': metodo,
                    'metodo_nome': metodo_nomes[metodo],
                    'dataset': dataset_nome,
                    'n_classes': n_classes,
                    'f1_macro': None,
                    'erro': str(e),
                })

    return resultados


# ========================================================================
# GERAÇÃO DE GRÁFICOS
# ========================================================================

def gerar_graficos(df_resultados, output_dir='.'):
    """Gera os 4 gráficos obrigatórios."""
    print(f"\n{'='*70}")
    print("GERANDO GRÁFICOS")
    print(f"{'='*70}")

    # Filtrar resultados válidos
    df_valid = df_resultados.dropna(subset=['f1_macro']).copy()

    # Cores por algoritmo
    cores_algo = {
        'LogReg': '#2196F3',
        'SVC': '#4CAF50',
        'Random Forest': '#FF9800',
        'KNN': '#9C27B0',
        'XGBoost': '#F44336',
        'ALMMo-0': '#000000',
    }

    # ------------------------------------------------------------------
    # GRÁFICO 1: Ranking geral por dataset
    # ------------------------------------------------------------------
    for ds in df_valid['dataset'].unique():
        df_ds = df_valid[df_valid['dataset'] == ds].copy()
        df_ds = df_ds.sort_values('f1_macro', ascending=True)
        df_ds['label'] = df_ds['algoritmo'] + ' (' + df_ds['metodo'] + ')'

        # Referências ALMMo-0 para este dataset
        if ds == 'v7':
            ref_baseline = 0.543
            ref_melhor = 0.598
            ref_aleatorio = 0.333
        else:  # v7_bin
            ref_baseline = 0.644  # ALMMo-0 binário Cost-Sensitive
            ref_melhor = 0.644
            ref_aleatorio = 0.500

        fig, ax = plt.subplots(figsize=(12, max(6, len(df_ds) * 0.35)))
        bars = ax.barh(
            range(len(df_ds)),
            df_ds['f1_macro'].values,
            color=[cores_algo.get(a, '#888') for a in df_ds['algoritmo']],
            alpha=0.8,
            height=0.7
        )
        ax.set_yticks(range(len(df_ds)))
        ax.set_yticklabels(df_ds['label'].values, fontsize=9)

        # Linhas de referência
        ax.axvline(ref_baseline, color='red', linestyle='--', linewidth=1.5,
                    label=f'ALMMo-0 baseline ({ref_baseline})')
        ax.axvline(ref_melhor, color='orange', linestyle='--', linewidth=1.5,
                    label=f'ALMMo-0 melhor ({ref_melhor})')
        ax.axvline(ref_aleatorio, color='gray', linestyle=':', linewidth=1,
                    label=f'Aleatório ({ref_aleatorio})')

        # Valores nas barras
        for i, v in enumerate(df_ds['f1_macro'].values):
            ax.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=8)

        ax.set_xlabel('F1-macro', fontsize=11)
        ax.set_title(f'Ranking F1-macro — Dataset {ds.upper()}', fontsize=13, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.set_xlim(0, min(1.0, df_ds['f1_macro'].max() + 0.08))
        plt.tight_layout()
        path = os.path.join(output_dir, f'grafico1_ranking_{ds}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ {path}")

    # ------------------------------------------------------------------
    # GRÁFICO 2: Heatmap F1-macro (algoritmo × tratamento) por dataset
    # ------------------------------------------------------------------
    for ds in df_valid['dataset'].unique():
        df_ds = df_valid[df_valid['dataset'] == ds]
        pivot = df_ds.pivot_table(
            index='algoritmo', columns='metodo', values='f1_macro', aggfunc='first'
        )
        # Ordenar colunas
        cols_order = [c for c in ['A', 'B', 'C', 'D', 'E'] if c in pivot.columns]
        pivot = pivot[cols_order]

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(
            pivot, annot=True, fmt='.3f', cmap='YlOrRd',
            linewidths=0.5, ax=ax, vmin=0.3, vmax=0.8,
            cbar_kws={'label': 'F1-macro'}
        )
        ax.set_title(f'Heatmap F1-macro — Dataset {ds.upper()}', fontsize=13, fontweight='bold')
        ax.set_xlabel('Método de Tratamento')
        ax.set_ylabel('Algoritmo')
        plt.tight_layout()
        path = os.path.join(output_dir, f'grafico2_heatmap_{ds}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ {path}")

    # ------------------------------------------------------------------
    # GRÁFICO 3: Trade-off Recall C0 vs Recall C1+C2 (só 3 classes)
    # ------------------------------------------------------------------
    for ds in ['v7']:  # Trade-off C0 vs C1+C2 só faz sentido com 3 classes
        df_ds = df_valid[(df_valid['dataset'] == ds) & (df_valid['n_classes'] == 3)].copy()
        if df_ds.empty:
            continue

        df_ds['recall_c1_c2_mean'] = df_ds.apply(
            lambda r: np.nanmean([
                r['recall_c1'] if r['recall_c1'] is not None else np.nan,
                r['recall_c2'] if r['recall_c2'] is not None else np.nan
            ]), axis=1
        )

        fig, ax = plt.subplots(figsize=(10, 8))

        for _, row in df_ds.iterrows():
            ax.scatter(
                row['recall_c0'], row['recall_c1_c2_mean'],
                color=cores_algo.get(row['algoritmo'], '#888'),
                s=80, alpha=0.7, edgecolors='black', linewidth=0.5
            )
            ax.annotate(
                f"{row['algoritmo']}({row['metodo']})",
                (row['recall_c0'], row['recall_c1_c2_mean']),
                fontsize=7, alpha=0.8,
                xytext=(5, 5), textcoords='offset points'
            )

        # ALMMo-0 referência
        if ds == 'v7':
            almmo_c0 = 0.771
            almmo_c1c2 = np.mean([0.341, 0.762])
            ax.scatter(almmo_c0, almmo_c1c2, color='black', s=150,
                       marker='*', zorder=5, label='ALMMo-0 v7 baseline')

        ax.set_xlabel('Recall C0 (Sem Irrigação)', fontsize=11)
        ax.set_ylabel('Recall Médio C1+C2 (Irrigação)', fontsize=11)
        ax.set_title(f'Trade-off Recall C0 vs C1+C2 — Dataset {ds.upper()}',
                      fontsize=13, fontweight='bold')
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.2, label='Equilíbrio perfeito')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(output_dir, f'grafico3_tradeoff_{ds}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ {path}")

    # ------------------------------------------------------------------
    # GRÁFICO 4: Comparação por dataset (melhor tratamento por algo)
    # ------------------------------------------------------------------
    melhor_por_algo_ds = df_valid.loc[
        df_valid.groupby(['algoritmo', 'dataset'])['f1_macro'].idxmax()
    ]

    datasets_presentes = sorted(melhor_por_algo_ds['dataset'].unique())
    algos_presentes = sorted(melhor_por_algo_ds['algoritmo'].unique())

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(algos_presentes))
    width = 0.25
    offsets = np.linspace(-width, width, len(datasets_presentes))

    for i, ds in enumerate(datasets_presentes):
        valores = []
        for algo in algos_presentes:
            val = melhor_por_algo_ds[
                (melhor_por_algo_ds['algoritmo'] == algo) &
                (melhor_por_algo_ds['dataset'] == ds)
            ]['f1_macro'].values
            valores.append(val[0] if len(val) > 0 else 0)

        bars = ax.bar(x + offsets[i], valores, width * 0.9,
                       label=f'Dataset {ds.upper()}', alpha=0.8)
        for j, v in enumerate(valores):
            if v > 0:
                ax.text(x[j] + offsets[i], v + 0.01, f'{v:.3f}',
                        ha='center', fontsize=7, rotation=45)

    # Linhas de referência ALMMo-0
    ax.axhline(0.543, color='red', linestyle='--', linewidth=1, alpha=0.7,
               label='ALMMo-0 v7 baseline (0.543)')
    ax.axhline(0.598, color='orange', linestyle='--', linewidth=1, alpha=0.7,
               label='ALMMo-0 v7 Cost-Sensitive (0.598)')
    ax.axhline(0.644, color='green', linestyle='--', linewidth=1, alpha=0.7,
               label='ALMMo-0 binário Cost-Sensitive (0.644)')

    ax.set_xlabel('Algoritmo', fontsize=11)
    ax.set_ylabel('F1-macro (melhor tratamento)', fontsize=11)
    ax.set_title('Comparação v7 (3 classes) vs v7_bin (binário) — Melhor Tratamento',
                  fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(algos_presentes, fontsize=10)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, 'grafico4_comparacao_datasets.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {path}")


# ========================================================================
# GERAÇÃO DO RELATÓRIO MARKDOWN
# ========================================================================

def gerar_relatorio(df_resultados, output_dir='.'):
    """Gera relatório completo em Markdown."""
    print(f"\n{'='*70}")
    print("GERANDO RELATÓRIO")
    print(f"{'='*70}")

    df = df_resultados.dropna(subset=['f1_macro']).copy()
    linhas = []

    def add(texto=''):
        linhas.append(texto)

    add("# Relatório de Benchmark: ALMMo-0 vs. Algoritmos Clássicos de ML")
    add()
    add("## 1. Objetivo")
    add()
    add("Este benchmark responde à pergunta: **o fraco desempenho em classes minoritárias ")
    add("é um problema do dataset ou do algoritmo ALMMo-0?**")
    add()
    add("Cinco algoritmos clássicos de ML foram avaliados nos mesmos datasets e com o ")
    add("mesmo protocolo de avaliação (Leave-Groups-Out) usado no cold start do ALMMo-0.")
    add()

    # ---- Tabelas por dataset ----
    for ds in sorted(df['dataset'].unique()):
        df_ds = df[df['dataset'] == ds]
        n_classes = df_ds['n_classes'].iloc[0]

        if ds == 'v7':
            titulo = "Dataset v7 — 3 classes, sem tratamento no dataset"
            ref_almmo = "ALMMo-0 baseline = 0.543 | ALMMo-0 Cost-Sensitive = 0.598"
        else:  # v7_bin
            titulo = "Dataset v7 binário — 2 classes (C1+C2 fundidas em classe 1)"
            ref_almmo = "ALMMo-0 binário Cost-Sensitive = 0.644"

        add(f"## 2.{['v7','v7_bin'].index(ds)+1}. {titulo}")
        add()
        add(f"**Referência:** {ref_almmo}")
        add()

        if n_classes == 3:
            add("| Algoritmo | Método | F1-macro | Rec.C0 | Rec.C1 | Rec.C2 | Prec.C1 | MAE | Adj% | Δ ALMMo-0 |")
            add("|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

            # Linha do ALMMo-0
            if ds == 'v7':
                add("| **ALMMo-0** | **A** | **0.543** | **0.771** | **0.341** | **0.762** | **0.125** | **0.294** | **87.4** | — |")
                add("| **ALMMo-0** | **B** | **0.598** | — | — | — | — | — | — | — |")

            for _, r in df_ds.sort_values(['algoritmo', 'metodo']).iterrows():
                delta = f"{r.get('delta_almmo_baseline', 0):+.4f}"
                mae = f"{r['mae_ordinal']:.4f}" if r.get('mae_ordinal') is not None else '—'
                adj = f"{r['erros_adj_pct']:.1f}" if r.get('erros_adj_pct') is not None else '—'
                rc2 = f"{r['recall_c2']:.4f}" if r.get('recall_c2') is not None else '—'
                add(f"| {r['algoritmo']} | {r['metodo']} | {r['f1_macro']:.4f} | "
                    f"{r['recall_c0']:.4f} | {r['recall_c1']:.4f} | {rc2} | "
                    f"{r['precision_c1']:.4f} | {mae} | {adj} | {delta} |")
        else:
            add("| Algoritmo | Método | F1-macro | Rec.C0 | Rec.C1 | Prec.C1 | Δ ALMMo-0 |")
            add("|---|:---:|:---:|:---:|:---:|:---:|:---:|")
            add("| **ALMMo-0** | **B** | **0.644** | — | — | — | — |")

            for _, r in df_ds.sort_values(['algoritmo', 'metodo']).iterrows():
                delta = f"{r.get('delta_almmo_baseline', 0):+.4f}"
                add(f"| {r['algoritmo']} | {r['metodo']} | {r['f1_macro']:.4f} | "
                    f"{r['recall_c0']:.4f} | {r['recall_c1']:.4f} | "
                    f"{r['precision_c1']:.4f} | {delta} |")

        add()

    # ---- Diagnóstico ----
    add("## 3. Diagnóstico")
    add()

    # Diagnóstico separado por cenário
    for ds_diag, ds_label, ref_almmo in [('v7', '3 classes', 0.598), ('v7_bin', 'binário', 0.644)]:
        df_diag = df[df['dataset'] == ds_diag]
        if df_diag.empty:
            continue

        melhor_f1_ds = df_diag['f1_macro'].max()
        melhor_row_ds = df_diag.loc[df_diag['f1_macro'].idxmax()]
        melhor_desc_ds = f"{melhor_row_ds['algoritmo']} com {melhor_row_ds['metodo_nome']}"

        add(f"### Cenário {ds_label} ({ds_diag})")
        add()
        add(f"**Melhor resultado:** F1-macro = {melhor_f1_ds:.4f} ({melhor_desc_ds})")
        add(f"**Referência ALMMo-0:** {ref_almmo}")
        add()

        if melhor_f1_ds >= 0.70:
            add("**Diagnóstico: PROBLEMA PREDOMINANTEMENTE DO ALGORITMO**")
            add()
            add(f"O melhor algoritmo clássico atingiu F1-macro = {melhor_f1_ds:.4f} (≥ 0.70). ")
            add("O dataset tem sinal discriminativo suficiente que algoritmos batch conseguem ")
            add("explorar, mas o ALMMo-0 não consegue na sua configuração atual.")
            add()
            add("**Recomendação:** Investigar se o ALMMo-0 pode incorporar Cost-Sensitive de ")
            add("forma mais agressiva, ou explorar arquitetura binária como solução de médio prazo.")
        elif melhor_f1_ds >= 0.60:
            add("**Diagnóstico: PROBLEMA MISTO (dataset + algoritmo)**")
            add()
            add(f"O melhor algoritmo clássico atingiu F1-macro = {melhor_f1_ds:.4f} (entre 0.60 e 0.70). ")
            add("Parte do problema é o dataset (desbalanceamento estrutural), parte é o algoritmo ")
            add("(limitações do ALMMo-0 online).")
            add()
            add("**Recomendação:** Prosseguir com o ALMMo-0 mas priorizar geração de dataset ")
            add("melhor calibrado (mais amostras de C1/C2 via calibração com dados reais de campo).")
        else:
            add("**Diagnóstico: PROBLEMA PREDOMINANTEMENTE DO DATASET**")
            add()
            add(f"O melhor algoritmo clássico atingiu F1-macro = {melhor_f1_ds:.4f} (< 0.60). ")
            add("O sinal discriminativo das 4 features é insuficiente para separar as classes ")
            add("com confiança, independentemente do algoritmo. O ALMMo-0 está competitivo ")
            add("dado suas restrições.")
            add()
            add("**Recomendação:** Focar em features adicionais (delta de tensão dia a dia, ")
            add("interação tensão × chuva) ou em mais dados simulados cobrindo casos-limite.")
        add()

    add()

    # ---- Comparações específicas ----
    add("### Comparações Específicas")
    add()

    # Random Forest vs ALMMo-0 (v7, 3 classes)
    rf_v7 = df[(df['algoritmo'] == 'Random Forest') & (df['dataset'] == 'v7')]
    rf_melhor = rf_v7['f1_macro'].max() if not rf_v7.empty else 0
    delta_rf = rf_melhor - 0.543
    if delta_rf >= 0.15:
        add(f"**Random Forest vs ALMMo-0 (3 classes):** Random Forest superou o ALMMo-0 em "
            f"{delta_rf*100:.1f} pontos percentuais (≥ 15pp). O ALMMo-0 tem desvantagem "
            f"estrutural significativa como classificador inicial. A escolha para o TCC "
            f"deve ser justificada pelas restrições de edge computing.")
    else:
        add(f"**Random Forest vs ALMMo-0 (3 classes):** Random Forest superou o ALMMo-0 em "
            f"{delta_rf*100:.1f} pontos percentuais (< 15pp). A diferença é moderada.")
    add()

    # KNN vs ALMMo-0 (v7, 3 classes)
    knn_v7 = df[(df['algoritmo'] == 'KNN') & (df['dataset'] == 'v7')]
    knn_melhor = knn_v7['f1_macro'].max() if not knn_v7.empty else None
    if knn_melhor is not None:
        delta_knn = abs(knn_melhor - 0.543)
        if delta_knn < 0.05:
            add(f"**KNN vs ALMMo-0:** Diferença de apenas {delta_knn*100:.1f}pp (< 5pp). "
                f"Resultado especialmente informativo — ambos usam distância euclidiana, "
                f"mas o KNN tem acesso a todos os dados de treino. A compactação de memória "
                f"do ALMMo-0 não causa perda significativa de informação.")
        else:
            add(f"**KNN vs ALMMo-0:** Diferença de {delta_knn*100:.1f}pp (≥ 5pp).")
    add()

    # v7_bin binário vs 3 classes
    if 'v7_bin' in df['dataset'].values:
        add("### Impacto da Reformulação Binária (v7 → v7_bin)")
        add()
        for algo in df['algoritmo'].unique():
            f1_3c = df[(df['algoritmo'] == algo) & (df['dataset'] == 'v7')]['f1_macro'].max()
            f1_2c = df[(df['algoritmo'] == algo) & (df['dataset'] == 'v7_bin')]['f1_macro'].max()
            if not np.isnan(f1_3c) and not np.isnan(f1_2c):
                ganho = f1_2c - f1_3c
                add(f"- **{algo}:** 3 classes={f1_3c:.4f} → binário={f1_2c:.4f} "
                    f"(Δ={ganho*100:+.1f}pp{' ✓ ≥10pp' if ganho >= 0.10 else ''})")

        todos_melhoram_10pp = True
        for algo in df['algoritmo'].unique():
            f1_3c = df[(df['algoritmo'] == algo) & (df['dataset'] == 'v7')]['f1_macro'].max()
            f1_2c = df[(df['algoritmo'] == algo) & (df['dataset'] == 'v7_bin')]['f1_macro'].max()
            if not np.isnan(f1_3c) and not np.isnan(f1_2c) and (f1_2c - f1_3c) < 0.10:
                todos_melhoram_10pp = False

        add()
        if todos_melhoram_10pp:
            add("**⚠ TODOS os algoritmos melhoram ≥ 10pp com formulação binária.** ")
            add("Recomendação forte: implementar ALMMo-0 binário para a fase de campo, ")
            add("com sub-classificação da classe 1 após acumulação de dados reais.")
        else:
            add("Nem todos os algoritmos melhoraram ≥ 10pp com a reformulação binária. ")
            add("O ganho é parcial e dependente do algoritmo/tratamento.")
    add()

    # ---- Referência a gráficos ----
    add("## 4. Gráficos")
    add()
    add("Os gráficos foram salvos no mesmo diretório deste relatório:")
    add()
    for ds in sorted(df['dataset'].unique()):
        add(f"- `grafico1_ranking_{ds}.png` — Ranking F1-macro dataset {ds.upper()}")
        add(f"- `grafico2_heatmap_{ds}.png` — Heatmap algoritmo × tratamento dataset {ds.upper()}")
    if 'v7' in df['dataset'].values:
        add(f"- `grafico3_tradeoff_v7.png` — Trade-off Recall C0 vs C1+C2 (apenas 3 classes)")
    add("- `grafico4_comparacao_datasets.png` — Comparação v7 (3 classes) vs v7_bin (binário)")
    add()

    # ---- Notas técnicas ----
    add("## 5. Notas Técnicas")
    add()
    add("- Split: Leave-Groups-Out com 6 grupos de teste (6, 10, 14, 18, 22, 28)")
    add("- Normalização: StandardScaler fitado no treino")
    add("- Todos os resampling (SMOTE, ADASYN) aplicados apenas ao treino")
    add("- Hiperparâmetros: defaults do scikit-learn, sem tuning")
    add("- Random state: 42 em todos os algoritmos e resampling")
    add("- KNN não inclui método B (Cost-Sensitive) por não ter class_weight nativo")
    add("- XGBoost multiclasse usa sample_weight via compute_sample_weight('balanced')")
    add("- Dataset v7_bin gerado a partir do v7: classes 1 e 2 fundidas → classe 1 (irrigação necessária)")

    # ADASYN fallbacks
    fallback_info = df[df['adasyn_fallback'] == True]
    if not fallback_info.empty:
        add()
        add("**ADASYN fallbacks acionados:**")
        for _, r in fallback_info.iterrows():
            add(f"- Dataset {r['dataset']}: ADASYN falhou, usou SMOTE integral como fallback")

    add()
    add("---")
    add("*Relatório gerado automaticamente por benchmark_script.py*")

    # Salvar
    path = os.path.join(output_dir, 'relatorio_benchmark.md')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(linhas))
    print(f"  ✓ {path}")


# ========================================================================
# MAIN
# ========================================================================

def main():
    t_inicio = time.time()

    print("=" * 70)
    print("BENCHMARK: ALMMo-0 vs. Algoritmos Clássicos de ML")
    print("=" * 70)

    # Verificar dataset v7
    if not os.path.exists(DATASET_V7):
        print(f"\n❌ ERRO: {DATASET_V7} não encontrado!")
        print(f"   Coloque o arquivo CSV no mesmo diretório deste script.")
        return

    df_v7 = pd.read_csv(DATASET_V7)
    print(f"\n✓ Dataset v7: {DATASET_V7} ({len(df_v7)} amostras)")
    print(f"  Features: {[c for c in df_v7.columns if c in FEATURES]}")
    print(f"  Classes (3): {dict(Counter(df_v7[TARGET]))}")

    # Criar cenário binário a partir do v7: C1 e C2 → classe 1
    df_v7_bin = df_v7.copy()
    df_v7_bin[TARGET] = (df_v7_bin[TARGET] >= 1).astype(int)  # 0 fica 0, 1 e 2 viram 1
    print(f"\n✓ Dataset v7_bin: gerado a partir do v7 (C1+C2 → classe 1)")
    print(f"  Classes (2): {dict(Counter(df_v7_bin[TARGET]))}")

    datasets_disponiveis = {
        'v7':     {'df': df_v7,     'n_classes': 3},
        'v7_bin': {'df': df_v7_bin, 'n_classes': 2},
    }

    # Executar benchmark para cada cenário
    todos_resultados = []
    for nome, config in datasets_disponiveis.items():
        resultados = executar_benchmark_dataset(
            config['df'], nome, config['n_classes']
        )
        todos_resultados.extend(resultados)

    # Consolidar resultados
    df_resultados = pd.DataFrame(todos_resultados)

    # Salvar CSV
    csv_path = 'benchmark_resultados.csv'
    df_resultados.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"\n✓ Resultados salvos em: {csv_path}")

    # Gerar gráficos
    gerar_graficos(df_resultados)

    # Gerar relatório
    gerar_relatorio(df_resultados)

    # Resumo final
    t_total = time.time() - t_inicio
    print(f"\n{'='*70}")
    print(f"CONCLUÍDO em {t_total:.1f} segundos")
    print(f"{'='*70}")

    df_valid = df_resultados.dropna(subset=['f1_macro'])
    if not df_valid.empty:
        # Melhor por cenário
        for ds in df_valid['dataset'].unique():
            df_ds = df_valid[df_valid['dataset'] == ds]
            best = df_ds.loc[df_ds['f1_macro'].idxmax()]
            ref = 0.543 if ds == 'v7' else 0.644
            print(f"\n  Melhor {ds}: {best['algoritmo']} ({best['metodo_nome']})")
            print(f"    F1-macro = {best['f1_macro']:.4f} (Δ ALMMo-0 = {best['f1_macro'] - ref:+.4f})")

    print(f"\nArquivos gerados:")
    print(f"  - benchmark_resultados.csv")
    print(f"  - relatorio_benchmark.md")
    print(f"  - grafico1_ranking_*.png")
    print(f"  - grafico2_heatmap_*.png")
    print(f"  - grafico3_tradeoff_v7.png (apenas 3 classes)")
    print(f"  - grafico4_comparacao_datasets.png")


if __name__ == '__main__':
    main()