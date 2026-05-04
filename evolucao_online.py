#!/usr/bin/env python3
"""
============================================================================
Modulo de Evolucao Online — Ensemble ALMMo-0 v12
============================================================================

Como funciona:
  1. O ensemble toma uma decisao (C0/C1/C2) no dia T.
  2. No dia T+1, o sensor de tensao registra o novo valor.
  3. O modulo calcula o delta real (tensao_hoje - tensao_ontem).
  4. Compara com a distribuicao esperada para a classe decidida no dia T.
     (distribuicao calculada a partir do dataset de treino — Imperatriz 2001-2023)
  5. Se o delta real for estatisticamente anomalo (|z| > LIMIAR_Z),
     a decisao do dia T foi provavelmente errada.
  6. Um pseudo-label e inferido pela direcao da anomalia.
  7. O ensemble e treinado com (features_dia_T, pseudo_label).

Nenhuma regra humana e codificada.
Os limiares emergem da propria distribuicao dos dados de treino.

Uso:
    from evolucao_online import EvolucaoOnline

    evolucao = EvolucaoOnline(modelos, verbose=True)

    # No loop diario:
    decisao = ensemble.predizer(features_hoje)
    evolucao.registrar(features_hoje, decisao['classe'])

    # No dia seguinte, antes de predizer:
    evento = evolucao.observar(tensao_hoje)
    if evento:
        print(evento)  # reporta se houve aprendizado

============================================================================
"""

import numpy as np
import json
import csv
from pathlib import Path
from datetime import datetime
from collections import Counter


# ============================================================================
# DISTRIBUICOES DE REFERENCIA
# Calculadas do dataset_cold_start_v11.csv (Imperatriz, 2001-2023)
# Coluna: delta do DIA SEGUINTE dado a classe decidida hoje
#
# Como foram calculadas:
#   df_sorted = df.sort_values(['grupo_id','dap'])
#   df['delta_proximo_dia'] = df.groupby('grupo_id')['delta_tensao_kpa'].shift(-1)
#   por classe: media, std, percentis
# ============================================================================
DIST_REFERENCIA = {
    # classe: {media, std, p5, p95, p99, n}
    0: {'media': +0.826, 'std': 2.636, 'p5': -4.12, 'p95': +4.43, 'p99': +6.38, 'n': 8458},
    1: {'media': +0.013, 'std': 0.498, 'p5': -1.33, 'p95': +0.71, 'p99': +1.20, 'n': 1723},
    2: {'media': -24.996, 'std': 9.422, 'p5': -47.03, 'p95': -15.92, 'p99': -15.42, 'n': 265},
}

# Limiar de anomalia em desvios padrao
# 2.0 = top/bottom 2.3% da distribuicao historica
# 2.5 = top/bottom 0.6% — mais conservador, menos retreinos
LIMIAR_Z_PADRAO = 2.0

NOMES_CLASSE = {0: 'C0-SemIrrig', 1: 'C1-Manutencao', 2: 'C2-Intensiva'}


# ============================================================================
# ALMMo-0 com metodo learn() (necessario para evolucao online)
# ============================================================================
class ALMMo0:
    def __init__(self, n_inputs=5, r_threshold=0.5, max_rules=50,
                 age_limit=100, epsilon=1e-8, n_classes=3,
                 min_rules_per_class=3):
        self.n_inputs = n_inputs
        self.r_threshold = r_threshold
        self.max_rules = max_rules
        self.age_limit = age_limit
        self.epsilon = epsilon
        self.n_classes = n_classes
        self.min_rules_per_class = min_rules_per_class
        self.rules = []
        self.input_mean = np.zeros(n_inputs)
        self.input_std = np.ones(n_inputs)
        self.n_samples_seen = 0
        self.n_rules_created = 0
        self.n_rules_pruned = 0

    def normalize(self, x):
        return (x - self.input_mean) / (self.input_std + 1e-10)

    def _dists(self, x_n):
        return np.array([
            np.sqrt(np.sum((x_n - np.array(r['center']))**2))
            for r in self.rules
        ])

    def _create(self, x_n, y):
        self.rules.append({
            'center': x_n.copy(),
            'consequent': int(y),
            'age': 0,
            'activations': 1
        })
        self.n_rules_created += 1

    def _update(self, idx, x_n, y, dist):
        r = self.rules[idx]
        r['activations'] += 1
        eta = 1.0 / r['activations']
        r['center'] = r['center'] + eta * (x_n - r['center'])
        if r['consequent'] != y:
            af = max(0.0, 1.0 - dist / self.r_threshold)
            ee = eta * af
            if ee > 0:
                nc = (1 - ee) * r['consequent'] + ee * y
                r['consequent'] = int(np.round(np.clip(nc, 0, self.n_classes - 1)))
        r['age'] = 0

    def _age(self, idx):
        for i, r in enumerate(self.rules):
            if i != idx:
                r['age'] += 1

    def _prune(self):
        n0 = len(self.rules)
        cc = Counter(r['consequent'] for r in self.rules)
        surv = []
        for r in self.rules:
            if r['age'] <= self.age_limit:
                surv.append(r)
            else:
                if cc[r['consequent']] > self.min_rules_per_class:
                    cc[r['consequent']] -= 1
                else:
                    r['age'] = 0
                    surv.append(r)
        if len(surv) > self.max_rules:
            surv.sort(key=lambda r: r['age'], reverse=True)
            cc2 = Counter(r['consequent'] for r in surv)
            kept = []
            for r in surv:
                if len(kept) >= self.max_rules:
                    if cc2[r['consequent']] <= self.min_rules_per_class:
                        kept.append(r)
                    else:
                        cc2[r['consequent']] -= 1
                else:
                    kept.append(r)
            surv = kept
        self.rules = surv
        self.n_rules_pruned += n0 - len(self.rules)

    def learn(self, x, y):
        """Atualiza o modelo com um novo par (features, label)."""
        self.n_samples_seen += 1
        x_n = self.normalize(x)
        if not self.rules:
            self._create(x_n, y)
            return
        d = self._dists(x_n)
        idx = int(np.argmin(d))
        dm = d[idx]
        if dm > self.r_threshold:
            self._create(x_n, y)
        else:
            self._update(idx, x_n, y, dm)
        self._age(idx)
        self._prune()

    def predict(self, x):
        x_n = self.normalize(x)
        if not self.rules:
            return 0, np.ones(self.n_classes) / self.n_classes
        d = self._dists(x_n)
        w = 1.0 / (d**2 + self.epsilon)
        v = np.zeros(self.n_classes)
        for i, r in enumerate(self.rules):
            v[r['consequent']] += w[i]
        s = v.sum()
        return int(np.argmax(v)), v / s if s > 0 else v

    def n_rules_por_classe(self):
        return {c: sum(1 for r in self.rules if r['consequent'] == c)
                for c in range(self.n_classes)}

    @classmethod
    def from_dict(cls, d):
        m = cls(
            n_inputs=d['n_inputs'],
            r_threshold=d['r_threshold'],
            max_rules=d['max_rules'],
            age_limit=d['age_limit'],
            n_classes=d.get('n_classes', 3),
            min_rules_per_class=d.get('min_rules_per_class', 3)
        )
        m.rules = [
            {**r, 'center': np.array(r['center'])}
            for r in d['rules']
        ]
        m.input_mean = np.array(d['input_mean'])
        m.input_std = np.array(d['input_std'])
        m.n_samples_seen = d.get('n_samples_seen', 0)
        m.n_rules_created = d.get('n_rules_created', 0)
        m.n_rules_pruned = d.get('n_rules_pruned', 0)
        return m


# ============================================================================
# MODULO DE EVOLUCAO ONLINE
# ============================================================================
class EvolucaoOnline:
    """
    Detecta anomalias no delta do dia seguinte e dispara retreino do ensemble.

    Parametros
    ----------
    modelos : list[ALMMo0]
        Lista de modelos do ensemble (com metodo learn()).
    limiar_z : float
        Limiar em desvios padrao para considerar anomalia. Padrao: 2.0.
    verbose : bool
        Se True, imprime eventos de aprendizado no terminal.
    log_path : str | None
        Caminho para CSV de log de eventos de aprendizado.
        Se None, nao salva log.
    """

    def __init__(self, modelos, limiar_z=LIMIAR_Z_PADRAO,
                 verbose=True, log_path='evolucao_online_log.csv'):
        self.modelos = modelos
        self.limiar_z = limiar_z
        self.verbose = verbose
        self.log_path = log_path

        # Estado do dia anterior (aguardando observacao)
        self._pendente = None   # (features_array, classe_decidida, data_str)

        # Distribuicoes de referencia (podem ser atualizadas via ancoragem)
        self.dist = {c: dict(v) for c, v in DIST_REFERENCIA.items()}

        # Contadores de sessao
        self.n_observacoes = 0
        self.n_anomalias   = 0
        self.n_retreinos   = 0

        # Inicializa log CSV
        if self.log_path:
            p = Path(self.log_path)
            if not p.exists():
                with open(p, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'data', 'classe_decidida', 'delta_real',
                        'delta_esperado', 'std_esperado', 'z_score',
                        'anomalia', 'pseudo_label',
                        'features_tensao', 'features_delta',
                        'n_regras_antes', 'n_regras_depois'
                    ])

    # ------------------------------------------------------------------
    # API PUBLICA
    # ------------------------------------------------------------------

    def registrar(self, features, classe_decidida, data=None):
        """
        Registra a decisao tomada hoje para avaliacao amanha.

        Chame APOS a decisao do ensemble, ANTES de avancar o dia.

        Parametros
        ----------
        features : array-like, shape (5,)
            [tensao_solo_kpa, chuva_acum_3d_mm, tmax_max_3d_c, dap, delta_tensao_kpa]
            NOTA: em simulacao AquaCrop, tensao_solo_kpa e shiftada (dia anterior).
            No campo real com sensor, use a leitura direta — sem shift.
        classe_decidida : int
            Classe escolhida pelo ensemble (0, 1 ou 2).
        data : str, opcional
            Data da decisao (para o log). Se None, usa datetime.now().
        """
        self._pendente = (
            np.array(features, dtype=float),
            int(classe_decidida),
            data or datetime.now().strftime('%Y-%m-%d')
        )

    def observar(self, tensao_hoje, tensao_ontem=None, delta_direto=None):
        """
        Observa a consequencia da decisao registrada ontem.

        Forneca tensao_hoje + tensao_ontem OU delta_direto.

        Parametros
        ----------
        tensao_hoje : float
            Tensao do solo medida hoje (kPa).
        tensao_ontem : float, opcional
            Tensao de ontem (para calcular delta internamente).
        delta_direto : float, opcional
            Delta ja calculado externamente (tensao_hoje - tensao_ontem).

        Retorna
        -------
        dict com resultado da observacao, ou None se nao ha pendente.
        """
        if self._pendente is None:
            return None

        features, classe_decidida, data = self._pendente
        self._pendente = None
        self.n_observacoes += 1

        # Calcular delta real
        if delta_direto is not None:
            delta_real = float(delta_direto)
        elif tensao_ontem is not None:
            delta_real = float(tensao_hoje) - float(tensao_ontem)
        else:
            raise ValueError("Forneca tensao_ontem ou delta_direto.")

        # Distribuicao de referencia para a classe decidida
        ref = self.dist[classe_decidida]
        delta_esp = ref['media']
        std_esp   = ref['std']

        # Z-score
        z = (delta_real - delta_esp) / (std_esp + 1e-10)

        resultado = {
            'data': data,
            'classe_decidida': classe_decidida,
            'nome_classe': NOMES_CLASSE[classe_decidida],
            'delta_real': round(delta_real, 3),
            'delta_esperado': round(delta_esp, 3),
            'z_score': round(z, 3),
            'anomalia': False,
            'pseudo_label': None,
            'retreino': False,
        }

        # Detecta anomalia
        if abs(z) >= self.limiar_z:
            self.n_anomalias += 1
            resultado['anomalia'] = True
            pseudo_label = self._inferir_pseudo_label(
                delta_real, z, classe_decidida
            )
            resultado['pseudo_label'] = pseudo_label
            resultado['nome_pseudo_label'] = NOMES_CLASSE.get(pseudo_label, '?')

            # Retreina apenas se o pseudo-label for diferente da decisao
            if pseudo_label != classe_decidida:
                n_antes = self._total_regras()
                self._treinar_ensemble(features, pseudo_label)
                n_depois = self._total_regras()
                self.n_retreinos += 1
                resultado['retreino'] = True
                resultado['n_regras_antes'] = n_antes
                resultado['n_regras_depois'] = n_depois

                if self.verbose:
                    self._print_evento(resultado, features)

            self._salvar_log(resultado, features)

        return resultado

    def status(self):
        """Retorna resumo do estado atual do modulo."""
        regras = {}
        for i, m in enumerate(self.modelos):
            rc = m.n_rules_por_classe()
            regras[i] = rc
        total_regras = self._total_regras()
        media_regras = total_regras / len(self.modelos) if self.modelos else 0

        return {
            'n_observacoes': self.n_observacoes,
            'n_anomalias':   self.n_anomalias,
            'n_retreinos':   self.n_retreinos,
            'taxa_anomalia_pct': round(self.n_anomalias / max(self.n_observacoes, 1) * 100, 1),
            'total_regras':  total_regras,
            'media_regras_por_modelo': round(media_regras, 1),
            'dist_referencia': {
                c: {'media': self.dist[c]['media'], 'std': self.dist[c]['std']}
                for c in self.dist
            }
        }

    def imprimir_status(self):
        s = self.status()
        sep = '-' * 50
        print(sep)
        print("EVOLUCAO ONLINE — Status")
        print(f"  Observacoes: {s['n_observacoes']} | Anomalias: {s['n_anomalias']} "
              f"({s['taxa_anomalia_pct']}%) | Retreinos: {s['n_retreinos']}")
        print(f"  Regras totais: {s['total_regras']} | Media por modelo: {s['media_regras_por_modelo']:.1f}")
        print(f"  Limiar Z: {self.limiar_z}")
        print("  Distribuicoes de referencia:")
        for c, d in s['dist_referencia'].items():
            print(f"    C{c}: delta_esperado = {d['media']:+.3f} ± {d['std']:.3f} kPa")
        print(sep)

    # ------------------------------------------------------------------
    # METODOS INTERNOS
    # ------------------------------------------------------------------

    def _inferir_pseudo_label(self, delta_real, z, classe_decidida):
        """
        Infere o pseudo-label com base na direcao e magnitude da anomalia.

        Logica direcional — sem regras de tensao codificadas:
          - Decidiu C0 (sem irrigar) e solo secou MUITO mais que o esperado
            (z >> 0) → deveria ter sido C2 (irrigacao era necessaria)

          - Decidiu C0 mas solo ficou estavel ou umidificou sem chuva
            (z << 0) → C0 era excessivamente conservador, confirmamos C0
            (nao retreina — nao ha acao diferente a recomendar)

          - Decidiu C1 (manutencao) mas solo continuou secando alem do esperado
            (z >> 0) → a irrigacao de manutencao nao foi suficiente → C2

          - Decidiu C2 (intensiva) mas delta nao foi suficientemente negativo
            (z >> 0, ou seja delta menos negativo que esperado) →
            irrigacao nao surtiu efeito esperado (solo muito seco, solo compactado,
            ou irrigacao subestimada) — nao retreina para evitar ruido

        Retorna
        -------
        int : pseudo_label (0, 1 ou 2)
        """
        if classe_decidida == 0:
            # Solo secou muito mais que o esperado apos C0
            # → o ponto de feature deste dia nao era C0
            if z > 0:
                return 2   # deveria ter irrigado intensivamente
            else:
                return 0   # anomalia negativa: chuva ou humidade nao prevista, C0 era certo

        elif classe_decidida == 1:
            # Irrigacao de manutencao nao segurou a tensao
            if z > 0:
                return 2   # precisava de irrigacao mais intensa
            else:
                return 0   # tensao caiu mais que esperado → C0 teria bastado

        elif classe_decidida == 2:
            # Irrigacao intensa aplicada mas resultado anomalo
            # Conservador: nao retreina (muitos fatores externos possiveis)
            return 2

        return classe_decidida

    def _treinar_ensemble(self, features, pseudo_label):
        """Chama learn() em todos os modelos do ensemble."""
        x = np.array(features, dtype=float)
        y = int(pseudo_label)
        for modelo in self.modelos:
            modelo.learn(x, y)

    def _total_regras(self):
        return sum(len(m.rules) for m in self.modelos)

    def _print_evento(self, resultado, features):
        print(f"\n[EVOLUCAO] {resultado['data']}")
        print(f"  Decisao:      {resultado['nome_classe']}")
        print(f"  Delta real:   {resultado['delta_real']:+.2f} kPa")
        print(f"  Delta esperd: {resultado['delta_esperado']:+.2f} ± {self.dist[resultado['classe_decidida']]['std']:.2f} kPa")
        print(f"  Z-score:      {resultado['z_score']:+.2f}  (limiar={self.limiar_z})")
        print(f"  Pseudo-label: {resultado.get('nome_pseudo_label','?')}  → retreinando ensemble")
        print(f"  Features:     tensao={features[0]:.1f} delta={features[4]:.2f} chuva={features[1]:.1f} tmax={features[2]:.1f} dap={features[3]:.0f}")
        if 'n_regras_antes' in resultado:
            print(f"  Regras:       {resultado['n_regras_antes']} → {resultado['n_regras_depois']}")

    def _salvar_log(self, resultado, features):
        if not self.log_path:
            return
        with open(self.log_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                resultado['data'],
                resultado['classe_decidida'],
                resultado['delta_real'],
                resultado['delta_esperado'],
                self.dist[resultado['classe_decidida']]['std'],
                resultado['z_score'],
                resultado['anomalia'],
                resultado.get('pseudo_label', ''),
                round(features[0], 2),   # tensao
                round(features[4], 3),   # delta
                resultado.get('n_regras_antes', ''),
                resultado.get('n_regras_depois', ''),
            ])


# ============================================================================
# FUNCAO AUXILIAR — recalcular distribuicoes do seu proprio dataset
# ============================================================================
def calcular_distribuicoes_do_dataset(csv_path):
    """
    Recalcula as distribuicoes de referencia a partir de um dataset local.
    Use se voce tiver um dataset diferente do Imperatriz 2001-2023.

    Parametros
    ----------
    csv_path : str
        Caminho para dataset_cold_start_vXX_full.csv
        (precisa ter colunas: grupo_id, dap, delta_tensao_kpa, classe_irrigacao)

    Retorna
    -------
    dict com distribuicoes por classe
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    df_sorted = df.sort_values(['grupo_id', 'dap']).copy()
    df_sorted['delta_proximo_dia'] = (
        df_sorted.groupby('grupo_id')['delta_tensao_kpa'].shift(-1)
    )
    df_seq = df_sorted.dropna(subset=['delta_proximo_dia'])

    dist = {}
    for c in sorted(df_seq['classe_irrigacao'].unique()):
        s = df_seq[df_seq['classe_irrigacao'] == c]['delta_proximo_dia']
        dist[int(c)] = {
            'media': round(float(s.mean()), 4),
            'std':   round(float(s.std()),  4),
            'p5':    round(float(s.quantile(0.05)), 4),
            'p95':   round(float(s.quantile(0.95)), 4),
            'p99':   round(float(s.quantile(0.99)), 4),
            'n':     int(len(s)),
        }
        print(f"  C{c}: n={dist[int(c)]['n']} | media={dist[int(c)]['media']:+.3f} "
              f"| std={dist[int(c)]['std']:.3f} | p95={dist[int(c)]['p95']:.2f}")

    return dist
