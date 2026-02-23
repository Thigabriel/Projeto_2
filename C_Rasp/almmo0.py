# almmo0.py — ALMMo-0 Classificador Evolucionário (compatível com memoria_cold_start_v7.pkl)
#
# O modelo é persistido como dict Python (não como instância de classe),
# o que garante compatibilidade máxima entre versões do código.
#
# Estrutura do pkl (formato real do cold start v7):
#   {
#     'rules'              : list[dict]  — center, consequent, age, activations
#     'input_mean'         : ndarray (4,)
#     'input_std'          : ndarray (4,)
#     'r_threshold'        : float
#     'max_rules'          : int
#     'age_limit'          : int
#     'n_inputs'           : int
#     'n_classes'          : int
#     'min_rules_per_class': int
#     'n_samples_seen'     : int
#     'n_rules_created'    : int
#     'n_rules_pruned'     : int
#     'created_at'         : str (ISO)
#     'saved_at'           : str (ISO)
#   }
#
# Activação: Cauchy sem scatter — 1 / (1 + dist² / r_threshold²)
#
# Melhorias M1+M2+M3 em learn():
#   M1 — Fusão de regras muito próximas (evita explosão em campo)
#   M2 — Protecção de regras com alto activation count (anti-esquecimento)
#   M3 — Garantia de min_rules_per_class por classe
#
# Interface pública:
#   modelo.predict(x)               → int
#   modelo.predict_com_confianca(x) → (int, float)
#   modelo.learn(x, label)          → None
#   modelo.rules                    → list[dict]
#   modelo.input_mean               → ndarray (R/W)
#   modelo.input_std                → ndarray (R/W)
#   modelo.salvar(path)             → None
#   modelo.info()                   → str
#   modelo.distribuicao_regras()    → dict

import numpy as np
import pickle
from datetime import datetime


class ALMMo0:
    """
    Wrapper em torno do dict pkl do cold start v7.
    Carrega, usa e actualiza o modelo preservando o formato original.
    """

    # Activações mínimas para protecção M2 (regra não é removida por idade)
    _M2_ACTIVATIONS_PROTEGIDAS = 5

    # Fracção de r_threshold abaixo da qual duas regras são fundidas (M1)
    _M1_FUSAO_FATOR = 0.5

    def __init__(self, estado):
        """
        Não usar directamente — usar ALMMo0.carregar() ou carregar_modelo().
        'estado' é o dict completo do pkl.
        """
        self._s = estado

    # ------------------------------------------------------------------
    # PROPRIEDADES PÚBLICAS
    # ------------------------------------------------------------------

    @property
    def rules(self):
        return self._s['rules']

    @property
    def input_mean(self):
        return self._s['input_mean']

    @input_mean.setter
    def input_mean(self, value):
        self._s['input_mean'] = np.asarray(value, dtype=float)

    @property
    def input_std(self):
        return self._s['input_std']

    @input_std.setter
    def input_std(self, value):
        self._s['input_std'] = np.asarray(value, dtype=float)

    @property
    def r_threshold(self):
        return float(self._s['r_threshold'])

    # ------------------------------------------------------------------
    # NORMALIZAÇÃO
    # ------------------------------------------------------------------

    def _normalizar(self, x):
        std_safe = np.where(self._s['input_std'] < 1e-8, 1.0, self._s['input_std'])
        return (np.asarray(x, dtype=float) - self._s['input_mean']) / std_safe

    # ------------------------------------------------------------------
    # ACTIVAÇÃO DE CAUCHY
    # ------------------------------------------------------------------

    def _activacao(self, x_norm, center):
        """
        Cauchy com parâmetro de largura = r_threshold.
        activation = 1 / (1 + ||x - center||² / r²)
        Fiel ao formato do pkl — sem scatter por regra.
        """
        dist_sq = float(np.sum((x_norm - np.asarray(center)) ** 2))
        return 1.0 / (1.0 + dist_sq / (self.r_threshold ** 2))

    # ------------------------------------------------------------------
    # PREDICT
    # ------------------------------------------------------------------

    def predict(self, x):
        """
        Classifica x (não normalizado).
        Retorna int: 0=sem irrigação, 1=moderada, 2=intensa.
        """
        if not self._s['rules']:
            return 0

        x_norm = self._normalizar(x)
        scores = np.zeros(self._s['n_classes'])

        for rule in self._s['rules']:
            act = self._activacao(x_norm, rule['center'])
            scores[rule['consequent']] += act
            rule['activations'] += 1

        return int(np.argmax(scores))

    def predict_com_confianca(self, x):
        """Retorna (classe, confianca) onde confianca ∈ [0, 1]."""
        if not self._s['rules']:
            return 0, 0.0

        x_norm = self._normalizar(x)
        scores = np.zeros(self._s['n_classes'])

        for rule in self._s['rules']:
            act = self._activacao(x_norm, rule['center'])
            scores[rule['consequent']] += act
            rule['activations'] += 1

        total = scores.sum()
        if total < 1e-10:
            return 0, 0.0

        classe    = int(np.argmax(scores))
        confianca = float(scores[classe] / total)
        return classe, confianca

    # ------------------------------------------------------------------
    # LEARN — actualização online
    # ------------------------------------------------------------------

    def learn(self, x, label):
        """
        Actualiza o modelo com um novo exemplo rotulado.

        1. Procura regra mais próxima com o mesmo label
        2. Se dist < r_threshold → absorve (actualiza centro por média online)
        3. Caso contrário → cria nova regra
        4. Pruning por idade (M2 protege regras activas)
        5. M1: fusão de regras próximas
        6. M3: garante min_rules_per_class
        """
        x      = np.asarray(x, dtype=float)
        x_norm = self._normalizar(x)
        label  = int(label)

        self._s['n_samples_seen'] += 1

        # Envelhecer todas as regras
        for rule in self._s['rules']:
            rule['age'] += 1

        # Procurar regra mais próxima com o mesmo consequente
        melhor_regra = None
        melhor_dist  = float('inf')

        for rule in self._s['rules']:
            if rule['consequent'] != label:
                continue
            dist = float(np.sqrt(np.sum(
                (x_norm - np.asarray(rule['center'])) ** 2
            )))
            if dist < melhor_dist:
                melhor_dist  = dist
                melhor_regra = rule

        if melhor_regra is not None and melhor_dist <= self.r_threshold:
            # Absorver: média online do centro
            n = melhor_regra['activations'] + 1
            delta = x_norm - np.asarray(melhor_regra['center'])
            melhor_regra['center']      = np.asarray(melhor_regra['center']) + delta / n
            melhor_regra['activations'] = n
            melhor_regra['age']         = 0  # renovar ao ser actualizada
        else:
            # Criar nova regra
            self._s['rules'].append({
                'center'     : x_norm.copy(),
                'consequent' : label,
                'age'        : 0,
                'activations': 1,
                'created_at' : datetime.now().isoformat(),
            })
            self._s['n_rules_created'] += 1

        # Pruning + melhorias
        self._pruning_por_idade()
        self._m1_fusao_regras()
        self._m3_garantir_minimo_por_classe()

    # ------------------------------------------------------------------
    # PRUNING POR IDADE (com M2)
    # ------------------------------------------------------------------

    def _pruning_por_idade(self):
        """
        Remove regras antigas COM poucas activações.
        M2: regras com activations >= _M2_ACTIVATIONS_PROTEGIDAS são imunes.
        """
        antes = len(self._s['rules'])
        self._s['rules'] = [
            r for r in self._s['rules']
            if (r['age'] < self._s['age_limit']
                or r['activations'] >= self._M2_ACTIVATIONS_PROTEGIDAS)
        ]
        self._s['n_rules_pruned'] += antes - len(self._s['rules'])

    # ------------------------------------------------------------------
    # M1 — Fusão de regras similares
    # ------------------------------------------------------------------

    def _m1_fusao_regras(self):
        """
        Funde pares de regras da mesma classe com centros muito próximos.
        Centro fundido é ponderado pelo activation count.
        """
        limite = self.r_threshold * self._M1_FUSAO_FATOR
        feito  = True

        while feito:
            feito = False
            for i in range(len(self._s['rules'])):
                for j in range(i + 1, len(self._s['rules'])):
                    ri = self._s['rules'][i]
                    rj = self._s['rules'][j]

                    if ri['consequent'] != rj['consequent']:
                        continue

                    ci   = np.asarray(ri['center'])
                    cj   = np.asarray(rj['center'])
                    dist = float(np.sqrt(np.sum((ci - cj) ** 2)))

                    if dist < limite:
                        ni, nj = ri['activations'], rj['activations']
                        total  = ni + nj
                        self._s['rules'][i] = {
                            'center'     : (ni * ci + nj * cj) / total,
                            'consequent' : ri['consequent'],
                            'age'        : min(ri['age'], rj['age']),
                            'activations': total,
                            'created_at' : ri.get('created_at', ''),
                        }
                        self._s['rules'].pop(j)
                        self._s['n_rules_pruned'] += 1
                        feito = True
                        break
                if feito:
                    break

    # ------------------------------------------------------------------
    # M3 — Garantir mínimo de regras por classe
    # ------------------------------------------------------------------

    def _m3_garantir_minimo_por_classe(self):
        """
        Se uma classe ficar abaixo de min_rules_per_class,
        duplica a regra mais activa com pequena perturbação.
        """
        for classe in range(self._s['n_classes']):
            regras_classe = [r for r in self._s['rules']
                             if r['consequent'] == classe]
            deficit = self._s['min_rules_per_class'] - len(regras_classe)

            if deficit <= 0 or not regras_classe:
                continue

            melhor = max(regras_classe, key=lambda r: r['activations'])
            n_dims = len(np.asarray(melhor['center']))

            for _ in range(min(deficit, 3)):
                pert = np.random.normal(0, self.r_threshold * 0.1, size=n_dims)
                self._s['rules'].append({
                    'center'     : np.asarray(melhor['center']) + pert,
                    'consequent' : classe,
                    'age'        : 0,
                    'activations': 1,
                    'created_at' : datetime.now().isoformat(),
                })
                self._s['n_rules_created'] += 1

    # ------------------------------------------------------------------
    # SERIALIZAÇÃO — preserva formato dict original
    # ------------------------------------------------------------------

    def salvar(self, path):
        """Guarda o modelo no formato dict original (compatível com pkl externo)."""
        self._s['saved_at'] = datetime.now().isoformat()
        with open(path, 'wb') as f:
            pickle.dump(self._s, f)

    # ------------------------------------------------------------------
    # UTILITÁRIOS
    # ------------------------------------------------------------------

    def distribuicao_regras(self):
        dist = {i: 0 for i in range(self._s['n_classes'])}
        for r in self._s['rules']:
            dist[r['consequent']] = dist.get(r['consequent'], 0) + 1
        return dist

    def info(self):
        dist = self.distribuicao_regras()
        return (f"ALMMo-0 | {len(self._s['rules'])} regras | "
                f"C0:{dist[0]} C1:{dist[1]} C2:{dist[2]} | "
                f"r={self.r_threshold} | seen={self._s['n_samples_seen']}")


# ------------------------------------------------------------------
# CARREGAMENTO COM FALLBACK
# ------------------------------------------------------------------

def carregar_modelo(pkl_campo, pkl_inicial):
    """
    Tenta carregar pkl_campo (modelo actualizado em campo).
    Se falhar, tenta pkl_inicial (cold start original).
    Lança RuntimeError se ambos falharem — não cria modelo falso.

    O pkl deve ser um dict no formato do cold start v7.
    """
    for path, label in [(pkl_campo, 'campo'), (pkl_inicial, 'inicial')]:
        try:
            with open(path, 'rb') as f:
                estado = pickle.load(f)

            campos = ['rules', 'input_mean', 'input_std', 'r_threshold', 'n_classes']
            for campo in campos:
                assert campo in estado, f"Campo ausente no pkl: '{campo}'"

            modelo = ALMMo0(estado)
            print(f"[ALMMo0] Carregado ({label}): {modelo.info()}")
            return modelo, label

        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"[ALMMo0] ERRO ao carregar {path}: {e}")

    raise RuntimeError(
        f"Nenhum pkl válido encontrado.\n"
        f"  pkl_campo  : {pkl_campo}\n"
        f"  pkl_inicial: {pkl_inicial}\n"
        f"Copie memoria_cold_start_v7.pkl para o directório do projecto."
    )
