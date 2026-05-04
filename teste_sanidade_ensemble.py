"""
TESTE DE SANIDADE — Ensemble ALMMo-0 v12
=========================================
Injeta vetores de features conhecidos directamente no ensemble,
sem NASA POWER API nem SimuladorArduino.
Verifica se o modelo detecta correctamente C0, C1 e C2.

Features (ordem): [tensao_solo_kpa, chuva_acum_3d_mm, tmax_max_3d_c, dap, delta_tensao_kpa]

Execução:
    python teste_sanidade_ensemble.py
"""
import pickle
import numpy as np
from collections import Counter

# =============================================================================
# ALMMo-0 (necessário para reconstruir modelos do pkl)
# =============================================================================
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

    def normalize(self, x): return (x - self.input_mean) / self.input_std

    def predict(self, x):
        x_n = self.normalize(x)
        d = np.array([np.sqrt(np.sum((x_n - np.array(r['center']))**2)) for r in self.rules])
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
        m.rules = d['rules']
        m.input_mean = np.array(d['input_mean'])
        m.input_std  = np.array(d['input_std'])
        m.n_samples_seen = d.get('n_samples_seen', 0)
        return m

# =============================================================================
# CARREGA ENSEMBLE
# =============================================================================
PKL_PATH = "memoria_cold_start_v12_ensemble.pkl"

with open(PKL_PATH, "rb") as f:
    raw = pickle.load(f)

modelos = [ALMMo0.from_dict(d) for d in raw["models"]]
print(f"Ensemble carregado: {len(modelos)} modelos\n")

# =============================================================================
# CASOS DE TESTE
# Cada caso: (label, vetor, classe_esperada)
# Vetor: [tensao_kPa, chuva_3d_mm, tmax_3d_C, dap, delta_kPa]
# =============================================================================
CASOS = [
    # --- C0: solo húmido, chuva recente, tensão caindo ---
    ("C0 — pós-chuva (tensão baixa, delta negativo)",
     [15.0, 40.0, 29.0, 45, -8.0],  0),

    ("C0 — solo estável, sem stress (tensão baixa, sem variação)",
     [20.0, 14.1, 30.4, 59, 0.83],  0),   # amostra central real do treino

    ("C0 — tensão moderada mas recuperando após evento",
     [28.0, 18.0, 31.0, 30, -3.5],  0),

    # --- C1: banda estreita 33–36 kPa, delta quasi-nulo ---
    ("C1 — tensão na banda de manutenção (34 kPa, delta ≈ 0)",
     [34.0, 0.0, 36.0, 65, 0.0],    1),   # amostra central real do treino

    ("C1 — tensão 35 kPa com delta muito pequeno",
     [35.2, 0.1, 35.5, 72, 0.3],    1),

    ("C1 — limite superior da banda C1",
     [36.0, 0.0, 37.0, 80, -0.2],   1),

    # --- C2: tensão moderada subindo rápido (delta alto) ---
    ("C2 — tensão 34 kPa subindo +4.3/dia (amostra central real)",
     [34.0, 0.0, 34.7, 67, 4.31],   2),   # amostra central real do treino

    ("C2 — tensão 38 kPa subindo +5/dia",
     [38.0, 0.0, 35.0, 65, 5.0],    2),

    ("C2 — tensão 45 kPa subindo +4.25/dia",
     [46.9, 1.5, 31.4, 54, 4.25],   2),   # amostra real do treino

    ("C2 — tensão 60 kPa subindo +7.7/dia (stress extremo)",
     [60.1, 0.1, 38.0, 103, 7.69],  2),   # amostra real do treino

    # --- Casos críticos da simulação de campo ---
    ("C2 — SIM DAP54: tensão 39 kPa subindo +3.6 (campo real)",
     [39.0, 2.0, 33.5, 54, 3.60],   2),

    ("C2 — SIM DAP56: tensão 46 kPa subindo +4.3 (campo real)",
     [45.6, 7.4, 34.5, 56, 4.34],   2),

    ("C2 — SIM 120 kPa delta=0 (stress extremo sem limite)",
     [120.0, 0.0, 36.0, 60, 0.0],   2),   # fisicamente insustentável sem irrigar
]

# =============================================================================
# VOTAÇÃO PONDERADA — custo assimétrico de errar em irrigação
# Não irrigar quando devia (stress da planta) custa mais do que
# irrigar sem necessidade (desperdício de água).
# =============================================================================
PESO_CLASSE = {0: 1.0, 1: 1.5, 2: 2.0}
NOMES = {0: "C0-SemIrrig", 1: "C1-Manutencao", 2: "C2-Intensiva"}
BAR = "─" * 80

print(f"Pesos de votação: C0={PESO_CLASSE[0]}  C1={PESO_CLASSE[1]}  C2={PESO_CLASSE[2]}\n")
print(BAR)
print(f"{'CASO':<52} {'ESPERADO':<14} {'OBTIDO':<14} {'RESULTADO'}")
print(BAR)

aprovados = 0
reprovados = 0
detalhes = []

for label, vetor, esperado in CASOS:
    x = np.array(vetor, dtype=float)
    votos_brutos = Counter(m.predict(x)[0] for m in modelos)
    score = {c: votos_brutos.get(c, 0) * PESO_CLASSE[c] for c in range(3)}
    classe_final = max(score, key=score.get)
    consenso = votos_brutos.get(classe_final, 0) / len(modelos) * 100

    ok = classe_final == esperado
    simbolo = "PASS" if ok else "FAIL"
    if ok:
        aprovados += 1
    else:
        reprovados += 1

    print(f"  {label[:50]:<50}  {NOMES[esperado]:<14} {NOMES[classe_final]:<14} {simbolo}  [{consenso:.0f}% votos brutos]")
    detalhes.append((label, vetor, esperado, classe_final, votos_brutos, score))

print(BAR)
print(f"\nRESULTADO: {aprovados}/{len(CASOS)} aprovados  |  {reprovados} reprovados\n")

if reprovados > 0:
    print("=== DETALHES DOS REPROVADOS ===")
    for label, vetor, esp, obt, votos, score in detalhes:
        if obt != esp:
            print(f"\n  [{label}]")
            print(f"  Vetor: tensao={vetor[0]} chuva={vetor[1]} tmax={vetor[2]} dap={vetor[3]} delta={vetor[4]}")
            print(f"  Votos brutos: {dict(votos)}")
            print(f"  Score ponderado: C0={score[0]:.1f}  C1={score[1]:.1f}  C2={score[2]:.1f}")
            print(f"  Esperado: {NOMES[esp]}  |  Obtido: {NOMES[obt]}")
else:
    print("Todos os casos passaram. O ensemble com votacao ponderada esta a funcionar correctamente.")
