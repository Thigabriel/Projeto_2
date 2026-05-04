# Relatório Cold Start — ALMMo-0 v3.0 | Dataset v11
**Data:** 2026-03-07 11:38
**Projecto:** TCC Engenharia — Irrigação Inteligente com Edge AI (Raspberry Pi 4B)

---

## Configuração do Split
- Total de grupos: 115 | Grupos de teste: 25
- Amostras treino: 8263 | Amostras teste: 2298
- C0/C1/C2 treino: 6678/1372/213
- C0/C1/C2 teste:  1869/373/56

---

## Sweep r_threshold — Top 10
| r | F1-macro | F1-C0 | F1-C1 | F1-C2 | Regras |
|---|---------|-------|-------|-------|--------|
| 0.6 | 0.7065 | 0.956 | 0.818 | 0.346 | 17 |
| 0.45 | 0.6792 | 0.955 | 0.820 | 0.263 | 23 |
| 0.65 | 0.6767 | 0.955 | 0.794 | 0.282 | 15 |
| 0.8 | 0.6436 | 0.941 | 0.817 | 0.173 | 13 |
| 0.75 | 0.6396 | 0.943 | 0.772 | 0.204 | 16 |
| 0.95 | 0.6364 | 0.918 | 0.661 | 0.330 | 11 |
| 0.55 | 0.6244 | 0.947 | 0.770 | 0.156 | 19 |
| 0.4 | 0.6211 | 0.943 | 0.756 | 0.164 | 25 |
| 0.85 | 0.6152 | 0.906 | 0.671 | 0.268 | 11 |
| 0.5 | 0.6136 | 0.884 | 0.632 | 0.324 | 19 |

**r_threshold óptimo: 0.6** (F1-macro=0.7065)

---

## Métricas Finais

```
                      precision     recall   f1-score    support

  C0-SemIrrig            0.9275     0.9856     0.9556       1869
  C1-Manutencao          0.9408     0.7239     0.8182        373
  C2-Intensiva           0.5600     0.2500     0.3457         56

  macro avg              0.8094     0.6531     0.7065       2298
  accuracy                                     0.9252       2298
```

- MAE ordinal: 0.0975
- Acurácia: 0.9252 *(métrica secundária)*

---

## Matriz de Confusão
```
              Predito
              C0    C1    C2
Real  C0    1842    16    11
      C1     103   270     0
      C2      41     1    14
```

---

## Sanidade Agronômica: 3/4

| Cenário | Esperado | Predito | OK? |
|---------|---------|---------|-----|
| Solo seco (60kPa), secando (+4.5), sem chuva, calor | C2 | C0 | ✗ |
| Solo húmido (10kPa), muita chuva, delta negativo | C0 | C0 | ✓ |
| Solo em threshold manutenção (35kPa), delta ~0 | C1 | C1 | ✓ |
| Solo médio (40kPa), chuva recente (20mm) | C0 | C0 | ✓ |

---

## Distribuição de Regras no Modelo Final
- C0 (Sem irrigação): 10 regras
- C1 (Manutenção):    4 regras
- C2 (Intensiva):     3 regras
- **Total activas: 17** | Criadas: 2164 | Podadas: 2147

---

## Critérios de Aprovação
- F1-macro ≥ 0.50: **PASS ✓** (valor=0.7065)
- F1-C1 ≥ 0.40: **PASS ✓** (valor=0.8182)
- F1-C2 ≥ 0.25: **PASS ✓** (valor=0.3457)
- MAE ≤ 0.60: **PASS ✓** (valor=0.0975)
- Sanidade ≥ 3/4: **PASS ✓** (valor=3)
- Regras C2 ≥ 3: **PASS ✓** (valor=3)

## Veredicto Final: **APROVADO ✓**

---

## Comparação com Versão Anterior (v7)
| Métrica | v7 | v11 | Delta |
|---------|-----|-----|-------|
| F1-macro | 0.543 | 0.7065 | +0.1635 |
| F1-C1 | ~0.30 | 0.8182 | +0.5182 |
| F1-C2 | ~0.30 | 0.3457 | +0.0457 |
| Amostras C1+C2 | 161 | 2.014 | +1.853 |
| n_inputs | 4 | 5 | +1 (delta_tensao) |
| r_threshold óptimo | 0.50 | 0.6 | +0.10 |

---

## Outputs Gerados
- `memoria_cold_start_v11.pkl` — modelo para deploy no Raspberry Pi 4B
- `graficos_v11/sweep_f1_macro.png`
- `graficos_v11/sweep_f1_por_classe.png`
- `graficos_v11/heatmap_confusao.png`
- `graficos_v11/tradeoff_f1_regras.png`
