# Relatório de Benchmark: ALMMo-0 vs. Algoritmos Clássicos de ML

## 1. Objetivo

Este benchmark responde à pergunta: **o fraco desempenho em classes minoritárias 
é um problema do dataset ou do algoritmo ALMMo-0?**

Cinco algoritmos clássicos de ML foram avaliados nos mesmos datasets e com o 
mesmo protocolo de avaliação (Leave-Groups-Out) usado no cold start do ALMMo-0.

## 2.1. Dataset v7 — 3 classes, sem tratamento no dataset

**Referência:** ALMMo-0 baseline = 0.543 | ALMMo-0 Cost-Sensitive = 0.598

| Algoritmo | Método | F1-macro | Rec.C0 | Rec.C1 | Rec.C2 | Prec.C1 | MAE | Adj% | Δ ALMMo-0 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **ALMMo-0** | **A** | **0.543** | **0.771** | **0.341** | **0.762** | **0.125** | **0.294** | **87.4** | — |
| **ALMMo-0** | **B** | **0.598** | — | — | — | — | — | — | — |
| KNN | A | 0.3256 | 0.9949 | 0.0244 | 0.0000 | 0.3333 | 0.1634 | 72.4 | -0.2174 |
| KNN | C | 0.3898 | 0.9040 | 0.0732 | 0.2500 | 0.1667 | 0.2715 | 60.2 | -0.1532 |
| KNN | D | 0.3864 | 0.7955 | 0.2195 | 0.2500 | 0.1837 | 0.3929 | 57.6 | -0.1566 |
| KNN | E | 0.3807 | 0.7677 | 0.2439 | 0.2500 | 0.1818 | 0.4283 | 56.3 | -0.1623 |
| LogReg | A | 0.3110 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1611 | 71.9 | -0.2320 |
| LogReg | B | 0.3300 | 0.5783 | 0.3902 | 0.2500 | 0.1429 | 0.6336 | 59.3 | -0.2130 |
| LogReg | C | 0.3110 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1611 | 71.9 | -0.2320 |
| LogReg | D | 0.3151 | 0.5227 | 0.4146 | 0.2500 | 0.1532 | 0.7329 | 52.4 | -0.2279 |
| LogReg | E | 0.3283 | 0.6111 | 0.3902 | 0.1250 | 0.1509 | 0.5982 | 59.6 | -0.2147 |
| Random Forest | A | 0.3234 | 0.9874 | 0.0244 | 0.0000 | 0.1667 | 0.1700 | 73.8 | -0.2196 |
| Random Forest | B | 0.3234 | 0.9874 | 0.0244 | 0.0000 | 0.1667 | 0.1700 | 73.8 | -0.2196 |
| Random Forest | C | 0.3796 | 0.9343 | 0.0488 | 0.1875 | 0.1818 | 0.2384 | 61.5 | -0.1634 |
| Random Forest | D | 0.3879 | 0.8687 | 0.1463 | 0.1875 | 0.1765 | 0.3024 | 63.0 | -0.1551 |
| Random Forest | E | 0.3801 | 0.8535 | 0.1463 | 0.1875 | 0.1818 | 0.3311 | 58.5 | -0.1629 |
| SVC | A | 0.3110 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1611 | 71.9 | -0.2320 |
| SVC | B | 0.3491 | 0.6742 | 0.2195 | 0.3125 | 0.1765 | 0.5960 | 43.0 | -0.1939 |
| SVC | C | 0.3110 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1611 | 71.9 | -0.2320 |
| SVC | D | 0.3165 | 0.5631 | 0.2439 | 0.3750 | 0.1370 | 0.7373 | 43.9 | -0.2265 |
| SVC | E | 0.3156 | 0.5909 | 0.1951 | 0.3750 | 0.1231 | 0.7064 | 43.9 | -0.2274 |
| XGBoost | A | 0.3234 | 0.9874 | 0.0244 | 0.0000 | 0.1667 | 0.1700 | 73.8 | -0.2196 |
| XGBoost | B | 0.3321 | 0.9722 | 0.0488 | 0.0000 | 0.1538 | 0.1810 | 75.8 | -0.2109 |
| XGBoost | C | 0.3646 | 0.9268 | 0.0976 | 0.0625 | 0.1739 | 0.2340 | 69.1 | -0.1784 |
| XGBoost | D | 0.3532 | 0.8535 | 0.1463 | 0.0625 | 0.1500 | 0.3245 | 63.9 | -0.1898 |
| XGBoost | E | 0.3859 | 0.8359 | 0.2195 | 0.1250 | 0.2368 | 0.3554 | 55.0 | -0.1571 |

## 2.2. Dataset v7 binário — 2 classes (C1+C2 fundidas em classe 1)

**Referência:** ALMMo-0 binário Cost-Sensitive = 0.644

| Algoritmo | Método | F1-macro | Rec.C0 | Rec.C1 | Prec.C1 | Δ ALMMo-0 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **ALMMo-0** | **B** | **0.644** | — | — | — | — |
| KNN | A | 0.4824 | 0.9949 | 0.0175 | 0.3333 | -0.1616 |
| KNN | C | 0.5338 | 0.9571 | 0.1053 | 0.2609 | -0.1102 |
| KNN | D | 0.5358 | 0.8409 | 0.2456 | 0.1818 | -0.1082 |
| KNN | E | 0.5438 | 0.8308 | 0.2807 | 0.1928 | -0.1002 |
| LogReg | A | 0.4664 | 1.0000 | 0.0000 | 0.0000 | -0.1776 |
| LogReg | B | 0.5400 | 0.7929 | 0.3333 | 0.1881 | -0.1040 |
| LogReg | C | 0.4664 | 1.0000 | 0.0000 | 0.0000 | -0.1776 |
| LogReg | D | 0.5335 | 0.7626 | 0.3684 | 0.1826 | -0.1105 |
| LogReg | E | 0.5221 | 0.7626 | 0.3333 | 0.1681 | -0.1219 |
| Random Forest | A | 0.4806 | 0.9899 | 0.0175 | 0.2000 | -0.1634 |
| Random Forest | B | 0.4797 | 0.9874 | 0.0175 | 0.1667 | -0.1643 |
| Random Forest | C | 0.4945 | 0.9848 | 0.0351 | 0.2500 | -0.1495 |
| Random Forest | D | 0.5499 | 0.8636 | 0.2456 | 0.2059 | -0.0941 |
| Random Forest | E | 0.5777 | 0.8712 | 0.2982 | 0.2500 | -0.0663 |
| SVC | A | 0.4664 | 1.0000 | 0.0000 | 0.0000 | -0.1776 |
| SVC | B | 0.5137 | 0.8864 | 0.1404 | 0.1509 | -0.1303 |
| SVC | C | 0.4664 | 1.0000 | 0.0000 | 0.0000 | -0.1776 |
| SVC | D | 0.5012 | 0.7677 | 0.2632 | 0.1402 | -0.1428 |
| SVC | E | 0.5059 | 0.8889 | 0.1228 | 0.1373 | -0.1381 |
| XGBoost | A | 0.4968 | 0.9899 | 0.0351 | 0.3333 | -0.1472 |
| XGBoost | B | 0.5046 | 0.9747 | 0.0526 | 0.2308 | -0.1394 |
| XGBoost | C | 0.5307 | 0.9722 | 0.0877 | 0.3125 | -0.1133 |
| XGBoost | D | 0.5532 | 0.8687 | 0.2456 | 0.2121 | -0.0908 |
| XGBoost | E | 0.5359 | 0.8662 | 0.2105 | 0.1846 | -0.1081 |

## 3. Diagnóstico

### Cenário 3 classes (v7)

**Melhor resultado:** F1-macro = 0.3898 (KNN com SMOTE Parcial 15%)
**Referência ALMMo-0:** 0.598

**Diagnóstico: PROBLEMA PREDOMINANTEMENTE DO DATASET**

O melhor algoritmo clássico atingiu F1-macro = 0.3898 (< 0.60). 
O sinal discriminativo das 4 features é insuficiente para separar as classes 
com confiança, independentemente do algoritmo. O ALMMo-0 está competitivo 
dado suas restrições.

**Recomendação:** Focar em features adicionais (delta de tensão dia a dia, 
interação tensão × chuva) ou em mais dados simulados cobrindo casos-limite.

### Cenário binário (v7_bin)

**Melhor resultado:** F1-macro = 0.5777 (Random Forest com ADASYN)
**Referência ALMMo-0:** 0.644

**Diagnóstico: PROBLEMA PREDOMINANTEMENTE DO DATASET**

O melhor algoritmo clássico atingiu F1-macro = 0.5777 (< 0.60). 
O sinal discriminativo das 4 features é insuficiente para separar as classes 
com confiança, independentemente do algoritmo. O ALMMo-0 está competitivo 
dado suas restrições.

**Recomendação:** Focar em features adicionais (delta de tensão dia a dia, 
interação tensão × chuva) ou em mais dados simulados cobrindo casos-limite.


### Comparações Específicas

**Random Forest vs ALMMo-0 (3 classes):** Random Forest superou o ALMMo-0 em -15.5 pontos percentuais (< 15pp). A diferença é moderada.

**KNN vs ALMMo-0:** Diferença de 15.3pp (≥ 5pp).

### Impacto da Reformulação Binária (v7 → v7_bin)

- **LogReg:** 3 classes=0.3300 → binário=0.5400 (Δ=+21.0pp ✓ ≥10pp)
- **SVC:** 3 classes=0.3491 → binário=0.5137 (Δ=+16.5pp ✓ ≥10pp)
- **Random Forest:** 3 classes=0.3879 → binário=0.5777 (Δ=+19.0pp ✓ ≥10pp)
- **KNN:** 3 classes=0.3898 → binário=0.5438 (Δ=+15.4pp ✓ ≥10pp)
- **XGBoost:** 3 classes=0.3859 → binário=0.5532 (Δ=+16.7pp ✓ ≥10pp)

**⚠ TODOS os algoritmos melhoram ≥ 10pp com formulação binária.** 
Recomendação forte: implementar ALMMo-0 binário para a fase de campo, 
com sub-classificação da classe 1 após acumulação de dados reais.

## 4. Gráficos

Os gráficos foram salvos no mesmo diretório deste relatório:

- `grafico1_ranking_v7.png` — Ranking F1-macro dataset V7
- `grafico2_heatmap_v7.png` — Heatmap algoritmo × tratamento dataset V7
- `grafico1_ranking_v7_bin.png` — Ranking F1-macro dataset V7_BIN
- `grafico2_heatmap_v7_bin.png` — Heatmap algoritmo × tratamento dataset V7_BIN
- `grafico3_tradeoff_v7.png` — Trade-off Recall C0 vs C1+C2 (apenas 3 classes)
- `grafico4_comparacao_datasets.png` — Comparação v7 (3 classes) vs v7_bin (binário)

## 5. Notas Técnicas

- Split: Leave-Groups-Out com 6 grupos de teste (6, 10, 14, 18, 22, 28)
- Normalização: StandardScaler fitado no treino
- Todos os resampling (SMOTE, ADASYN) aplicados apenas ao treino
- Hiperparâmetros: defaults do scikit-learn, sem tuning
- Random state: 42 em todos os algoritmos e resampling
- KNN não inclui método B (Cost-Sensitive) por não ter class_weight nativo
- XGBoost multiclasse usa sample_weight via compute_sample_weight('balanced')
- Dataset v7_bin gerado a partir do v7: classes 1 e 2 fundidas → classe 1 (irrigação necessária)

---
*Relatório gerado automaticamente por benchmark_script.py*