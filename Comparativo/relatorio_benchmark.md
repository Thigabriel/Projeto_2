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
| KNN | A | 0.5245 | 0.9299 | 0.0732 | 0.4286 | 0.0833 | 0.1773 | 84.5 | -0.0185 |
| KNN | C | 0.5830 | 0.8103 | 0.4146 | 0.8095 | 0.1789 | 0.2523 | 85.0 | +0.0400 |
| KNN | D | 0.5685 | 0.7732 | 0.4878 | 0.8571 | 0.1818 | 0.2870 | 82.8 | +0.0255 |
| KNN | E | 0.5667 | 0.7670 | 0.5122 | 0.8571 | 0.1842 | 0.2907 | 83.1 | +0.0237 |
| LogReg | A | 0.3642 | 0.9938 | 0.0000 | 0.0952 | 0.0000 | 0.1554 | 65.1 | -0.1788 |
| LogReg | B | 0.1928 | 0.0536 | 0.9268 | 0.9048 | 0.0907 | 0.9945 | 82.8 | -0.3502 |
| LogReg | C | 0.4238 | 0.9134 | 0.0000 | 0.6667 | 0.0000 | 0.2541 | 45.6 | -0.1192 |
| LogReg | D | 0.2016 | 0.0701 | 0.9268 | 0.8571 | 0.0916 | 0.9762 | 83.2 | -0.3414 |
| LogReg | E | 0.1952 | 0.0701 | 0.9024 | 0.8571 | 0.0907 | 0.9890 | 81.9 | -0.3478 |
| Random Forest | A | 0.4110 | 0.9546 | 0.0488 | 0.1429 | 0.0833 | 0.1773 | 77.2 | -0.1320 |
| Random Forest | B | 0.4507 | 0.9505 | 0.0488 | 0.2381 | 0.0800 | 0.1755 | 78.5 | -0.0923 |
| Random Forest | C | 0.5623 | 0.9010 | 0.2683 | 0.5238 | 0.2200 | 0.1956 | 78.4 | +0.0193 |
| Random Forest | D | 0.5353 | 0.8639 | 0.2195 | 0.6190 | 0.1475 | 0.2340 | 79.2 | -0.0077 |
| Random Forest | E | 0.5363 | 0.8619 | 0.2439 | 0.6667 | 0.1724 | 0.2395 | 75.2 | -0.0067 |
| SVC | A | 0.3439 | 1.0000 | 0.0000 | 0.0476 | 0.0000 | 0.1481 | 67.2 | -0.1991 |
| SVC | B | 0.3933 | 0.4247 | 0.9024 | 0.9048 | 0.1510 | 0.6545 | 74.4 | -0.1497 |
| SVC | C | 0.4740 | 0.7443 | 0.2683 | 0.8095 | 0.1111 | 0.3620 | 74.7 | -0.0690 |
| SVC | D | 0.4209 | 0.4784 | 0.8780 | 0.9048 | 0.1614 | 0.5996 | 73.8 | -0.1221 |
| SVC | E | 0.4166 | 0.4577 | 0.9024 | 0.9524 | 0.1595 | 0.6161 | 74.3 | -0.1264 |
| XGBoost | A | 0.5041 | 0.9320 | 0.1463 | 0.3333 | 0.1714 | 0.1828 | 78.0 | -0.0389 |
| XGBoost | B | 0.5541 | 0.9093 | 0.3171 | 0.3810 | 0.2500 | 0.1883 | 78.8 | +0.0111 |
| XGBoost | C | 0.5822 | 0.8804 | 0.3171 | 0.6190 | 0.2131 | 0.2048 | 80.9 | +0.0392 |
| XGBoost | D | 0.5800 | 0.8639 | 0.3659 | 0.6190 | 0.2113 | 0.2157 | 82.0 | +0.0370 |
| XGBoost | E | 0.5976 | 0.8515 | 0.4146 | 0.7143 | 0.2237 | 0.2212 | 81.4 | +0.0546 |

## 2.2. Dataset v7 binário — 2 classes (C1+C2 fundidas em classe 1)

**Referência:** ALMMo-0 binário Cost-Sensitive = 0.644

| Algoritmo | Método | F1-macro | Rec.C0 | Rec.C1 | Prec.C1 | Δ ALMMo-0 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **ALMMo-0** | **B** | **0.644** | — | — | — | — |
| KNN | A | 0.5723 | 0.9237 | 0.2097 | 0.2600 | -0.0717 |
| KNN | C | 0.6273 | 0.8454 | 0.5000 | 0.2925 | -0.0167 |
| KNN | D | 0.6244 | 0.7814 | 0.6613 | 0.2789 | -0.0196 |
| KNN | E | 0.6303 | 0.7835 | 0.6774 | 0.2857 | -0.0137 |
| LogReg | A | 0.4992 | 0.9938 | 0.0323 | 0.4000 | -0.1448 |
| LogReg | B | 0.1815 | 0.0804 | 0.9839 | 0.1203 | -0.4625 |
| LogReg | C | 0.5981 | 0.9443 | 0.2258 | 0.3415 | -0.0459 |
| LogReg | D | 0.1817 | 0.0825 | 0.9677 | 0.1188 | -0.4623 |
| LogReg | E | 0.1778 | 0.0784 | 0.9677 | 0.1183 | -0.4662 |
| Random Forest | A | 0.5461 | 0.9505 | 0.1290 | 0.2500 | -0.0979 |
| Random Forest | B | 0.5123 | 0.9464 | 0.0806 | 0.1613 | -0.1317 |
| Random Forest | C | 0.5993 | 0.9175 | 0.2742 | 0.2982 | -0.0447 |
| Random Forest | D | 0.6171 | 0.8742 | 0.4032 | 0.2907 | -0.0269 |
| Random Forest | E | 0.6051 | 0.8825 | 0.3548 | 0.2785 | -0.0389 |
| SVC | A | 0.4863 | 1.0000 | 0.0161 | 1.0000 | -0.1577 |
| SVC | B | 0.4800 | 0.4825 | 0.9355 | 0.1877 | -0.1640 |
| SVC | C | 0.5465 | 0.8000 | 0.3548 | 0.1849 | -0.0975 |
| SVC | D | 0.4797 | 0.5216 | 0.7903 | 0.1744 | -0.1643 |
| SVC | E | 0.4949 | 0.5052 | 0.9355 | 0.1946 | -0.1491 |
| XGBoost | A | 0.5891 | 0.9340 | 0.2258 | 0.3043 | -0.0549 |
| XGBoost | B | 0.6195 | 0.9072 | 0.3387 | 0.3182 | -0.0245 |
| XGBoost | C | 0.5816 | 0.9052 | 0.2581 | 0.2581 | -0.0624 |
| XGBoost | D | 0.6309 | 0.8763 | 0.4355 | 0.3103 | -0.0131 |
| XGBoost | E | 0.6085 | 0.8866 | 0.3548 | 0.2857 | -0.0355 |

## 3. Diagnóstico

### Cenário 3 classes (v7)

**Melhor resultado:** F1-macro = 0.5976 (XGBoost com ADASYN)
**Referência ALMMo-0:** 0.598

**Diagnóstico: PROBLEMA PREDOMINANTEMENTE DO DATASET**

O melhor algoritmo clássico atingiu F1-macro = 0.5976 (< 0.60). 
O sinal discriminativo das 4 features é insuficiente para separar as classes 
com confiança, independentemente do algoritmo. O ALMMo-0 está competitivo 
dado suas restrições.

**Recomendação:** Focar em features adicionais (delta de tensão dia a dia, 
interação tensão × chuva) ou em mais dados simulados cobrindo casos-limite.

### Cenário binário (v7_bin)

**Melhor resultado:** F1-macro = 0.6309 (XGBoost com SMOTE Integral)
**Referência ALMMo-0:** 0.644

**Diagnóstico: PROBLEMA MISTO (dataset + algoritmo)**

O melhor algoritmo clássico atingiu F1-macro = 0.6309 (entre 0.60 e 0.70). 
Parte do problema é o dataset (desbalanceamento estrutural), parte é o algoritmo 
(limitações do ALMMo-0 online).

**Recomendação:** Prosseguir com o ALMMo-0 mas priorizar geração de dataset 
melhor calibrado (mais amostras de C1/C2 via calibração com dados reais de campo).


### Comparações Específicas

**Random Forest vs ALMMo-0 (3 classes):** Random Forest superou o ALMMo-0 em 1.9 pontos percentuais (< 15pp). A diferença é moderada.

**KNN vs ALMMo-0:** Diferença de apenas 4.0pp (< 5pp). Resultado especialmente informativo — ambos usam distância euclidiana, mas o KNN tem acesso a todos os dados de treino. A compactação de memória do ALMMo-0 não causa perda significativa de informação.

### Impacto da Reformulação Binária (v7 → v7_bin)

- **LogReg:** 3 classes=0.4238 → binário=0.5981 (Δ=+17.4pp ✓ ≥10pp)
- **SVC:** 3 classes=0.4740 → binário=0.5465 (Δ=+7.3pp)
- **Random Forest:** 3 classes=0.5623 → binário=0.6171 (Δ=+5.5pp)
- **KNN:** 3 classes=0.5830 → binário=0.6303 (Δ=+4.7pp)
- **XGBoost:** 3 classes=0.5976 → binário=0.6309 (Δ=+3.3pp)

Nem todos os algoritmos melhoraram ≥ 10pp com a reformulação binária. 
O ganho é parcial e dependente do algoritmo/tratamento.

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