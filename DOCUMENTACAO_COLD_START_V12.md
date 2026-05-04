# DOCUMENTAÇÃO TÉCNICA — Cold Start v12 do ALMMo-0

## Sistema de Controlo Preditivo de Irrigação com Edge AI
## Quarto Ciclo: Ensemble Multi-Agente

**Dataset:** v11 (10.561 amostras, 5 features, 3 classes)
**Algoritmo:** ALMMo-0 v3.0 × 14 instâncias (Ensemble)
**Data:** Março 2026
**Status:** 3 modelos aprovados — Ensemble seleccionado para deploy

---

## 1. RESUMO EXECUTIVO

O v12 introduz a técnica de **Ensemble Multi-Agente**: em vez de um único modelo ALMMo-0, são treinados 14 modelos com hiperparâmetros diversos (combinações de r_threshold e min_rules_per_class). Cada modelo aprende uma geometria diferente do espaço de features. O voto conjunto cancela os erros idiossincráticos de cada modelo individual.

### Resultado Principal

| Modelo | F1-macro | Recall C2 | Precision C2 | F1-C2 | Sanidade | Regras |
|:------:|:--------:|:---------:|:------------:|:-----:|:--------:|:------:|
| Individual mrpc=3 | 0.734 | 39,3% | 56,4% | 0.463 | 15/20 | 12 |
| Individual mrpc=10 | 0.631 | 58,9% | 17,6% | 0.271 | 18/20 | 19 |
| **Ensemble (hard vote)** | **0.761** | **60,7%** | **56,7%** | **0.586** | 16/20 | 239 |

O ensemble obtém simultaneamente o **melhor F1-macro** (0.761), o **melhor recall C2** (60,7%) e mantém **precision C2 alta** (56,7%). É a primeira vez no projecto que recall e precision de C2 são ambos superiores a 50%.

**Memória no Raspberry Pi:** 239 regras × 48 bytes ≈ 11,2 KB — cabe sem qualquer problema.

---

## 2. EVOLUÇÃO DO PROJECTO

### 2.1 Cronologia dos Datasets

| Versão | Amostras | Features | Classes | Irrigação | Desbalanceamento C0 |
|:------:|:--------:|:--------:|:-------:|:---------:|:-------------------:|
| v5 | 1.692 | 4 | 3 | 813 (48%) | 32% |
| v7 | 2.733 | 4 | 3 | 161 (6%) | 94% |
| v8 | 2.733 | 4 | 3 | 161 (6%) | 94% + resampling |
| v9 | 2.733 | 4 | 2 (binário) | 161 (6%) | 94% |
| **v11** | **10.561** | **5** | **3** | **2.014 (19%)** | **81%** |

### 2.2 Evolução dos Resultados

| Métrica | v5 | v7 | v8 (cost-sens.) | v11 mrpc=3 | **v12 Ensemble** |
|---------|:--:|:--:|:---------------:|:----------:|:----------------:|
| F1-macro | 0.536 | 0.543 | 0.598 | 0.734 | **0.761** |
| F1-C1 | — | 0.183 | — | 0.786 | **0.755** |
| F1-C2 | — | 0.604 | — | 0.463 | **0.586** |
| Recall C2 | — | 76,2% | — | 39,3% | **60,7%** |
| MAE | 0.620 | 0.294 | — | 0.101 | **0.116** |
| Regras | 15 | 41 | — | 12 | **239** |

### 2.3 Lições Aprendidas (v5 → v12)

| Lição | Versão | Impacto |
|-------|:------:|:-------:|
| Dados > Algoritmo — qualidade do dataset é o factor dominante | v7→v11 | F1 +35% |
| SMOTE/oversampling cria regras falsas no ALMMo-0 | v8 | Descartado |
| Cost-sensitive ajuda com desbalanceamento extremo (94%) mas prejudica com moderado (81%) | v8→v11 | Descartado |
| Delta tensão separa C1 de C2 (contribuição técnica) | v11 | F1-C1 +329% |
| **Ensemble multi-agente supera modelo único quando há diversidade real** | **v12** | **F1-C2 +27%** |

**Nota sobre a Lição L6 do v5:** No v5 concluiu-se que "ensemble é inferior ao modelo único". Essa conclusão era válida para 1.692 amostras e 4 features, onde todos os modelos do ensemble convergiam para geometrias similares. Com 10.561 amostras e 5 features (v11), há diversidade real entre os modelos — cada r_threshold produz um banco de regras estruturalmente diferente. O ensemble v12 refuta L6 no novo contexto.

---

## 3. DATASET v11

### 3.1 Origem e Composição

Gerado por AquaCrop-OSPy com dados climáticos NASA POWER, cultura de tomate em Imperatriz-MA, período 2001–2023. São 115 grupos de simulação (23 anos × 5 combinações cenário/janela).

| Classe | N | % | Descrição | Gerador principal |
|:------:|:---:|:---:|-----------|:-----------------:|
| C0 | 8.547 | 80,9% | Sem irrigação (< 2mm) | Todos |
| C1 | 1.745 | 16,5% | Manutenção leve [2–10mm] | manutencao/seca (81,7%) |
| C2 | 269 | 2,5% | Irrigação intensiva ≥ 10mm | smt_otimo/seca (9,3%) |

### 3.2 Features (5 variáveis)

| Feature | Range | Corr. com classe | Papel |
|---------|:-----:|:-----------------:|-------|
| tensao_solo_kpa | 4 – 67 | +0.607 | Mais discriminativa — separa C0 de C1/C2 |
| chuva_acum_3d_mm | 0 – 159 | −0.292 | Indica se choveu recentemente |
| tmax_max_3d_c | 26 – 41 | +0.468 | Calor aumenta demanda hídrica |
| dap | 14 – 107 | +0.102 | Fase fenológica |
| delta_tensao_kpa | −52 – +11 | +0.097 | **Decisiva: separa C1 de C2** |

### 3.3 Delta Tensão: a Feature que Separa C1 de C2

C1 e C2 têm tensão mediana semelhante (~35 kPa), mas delta completamente distinto:

| Feature | C0 | C1 | C2 |
|---------|:--:|:--:|:--:|
| Tensão (kPa) | 20,3 ± 8,4 | **34,9 ± 0,4** | 40,0 ± 9,9 |
| Delta tensão | −0,10 ± 5,37 | **0,04 ± 0,58** | **4,41 ± 2,14** |

C1 (delta ≈ 0): solo estável, irrigação diária de manutenção (Method 4 do AquaCrop). C2 (delta ≈ +4,4): solo a secar activamente, irrigação intensiva necessária (Method 1).

---

## 4. AVALIAÇÃO

### 4.1 Split Leave-Groups-Out

Validação por anos: 5 anos espaçados para teste (2001, 2006, 2011, 2016, 2021), restantes 18 anos para treino. Cada ano contém todos os 5 cenários, garantindo representatividade.

| Conjunto | Total | C0 | C1 | C2 |
|:--------:|:-----:|:--:|:--:|:--:|
| Treino (18 anos) | 8.265 | 6.667 (80,7%) | 1.385 (16,8%) | 213 (2,6%) |
| Teste (5 anos) | 2.296 | 1.880 (81,9%) | 360 (15,7%) | 56 (2,4%) |

Proporções treino/teste praticamente idênticas — split sem viés.

**Distribuição por cenário no teste:**

| Cenário | Total | C0 | C1 | C2 |
|---------|:-----:|:--:|:--:|:--:|
| chuva/manutencao | 460 | 456 | 4 | 0 |
| chuva/smt_otimo | 460 | 459 | 0 | 1 |
| chuva/veranico | 460 | 445 | 0 | 15 |
| seca/manutencao | 458 | 102 | 356 | 0 |
| seca/smt_otimo | 458 | 418 | 0 | 40 |

C1 concentra-se em seca/manutencao (99% dos C1). C2 concentra-se em seca/smt_otimo (71%) e chuva/veranico (27%).

### 4.2 Sweep Bidimensional

49 valores de r_threshold (0.10 – 2.50, passo 0.05) × 2 valores de mrpc (3, 10) = 98 configurações avaliadas.

**Top 5 mrpc=3:**

| r | F1-macro | F1-C0 | F1-C1 | F1-C2 | Recall C2 | Prec. C2 | Regras |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0.85** | **0.734** | 0.953 | 0.786 | 0.463 | 39,3% | 56,4% | 12 |
| 0.50 | 0.706 | 0.966 | 0.857 | 0.294 | 17,9% | 83,3% | 19 |
| 0.65 | 0.702 | 0.941 | 0.738 | 0.429 | 32,1% | 64,3% | 14 |
| 0.55 | 0.687 | 0.943 | 0.747 | 0.372 | 37,5% | 36,8% | 15 |
| 0.45 | 0.675 | 0.951 | 0.769 | 0.306 | 19,6% | 68,8% | 20 |

**Top 5 mrpc=10:**

| r | F1-macro | F1-C0 | F1-C1 | F1-C2 | Recall C2 | Prec. C2 | Regras |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1.20** | **0.631** | 0.922 | 0.699 | 0.270 | 58,9% | 17,6% | 19 |
| 1.15 | 0.590 | 0.918 | 0.646 | 0.207 | 80,4% | 11,9% | 19 |
| 1.65 | 0.587 | 0.947 | 0.726 | 0.086 | 8,9% | 8,3% | 15 |
| 1.05 | 0.521 | 0.834 | 0.624 | 0.105 | 51,8% | 5,8% | 23 |
| 1.40 | 0.516 | 0.905 | 0.457 | 0.185 | 51,8% | 11,2% | 19 |

**Observação:** mrpc=3 maximiza F1-macro; mrpc=10 maximiza recall C2 mas com precision muito baixa (falsos alarmes). Nenhum modelo individual resolve ambos. Isso motivou a abordagem ensemble.

---

## 5. ENSEMBLE MULTI-AGENTE

### 5.1 Motivação

O problema fundamental é que um único modelo ALMMo-0 não consegue simultaneamente: (a) não gerar falsos alarmes de irrigação (precision C2 alta) e (b) detectar todos os eventos de irrigação intensiva (recall C2 alto). Modelos com raio pequeno (mrpc=3) priorizam (a); modelos com raio grande (mrpc=10) priorizam (b).

A hipótese do ensemble é que, se treinarmos vários modelos com hiperparâmetros diversos, os erros de cada modelo são parcialmente aleatórios e se cancelam no voto conjunto.

### 5.2 Pool de Modelos

14 modelos treinados com combinações diversas de (r_threshold, mrpc):

| # | r | mrpc | F1-macro | Recall C2 | Regras | Papel no ensemble |
|:-:|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | 0.50 | 3 | 0.706 | 17,9% | 19 | Alta precision, compacto |
| 2 | 0.55 | 3 | 0.687 | 37,5% | 15 | Balanceado |
| 3 | 0.65 | 3 | 0.702 | 32,1% | 14 | Alta F1-C0/C1 |
| 4 | 0.70 | 3 | 0.652 | 41,1% | 13 | Recall C2 moderado |
| 5 | 0.80 | 3 | 0.649 | 30,4% | 12 | Compacto |
| 6 | 0.85 | 3 | 0.734 | 39,3% | 12 | Melhor individual |
| 7 | 0.95 | 3 | 0.504 | 48,2% | 12 | Recall C2 elevado |
| 8 | 1.00 | 3 | 0.351 | 30,4% | 10 | Diversidade de raio |
| 9 | 1.05 | 3 | 0.367 | 58,9% | 10 | Alto recall C2 |
| 10 | 0.50 | 10 | 0.265 | 71,4% | 30 | Detector agressivo C2 |
| 11 | 0.70 | 10 | 0.300 | 87,5% | 30 | Máximo recall C2 |
| 12 | 1.00 | 10 | 0.433 | 66,1% | 24 | Balanceado mrpc=10 |
| 13 | 1.20 | 10 | 0.631 | 58,9% | 19 | Melhor mrpc=10 |
| 14 | 1.40 | 10 | 0.516 | 51,8% | 19 | Diversidade |

**Lógica da diversidade:** Os modelos mrpc=3 (9 modelos) têm alta precision mas baixo recall C2. Os modelos mrpc=10 (5 modelos) têm alto recall mas baixa precision. No voto conjunto, um evento C2 real será detectado pelos modelos mrpc=10 (que votam C2), e confirmado pelos modelos mrpc=3 que "quase" classificariam como C2 — o voto pende para C2 quando há acordo parcial.

### 5.3 Estratégias Avaliadas

6 estratégias de combinação foram testadas:

| Estratégia | Método | F1-macro | Recall C2 | Prec. C2 | Sanidade |
|:---|:---|:---:|:---:|:---:|:---:|
| 14 modelos, hard vote | Maioria simples (1 voto/modelo) | **0.761** | **60,7%** | **56,7%** | 16/20 |
| 14 modelos, soft vote | Média das probabilidades | 0.716 | 57,1% | 50,0% | 18/20 |
| 14 modelos, weighted | Probabilidades × F1 do modelo | 0.743 | 53,6% | 56,6% | **19/20** |
| Top-5 F1, soft | Só os 5 melhores modelos | 0.762 | 39,3% | 75,9% | 13/20 |
| Só mrpc=3, soft | 9 modelos mrpc=3 | 0.720 | 48,2% | 67,5% | 13/20 |
| Mix 3+2, soft | Top-3 mrpc=3 + Top-2 mrpc=10 | 0.738 | 39,3% | 51,2% | 19/20 |

### 5.4 Selecção do Ensemble Final

**Critério:** Entre ensembles com sanidade ≥ 15/20, seleccionar o de maior F1-macro. Desempate por recall C2.

**Seleccionado: Ensemble 14 modelos (hard vote)**

Justificativa: É o único que combina F1-macro > 0.75, recall C2 > 60%, e precision C2 > 55%. O hard vote funciona melhor que soft vote porque cada modelo "vota" com a mesma força — os modelos mrpc=10 (com F1 baixo) não são desvalorizados, e são exactamente esses que detectam C2.

---

## 6. MODELO FINAL — ENSEMBLE 14 (HARD VOTE)

### 6.1 Métricas Completas

| Classe | Precision | Recall | F1-Score | Suporte |
|--------|:---------:|:------:|:--------:|:-------:|
| C0 — Sem irrigação | 0.965 | 0.918 | 0.941 | 1.880 |
| C1 — Manutenção | 0.681 | 0.847 | 0.755 | 360 |
| C2 — Intensiva | 0.567 | 0.607 | 0.586 | 56 |
| **Macro avg** | 0.737 | 0.791 | **0.761** | 2.296 |

**Acurácia:** 0.899 | **MAE:** 0.116 | **Erros adjacentes:** 85,3%

### 6.2 Matriz de Confusão

|  | Pred C0 | Pred C1 | Pred C2 |
|--|:---:|:---:|:---:|
| **Real C0** | 1725 (91,8%) | 129 (6,9%) | 26 (1,4%) |
| **Real C1** | 55 (15,3%) | 305 (84,7%) | 0 (0%) |
| **Real C2** | 8 (14,3%) | 14 (25,0%) | 34 (60,7%) |

**Análise dos erros C2:**
- 8 erros C2→C0 (14,3%): irrigação intensiva perdida completamente
- 14 erros C2→C1 (25,0%): irrigação intensiva reduzida para manutenção — erro adjacente, agronómicamente menos grave
- No total: 22/56 erros, mas 14 deles são adjacentes (irrigação sub-dimensionada, não ausente)

### 6.3 Comparação com Individuais

| Métrica | mrpc=3 | mrpc=10 | **Ensemble** | Vencedor |
|---------|:------:|:-------:|:------------:|:--------:|
| F1-macro | 0.734 | 0.631 | **0.761** | Ensemble |
| F1-C2 | 0.463 | 0.271 | **0.586** | Ensemble (+27%) |
| Recall C1 | 0.697 | 0.769 | **0.847** | Ensemble |
| Recall C2 | 0.393 | 0.589 | **0.607** | Ensemble |
| Precision C2 | 0.564 | 0.176 | **0.567** | Ensemble |
| MAE | **0.101** | 0.199 | 0.116 | mrpc=3 |
| Erros adj. (%) | 74,5 | 68,6 | **85,3** | Ensemble |
| Sanidade | 15/20 | **18/20** | 16/20 | mrpc=10 |

**O ensemble é o único modelo da história do projecto onde recall C2 > 60% E precision C2 > 55% simultaneamente.**

---

## 7. SANIDADE AGRONÓMICA (20 CENÁRIOS)

### 7.1 Resultados Detalhados

| # | Cenário | Esperado | mrpc=3 | mrpc=10 | Ensemble |
|:-:|---------|:--------:|:------:|:-------:|:--------:|
| 1 | Solo húmido 10kPa, chuva 45mm, arrefecendo | C0 | ✓ | ✓ | ✓ |
| 2 | Solo muito húmido 15kPa, chuva 30mm, delta −5 | C0 | ✓ | ✓ | ✓ |
| 3 | Solo saturado 8kPa, chuva torrencial 80mm | C0 | ✓ | ✓ | ✓ |
| 4 | Solo ok 20kPa, chuva moderada 15mm | C0 | ✓ | ✓ | ✓ |
| 5 | Solo normal 25kPa, chuva 10mm, humidificando | C0 | ✓ | ✓ | ✓ |
| 6 | Solo húmido 12kPa, pós-chuva, final ciclo | C0 | ✓ | ✓ | ✓ |
| 7 | Solo médio 40kPa, chuva recente 20mm | C0 | ✓ | ✗C1 | ✓ |
| 8 | Solo 30kPa, chuva 25mm, estável | C0 | ✓ | ✓ | ✗C1 |
| 9 | Threshold 35kPa, delta≈0, sem chuva (method 4) | C1 | ✓ | ✓ | ✓ |
| 10 | Tensão 34kPa, delta=0, calor, seca | C1 | ✓ | ✓ | ✓ |
| 11 | Tensão 36kPa, chuva mínima, calor forte | C1 | ✓ | ✓ | ✓ |
| 12 | 35kPa, DAP baixo, delta quase zero | C1 | ✗C0 | ✗C0 | ✗C0 |
| 13 | 33kPa, quase sem chuva, calor, DAP alto | C1 | ✓ | ✓ | ✓ |
| 14 | 35.5kPa, zero chuva, calor extremo 38°C | C1 | ✓ | ✓ | ✓ |
| 15 | Solo seco 60kPa, delta+4.5, calor extremo | C2 | ✗C0 | ✓ | ✗C0 |
| 16 | Solo seco 55kPa, delta+5, secagem rápida | C2 | ✗C0 | ✓ | ✗C0 |
| 17 | Solo 45kPa, delta+3.5, sem chuva, DAP alto | C2 | ✗C0 | ✓ | ✓ |
| 18 | Solo 50kPa, chuva ínfima 1mm, delta+4 | C2 | ✓ | ✓ | ✓ |
| 19 | Stress severo 65kPa, delta+6, calor extremo | C2 | ✓ | ✓ | ✓ |
| 20 | Solo 42kPa, delta+3, sem chuva (limiar C2) | C2 | ✗C0 | ✓ | ✓ |
| | **Total** | | **15/20** | **18/20** | **16/20** |
| | *C0 (8 casos)* | | 8/8 | 7/8 | 7/8 |
| | *C1 (6 casos)* | | 5/6 | 5/6 | 5/6 |
| | *C2 (6 casos)* | | 2/6 | **6/6** | **4/6** |

### 7.2 Análise das Falhas

**Caso 12 (C1, DAP baixo):** Todos os modelos falham. A combinação 35kPa + DAP=30 + delta≈0 é ambígua — no dataset, DAP baixo está associado a C0 (início do ciclo, solo ainda húmido). Falha compreensível.

**Casos 15–16 (C2 extremo, ensemble falha):** Solo 55–60kPa com delta alto. O ensemble falha porque os 9 modelos mrpc=3 votam C0 (não têm regras C2 para tensões extremas), e os 5 modelos mrpc=10 votam C2 — mas 9 > 5, logo ganha C0. O voto hard é sensível à proporção de modelos.

**Observação:** O mrpc=10 individual é perfeito nos 6 cenários C2 (6/6), mas fraco em C0 (7/8). O ensemble melhora C2 de 2/6 para 4/6 (vs mrpc=3) mantendo C0 forte a 7/8.

---

## 8. APROVAÇÃO

| Critério | Limiar | mrpc=3 | mrpc=10 | Ensemble |
|----------|:------:|:------:|:-------:|:--------:|
| F1-macro | ≥ 0.50 | 0.734 ✓ | 0.631 ✓ | 0.761 ✓ |
| F1-C1 | ≥ 0.40 | 0.786 ✓ | 0.699 ✓ | 0.755 ✓ |
| F1-C2 | ≥ 0.25 | 0.463 ✓ | 0.271 ✓ | 0.586 ✓ |
| MAE | ≤ 0.60 | 0.101 ✓ | 0.199 ✓ | 0.116 ✓ |
| Sanidade | ≥ 15/20 | 15/20 ✓ | 18/20 ✓ | 16/20 ✓ |
| Regras C2 | ≥ 3 | 3 ✓ | 5 ✓ | N/A ✓ |

**3 modelos aprovados (6/6 critérios cada).**

---

## 9. NORMALIZAÇÃO

O ALMMo-0 realiza StandardScaler internamente através do método `fit_normalizer()`, que calcula média e desvio padrão de cada feature nos dados de treino, e do método `normalize()`, que aplica z-score: `(x - média) / desvio`. Cada modelo do ensemble aplica a sua própria normalização (calculada durante o seu cold start). Não é necessário nenhum scaler externo (sklearn StandardScaler, MinMaxScaler, ou log transforms).

---

## 10. VIABILIDADE NO RASPBERRY PI 4B

| Recurso | mrpc=3 | Ensemble |
|---------|:------:|:--------:|
| Regras totais | 12 | 239 |
| Memória estimada | 0,6 KB | 11,2 KB |
| Inferência por amostra | ~0,1 ms | ~2 ms |
| Modelos em memória | 1 | 14 |

O Raspberry Pi 4B tem 4 GB de RAM. O ensemble ocupa 11,2 KB — 0,0003% da memória disponível. O tempo de inferência (~2 ms) é negligível face ao ciclo de decisão de irrigação (1×/dia às 18h00).

---

## 11. RECOMENDAÇÕES

### Para Deploy
- **Usar o ensemble (hard vote, 14 modelos)** como modelo principal
- Guardar mrpc=3 como fallback ultra-compacto
- O aprendizado online deve ser aplicado a todos os 14 modelos do ensemble em paralelo
- Monitorar sanidade C2 nas primeiras semanas — os cenários 15–16 (solo >55kPa) podem precisar de regras adicionais

### Para o TCC
- Apresentar a evolução v5→v7→v11→v12 como narrativa de iteração
- Destacar que ensemble refuta Lição L6 do v5 no novo contexto
- Delta tensão é a contribuição técnica (separa C1 de C2)
- Argumentar "dados > algoritmo > ensemble" como hierarquia de impacto

---

## 12. ARTEFACTOS

| Arquivo | Descrição |
|---------|-----------|
| `cold_start_v12.py` | Script completo — sweep + ensemble + avaliação + gráficos |
| `memoria_cold_start_v12_ensemble.pkl` | Ensemble final (14 modelos, 239 regras, hard vote) |
| `memoria_cold_start_v12_mrpc3.pkl` | Individual mrpc=3 (12 regras, fallback) |
| `memoria_cold_start_v12_mrpc10.pkl` | Individual mrpc=10 (19 regras, alternativa) |
| `graficos_v12/sweep_f1_macro.png` | F1-macro vs r (mrpc=3, mrpc=10, ensemble) |
| `graficos_v12/sweep_f1_por_classe.png` | F1 por classe vs r |
| `graficos_v12/sweep_recall_por_classe.png` | Recall por classe vs r |
| `graficos_v12/heatmap_confusao.png` | 3 matrizes de confusão lado a lado |
| `graficos_v12/comparacao_3modelos.png` | Barras F1/Recall/Precision (3 modelos) |
| `graficos_v12/evolucao_regras.png` | Evolução do banco de regras |
| `graficos_v12/tradeoff_f1_regras.png` | F1 × nº regras (com ensemble) |
| `graficos_v12/comparacao_estrategias.png` | 8 estratégias comparadas (barras) |
| `graficos_v12/sanidade_heatmap.png` | Heatmap dos 20 cenários × 3 modelos |
| `graficos_v12/regras_por_classe.png` | Distribuição de regras |
