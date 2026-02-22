# Relatório de Correções — Dataset Cold Start v6
**Projeto:** Sistema ALMMo-0 — Irrigação Inteligente de Tomate | Imperatriz-MA
**Referência de solo:** Saxton & Rawls (2006), S=65.0% C=10.0% OM=3.0%
θ_CC=0.1864 m³/m³ | θ_PM=0.0853 m³/m³ | A=0.0090 | B=4.8825
---
## Rotulagem — 3 classes, limiar = mediana do cenário ótimo
**Justificativa:** os três cenários têm distribuições de IrrDay estruturalmente incompatíveis (déficit truncado em 5mm, excesso concentrado acima de 10mm). Qualquer fatiamento em 5 classes produz limiares estreitos demais para sensores reais. A solução é reduzir para 3 classes com significado agronômico claro: sem irrigação, irrigação moderada e irrigação intensa. O limiar único é a mediana do IrrDay > 0 do cenário ótimo — único cenário com distribuição não truncada.

**Limiar adotado:** mediana do cenário ótimo = 4.85 mm (n=534 eventos de irrigação)

**Regra de rotulagem:**
- Classe 0: IrrDay = 0,0 mm (sem irrigação)
- Classe 1: 0 < IrrDay ≤ 4.85 mm (irrigação moderada)
- Classe 2: IrrDay > 4.85 mm (irrigação intensa)


**Distribuição final das classes (dataset completo):**
| Classe | Contagem | % |
|--------|----------|---|
| 0 | 119 | 7.4% |
| 1 | 716 | 44.6% |
| 2 | 770 | 48.0% |

**Distribuição por cenário:**
classe_irrigacao      0    1    2
cenario                          
cenario1_otimo_v6    28  254  253
cenario2_deficit_v6  78  218  239
cenario3_excesso_v6  13  244  278
---
## PROBLEMA 2 — Feature de Chuva Selecionada
**Correlações testadas com classe_irrigacao:**
| Variante | Correlação Pearson | |Corr| |
|----------|-------------------|--------|
| chuva_mm | -0.1813 | 0.1813 |
| chuva_acum_3d_mm | -0.1693 | 0.1693 |
| chuva_acum_7d_mm | -0.1546 | 0.1546 |
| chuva_bool | -0.2445 | 0.2445 |

**Feature selecionada:** `chuva_bool`
---
## PROBLEMA 3 — Tensão Cravada em 1500 kPa
**Amostras com tensao_solo_kpa ≥ 1490 kPa:** 0 (0.0%)
**Interpretação:** O clamping em 1500 kPa ocorre quando θ ≤ θ_PM (ponto de murchamento permanente). Múltiplos graus de secura acima de 1500 kPa são indistinguíveis pela equação S&R — limitação inerente à curva de retenção. Esses valores concentram-se no cenário de déficit hídrico e nos DAPs finais de períodos sem chuva. Não é necessário corrigir — declarar como limitação.
---
## Arquivos Gerados
- `dataset_cold_start_v6.csv` — 5 colunas para ALMMo-0
- `dataset_cold_start_completo_v6.csv` — com metadados de cenário/ano
- `graficos_v6/histograma_IrrDay.png`
- `graficos_v6/distribuicao_classes_v6.png`
- `graficos_v6/correlacao_features_v6.png`
