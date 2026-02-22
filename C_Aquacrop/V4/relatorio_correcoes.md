# Relatório de Correções — Dataset Cold Start v4
**Projeto:** Sistema ALMMo-0 — Irrigação Inteligente de Tomate | Imperatriz-MA
**Referência de solo:** Saxton & Rawls (2006), S=65.0% C=10.0% OM=3.0%
θ_CC=0.1864 m³/m³ | θ_PM=0.0853 m³/m³ | A=0.0090 | B=4.8825
---
## Rotulagem — 5 classes por P25/P50/P75 do cenário ótimo
**Justificativa:** percentis calculados apenas sobre o cenário ótimo (NetIrrSMT=65%) porque o déficit tem IrrDay truncado em 5mm e o excesso tem IrrDay deslocado para cima. O cenário ótimo é o único com distribuição não truncada do IrrDay.

**Limiares adotados (IrrDay > 0, n=534 eventos do cenário ótimo):**
- Classe 0: IrrDay = 0,0 mm (sem irrigação)
- Classe 1: 0 < IrrDay ≤ 4.80 mm (P25)
- Classe 2: 4.80 < IrrDay ≤ 5.55 mm (P50)
- Classe 3: 5.55 < IrrDay ≤ 6.52 mm (P75)
- Classe 4: IrrDay > 6.52 mm

Amplitude P25→P75: 1.71 mm (vs 4,32 mm no v3 com percentis combinados)


**Distribuição final das classes (dataset completo):**
| Classe | Contagem | % |
|--------|----------|---|
| 0 | 537 | 31.7% |
| 1 | 200 | 11.8% |
| 2 | 146 | 8.6% |
| 3 | 161 | 9.5% |
| 4 | 648 | 38.3% |

**Distribuição por cenário:**
classe_irrigacao       0    1    2    3    4
cenario                                     
cenario1_otimo_v4     30  134  133  133  134
cenario2_deficit_v4  500   61    3    0    0
cenario3_excesso_v4    7    5   10   28  514
---
## PROBLEMA 2 — Feature de Chuva Selecionada
**Correlações testadas com classe_irrigacao:**
| Variante | Correlação Pearson | |Corr| |
|----------|-------------------|--------|
| chuva_mm | -0.0981 | 0.0981 |
| chuva_acum_3d_mm | -0.0547 | 0.0547 |
| chuva_acum_7d_mm | -0.0388 | 0.0388 |
| chuva_bool | -0.1037 | 0.1037 |

**Feature selecionada:** `chuva_acum_3d_mm`
**Limitação declarada:** nenhuma variante de chuva atingiu |corr| ≥ 0,15. O AquaCrop com NetIrrSMT toma decisões de irrigação baseadas na umidade do solo (Wr), não diretamente na chuva recente. A chuva já está incorporada indiretamente na tensão do solo — por isso a correlação direta chuva→classe é baixa. Feature mantida como informação auxiliar para o modelo.
---
## PROBLEMA 3 — Tensão Cravada em 1500 kPa
**Amostras com tensao_solo_kpa ≥ 1490 kPa:** 32 (1.9%)
**Interpretação:** O clamping em 1500 kPa ocorre quando θ ≤ θ_PM (ponto de murchamento permanente). Múltiplos graus de secura acima de 1500 kPa são indistinguíveis pela equação S&R — limitação inerente à curva de retenção. Esses valores concentram-se no cenário de déficit hídrico e nos DAPs finais de períodos sem chuva. Não é necessário corrigir — declarar como limitação.
---
## Arquivos Gerados
- `dataset_cold_start_v4.csv` — 5 colunas para ALMMo-0
- `dataset_cold_start_completo_v3.csv` — com metadados de cenário/ano
- `graficos_v4/histograma_IrrDay.png`
- `graficos_v4/distribuicao_classes_v4.png`
- `graficos_v4/correlacao_features_v4.png`
