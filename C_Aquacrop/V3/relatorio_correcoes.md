# Relatório de Correções — Dataset Cold Start v3
**Projeto:** Sistema ALMMo-0 — Irrigação Inteligente de Tomate | Imperatriz-MA
**Referência de solo:** Saxton & Rawls (2006), S=65.0% C=10.0% OM=3.0%
θ_CC=0.1864 m³/m³ | θ_PM=0.0853 m³/m³ | A=0.0090 | B=4.8825
---
## PROBLEMA 1 — Rotulagem por Percentis Reais do IrrDay
**Número de classes adotado:** 5
**Limiares adotados (IrrDay > 0):**
- Classe 0: IrrDay = 0,0 mm (sem irrigação)
- Classe 1: 0 < IrrDay ≤ 5.03 mm (P20)
- Classe 2: 5.03 < IrrDay ≤ 6.27 mm (P40)
- Classe 3: 6.27 < IrrDay ≤ 9.35 mm (P60)
- Classe 4: IrrDay > 9.35 mm (acima de P60)

**Distribuição final das classes (dataset completo):**
| Classe | Contagem | % |
|--------|----------|---|
| 0 | 537 | 31.7% |
| 1 | 231 | 13.7% |
| 2 | 231 | 13.7% |
| 3 | 231 | 13.7% |
| 4 | 462 | 27.3% |

**Distribuição por cenário:**
classe_irrigacao       0    1    2    3    4
cenario                                     
cenario1_otimo_v3     30  159  204  140   31
cenario2_deficit_v3  500   64    0    0    0
cenario3_excesso_v3    7    8   27   91  431
---
## PROBLEMA 2 — Feature de Chuva Selecionada
**Correlações testadas com classe_irrigacao:**
| Variante | Correlação Pearson | |Corr| |
|----------|-------------------|--------|
| chuva_mm | -0.0960 | 0.0960 |
| chuva_acum_3d_mm | -0.0595 | 0.0595 |
| chuva_acum_7d_mm | -0.0434 | 0.0434 |
| chuva_bool | -0.1031 | 0.1031 |

**Feature selecionada:** `chuva_acum_3d_mm`
**Limitação declarada:** nenhuma variante de chuva atingiu |corr| ≥ 0,15. O AquaCrop com NetIrrSMT toma decisões de irrigação baseadas na umidade do solo (Wr), não diretamente na chuva recente. A chuva já está incorporada indiretamente na tensão do solo — por isso a correlação direta chuva→classe é baixa. Feature mantida como informação auxiliar para o modelo.
---
## PROBLEMA 3 — Tensão Cravada em 1500 kPa
**Amostras com tensao_solo_kpa ≥ 1490 kPa:** 32 (1.9%)
**Interpretação:** O clamping em 1500 kPa ocorre quando θ ≤ θ_PM (ponto de murchamento permanente). Múltiplos graus de secura acima de 1500 kPa são indistinguíveis pela equação S&R — limitação inerente à curva de retenção. Esses valores concentram-se no cenário de déficit hídrico e nos DAPs finais de períodos sem chuva. Não é necessário corrigir — declarar como limitação.
---
## Arquivos Gerados
- `dataset_cold_start_v3.csv` — 5 colunas para ALMMo-0
- `dataset_cold_start_completo_v3.csv` — com metadados de cenário/ano
- `graficos_v3/histograma_IrrDay.png`
- `graficos_v3/distribuicao_classes_v3.png`
- `graficos_v3/correlacao_features_v3.png`
