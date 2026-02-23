# Relatório de Validação Hardware-in-the-Loop (HIL)
**Data de execução:** 2026-02-22 15:46:41
**Modelo carregado de:** `inicial`

## Secção 1 — Validação de Hardware

**Cenário 1 (Seca Progressiva):**
- Tempo carregamento pkl: `0.9 ms`
- Tempo médio de inferência: `7.89 ms/ciclo`
**Cenário 2 (Evento de Chuva):**
- Tempo carregamento pkl: `0.9 ms`
- Tempo médio de inferência: `4.26 ms/ciclo`
**Cenário 3 (Cegueira C1 e Dead Zone do Feedback):**
- Tempo carregamento pkl: `0.9 ms`
- Tempo médio de inferência: `4.90 ms/ciclo`
**Cenário 4 (Stress Crítico e Recuperação):**
- Tempo carregamento pkl: `0.9 ms`
- Tempo médio de inferência: `4.34 ms/ciclo`
- Fonte do modelo: `inicial`

## Secção 2 — Validação de Integração

- API Open-Meteo: `api`
- Latência: `904 ms` ✅
- Chuva acum. 3d (Imperatriz-MA): `32.1 mm`
- Tmax 3d: `31.9 °C`
- Cache JSON: `cache_meteo.json` — ✅

## Secção 3 — Resultados por Cenário

### Cenário 1 — Seca Progressiva
*Solo começa na capacidade de campo e seca sem chuva.*

**Comportamento esperado:** Dias 1-3: C0 (solo ainda húmido). Dias 4-8: C1 (stress moderado). Dias 9-14: C2 (stress severo). Escalamento monotónico esperado.

| Dia | kPa | θ | C manhã | C final | Irr mm | Motivo 18h | Regras | Flags |
|-----|-----|---|---------|---------|--------|------------|--------|-------|
| 1 | 47.3 | 0.173 | C0 | C0 | 0.0 | executado_normal | 41 (21/10/10) |  |
| 2 | 61.9 | 0.1637 | C0 | C0 | 0.0 | executado_normal | 41 (21/10/10) |  |
| 3 | 79.0 | 0.1557 | C0 | C0 | 0.0 | executado_normal | 41 (21/10/10) |  |
| 4 | 102.2 | 0.1477 | C0 | C0 | 0.0 | executado_normal | 41 (21/10/10) |  |
| 5 | 134.1 | 0.1397 | C2 | C2 | 7.0 | executado_normal | 42 (21/11/10) | 📚fb  |
| 6 | 89.7 | 0.1517 | C0 | C0 | 0.0 | executado_normal | 42 (21/11/10) |  |
| 7 | 116.8 | 0.1437 | C1 | C1 | 3.0 | executado_normal | 42 (21/11/10) |  |
| 8 | 114.6 | 0.1443 | C0 | C0 | 0.0 | executado_normal | 42 (21/11/10) |  |
| 9 | 151.4 | 0.1363 | C0 | C0 | 0.0 | executado_normal | 43 (21/12/10) | ⚠️range 📚fb  |
| 10 | 203.4 | 0.1283 | C0 | C0 | 0.0 | executado_normal | 43 (21/12/10) | ⚠️range  |
| 11 | 278.6 | 0.1203 | C2 | C2 | 7.0 | executado_normal | 44 (21/13/10) | ⚠️range 📚fb  |
| 12 | 175.1 | 0.1323 | C1 | C1 | 3.0 | executado_normal | 44 (21/13/10) | ⚠️range  |
| 13 | 171.5 | 0.1329 | C1 | C1 | 3.0 | executado_normal | 44 (21/13/10) | ⚠️range  |
| 14 | 167.9 | 0.1334 | C1 | C1 | 3.0 | executado_normal | 45 (21/13/11) | ⚠️range 📚fb  |

### Cenário 2 — Evento de Chuva
*Solo seco recebe chuva intensa no dia 4.*

**Comportamento esperado:** Dias 1-3: C1 ou C2 (solo seco). Dias 4-7: C0 (chuva acumulada alta). Dias 8-10: retorno gradual a C1.

| Dia | kPa | θ | C manhã | C final | Irr mm | Motivo 18h | Regras | Flags |
|-----|-----|---|---------|---------|--------|------------|--------|-------|
| 1 | 347.1 | 0.115 | C2 | C2 | 7.0 | executado_normal | 41 (21/10/10) | ⚠️range  |
| 2 | 209.0 | 0.1276 | C0 | C0 | 0.0 | executado_normal | 41 (21/10/10) | ⚠️range  |
| 3 | 267.2 | 0.1213 | C0 | C0 | 0.0 | executado_normal | 41 (21/10/10) | ⚠️range  |
| 4 | 348.2 | 0.1149 | C0 | C0 | 0.0 | mantido_stress_persistente | 42 (21/11/10) | ⚠️range 📚fb 🌧️ |
| 5 | 40.7 | 0.1784 | C0 | C0 | 0.0 | cancelado_chuva_solo_ok | 42 (21/11/10) | 🌧️ |
| 6 | 33.0 | 0.1957 | C0 | C0 | 0.0 | cancelado_chuva_solo_ok | 42 (21/11/10) | 🌧️ |
| 7 | 33.0 | 0.1989 | C0 | C0 | 0.0 | executado_normal | 42 (21/11/10) |  |
| 8 | 33.0 | 0.192 | C0 | C0 | 0.0 | executado_normal | 42 (21/11/10) |  |
| 9 | 34.1 | 0.1849 | C0 | C0 | 0.0 | executado_normal | 42 (21/11/10) |  |
| 10 | 41.4 | 0.1777 | C1 | C1 | 3.0 | executado_normal | 42 (21/11/10) |  |

### Cenário 3 — Cegueira C1 e Dead Zone do Feedback
*Demonstra o comportamento real do sistema: o modelo tem cegueira para C1 (precision=12.5%) e oscila entre C0 e C2 na zona 40-90 kPa. O feedback nao actua nesta zona (dead zone por design). Este e um achado de investigacao para o TCC.*

**Comportamento esperado:** Dias 1-3: C0 (tensao ~43-72 kPa — dead zone, feedback inactivo). Dias 4+: oscilacao C0-C2 (cegueira C1 — modelo nunca usa irrigacao moderada). C1 ausente ou raro em todos os 21 dias. Feedback: 0 eventos esperados — dead zone cobre a zona do vies C1. ACHADO TCC: feedback calibrado para stress critico (>90 kPa), nao para optimizacao de stress moderado. Solucao: calibrar threshold com dados de campo reais.

| Dia | kPa | θ | C manhã | C final | Irr mm | Motivo 18h | Regras | Flags |
|-----|-----|---|---------|---------|--------|------------|--------|-------|
| 1 | 53.5 | 0.1687 | C0 | C0 | 0.0 | executado_normal | 41 (21/10/10) |  |
| 2 | 69.4 | 0.1599 | C0 | C0 | 0.0 | executado_normal | 41 (21/10/10) |  |
| 3 | 88.0 | 0.1523 | C0 | C0 | 0.0 | executado_normal | 41 (21/10/10) |  |
| 4 | 113.5 | 0.1446 | C0 | C0 | 0.0 | executado_normal | 41 (21/10/10) |  |
| 5 | 149.1 | 0.1367 | C2 | C2 | 7.0 | executado_normal | 42 (21/11/10) | ⚠️range 📚fb  |
| 6 | 98.9 | 0.1487 | C0 | C0 | 0.0 | executado_normal | 42 (21/11/10) |  |
| 7 | 129.5 | 0.1407 | C1 | C1 | 3.0 | executado_normal | 42 (21/11/10) |  |
| 8 | 127.0 | 0.1413 | C0 | C0 | 0.0 | executado_normal | 42 (21/11/10) |  |
| 9 | 168.8 | 0.1333 | C0 | C0 | 0.0 | executado_normal | 43 (21/12/10) | ⚠️range 📚fb  |
| 10 | 228.3 | 0.1253 | C0 | C0 | 0.0 | executado_normal | 43 (21/12/10) | ⚠️range  |
| 11 | 315.1 | 0.1173 | C2 | C2 | 7.0 | executado_normal | 44 (21/13/10) | ⚠️range 📚fb  |
| 12 | 195.9 | 0.1293 | C1 | C1 | 3.0 | executado_normal | 44 (21/13/10) | ⚠️range  |
| 13 | 191.7 | 0.1299 | C1 | C1 | 3.0 | executado_normal | 44 (21/13/10) | ⚠️range  |
| 14 | 187.6 | 0.1304 | C1 | C1 | 3.0 | executado_normal | 45 (21/13/11) | ⚠️range 📚fb  |
| 15 | 183.7 | 0.131 | C2 | C2 | 7.0 | executado_normal | 45 (21/13/11) | ⚠️range  |
| 16 | 119.7 | 0.143 | C1 | C1 | 3.0 | executado_normal | 45 (21/13/11) |  |
| 17 | 117.4 | 0.1436 | C1 | C1 | 3.0 | executado_normal | 45 (21/13/11) |  |
| 18 | 115.2 | 0.1442 | C1 | C1 | 3.0 | executado_normal | 45 (21/13/11) |  |
| 19 | 113.0 | 0.1447 | C1 | C1 | 3.0 | executado_normal | 45 (21/13/11) |  |
| 20 | 110.8 | 0.1453 | C1 | C1 | 3.0 | executado_normal | 45 (21/13/11) |  |
| 21 | 108.7 | 0.1459 | C1 | C1 | 3.0 | executado_normal | 45 (21/13/11) |  |

### Cenário 4 — Stress Crítico e Recuperação
*Solo próximo do PM. Irrigação intensa recupera.*

**Comportamento esperado:** Dias 1-3: C2 (tensão >90 kPa, acima do range de treino). tensao_acima_range=True no log. Após irrigação intensa: tensão cai, predições voltam a C1/C0.

| Dia | kPa | θ | C manhã | C final | Irr mm | Motivo 18h | Regras | Flags |
|-----|-----|---|---------|---------|--------|------------|--------|-------|
| 1 | 1114.4 | 0.0906 | C2 | C2 | 7.0 | executado_normal | 41 (21/10/10) | ⚠️range  |
| 2 | 645.1 | 0.1013 | C2 | C2 | 7.0 | executado_normal | 41 (21/10/10) | ⚠️range  |
| 3 | 373.4 | 0.1133 | C0 | C0 | 0.0 | executado_normal | 41 (21/10/10) | ⚠️range  |
| 4 | 533.9 | 0.1053 | C0 | C0 | 0.0 | executado_normal | 41 (21/10/10) | ⚠️range  |
| 5 | 785.3 | 0.0973 | C2 | C2 | 7.0 | executado_normal | 42 (21/11/10) | ⚠️range 📚fb  |
| 6 | 445.0 | 0.1093 | C1 | C1 | 3.0 | executado_normal | 42 (21/11/10) | ⚠️range  |
| 7 | 433.9 | 0.1099 | C1 | C1 | 3.0 | executado_normal | 42 (21/11/10) | ⚠️range  |
| 8 | 423.0 | 0.1104 | C1 | C1 | 3.0 | executado_normal | 43 (21/11/11) | ⚠️range 📚fb  |
| 9 | 412.5 | 0.111 | C2 | C2 | 7.0 | executado_normal | 43 (21/11/11) | ⚠️range  |
| 10 | 249.9 | 0.123 | C1 | C1 | 3.0 | executado_normal | 43 (21/11/11) | ⚠️range  |

## Secção 4 — Análise do Aprendizado Online (Cenário 3)

### % de dias C1+C2 por semana

| Semana | Dias | C0 | C1+C2 | % C1+C2 |
|--------|------|----|-------|---------|
| Semana 1 | 7 | 5 | 2 | 29% |
| Semana 2 | 7 | 3 | 4 | 57% |
| Semana 3 | 7 | 0 | 7 | 100% |

**MÉTRICA CHAVE — % C1+C2 aumentou da semana 1 para semana 3?** ✅ PASS (29% → 100%)

**Total de eventos de feedback:** 4 em 21 dias

## Secção 5 — Dead Zone Documentada

A dead zone corresponde a tensões entre 40–90 kPa onde o mecanismo de feedback não actua (tensão dentro de limites aceitáveis, sem tendência clara).
Esta é uma limitação conhecida e intencional — erros nesta faixa serão corrigidos por dados de campo reais com o sensor físico calibrado.

**Cenário 1:** 4 dias na dead zone sem feedback
**Cenário 2:** 2 dias na dead zone sem feedback
**Cenário 3:** 3 dias na dead zone sem feedback
**Cenário 4:** 0 dias na dead zone sem feedback

---
*Relatório gerado automaticamente por `main_hil.py`.*
*Sistema: ALMMo-0 + Saxton & Rawls + Open-Meteo + Dupla Confirmação 18h*