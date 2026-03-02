# Relatorio de Validacao — dataset_cold_start_v11.csv

**Data:** 2026-03-01 21:23
**Amostras:** 14784 | **Grupos:** 161
**Periodo:** 2001-2023 (23 anos)
**Limiar ruido:** 2.0mm | **Classes:** C0/<2.0, C1/[2.0,10.0), C2/[10.0,30.0), C3/>=30.0mm
**Features:** tensao_solo_kpa, chuva_acum_3d_mm, tmax_max_3d_c, dap, delta_tensao_kpa

## Cenarios

- **smt_otimo** (method 1): SMT otimo (method 1) — irriga sob demanda
- **manutencao** (method 4): Manutencao (method 4) — net irrigation diaria, alvo 70% TAW
- **intervalo** (method 2): Intervalo fixo (method 2) — a cada 5 dias
- **veranico** (method 1): Veranico (method 1 + precip fev-mar x 0.20)

## Distribuicao de Classes

- Sem irrigacao (C0): 12177 (82.4%)
- Manutencao (C1): 1871 (12.7%)
- Suplementar (C2): 467 (3.2%)
- Intensiva (C3): 269 (1.8%)

**v7:** C0=94.1%, C1=4.1%, C2=1.8% | **v10:** C0=94.9%, C1=3.4%, C2=1.7%

### Por Cenario x Janela

- smt_otimo/chuva (n=2115): C0=2109, C1=0, C2=0, C3=6 | dose=41.4mm
- smt_otimo/seca (n=2108): C0=1912, C1=0, C2=0, C3=196 | dose=44.6mm
- manutencao/chuva (n=2115): C0=2093, C1=22, C2=0, C3=0 | dose=3.6mm
- manutencao/seca (n=2108): C0=385, C1=1723, C2=0, C3=0 | dose=5.3mm
- intervalo/chuva (n=2115): C0=1944, C1=115, C2=56, C3=0 | dose=8.5mm
- intervalo/seca (n=2108): C0=1686, C1=11, C2=411, C3=0 | dose=21.8mm
- veranico/chuva (n=2115): C0=2048, C1=0, C2=0, C3=67 | dose=41.3mm

### Por Metodo

- smt_otimo (m1): C0=95.2%, C1=0.0%, C2=0.0%, C3=4.8% | tensao std=9.9
- manutencao (m4): C0=58.7%, C1=41.3%, C2=0.0%, C3=0.0% | tensao std=9.9
- intervalo (m2): C0=86.0%, C1=3.0%, C2=11.1%, C3=0.0% | tensao std=17.5
- veranico (m1): C0=96.8%, C1=0.0%, C2=0.0%, C3=3.2% | tensao std=8.5

## Features

- tensao_solo_kpa: min=4.18, max=156.30, med=18.65, std=12.47
- chuva_acum_3d_mm: min=0.00, max=158.96, med=3.66, std=16.71
- tmax_max_3d_c: min=25.71, max=40.93, med=31.19, std=3.18
- dap: min=14.00, max=107.00, med=59.00, std=26.54
- delta_tensao_kpa: min=-72.79, max=21.80, med=0.77, std=5.67

## Correlacoes

- tensao_solo_kpa: 0.4199
- chuva_acum_3d_mm: -0.2475
- tmax_max_3d_c: 0.3400
- dap: 0.0658
- delta_tensao_kpa: 0.1517

## Tensao por Classe

- C0 (Sem irrigacao): 16.9 kPa (n=12177, std=10.9)
- C1 (Manutencao): 35.0 kPa (n=1871, std=4.7)
- C2 (Suplementar): 24.0 kPa (n=467, std=26.5)
- C3 (Intensiva): 34.6 kPa (n=269, std=9.9)

## Criterios

- C0>=5%: PASS (82.4%)
- C3>=0.5%: PASS (1.8%)
- Corr tensao [0.10,0.70]: PASS (0.4199)
- |Corr chuva|>=0.05: PASS (0.2475)
- Dias chuva>0>=500: PASS (11918)
- Tensao C0<C1<C2<C3: FAIL (C0=16.9, C1=35.0, C2=24.0, C3=34.6)
- Sem NaN: PASS (0)
- Tensao<1500: PASS (156.3)
- Std tensao>=8: PASS (12.5)
- Veranico>smt: PASS
- Diversidade: PASS (manut=41.3% vs smt=4.8%)

## Veredicto

**REPROVADO** — 10/11 criterios OK