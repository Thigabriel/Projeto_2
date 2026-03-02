# Relatorio de Validacao — dataset_cold_start_v11.csv

**Data:** 2026-03-02 02:08
**Amostras:** 10561 | **Grupos:** 115
**Periodo:** 2001-2023 (23 anos)
**Limiar ruido:** 2.0mm | **Classes:** C0/<2.0, C1/[2.0,10.0), C2/>=10.0mm
**Features:** tensao_solo_kpa, chuva_acum_3d_mm, tmax_max_3d_c, dap, delta_tensao_kpa

## Cenarios

- **smt_otimo** (method 1): SMT otimo (method 1) — irriga sob demanda
- **manutencao** (method 4): Manutencao (method 4) — net irrigation diaria, alvo 70% TAW
- **veranico** (method 1): Veranico (method 1 + precip fev-mar x 0.20)

## Distribuicao de Classes

- Sem irrigacao (C0): 8547 (80.9%)
- Manutencao (C1): 1745 (16.5%)
- Intensiva (C2): 269 (2.5%)

**v7:** C0=94.1%, C1=4.1%, C2=1.8%

### Por Cenario x Janela

- smt_otimo/chuva (n=2115): C0=2109, C1=0, C2=6 | dose=41.4mm
- smt_otimo/seca (n=2108): C0=1912, C1=0, C2=196 | dose=44.6mm
- manutencao/chuva (n=2115): C0=2093, C1=22, C2=0 | dose=3.6mm
- manutencao/seca (n=2108): C0=385, C1=1723, C2=0 | dose=5.3mm
- veranico/chuva (n=2115): C0=2048, C1=0, C2=67 | dose=41.3mm

### Por Metodo

- smt_otimo (m1): C0=95.2%, C1=0.0%, C2=4.8% | tensao std=9.9
- manutencao (m4): C0=58.7%, C1=41.3%, C2=0.0% | tensao std=9.9
- veranico (m1): C0=96.8%, C1=0.0%, C2=3.2% | tensao std=8.5

## Features

- tensao_solo_kpa: min=4.18, max=67.09, med=20.25, std=9.78
- chuva_acum_3d_mm: min=0.00, max=158.96, med=3.88, std=16.51
- tmax_max_3d_c: min=25.71, max=40.93, med=31.10, std=3.13
- dap: min=14.00, max=107.00, med=59.00, std=26.54
- delta_tensao_kpa: min=-52.49, max=10.78, med=0.59, std=4.90

## Correlacoes

- tensao_solo_kpa: 0.6067
- chuva_acum_3d_mm: -0.2921
- tmax_max_3d_c: 0.4676
- dap: 0.1018
- delta_tensao_kpa: 0.0970

## Tensao por Classe

- C0 (Sem irrigacao): 17.6 kPa (n=8547, std=8.4)
- C1 (Manutencao): 35.0 kPa (n=1745, std=0.4)
- C2 (Intensiva): 34.6 kPa (n=269, std=9.9)

## Criterios

- C0>=5%: PASS (80.9%)
- C2>=0.5%: PASS (2.5%)
- Corr tensao [0.10,0.70]: PASS (0.6067)
- |Corr chuva|>=0.05: PASS (0.2921)
- Dias chuva>0>=500: PASS (8650)
- Tensao C0<C1<C2: FAIL (C0=17.6, C1=35.0, C2=34.6)
- Sem NaN: PASS (0)
- Tensao<1500: PASS (67.1)
- Std tensao>=8: PASS (9.8)
- Veranico>smt: PASS
- Diversidade: PASS (manut=41.3% vs smt=4.8%)

## Veredicto

**REPROVADO** — 10/11 criterios OK