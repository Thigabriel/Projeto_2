# Relatorio de Validacao — dataset_cold_start_v7.csv

**Amostras:** 2733
**Limiar rotulagem:** 40.98 mm
**Janelas:** chuva (Jan 5) + seca (Jun 1)
**Metodo irrigacao:** method=1 (SMT por estagio + MaxIrr cap)
**Cenarios:**
  - excesso: SMT=[80, 80, 80, 70], MaxIrr=100mm, MaxSeason=10000mm
  - otimo: SMT=[60, 60, 70, 50], MaxIrr=100mm, MaxSeason=10000mm
  - deficit: SMT=[40, 40, 50, 30], MaxIrr=100mm, MaxSeason=10000mm

## ETo por Ano

- 2019: Jan-Jun ETo=3.95 prec=826mm | Jul-Dez ETo=5.62 prec=202mm
- 2020: Jan-Jun ETo=3.94 prec=1239mm | Jul-Dez ETo=5.27 prec=388mm
- 2021: Jan-Jun ETo=3.94 prec=712mm | Jul-Dez ETo=5.13 prec=741mm
- 2022: Jan-Jun ETo=3.84 prec=904mm | Jul-Dez ETo=5.27 prec=522mm
- 2023: Jan-Jun ETo=4.08 prec=624mm | Jul-Dez ETo=6.12 prec=200mm

## Distribuicao de Classes

- Classe 0: 2572 (94.1%)
- Classe 1: 111 (4.1%)
- Classe 2: 50 (1.8%)

### Por Cenario x Janela

- excesso/chuva (n=458): C0=453, C1=5, C2=0
- excesso/seca (n=453): C0=368, C1=83, C2=2
- otimo/chuva (n=458): C0=457, C1=0, C2=1
- otimo/seca (n=453): C0=408, C1=23, C2=22
- deficit/chuva (n=458): C0=458, C1=0, C2=0
- deficit/seca (n=453): C0=428, C1=0, C2=25

## Estatisticas das Features

- tensao_solo_kpa: min=4.92, max=136.15, median=18.24, std=14.47
- chuva_acum_3d_mm: min=0.00, max=107.38, median=2.07, std=17.66
- tmax_max_3d_c: min=27.95, max=40.41, median=32.02, std=3.35
- dap: min=14.00, max=107.00, median=59.00, std=26.33

### Tensao por Janela

- chuva: min=4.9, max=54.8, median=15.4, std=5.3 kPa
- seca: min=13.7, max=136.1, median=23.4, std=17.5 kPa

## Correlacoes com classe_irrigacao

- tensao_solo_kpa: 0.3693
- chuva_acum_3d_mm: -0.1414
- tmax_max_3d_c: 0.2188
- dap: 0.0350

## Tensao Mediana por Classe

- Classe 0: 17.6 kPa (n=2572, std=12.9)
- Classe 1: 23.7 kPa (n=111, std=5.5)
- Classe 2: 62.2 kPa (n=50, std=28.6)

## Chuva

- chuva_3d > 0mm: 2004 dias
- chuva_3d > 5mm: 1155 dias

## Criterios de Aprovacao

- C0 >= 5%: PASS (94.1%)
- Corr tensao [+0.15,+0.7]: PASS (0.3693)
- |Corr chuva| >= 0.10: PASS (0.1414)
- Dias chuva>0 >= 100: PASS (2004)
- Tensao C2>C0: PASS (C0=17.6, C2=62.2 kPa)
- Sem NaN: PASS (0)
- Tensao<1500: PASS (max=136.1)
- Desvio padrao tensao >= 10: PASS (14.5)

## Limitacoes Conhecidas

- Profundidade radicular dinamica mitiga mas nao elimina discrepancia canteiro 35cm
- Parametros TomatoGDD calibrados para clima mediterranico
- Duas janelas de plantio: modelo aprende com chuva e seca
- NASA POWER: resolucao ~50km, pode diferir de estacao local

## Referencias

- ALLEN, R.G. et al. FAO Irrigation Paper 56, 1998.
- FOSTER, T. et al. Agric. Water Manag., v.251, 2021.
- SAXTON, K.E.; RAWLS, W.J. Soil Sci. Soc. Am. J. 70:1569-1578, 2006.
- NASA POWER. https://power.larc.nasa.gov/

## Veredicto

**APROVADO** — 8/8 criterios OK