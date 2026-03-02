# Relatorio de Validacao — dataset_cold_start_v10.csv

**Data:** 2026-03-01 20:13
**Amostras:** 10554
**Grupos:** 115
**Limiar rotulagem:** 41.04 mm
**Periodo:** 2001-2023 (23 anos)
**Janelas:** chuva (Jan 5) + seca (Jun 1)
**Cenarios chuva:** excesso + veranico (fev-mar × 0.2)
**Cenarios seca:** excesso, otimo, deficit
**Nota:** otimo/chuva e deficit/chuva removidos (geravam <0.3% irrigacao)
**Features:** tensao_solo_kpa, chuva_acum_3d_mm, tmax_max_3d_c, dap, delta_tensao_kpa

## Configuracao dos Cenarios

- excesso: SMT=[80, 80, 80, 70], MaxIrr=100mm
- otimo: SMT=[60, 60, 70, 50], MaxIrr=100mm
- deficit: SMT=[40, 40, 50, 30], MaxIrr=100mm
- veranico: SMT=[60, 60, 70, 50], MaxIrr=100mm (precip fev-mar × 0.2, janela chuva apenas)

## Distribuicao de Classes

- Classe 0: 9782 (92.7%)
- Classe 1: 533 (5.1%)
- Classe 2: 239 (2.3%)

**Comparacao com v7:** C0=94.1%, C1=4.1%, C2=1.8% (2733 amostras, 30 grupos)
**Comparacao com v10-anterior:** C0=94.9%, C1=3.4%, C2=1.7% (14784 amostras, 161 grupos)

### Por Cenario x Janela

- excesso/chuva (n=2115): C0=2093, C1=22, C2=0 (1.0% irrig)
- excesso/seca (n=2108): C0=1735, C1=369, C2=4 (17.7% irrig)
- otimo/seca (n=2108): C0=1912, C1=98, C2=98 (9.3% irrig)
- deficit/seca (n=2108): C0=1994, C1=0, C2=114 (5.4% irrig)
- veranico/chuva (n=2115): C0=2048, C1=44, C2=23 (3.2% irrig)

## Estatisticas das Features

- tensao_solo_kpa: min=4.18, max=140.53, median=20.35, std=15.13
- chuva_acum_3d_mm: min=0.00, max=158.96, median=0.68, std=13.52
- tmax_max_3d_c: min=25.71, max=40.93, median=32.25, std=3.36
- dap: min=14.00, max=107.00, median=59.00, std=26.53
- delta_tensao_kpa: min=-125.55, max=17.29, median=1.89, std=9.51

## Correlacoes com classe_irrigacao

- tensao_solo_kpa: 0.3440
- chuva_acum_3d_mm: -0.1233
- tmax_max_3d_c: 0.1597
- dap: 0.0394
- delta_tensao_kpa: 0.1420

## Tensao Mediana por Classe

- Classe 0: 19.6 kPa (n=9782, std=13.7)
- Classe 1: 24.5 kPa (n=533, std=5.2)
- Classe 2: 60.6 kPa (n=239, std=26.7)

## Delta Tensao Mediana por Classe

- Classe 0: 1.76 kPa (std=9.76)
- Classe 1: 3.07 kPa (std=1.23)
- Classe 2: 7.80 kPa (std=3.35)

## Chuva

- chuva_3d > 0mm: 7690 dias
- chuva_3d > 5mm: 3201 dias

## Criterio Veranico

- Irrigacao janela chuva: excesso=1.0%, veranico=3.2%
- Veranico > excesso: PASS
- Veranico >= 4% irrigacao: FAIL (3.2%)

## Criterios de Aprovacao

- C0 >= 5%: PASS (92.7%)
- Corr tensao [+0.15,+0.7]: PASS (0.3440)
- |Corr chuva| >= 0.10: PASS (0.1233)
- Dias chuva>0 >= 500: PASS (7690)
- Tensao C2>C0: PASS (C0=19.6, C2=60.6 kPa)
- Sem NaN: PASS (0)
- Tensao<1500: PASS (max=140.5)
- Desvio padrao tensao >= 12: PASS (15.1)
- Veranico > otimo (chuva): FAIL

## Limitacoes Conhecidas

- Profundidade radicular dinamica mitiga mas nao elimina discrepancia canteiro 35cm
- Parametros TomatoGDD calibrados para clima mediterranico
- NASA POWER: resolucao ~50km, pode diferir de estacao local
- Cenario veranico assume reducao uniforme fev-mar a 20% (veranicos reais sao episodicos)
- Cenarios otimo/chuva e deficit/chuva removidos por improdutividade (<0.3% irrigacao)
- Desbalanceamento C0 permanece estrutural — tratar no pipeline de treino

## Referencias

- ALLEN, R.G. et al. FAO Irrigation Paper 56, 1998.
- FOSTER, T. et al. Agric. Water Manag., v.251, 2021.
- SAXTON, K.E.; RAWLS, W.J. Soil Sci. Soc. Am. J. 70:1569-1578, 2006.
- NASA POWER. https://power.larc.nasa.gov/
- Embrapa. Veranicos na Regiao dos Cerrados Brasileiros.

## Veredicto

**REPROVADO** — 8/9 criterios OK