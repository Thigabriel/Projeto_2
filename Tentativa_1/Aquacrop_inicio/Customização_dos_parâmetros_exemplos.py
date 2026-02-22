# Parâmetros Customizáveis do AquaCrop-OSPy (Soil e Crop) e Exemplo de Uso

# ===============================================
# == Parâmetros Customizáveis da Classe Soil ==
# ===============================================
# Referência: Apêndice B do AquaCrop_OSPy_Notebook_1.ipynb

# --- Parâmetros Gerais (no construtor Soil(...)) ---
# soilType (str): Tipo de solo. Padrão ('SandyLoam', etc.) ou 'custom'. Obrigatório.
# dz (list): Espessura (m) de cada compartimento do solo. Padrão: [0.1]*12.
# CalcSHP (int): Calcular propriedades hidráulicas (0=Não, 1=Sim). Padrão: 0.
# AdjREW (int): Ajustar valor padrão de REW (0=Não, 1=Sim). Padrão: 1.
# REW (float): Água facilmente evaporável (mm). Padrão: 9.0.
# CalcCN (int): Calcular número da curva (0=Não, 1=Sim). Padrão: 1.
# CN (float): Número da Curva (Curve Number). Padrão: 61.0.
# zRes (float): Profundidade da camada restritiva (m) (negativo se ausente). Padrão: -999.
# EvapZsurf (float): Espessura da camada superficial de evaporação (m). Padrão: 0.04.
# EvapZmin (float): Espessura mínima da camada completa de evaporação superficial (m). Padrão: 0.15.
# EvapZmax (float): Espessura máxima da camada completa de evaporação superficial (m). Padrão: 0.30.
# Kex (float): Coeficiente máximo de evaporação do solo. Padrão: 1.1.
# fevap (int): Fator de forma para redução da evaporação no estágio 2. Padrão: 4.
# fWrelExp (float): Umidade relativa (Wrel) na qual a camada de evaporação expande. Padrão: 0.4.
# fwcc (float): Coef. máx. para redução de evaporação por abrigo do dossel murcho. Padrão: 50.
# zCN (float): Espessura superficial (m) para cálculo de umidade e ajuste do CN. Padrão: 0.3.
# zGerm (float): Espessura superficial (m) para cálculo de umidade para germinação. Padrão: 0.3.
# AdjCN (int): Ajustar CN por umidade antecedente (0=Não, 1=Sim). Padrão: 1.
# fshape_cr (float): Fator de forma da ascensão capilar. Padrão: 16.
# zTop (float): Espessura superficial (m) para comparações de estresse hídrico. Padrão: 0.1.

# --- Parâmetros por Camada (método .add_layer(...)) ---
# thickness (float): Espessura da camada (m).
# thWP (float): Umidade no Ponto de Murcha (m³/m³).
# thFC (float): Umidade na Capacidade de Campo (m³/m³).
# thS (float): Umidade na Saturação (m³/m³).
# Ksat (float): Condutividade hidráulica saturada (mm/dia).
# penetrability (float): Penetrabilidade (%).

# --- Parâmetros por Camada (método .add_layer_from_texture(...)) ---
# thickness (float): Espessura da camada (m).
# Sand (float): Percentual de Areia (%).
# Clay (float): Percentual de Argila (%).
# OrgMat (float): Percentual de Matéria Orgânica (%).
# penetrability (float): Penetrabilidade (%).

# ==============================================
# == Parâmetros Customizáveis da Classe Crop ==
# ==============================================
# Referência: Apêndice C do AquaCrop_OSPy_Notebook_1.ipynb

# --- Parâmetros Gerais e Fenologia ---
# Name (str): Nome da cultura.
# CropType (int): Tipo (1=Folhosa, 2=Raiz/Tubérculo, 3=Fruto/Grão).
# PlantMethod (int): Método de plantio (0=Transplantado, 1=Semeadura).
# CalendarType (int): Calendário (1=Dias corridos, 2=GDD).
# SwitchGDD (int): Converter calendário para GDD (0=Não, 1=Sim).
# PlantingDate (str): Data de plantio (MM/DD).
# HarvestDate (str): Última data de colheita (MM/DD).
# Emergence (float): Dias/GDD para emergência.
# MaxRooting (float): Dias/GDD para enraizamento máximo.
# Senescence (float): Dias/GDD para senescência.
# Maturity (float): Dias/GDD para maturidade.
# HIstart (float): Dias/GDD para início da formação do rendimento.
# Flowering (float): Duração da floração (dias/GDD, -999 se não aplicável).
# YldForm (float): Duração da formação do rendimento (dias/GDD).

# --- Parâmetros de Temperatura ---
# GDDmethod (int): Método de cálculo de GDD.
# Tbase (float): Temperatura base (°C).
# Tupp (float): Temperatura superior (°C).
# PolHeatStress (int): Polinização afetada por calor (0=Não, 1=Sim).
# Tmax_up (float): Tmax (°C) onde polinização começa a falhar.
# Tmax_lo (float): Tmax (°C) onde polinização falha totalmente.
# PolColdStress (int): Polinização afetada por frio (0=Não, 1=Sim).
# Tmin_up (float): Tmin (°C) onde polinização começa a falhar.
# Tmin_lo (float): Tmin (°C) onde polinização falha totalmente.
# TrColdStress (int): Transpiração afetada por frio (0=Não, 1=Sim).
# GDD_up (float): GDDs mínimos (°C/dia) para transpiração potencial total.
# GDD_lo (float): GDDs (°C/dia) para transpiração zero.

# --- Parâmetros Radiculares ---
# Zmin (float): Profundidade mínima efetiva raiz (m).
# Zmax (float): Profundidade máxima raiz (m).
# fshape_r (float): Fator de forma da expansão radicular.
# SxTopQ (float): Extração máxima de água no topo da ZR (m³/m³/dia).
# SxBotQ (float): Extração máxima de água na base da ZR (m³/m³/dia).
# PctZmin (float): % inicial de Zmin (%). Padrão: 70.
# fshape_ex (float): Fator de forma do efeito do estresse hídrico na expansão radicular. Padrão: -6.

# --- Parâmetros do Dossel ---
# SeedSize (float): Área coberta por plântula (cm²).
# PlantPop (float): Densidade de plantas (plantas/ha).
# CCx (float): Cobertura máxima do dossel (fração).
# CDC (float): Coeficiente de declínio do dossel (fração/GDD ou fração/dia).
# CGC (float): Coeficiente de crescimento do dossel (fração/GDD).
# Kcb (float): Coeficiente de cultura (basal) máx.
# fage (float): Declínio de Kcb por envelhecimento (%/dia).
# GermThr (float): Umidade mínima para germinação (fração TAW). Padrão: 0.2.
# CCmin (float): CC mínima para formação de rendimento. Padrão: 0.05.

# --- Parâmetros de Produtividade e HI ---
# WP (float): Produtividade da água normalizada (g/m²).
# WPy (float): Ajuste de WP na formação de rendimento (% de WP).
# fsink (float): Efeito de CO2 elevado (%/100 ppm).
# HI0 (float): Índice de colheita de referência (%).
# dHI_pre (float): Aumento de HI por estresse pré-floração (%).
# a_HI (float): Coef. impacto positivo no HI por restrição vegetativa.
# b_HI (float): Coef. impacto negativo no HI por fechamento estomático.
# dHI0 (float): Aumento máx. permitido de HI (%).
# Determinant (int): Determinância (0=Indeterminada, 1=Determinada).
# exc (float): Excesso de frutos potenciais (%).
# MaxFlowPct (float): % da floração onde ocorre o pico. Padrão: 33.3.
# HIini (float): HI inicial (%). Padrão: 0.01.

# --- Parâmetros de Estresse Hídrico ---
# p_up1 (float): Limiar superior p/ expansão do dossel (%).
# p_up2 (float): Limiar superior p/ controle estomático (%).
# p_up3 (float): Limiar superior p/ senescência (%).
# p_up4 (float): Limiar superior p/ polinização (%).
# p_lo1 (float): Limiar inferior p/ expansão do dossel (%).
# p_lo2 (float): Limiar inferior p/ controle estomático (%).
# p_lo3 (float): Limiar inferior p/ senescência (%).
# p_lo4 (float): Limiar inferior p/ polinização (%).
# fshape_w1 (float): Fator de forma p/ estresse na expansão.
# fshape_w2 (float): Fator de forma p/ estresse no controle estomático.
# fshape_w3 (float): Fator de forma p/ estresse na senescência.
# fshape_w4 (float): Fator de forma p/ estresse na polinização.
# ETadj (int): Ajustar limiares de estresse pela ETo (0=Não, 1=Sim). Padrão: 1.
# beta (float): Redução (%) de p_lo3 na senescência precoce. Padrão: 12.
# a_Tr (float): Expoente para ajuste de Kcx pós-senescência. Padrão: 1.

# --- Parâmetros de Aeração ---
# Aer (float): Vol (%) abaixo de SAT para estresse de aeração. Padrão: 5.
# LagAer (int): Dias de atraso para efeito do estresse de aeração. Padrão: 3.

# --- Parâmetros de CO2 (raramente modificados) ---
# bsted (float): Ajuste WP CO2 (Steduto). Padrão: 0.000138.
# bface (float): Ajuste WP CO2 (FACE). Padrão: 0.001165.


# ====================================================
# == Exemplo de Código Python usando Customização ==
# ====================================================

# Importar bibliotecas necessárias
from aquacrop import AquaCropModel, Soil, Crop, InitialWaterContent
from aquacrop.utils import prepare_weather, get_filepath
import pandas as pd

# --- 1. Definir Clima (usando arquivo interno como exemplo) ---
weather_file_path = get_filepath('cordoba_climate.txt') # Usando Córdoba
weather_df = prepare_weather(weather_file_path)

# --- 2. Criar Solo Customizado ---
# Solo com duas camadas texturizadas e CN/REW específicos
solo_meu = Soil(soil_type='custom', cn=78, rew=8.5)

# Camada 1: 40cm, Franco-argilo-arenoso
solo_meu.add_layer_from_texture(thickness=0.4,
                                Sand=55, Clay=25, OrgMat=2.0,
                                penetrability=90)

# Camada 2: 80cm, Argiloso-limoso
solo_meu.add_layer_from_texture(thickness=0.8,
                                Sand=15, Clay=45, OrgMat=2.5,
                                penetrability=75)

print("Perfil do Solo Customizado Criado:")
print(solo_meu.profile)
print(f"CN: {solo_meu.cn}, REW: {solo_meu.rew}")

# --- 3. Criar Cultura Customizada (baseada em Trigo) ---
# Trigo com ciclo ajustado, raiz mais profunda e sensibilidade ao estresse alterada
trigo_meu = Crop(crop_type='Wheat',          # Baseado no trigo padrão
                 planting_date='10/20',      # Plantio em 20 de Outubro
                 Maturity=2100,            # Ajuste GDD para maturidade (valor exemplo)
                 MaxRooting=1600,          # Ajuste GDD para enraizamento max (valor exemplo)
                 Zmax=1.3,                 # Raiz máxima de 1.3m
                 p_up2=0.60,               # Limiar superior para estresse estomático (ex: 60% TAW)
                 p_lo2=0.85,               # Limiar inferior para estresse estomático (ex: 85% TAW)
                 Name='Trigo Local'        # Dar um nome descritivo (opcional)
                 )

print(f"\nCultura Customizada: {trigo_meu.Name}")
print(f"  Data Plantio: {trigo_meu.planting_date}")
print(f"  Zmax: {trigo_meu.Zmax} m")
print(f"  Limiares Estresse Estomático (p_up, p_lo): ({trigo_meu.p_up2}, {trigo_meu.p_lo2})")

# --- 4. Definir Conteúdo Inicial de Água (para solo de 2 camadas) ---
# Camada 1: 70% TAW, Camada 2: 50% TAW
# O número de valores na lista 'value' deve corresponder ao número de camadas definidas no solo
iwc_meu = InitialWaterContent(wc_type='Pct',        # Definido como % TAW
                              method='Layer',       # Aplicar por camada
                              depth_layer=[1, 2],   # Índices das camadas (1ª e 2ª)
                              value=[70, 50]        # Valores de % TAW para cada camada
                              )
print(f"\nConteúdo Inicial de Água: Camada 1 a {iwc_meu.value[0]}% TAW, Camada 2 a {iwc_meu.value[1]}% TAW")


# --- 5. Configurar e Rodar o Modelo AquaCrop ---
# Usar um período de simulação menor para exemplo rápido
sim_start = '1980/10/20' # Começa na data de plantio
sim_end = '1982/06/30'   # Simula 2 estações completas

modelo_custom = AquaCropModel(
                    sim_start_time=sim_start,
                    sim_end_time=sim_end,
                    weather_df=weather_df,
                    soil=solo_meu,             # Usa o solo customizado
                    crop=trigo_meu,            # Usa a cultura customizada
                    initial_water_content=iwc_meu # Usa o IWC customizado
                )

# Rodar a simulação
modelo_custom.run_model(till_termination=True)

# --- 6. Exibir Resultados Finais ---
print("\nResultados Finais da Simulação Customizada:")
print(modelo_custom._outputs.final_stats)

# Você pode então adicionar código para plotar ou analisar os resultados diários
# Exemplo: Acessar fluxo de água diário
# print("\nFluxo de Água Diário (primeiros 5 dias):")
# Adicionar coluna Date primeiro, como discutido anteriormente
# datas_sim_custom = weather_df[(weather_df['Date'] >= pd.to_datetime(sim_start)) & (weather_df['Date'] <= pd.to_datetime(sim_end))]['Date']
# modelo_custom._outputs.water_flux['Date'] = datas_sim_custom.iloc[:len(modelo_custom._outputs.water_flux)].values
# print(modelo_custom._outputs.water_flux[['Date', 'precipitation', 'transpiration', 'storage']].head())