# Parâmetros Customizáveis do AquaCrop-OSPy (Manejo) e Exemplo de Uso

# ====================================================
# == Parâmetros Customizáveis da Classe FieldMngt ==
# ====================================================
# Define práticas de manejo da superfície do solo.
# Referência: Apêndice D do AquaCrop_OSPy_Notebook_1.ipynb

# Mulches (bool): Superfície coberta com cobertura morta (mulch)? (True/False). Padrão: False.
# MulchPct (float): Percentagem da área coberta por mulches (%). Padrão: 50.
# fMulch (float): Fator de ajuste da evaporação do solo devido ao mulch (0-1). Padrão: 0.5.
# Bunds (bool): Presença de diques/taipas na superfície? (True/False). Padrão: False.
# zBund (float): Altura dos diques (m). Padrão: 0.
# BundWater (float): Altura inicial da água nos diques (mm). Padrão: 0.
# CNadj (bool): Condições de campo afetam o Número da Curva (CN)? (True/False). Padrão: False.
# CNadjPct (float): Variação percentual no CN devido ao manejo (%). Padrão: 0.
# SRinhb (bool): Manejo inibe completamente o escoamento superficial? (True/False). Padrão: False.

# =======================================================
# == Parâmetros Customizáveis da Classe GroundWater ==
# =======================================================
# Simula a presença e profundidade do lençol freático.
# Referência: Apêndice D do AquaCrop_OSPy_Notebook_1.ipynb

# WaterTable (str): Considerar lençol freático ('Y'=Sim, 'N'=Não). Padrão: 'N'.
# Method (str): Método de entrada dos dados ('Constant'=Profundidade fixa, 'Variable'=Variável no tempo). Padrão: 'Constant'.
# dates (list[str]): Datas ('YYYY/MM/DD' ou 'YYYYMMDD') das observações (se Method='Variable'). Padrão: [].
# values (list[float]): Profundidades (m) correspondentes às datas. Se Method='Constant', lista com um valor. Padrão: [].

# ============================================================
# == Parâmetros Customizáveis da Classe IrrigationManagement ==
# ============================================================
# Define como e quando a irrigação é aplicada.
# Referência: Código fonte (entities/irrigationManagement.py) e documentação/exemplos.

# IrrMethod (int): Método de irrigação.
#   0 = Sem irrigação (Rainfed).
#   1 = Calendário Fixo (Schedule): Irriga quantidades fixas em datas específicas.
#   2 = Limiar de Depleção (Soil Moisture Threshold): Irriga quando a depleção na ZR atinge um limiar (%). **Mais relevante para sistemas automáticos.**
#   3 = Nível Alvo (Target Soil Moisture): Irriga para retornar a umidade a um nível alvo (e.g., FC).
#   4 = Nível de Arrozal (Paddy Rice): Manejo específico para arroz inundado.

# --- Parâmetros Comuns ---
# AppEff (float): Eficiência de aplicação da irrigação (%). Padrão: 100.
# WetSurf (float): Percentagem da superfície do solo molhada pela irrigação (%). Padrão: 100.

# --- Parâmetros para IrrMethod=1 (Schedule) ---
# Schedule (pd.DataFrame ou dict): Tabela/Dicionário com datas ('Date', formato datetime ou str 'YYYY/MM/DD') e profundidades ('Depth', mm).

# --- Parâmetros para IrrMethod=2 (Threshold) ---
# SMT (list[float]): Limiares de depleção (% TAW - Água Total Disponível) que acionam a irrigação, geralmente para 4 estágios (ex: [emergência-vegetativo, vegetativo-floração, floração-maturação, maturação-colheita]). Ex: [30, 50, 60, 70] significa irrigar quando 30% da TAW for consumida no estágio inicial, 50% no vegetativo, etc.
# MaxIrr (float): Quantidade máxima de irrigação aplicada por evento (mm). Padrão: 1000 (sem limite prático).
# IrrInterval (int): Intervalo mínimo entre duas irrigações (dias). Padrão: 1.
# NetIrrSMT (int): Usar Depleção Líquida (considera chuva) para o limiar? (0=Não, 1=Sim). Padrão: 0 (Usa depleção total).
# DepletionField (str): Define qual campo de depleção usar ('Depletion' ou 'RelativeDepletion'). Padrão: 'Depletion'.
# AmountType (str): Define como calcular a quantidade a irrigar ('Variable' para reabastecer até FC, 'Fixed' para usar MaxIrr). Padrão: 'Variable'.
# FixedAmount (float): Quantidade fixa a aplicar se AmountType='Fixed' (mm). Padrão: 10.

# --- Parâmetros para IrrMethod=3 (Target) ---
# (Similar a IrrMethod=2, mas o alvo pode ser especificado, ex: irrigar sempre para voltar a 'FC')
# Verificar documentação para detalhes específicos.

# --- Parâmetros para IrrMethod=4 (Paddy Rice) ---
# Parâmetros específicos para manejo de água em arrozal (altura da lâmina d'água, etc.).
# Verificar documentação para detalhes específicos.

# ========================================================
# == Exemplo de Código Python usando Manejo Customizado ==
# ========================================================

# Importar bibliotecas
from aquacrop import AquaCropModel, Soil, Crop, InitialWaterContent, FieldMngt, GroundWater, IrrigationManagement
from aquacrop.utils import prepare_weather, get_filepath
import pandas as pd

# --- 1. Definir Clima ---
weather_file_path = get_filepath('cordoba_climate.txt')
weather_df = prepare_weather(weather_file_path)

# --- 2. Definir Solo (Exemplo Simples) ---
solo = Soil(soil_type='Loam') # Solo Franco

# --- 3. Definir Cultura (Exemplo Simples) ---
milho = Crop(crop_type='Maize', planting_date='11/01') # Milho plantado em 01/Nov

# --- 4. Definir Conteúdo Inicial de Água ---
iwc = InitialWaterContent(value=['FC']) # Começa na Capacidade de Campo

# --- 5. Definir Manejo de Campo (Exemplo com Mulch) ---
manejo_campo = FieldMngt(Mulches=True,   # Habilita mulch
                         MulchPct=80,    # 80% de cobertura
                         fMulch=0.3)     # Fator de redução de evaporação

print("Manejo de Campo: Mulch habilitado com 80% de cobertura.")

# --- 6. Definir Lençol Freático (Exemplo Variável) ---
# Simula um lençol freático que varia de 2.5m para 2.0m e depois volta para 2.5m
gw = GroundWater(WaterTable='Y',       # Habilita lençol freático
                 Method='Variable',    # Profundidade variável
                 dates=['1980/11/01', '1981/03/01', '1981/07/01'], # Datas das medições
                 values=[2.5, 2.0, 2.5]  # Profundidades (m) nessas datas
                 )
print("\nLençol Freático: Habilitado, profundidade variável entre 2.0m e 2.5m.")

# --- 7. Definir Manejo de Irrigação (Método Limiar - IrrMethod=2) ---
# Irriga quando 50% da Água Total Disponível (TAW) na zona radicular for consumida.
# Reabastece o solo até a Capacidade de Campo (AmountType='Variable').
irrig_limiar = IrrigationManagement(IrrMethod=2,      # Método por Limiar de Depleção
                                    SMT=[50,50,50,50], # Limiar de 50% TAW para todos os estágios
                                    AppEff=90,        # Eficiência de aplicação de 90%
                                    MaxIrr=50,        # Limita a 50mm por evento (se AmountType='Fixed') - aqui não terá efeito
                                    AmountType='Variable' # Irrigar para voltar à Capacidade de Campo
                                    )
print(f"\nManejo de Irrigação: Método Limiar ({irrig_limiar.SMT[0]}% TAW), Eficiência {irrig_limiar.AppEff}%.")


# --- 8. Configurar e Rodar o Modelo AquaCrop com Manejo ---
sim_start = '1980/11/01' # Começa na data de plantio
sim_end = '1982/05/30'   # Simula 1 estação completa e parte da segunda

modelo_manejo = AquaCropModel(
                    sim_start_time=sim_start,
                    sim_end_time=sim_end,
                    weather_df=weather_df,
                    soil=solo,
                    crop=milho,
                    initial_water_content=iwc,
                    field_management=manejo_campo,      # Aplica manejo de campo
                    groundwater=gw,                     # Aplica lençol freático
                    irrigation_management=irrig_limiar  # Aplica manejo de irrigação
                )

# Rodar a simulação
modelo_manejo.run_model(till_termination=True)

# --- 9. Exibir Resultados Finais (com irrigação) ---
print("\nResultados Finais da Simulação com Manejo:")
print(modelo_manejo._outputs.final_stats)

# --- 10. Analisar Irrigação Aplicada (Exemplo) ---
# Adicionar coluna 'Date' se necessário
if 'Date' not in modelo_manejo._outputs.water_flux.columns:
     datas_sim_manejo = weather_df[(weather_df['Date'] >= pd.to_datetime(sim_start)) & (weather_df['Date'] <= pd.to_datetime(sim_end))]['Date']
     modelo_manejo._outputs.water_flux['Date'] = datas_sim_manejo.iloc[:len(modelo_manejo._outputs.water_flux)].values

# Filtrar dias onde houve irrigação
dias_irrigados = modelo_manejo._outputs.water_flux[modelo_manejo._outputs.water_flux['irrigation'] > 0]

print("\nDias e Quantidades de Irrigação Aplicadas (Primeiros 10 eventos):")
print(dias_irrigados[['Date', 'irrigation']].head(10))

# Calcular irrigação total na primeira estação
irrig_total_s0 = modelo_manejo._outputs.final_stats.loc[0, 'Seasonal irrigation (mm)']
print(f"\nIrrigação Total na Estação 0: {irrig_total_s0:.1f} mm")

# Plotar armazenamento e irrigação juntos (Exemplo)
import matplotlib.pyplot as plt

fig, ax1 = plt.subplots(figsize=(12, 6))

color = 'tab:blue'
ax1.set_xlabel('Data')
ax1.set_ylabel('Armazenamento (mm)', color=color)
ax1.plot(modelo_manejo._outputs.water_flux['Date'], modelo_manejo._outputs.water_flux['storage'], color=color, label='Armazenamento')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx() # Compartilha o mesmo eixo x
color = 'tab:red'
ax2.set_ylabel('Irrigação (mm)', color=color)
ax2.bar(modelo_manejo._outputs.water_flux['Date'], modelo_manejo._outputs.water_flux['irrigation'], color=color, alpha=0.6, label='Irrigação', width=1.0)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(bottom=0) # Garante que as barras comecem em 0

plt.title('Armazenamento de Água no Solo e Eventos de Irrigação (Limiar 50% TAW)')
fig.tight_layout()
plt.show()