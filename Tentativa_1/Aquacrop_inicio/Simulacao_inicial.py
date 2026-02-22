from aquacrop import AquaCropModel, Soil, Crop, InitialWaterContent, FieldMngt, GroundWater, IrrigationManagement
from aquacrop.utils import prepare_weather, get_filepath
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Cinco componentes principais são necessários para configurar uma simulação do AquaCrop usando a biblioteca AquaCrop-OSPy:

# a.  **Dados Climáticos (Weather Data):**
# * Necessita de dados diários de Temperatura Mínima (°C), Temperatura Máxima (°C), Precipitação (mm) e Evapotranspiração de Referência (ETo, mm).
# * Esses dados são geralmente lidos de um arquivo de texto (ex: `.txt` ou `.CLI`). O notebook `AquaCrop_OSPy_Notebook_1.ipynb` mostra o formato esperado (delimitado por espaços ou tabulações, com colunas específicas).
# * A função `prepare_weather()` da biblioteca é usada para ler o arquivo e formatá-lo corretamente em um DataFrame pandas.
# * A biblioteca inclui arquivos climáticos de exemplo (como `tunis_climate.txt`). A função `get_filepath()` pode ser usada para acessar o caminho desses arquivos internos.
# ```python
# Exemplo usando arquivo interno
weather_file_path = 'imperatriz_climate.txt'
weather_df = prepare_weather(weather_file_path)
# print(weather_df.head())
#


# b.  #**Solo (Soil):**
# * Definido através da classe `Soil`. Contém as propriedades hidráulicas e de composição do solo.
# * Você pode usar tipos de solo pré-definidos (como 'SandyLoam', 'Clay', 'Loam', etc.) baseados nos padrões do AquaCrop.
# * Ou pode definir um solo personalizado, especificando suas camadas, texturas (areia, argila, matéria orgânica) ou propriedades hidráulicas (Ponto de Murcha - WP, Capacidade de Campo - FC, Saturação - SAT, Condutividade Hidráulica Saturada - Ksat). O notebook detalha isso no Apêndice B.
# ```python
# Exemplo usando tipo pré-definido
solo_arenoso_franco = Soil(soil_type='SandyLoam')

#Exemplo de solo personalizado
#solo_custom = Soil(soil_type='custom', cn=78, rew=8.5)
#solo_custom.add_layer_from_texture(thickness=0.4,
#                                Sand=55, Clay=25, OrgMat=2.0,
#                                penetrability=90)

#solo_custom.add_layer_from_texture(thickness=0.8,
#                               Sand=15, Clay=45, OrgMat=2.5,
#                                penetrability=75)



# c.  #**Cultura (Crop):**
# * Definida pela classe `Crop`. Contém os parâmetros fenológicos e fisiológicos da cultura.
# * Requer o nome da cultura e a data de plantio (`planting_date='MM/DD'`).
# * Inclui tipos pré-definidos como 'Wheat' (Trigo), 'Maize' (Milho), 'Rice' (Arroz), 'Potato' (Batata).
# * Parâmetros como coeficientes de crescimento (CGC, CDC), profundidade máxima de raiz (Zmax), datas fenológicas (emergência, senescência, maturidade), etc., podem ser customizados. O notebook detalha no Apêndice C.
# ```python
# Exemplo usando tipo pré-definido
# Plantio em 1º de Outubro
trigo = Crop(c_name='Tomato', planting_date='01/07')
       
# ```


# d.  #**Conteúdo Inicial de Água (Initial Water Content - IWC):**
# * Definido pela classe `InitialWaterContent`. Especifica a umidade do solo no início da simulação.
# * Pode ser definido como uma proporção ('Prop') das constantes do solo ('WP', 'FC', 'SAT'), como percentual da Água Total Disponível ('Pct' - % TAW), ou como um valor numérico ('Num' - m³/m³).
# * O padrão é iniciar com o perfil de solo na Capacidade de Campo ('FC').
# ```python
# Exemplo: Iniciar na Capacidade de Campo
iwc = InitialWaterContent(value=['FC'])

manejo_campo= FieldMngt(mulches=True,   
                        mulch_pct=80,    
                        f_mulch=0.3 ) 

gw = GroundWater(water_table='Y',
            dates=[f'{2022}/10/03'],
            values=[2])

irrig_limiar = IrrigationManagement(irrigation_method=2,     
                                    SMT=[50], 
                                    AppEff=90,        
                                    MaxIrr=50,        
                                    AmountType='Variable',
                                    IrrInterval = 2
                                    )

# e.  #**Período de Simulação:**
# * As datas de início (`sim_start_time`) e fim (`sim_end_time`) da simulação são #definidas ao criar o objeto do modelo principal, no formato 'AAAA/MM/DD'.


modelo = AquaCropModel(
    sim_start_time='2022/01/01',
    sim_end_time='2024/04/30',
    weather_df=weather_df,
    soil=solo_arenoso_franco,
    crop=trigo ,
    initial_water_content=iwc,
    field_management= manejo_campo,
    groundwater=gw,
    irrigation_management=irrig_limiar
)

modelo.run_model(till_termination=True)




df_saida1 = modelo.get_water_flux()
df_saida2 = modelo.get_crop_growth()
df_saida3 = modelo.get_water_storage()


data_inicio_sim = modelo._clock_struct.simulation_start_date
data_fim_sim = modelo._clock_struct.simulation_end_date

datas_simulacao = weather_df[
    (weather_df['Date'] >= data_inicio_sim) &
    (weather_df['Date'] <= data_fim_sim)
]['Date']


df_saida1['Date'] = datas_simulacao.iloc[:len(df_saida1)].values
df_saida2['Date'] = datas_simulacao.iloc[:len(df_saida2)].values
df_saida3['Date'] = datas_simulacao.iloc[:len(df_saida3)].values 


data_inicio = '2022-01-01'
data_fim = '2022-04-30'

Planta = df_saida2[ 
    (df_saida2['Date'] >= data_inicio) &
    (df_saida2['Date'] <= data_fim)
]

Agua = df_saida1[
    (df_saida1['Date'] >= data_inicio) &
    (df_saida1['Date'] <= data_fim)
]

Solo = df_saida3[ 
    (df_saida3['Date'] >= data_inicio) &
    (df_saida3['Date'] <= data_fim)
]


print(Agua[['Date','IrrDay']].head(20))
print(Solo[['Date','th1']].head(20))




