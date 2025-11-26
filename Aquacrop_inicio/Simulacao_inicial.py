from aquacrop import AquaCropModel, Soil, Crop, InitialWaterContent
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
weather_file_path = get_filepath('cordoba_climate.txt')
weather_df = prepare_weather(weather_file_path)
# print(weather_df.head())
# ```


# b.  #**Solo (Soil):**
# * Definido através da classe `Soil`. Contém as propriedades hidráulicas e de composição do solo.
# * Você pode usar tipos de solo pré-definidos (como 'SandyLoam', 'Clay', 'Loam', etc.) baseados nos padrões do AquaCrop.
# * Ou pode definir um solo personalizado, especificando suas camadas, texturas (areia, argila, matéria orgânica) ou propriedades hidráulicas (Ponto de Murcha - WP, Capacidade de Campo - FC, Saturação - SAT, Condutividade Hidráulica Saturada - Ksat). O notebook detalha isso no Apêndice B.
# ```python
# Exemplo usando tipo pré-definido
solo_arenoso_franco = Soil(soil_type='SandyLoam')
# ```


# c.  #**Cultura (Crop):**
# * Definida pela classe `Crop`. Contém os parâmetros fenológicos e fisiológicos da cultura.
# * Requer o nome da cultura e a data de plantio (`planting_date='MM/DD'`).
# * Inclui tipos pré-definidos como 'Wheat' (Trigo), 'Maize' (Milho), 'Rice' (Arroz), 'Potato' (Batata).
# * Parâmetros como coeficientes de crescimento (CGC, CDC), profundidade máxima de raiz (Zmax), datas fenológicas (emergência, senescência, maturidade), etc., podem ser customizados. O notebook detalha no Apêndice C.
# ```python
# Exemplo usando tipo pré-definido
# Plantio em 1º de Outubro
trigo = Crop(c_name='Wheat', planting_date='05/01')
# ```


# d.  #**Conteúdo Inicial de Água (Initial Water Content - IWC):**
# * Definido pela classe `InitialWaterContent`. Especifica a umidade do solo no início da simulação.
# * Pode ser definido como uma proporção ('Prop') das constantes do solo ('WP', 'FC', 'SAT'), como percentual da Água Total Disponível ('Pct' - % TAW), ou como um valor numérico ('Num' - m³/m³).
# * O padrão é iniciar com o perfil de solo na Capacidade de Campo ('FC').
# ```python
# Exemplo: Iniciar na Capacidade de Campo
iwc = InitialWaterContent(value=['FC'])
# ```


# e.  #**Período de Simulação:**
# * As datas de início (`sim_start_time`) e fim (`sim_end_time`) da simulação são #definidas ao criar o objeto do modelo principal, no formato 'AAAA/MM/DD'.


modelo = AquaCropModel(
    sim_start_time='1991/05/01',
    sim_end_time='2004/05/30',
    weather_df=weather_df,
    soil=solo_arenoso_franco,
    crop=trigo,
    initial_water_content=iwc
)

modelo.run_model(till_termination=True)





