import pandas as pd
import sys
import numpy as np
import math
from eto import ETo  # Agora, este 'ETo' é o correto!


def carregar_dados_brutos_inmet(caminho_arquivo, linhas_pular=8, separador=';', decimal=','):
    """
    Carrega um arquivo CSV padrão do INMET.
    """
    try:
        df = pd.read_csv(
            caminho_arquivo,
            sep=separador,
            decimal=decimal,
            skiprows=linhas_pular,
            encoding='latin-1'
        )
        print("Arquivo bruto carregado com sucesso (encoding 'latin-1')!")
        print("-" * 30)
        return df
    except Exception as e:
        print(f"Erro ao tentar carregar com 'latin-1': {e}")
        try:
            df = pd.read_csv(
                caminho_arquivo,
                sep=separador,
                decimal=decimal,
                skiprows=linhas_pular,
                encoding='utf-8'
            )
            print("Arquivo bruto carregado com sucesso (encoding 'utf-8')!")
            print("-" * 30)
            return df
        except Exception as e_utf:
            print(f"Erro ao tentar carregar com 'utf-8': {e_utf}")
            print("Não foi possível carregar o arquivo.")
            return None


def agregar_para_diario(df_bruto):
    """
    Converte dados horários do INMET em dados diários agregados.
    """
    try:
        df = df_bruto.copy()
        if 'Unnamed: 19' in df.columns:
            df = df.drop(columns=['Unnamed: 19'])

        # 2. Converter colunas de Data e Hora para um índice datetime
        df['Hora_Formatada'] = df['Hora UTC'].str.slice(
            0, 4).str.replace(r'(\d{2})(\d{2})', r'\1:\2', regex=True)
        df['datetime'] = pd.to_datetime(
            df['Data'] + ' ' + df['Hora_Formatada'], format='%Y/%m/%d %H:%M')
        df = df.set_index('datetime')
        print("Índice de data/hora criado com sucesso.")

        # 3. Definir as regras de agregação
        regras_agregacao = {
            'PRECIPITAÇÃO TOTAL, HORÁRIO (mm)': 'sum',
            'TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)': 'max',
            'TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)': 'min',
            'UMIDADE RELATIVA DO AR, HORARIA (%)': 'mean',
            'VENTO, VELOCIDADE HORARIA (m/s)': 'mean',
            'RADIACAO GLOBAL (Kj/m²)': 'sum'
        }

        # 4. Aplicar a agregação diária
        df_diario = df.resample('D').agg(regras_agregacao)
        print("Agregação diária concluída.")

        # 5. Renomear colunas
        df_diario = df_diario.rename(columns={
            'PRECIPITAÇÃO TOTAL, HORÁRIO (mm)': 'Chuva',
            'TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)': 'Tmax',
            'TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)': 'Tmin',
            'UMIDADE RELATIVA DO AR, HORARIA (%)': 'RH',
            'VENTO, VELOCIDADE HORARIA (m/s)': 'Vento_10m',  # Nome temporário
            'RADIACAO GLOBAL (Kj/m²)': 'Rs_Kj_dia'  # Nome temporário
        })

        # 6. Converter Radiação (Rs) de Kj/dia para MJ/dia
        df_diario['Rs'] = df_diario['Rs_Kj_dia'] / 1000.0

        # 7. Converter Vento de 10m para 2m (Padrão FAO Penman-Monteith)
        df_diario['Vento_2m'] = df_diario['Vento_10m'] * 0.748
        print("Vento convertido de 10m para 2m.")

        # 8. Arredondar valores para melhor visualização (opcional)
        df_diario = df_diario.round(2)

        # 9. Remover colunas intermediárias
        df_diario = df_diario.drop(columns=['Rs_Kj_dia', 'Vento_10m'])

        return df_diario

    except Exception as e:
        print(f"\nOcorreu um erro inesperado durante a agregação: {e}")
        return None


def calcular_eto_penman_monteith(df_diario, latitude_graus, altitude_metros):
    """
    Calcula a ETo (Penman-Monteith) para cada dia no DataFrame.
    Usa a biblioteca 'ETo' (a correta).
    """
    try:
        print(f"\nIniciando cálculo da ETo (Penman-Monteith) com a biblioteca 'ETo'...")
        print(f"  Latitude: {latitude_graus}°, Altitude: {altitude_metros}m")

        # 1. Converter latitude de graus para radianos
        lat_radianos = math.radians(latitude_graus)

        # 2. Criar uma função 'wrapper' para aplicar em cada linha
        def calcular_eto_linha(linha):
            # A. Criar a instância da classe ETo
            calc = ETo()

            # B. Extrair parâmetros
            dia_do_ano = linha.name.dayofyear
            tmin_val = linha['Tmin']
            tmax_val = linha['Tmax']
            rh_mean_val = linha['RH']
            vento_2m = linha['Vento_2m']
            rs_mj = linha['Rs']

            # C. Carregar parâmetros na instância (usando param_est)
            #    Agora a biblioteca correta está instalada,
            #    ela VAI reconhecer 't_min' e 't_max'.
            calc.param_est(
                t_min=tmin_val,
                t_max=tmax_val,
                rh_mean=rh_mean_val,
                wind_speed=vento_2m,
                sol_rad=rs_mj,
                lat=lat_radianos,
                elevation=altitude_metros,
                doy=dia_do_ano
            )

            # D. Chamar o método de cálculo (que agora existe)
            eto_val = calc.eto_fao56()
            return eto_val

        # 3. Aplicar a função a cada linha do DataFrame
        df_diario['ETo'] = df_diario.apply(calcular_eto_linha, axis=1)

        # Arredondar a ETo final
        df_diario['ETo'] = df_diario['ETo'].round(2)

        print("Cálculo da ETo concluído com sucesso.")
        return df_diario

    except ImportError:
        print("\nERRO: A biblioteca 'ETo' (com T maiúsculo) não foi encontrada.")
        print("Por favor, instale-a usando: pip install ETo")
        return None
    except Exception as e:
        print(f"\nOcorreu um erro durante o cálculo da ETo: {e}")
        return None

# --- INÍCIO DA EXECUÇÃO ---


caminho_arquivo = r"C:\Users\thiga\OneDrive\Documentos\GitHub\Projeto_2\Dados__inmet\IMPERATRIZ__2022.CSV"
linhas_cabecalho = 8
separador_coluna = ';'
separador_decimal = ','

# 2. Metadados da Estação (Imperatriz - MA)
LATITUDE_ESTACAO = -5.53638888
ALTITUDE_ESTACAO = 126.33

# 3. Carregar os dados brutos
dados_brutos = carregar_dados_brutos_inmet(
    caminho_arquivo,
    linhas_pular=linhas_cabecalho,
    separador=separador_coluna,
    decimal=separador_decimal
)

# 4. Se carregou, agregar para diário
if dados_brutos is not None:
    dados_diarios = agregar_para_diario(dados_brutos)

    # 5. Se agregou, calcular a ETo
    if dados_diarios is not None:
        dados_com_eto = calcular_eto_penman_monteith(
            dados_diarios,
            latitude_graus=LATITUDE_ESTACAO,
            altitude_metros=ALTITUDE_ESTACAO
        )

        if dados_com_eto is not None:
            print("\n--- DADOS FINAIS COM ETo (PRIMEIRAS 5 LINHAS) ---")
            print(dados_com_eto.head())

            print("\n--- INFORMAÇÕES DOS DADOS FINAIS (COLUNAS E DADOS FANTES) ---")
            dados_com_eto.info(buf=sys.stdout)
