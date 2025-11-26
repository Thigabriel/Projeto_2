import pandas as pd
import sys
import numpy as np


def carregar_dados_brutos_inmet(caminho_arquivo, linhas_pular=8, separador=';', decimal=','):
    """
    Carrega um arquivo CSV padrão do INMET.

    Argumentos:
    caminho_arquivo (str): O caminho completo para o seu arquivo .csv.
    linhas_pular (int): Número de linhas de cabeçalho antes dos nomes das colunas.
    separador (str): O caractere usado para separar as colunas.
    decimal (str): O caractere usado para indicar casas decimais.
    """
    try:
        # Tenta carregar com encoding 'latin-1', comum em arquivos portugueses
        df = pd.read_csv(
            caminho_arquivo,
            sep=separador,
            decimal=decimal,
            skiprows=linhas_pular,
            encoding='latin-1'
        )
        print("Arquivo carregado com sucesso (encoding 'latin-1')!")
        print("-" * 30)
        return df
    except Exception as e:
        print(f"Erro ao tentar carregar com 'latin-1': {e}")
        try:
            # Se falhar, tenta com 'utf-8'
            df = pd.read_csv(
                caminho_arquivo,
                sep=separador,
                decimal=decimal,
                skiprows=linhas_pular,
                encoding='utf-8'
            )
            print("Arquivo carregado com sucesso (encoding 'utf-8')!")
            print("-" * 30)
            return df
        except Exception as e_utf:
            print(f"Erro ao tentar carregar com 'utf-8': {e_utf}")
            print("Não foi possível carregar o arquivo. Verifique os parâmetros:")
            print(f"  - Caminho: {caminho_arquivo}")
            print(f"  - Linhas para pular: {linhas_pular}")
            print(f"  - Separador: '{separador}'")
            print(f"  - Decimal: '{decimal}'")
            return None


def agregar_para_diario(df_bruto):
    """
    Converte dados horários do INMET em dados diários agregados.
    """
    try:
        # 0. Fazer uma cópia para evitar sobreescrever o original
        df = df_bruto.copy()

        # 1. Limpar a coluna 'Unnamed: 19' se ela existir
        if 'Unnamed: 19' in df.columns:
            df = df.drop(columns=['Unnamed: 19'])

        # 2. Converter colunas de Data e Hora para um índice datetime
        # A 'Hora UTC' está como '0000 UTC', '0100 UTC', etc.
        # Vamos formatar para '00:00', '01:00'
        df['Hora_Formatada'] = df['Hora UTC'].str.slice(
            0, 4).str.replace(r'(\d{2})(\d{2})', r'\1:\2', regex=True)

        # Combinar Data e Hora Formatada
        df['datetime'] = pd.to_datetime(
            df['Data'] + ' ' + df['Hora_Formatada'], format='%Y/%m/%d %H:%M')

        # Definir como índice
        df = df.set_index('datetime')
        print("Índice de data/hora criado com sucesso.")

        # 3. Definir as regras de agregação (mapeamento exato das colunas)
        regras_agregacao = {
            'PRECIPITAÇÃO TOTAL, HORÁRIO (mm)': 'sum',
            'TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)': 'max',
            'TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)': 'min',
            'UMIDADE RELATIVA DO AR, HORARIA (%)': 'mean',
            'VENTO, VELOCIDADE HORARIA (m/s)': 'mean',
            # Vamos somar o total de Kj do dia
            'RADIACAO GLOBAL (Kj/m²)': 'sum'
        }

        # 4. Aplicar a agregação diária (resample por 'D' = Dia)
        df_diario = df.resample('D').agg(regras_agregacao)
        print("Agregação diária concluída.")

        # 5. Renomear colunas para o padrão AquaCrop
        df_diario = df_diario.rename(columns={
            'PRECIPITAÇÃO TOTAL, HORÁRIO (mm)': 'Chuva',
            'TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)': 'Tmax',
            'TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)': 'Tmin',
            'UMIDADE RELATIVA DO AR, HORARIA (%)': 'RH',
            'VENTO, VELOCIDADE HORARIA (m/s)': 'Vento',
            'RADIACAO GLOBAL (Kj/m²)': 'Rs_Kj_dia'  # Nome temporário
        })

        # 6. Converter Radiação (Rs) de Kj/dia para MJ/dia
        # A unidade padrão do AquaCrop/Penman-Monteith é MJ/m²/dia
        df_diario['Rs'] = df_diario['Rs_Kj_dia'] / 1000.0

        # Devido aos dados faltantes, onde a soma foi 0 (mas deveria ser NaN),
        # vamos corrigir. Se a soma foi 0, mas havia dados horários,
        # (na verdade, a radiação pode ser 0 à noite).
        # A radiação faltante (4624 non-null) já deve ter sido propagada
        # como NaN pela 'sum' (se todos os valores daquele dia eram NaN)
        # ou como um valor (se alguns eram NaN).

        # Vamos substituir dias onde a soma deu 0 (mas não era noite o dia todo)
        # por NaN, se a contagem de não-nulos for 0 naquele dia.
        # O .resample().agg() já lida bem com NaNs (skipna=True por padrão).
        # O problema é que o INMET pode registrar 0 em vez de NaN.
        # No seu caso (4624 non-null), muitos dias terão Rs = 0.0

        # Se um dia inteiro teve medições NaN, a soma será 0.
        # Vamos converter somas 0 em NaN, exceto se a Tmax for muito baixa (noite polar, não é o caso).
        # No seu caso: 'RADIACAO GLOBAL (Kj/m²)' (4624 non-null)
        # O resample().sum() vai dar 0 para dias que só tinham NaN.
        # Vamos verificar quantos dias ficaram com 0

        dias_com_zero_Rs = df_diario[df_diario['Rs'] == 0.0].shape[0]
        print(
            f"Detectamos {dias_com_zero_Rs} dias onde a Radiação Solar (Rs) foi 0.0.")

        # Vamos converter Rs=0.0 para NaN, pois é muito provável que sejam dados faltantes
        # (exceto talvez em dias de chuva extrema, mas é mais seguro tratar como faltante)
        df_diario['Rs'] = df_diario['Rs'].replace(0.0, np.nan)
        print("Valores de Rs=0.0 foram convertidos para NaN (Dado Faltante).")

        # 7. Remover colunas intermediárias
        df_diario = df_diario.drop(columns=['Rs_Kj_dia'])

        return df_diario

    except KeyError as e:
        print(f"\nERRO: Nome da coluna não encontrado: {e}")
        print("Verifique se os nomes das colunas no seu CSV são exatamente os esperados.")
        return None
    except Exception as e:
        print(f"\nOcorreu um erro inesperado durante a agregação: {e}")
        return None


caminho_arquivo = r"C:\Users\thiga\OneDrive\Documentos\GitHub\Projeto_2\Dados__inmet\IMPERATRIZ__2022.CSV"
linhas_cabecalho = 8
separador_coluna = ';'
separador_decimal = ','


dados_brutos = carregar_dados_brutos_inmet(
    caminho_arquivo,
    linhas_pular=linhas_cabecalho,
    separador=separador_coluna,
    decimal=separador_decimal
)

if dados_brutos is not None:
    dados_diarios = agregar_para_diario(dados_brutos)

    if dados_diarios is not None:
        print("\n--- DADOS DIÁRIOS AGREGADOS (PRIMEIRAS 5 LINHAS) ---")
        print(dados_diarios.head())

        print("\n--- INFORMAÇÕES DOS DADOS DIÁRIOS (COLUNAS E DADOS FALTANTES) ---")
        dados_diarios.info(buf=sys.stdout)
