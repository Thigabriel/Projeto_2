import sys
import math
import numpy as np
import pandas as pd
import inspect

# --- INÍCIO DO DIAGNÓSTICO ---
try:
    from eto import ETo
    import eto as eto_module
    print(f"Diagnóstico: Biblioteca 'ETo' (científica) importada com sucesso de:")
    print(f"Local: {eto_module.__file__}")
except ImportError as e:
    print(f"ERRO CRÍTICO: Não foi possível importar 'from eto import ETo'.")
    print(f"Erro: {e}")
    print("Por favor, rode 'pip install eto' (com 'e' minúsculo).")
    sys.exit()  # Para o script
except Exception as e:
    print(f"Um erro inesperado ocorreu durante a importação: {e}")
    sys.exit()
# --- FIM DO DIAGNÓSTICO ---


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
    try:
        print(f"\nIniciando cálculo da ETo (Penman-Monteith) com a biblioteca 'eto'...")
        print(f"  Latitude: {latitude_graus}°, Altitude: {altitude_metros}m")

        # Preparar DataFrame no formato esperado por param_est (usa índice diário)
        df_input = pd.DataFrame({
            'Tmin': df_diario['Tmin'].astype(float),
            'Tmax': df_diario['Tmax'].astype(float),
            'RH': df_diario['RH'].astype(float),
            'Rs': df_diario['Rs'].astype(float),        # MJ/m²/dia
            'uz': df_diario['Vento_2m'].astype(float)   # m/s em 2 m
        }, index=df_diario.index)

        calc = ETo()

        # Chamar param_est passando o DataFrame (conforme assinatura detectada)
        try:
            out = calc.param_est(df_input, freq='D', z_msl=float(
                altitude_metros), lat=float(latitude_graus), z_u=2)
        except Exception as e:
            # diagnóstico detalhado se falhar
            print("DEBUG: falha ao chamar calc.param_est(df, ...):", e)
            try:
                print("DEBUG: assinatura param_est:",
                      inspect.signature(calc.param_est))
            except Exception:
                pass
            raise

        # Tratar os possíveis tipos de retorno
        # 1) DataFrame retornado contendo coluna de ETo
        if isinstance(out, pd.DataFrame):
            candidate_cols = ['ETo', 'eto', 'ET0', 'ETc', 'eto_fao56']
            for c in candidate_cols:
                if c in out.columns:
                    df_diario['ETo'] = out[c].astype(float).values
                    df_diario['ETo'] = df_diario['ETo'].round(2)
                    print("Cálculo da ETo concluído com sucesso (retorno DataFrame).")
                    return df_diario
            # se DataFrame tem única coluna, assumir que é ETo
            if out.shape[1] == 1:
                df_diario['ETo'] = out.iloc[:, 0].astype(float).values
                df_diario['ETo'] = df_diario['ETo'].round(2)
                print("Cálculo da ETo concluído com sucesso (DataFrame com 1 coluna).")
                return df_diario

        # 2) Series / ndarray
        if isinstance(out, (pd.Series, np.ndarray)):
            df_diario['ETo'] = np.array(out).astype(float)
            df_diario['ETo'] = df_diario['ETo'].round(2)
            print("Cálculo da ETo concluído com sucesso (retorno Series/ndarray).")
            return df_diario

        # 3) valor escalar
        if isinstance(out, (int, float, np.floating)):
            df_diario['ETo'] = float(out)
            df_diario['ETo'] = df_diario['ETo'].round(2)
            print("Cálculo da ETo concluído (retorno escalar).")
            return df_diario

        # 4) fallback: tentar obter resultado via método eto_fao56() do objeto
        try:
            eto_res = calc.eto_fao56()
            if isinstance(eto_res, (pd.Series, np.ndarray)):
                df_diario['ETo'] = np.array(eto_res).astype(float)
                df_diario['ETo'] = df_diario['ETo'].round(2)
                print("Cálculo da ETo concluído com sucesso (fallback eto_fao56).")
                return df_diario
        except Exception:
            pass

        # diagnóstico final antes de falhar
        print("DEBUG: param_est retornou tipo inesperado:", type(out))
        if hasattr(out, 'columns'):
            print("DEBUG: colunas retornadas:", getattr(out, 'columns', None))

        raise RuntimeError(
            "param_est retornou resultado inesperado; verifique API/versão da biblioteca 'eto'.")

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
