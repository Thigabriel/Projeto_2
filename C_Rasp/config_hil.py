# config_hil.py — Parâmetros editáveis do protocolo HIL
# Editar antes de executar main_hil.py

# === PLANTIO ===
DATA_PLANTIO = "2025-06-01"   # data real de plantio (actualizar)
DAP_MAXIMO   = 107            # dias — após este DAP o sistema hiberna

# === LOCALIZAÇÃO ===
LATITUDE     = -5.5253
LONGITUDE    = -47.4825
TIMEZONE     = "America/Fortaleza"

# === MODELO ===
PKL_INICIAL  = "memoria_cold_start_v7.pkl"
PKL_CAMPO    = "memoria_campo.pkl"

# === SAXTON & RAWLS — Franco-Arenoso S=65%, C=10%, OM=3% (não alterar) ===
THETA_CC     = 0.1864   # m³/m³ — capacidade de campo (33 kPa)
THETA_PM     = 0.0853   # m³/m³ — ponto de murchamento (1500 kPa)
THETA_SAT    = 0.3863   # m³/m³ — saturação
A_SAXTON     = 0.0090
B_SAXTON     = 4.8825

# === LIMIARES AGRONÓMICOS — Tomate (não alterar) ===
TENSAO_ENCHARCADO  = 15.0   # kPa — abaixo: risco anóxia
TENSAO_OPTIMA_MIN  = 10.0   # kPa
TENSAO_OPTIMA_MAX  = 40.0   # kPa — limiar de cancelamento por chuva
TENSAO_STRESS_MOD  = 60.0   # kPa — irrigar moderadamente
TENSAO_STRESS_SEV  = 90.0   # kPa — irrigar intensamente
TENSAO_RANGE_MAX   = 136.0  # kPa — limite superior do range de treino

# === BOMBA / IRRIGAÇÃO ===
CAUDAL_LPM   = 2.0    # L/min — calibrar em campo
AREA_M2      = 1.0    # m² do canteiro
VOLUMES_MM   = {0: 0.0, 1: 3.0, 2: 7.0}  # mm por classe

# === NORMALIZADOR ONLINE ===
# n_inicial: peso dos dados simulados vs dados reais
# 50 = compromisso entre estabilidade inicial e adaptação em campo
NORM_N_INICIAL = 50

# === FEEDBACK ===
FEEDBACK_JANELA_DIAS    = 2   # dias para avaliar efeito da decisão
FEEDBACK_MIN_OCORRENCIAS = 2  # eventos antes de retreinar (anti-degeneração)

# === SIMULAÇÃO HIL ===
# Hora simulada de decisão (manhã) e acção (tarde)
HORA_DECISAO = 6    # 06h00
HORA_ACCAO   = 18   # 18h00

# === PATHS ===
import os
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
RESULTADOS_DIR = os.path.join(BASE_DIR, "resultados_hil")
LOG_SISTEMA   = os.path.join(BASE_DIR, "log_sistema.txt")
CACHE_METEO   = os.path.join(BASE_DIR, "cache_meteo.json")
