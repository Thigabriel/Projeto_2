# simulador_sensor.py — Gerador determinístico de θ para validação HIL
#
# Substitui sensores.py durante o protocolo HIL.
# Quando o sensor físico for adquirido e calibrado, apenas este ficheiro
# é substituído — o resto do sistema não muda.
#
# Também inclui:
#   - SimuladorChuva: simula sensor de chuva físico (reed switch / pluviômetro)
#   - umidade_para_tensao_kpa: conversão Saxton & Rawls

import numpy as np
from config_hil import (
    THETA_CC, THETA_PM, THETA_SAT, A_SAXTON, B_SAXTON,
    TENSAO_OPTIMA_MAX
)


# ------------------------------------------------------------------
# CONVERSÃO SAXTON & RAWLS
# ------------------------------------------------------------------

def umidade_para_tensao_kpa(theta_vol):
    """
    Converte umidade volumétrica (m³/m³) → tensão do solo (kPa).
    Saxton & Rawls (2006) — Solo Franco-Arenoso S=65%, C=10%, OM=3%.

    Range válido: 33 kPa (CC) a 1500 kPa (PM).
    Tensões > 136 kPa estão fora do range de treino do modelo — registar no log.
    """
    theta_safe = np.clip(theta_vol, THETA_PM, THETA_CC)
    tensao     = A_SAXTON * (theta_safe ** (-B_SAXTON))
    return float(np.clip(tensao, 33.0, 1500.0))


# ------------------------------------------------------------------
# SIMULADOR DO SENSOR CAPACITIVO DE SOLO
# ------------------------------------------------------------------

class SimuladorSensor:
    """
    Substitui o sensor capacitivo físico durante validação HIL.

    Simula a dinâmica real do solo usando:
      - Secagem por evapotranspiração (ETc varia com DAP)
      - Recarga por irrigação (efeito no dia seguinte)
      - Recarga por chuva (70% de eficiência — resto é runoff/evaporação)
      - Ruído de sensor ±2% (realista para sensor capacitivo comum)

    Parâmetros Saxton & Rawls — Franco-Arenoso S=65%, C=10%, OM=3%:
      θ_CC  = 0.1864 m³/m³  (capacidade de campo, ~33 kPa)
      θ_PM  = 0.0853 m³/m³  (ponto de murchamento, ~1500 kPa)
      ETc típica Imperatriz-MA: ~5-7 mm/dia → ~0.008 m³/m³/dia
    """

    TAXA_SECAGEM_BASE  = 0.008   # m³/m³/dia sem irrigação (ETc ~5mm/dia)
    PROFUNDIDADE_M     = 0.35    # m — profundidade do canteiro experimental
    EFICIENCIA_CHUVA   = 0.70    # 70% — restante é runoff/evaporação
    RUIDO_STD          = 0.002   # m³/m³ — ±2% — realista para sensor capacitivo

    def __init__(self, theta_inicial=None, seed=42):
        np.random.seed(seed)
        self.theta    = theta_inicial if theta_inicial is not None else THETA_CC
        self.historico = []

    def ler_theta(self, irrigou_mm=0.0, chuva_mm=0.0, dap=60,
                  adicionar_ruido=True):
        """
        Simula leitura do sensor para o momento actual.

        Aplica (por esta ordem):
          1. Secagem por ETc (aumenta com DAP — planta maior)
          2. Recarga por irrigação (passada como parâmetro do dia anterior)
          3. Recarga por chuva do dia anterior (com delay de 1 dia — realista)
          4. Ruído de sensor

        Args:
            irrigou_mm      : mm aplicados ONTEM (efeito no sensor hoje)
            chuva_mm        : mm de chuva ONTEM (efeito no sensor hoje)
                              NOTA: chuva de hoje é gerida pelo SimuladorChuva
                              e só afecta o sensor amanhã.
            dap             : dias após plantio (afecta ETc)
            adicionar_ruido : False para testes determinísticos puros

        Returns:
            float — θ em m³/m³ (o que o sensor físico mediria)
        """
        # 1. Secagem por ETc
        # ETc aumenta com DAP até ao pico (fase produtiva ~DAP 60-80)
        # Normalizado para que DAP 60 = taxa base completa
        fator_dap = min(1.0, dap / 60.0)
        secagem   = self.TAXA_SECAGEM_BASE * fator_dap
        self.theta -= secagem

        # 2. Recarga por irrigação do dia anterior
        if irrigou_mm > 0.0:
            recarga_irr = irrigou_mm / (1000.0 * self.PROFUNDIDADE_M)
            self.theta += recarga_irr

        # 3. Recarga por chuva do dia anterior (70% eficiência)
        if chuva_mm > 0.0:
            recarga_chuva = (chuva_mm * self.EFICIENCIA_CHUVA) / (1000.0 * self.PROFUNDIDADE_M)
            self.theta   += recarga_chuva

        # Limites físicos — solo não pode exceder saturação nem secar abaixo do PM
        # (PM*1.05 como margem de segurança — sensor capacitivo perde linearidade próximo do PM)
        self.theta = float(np.clip(self.theta, THETA_PM * 1.05, THETA_SAT))

        # 4. Ruído de sensor
        if adicionar_ruido:
            ruido      = np.random.normal(0.0, self.RUIDO_STD)
            theta_lido = float(np.clip(self.theta + ruido, THETA_PM, THETA_SAT))
        else:
            theta_lido = self.theta

        self.historico.append({
            'theta_real'  : round(self.theta, 5),
            'theta_lido'  : round(theta_lido, 5),
            'irrigou_mm'  : irrigou_mm,
            'chuva_mm'    : chuva_mm,
            'dap'         : dap,
        })

        return float(theta_lido)

    def ler_tensao_kpa(self, irrigou_mm=0.0, chuva_mm=0.0, dap=60,
                       adicionar_ruido=True):
        """Atalho: lê θ e converte directamente para tensão em kPa."""
        theta = self.ler_theta(irrigou_mm, chuva_mm, dap, adicionar_ruido)
        return theta, umidade_para_tensao_kpa(theta)

    def estado_actual(self):
        """Retorna estado actual do solo (para debug/log)."""
        return {
            'theta'   : round(self.theta, 4),
            'tensao'  : round(umidade_para_tensao_kpa(self.theta), 1),
        }

    def reset(self, theta_inicial):
        """Reinicia o simulador (para novo cenário)."""
        self.theta     = float(theta_inicial)
        self.historico = []


# ------------------------------------------------------------------
# SIMULADOR DO SENSOR DE CHUVA
# ------------------------------------------------------------------

class SimuladorChuva:
    """
    Simula o sensor físico de chuva (ex: pluviômetro de báscula ou reed switch).

    Na arquitectura do sistema, o sensor de chuva é consultado às 18h
    para decidir se cancela a irrigação planeada de manhã.

    Papel no sistema:
      - NÃO substitui chuva_acum_3d_mm da API (que usa dados históricos reais)
      - Detecta se choveu HOJE (entre a decisão da manhã e a acção das 18h)
      - Actua como GATE de segurança: cancela irrigação SE choveu E solo ok

    Em HIL, o cenário define precipitacao_diaria[]. O simulador usa esse
    valor como "chuva de hoje" com uma componente aleatória menor para
    simular variabilidade de leitura do sensor.
    """

    # Limiar mínimo de precipitação que o sensor detecta (mm)
    # Abaixo disto o sensor reporta "sem chuva" (garoa não conta)
    LIMIAR_DETECCAO_MM = 1.0

    def __init__(self, seed=42):
        self._rng          = np.random.RandomState(seed + 100)
        self.chuva_detectada_hoje = 0.0   # mm detectados no dia actual
        self._historico    = []

    def simular_leitura_18h(self, chuva_real_mm):
        """
        Simula o que o sensor de chuva reporta às 18h.

        Args:
            chuva_real_mm : precipitação real do dia (do cenário)

        Returns:
            (detectou: bool, mm_estimados: float)
        """
        if chuva_real_mm <= 0.0:
            # Sem chuva — sensor reporta 0 (com possibilidade de falso positivo <1%)
            falso_positivo = self._rng.random() < 0.01
            mm = self._rng.uniform(0.5, 1.5) if falso_positivo else 0.0
        else:
            # Com chuva — sensor tem ±15% de erro de medição (mecânico)
            erro = self._rng.uniform(0.85, 1.15)
            mm   = chuva_real_mm * erro
            # Pequena prob de falha de leitura (báscula encravada)
            if self._rng.random() < 0.03:
                mm = 0.0  # falha de sensor — reporta sem chuva

        detectou = mm >= self.LIMIAR_DETECCAO_MM
        self.chuva_detectada_hoje = mm

        self._historico.append({
            'chuva_real': chuva_real_mm,
            'chuva_lida': round(mm, 1),
            'detectou'  : detectou,
        })

        return detectou, round(mm, 1)

    def reset_diario(self):
        """Chamar no início de cada dia para reiniciar o contador diário."""
        self.chuva_detectada_hoje = 0.0


# ------------------------------------------------------------------
# LÓGICA DE DECISÃO ÀS 18h — DUPLA CONFIRMAÇÃO
# ------------------------------------------------------------------

def decidir_accao_18h(classe_manha, sensor_chuva, sensor_solo,
                      chuva_real_mm, irrigou_ontem_mm, chuva_ontem_mm, dap):
    """
    Implementa a lógica de dupla confirmação às 18h.

    Regra:
      cancelar = choveu hoje E tensão actual ≤ TENSAO_OPTIMA_MAX (40 kPa)
      manter   = não choveu  OU tensão ainda alta (stress persistente)

    Args:
        classe_manha    : int — decisão tomada de manhã (0, 1, 2)
        sensor_chuva    : SimuladorChuva
        sensor_solo     : SimuladorSensor (para leitura das 18h)
        chuva_real_mm   : float — chuva real do dia (do cenário HIL)
        irrigou_ontem_mm: float — irrigação de ontem (para actualizar solo)
        chuva_ontem_mm  : float — chuva de ontem (para actualizar solo)
        dap             : int — dias após plantio

    Returns:
        dict com:
          'classe_final'  : int — classe executada
          'irrigou_mm'    : float — volume real aplicado
          'motivo'        : str — razão da decisão
          'choveu'        : bool
          'mm_chuva'      : float — mm detectados pelo sensor
          'tensao_18h'    : float — tensão lida às 18h
    """
    # 1. Consultar sensor de chuva (o que detectou durante o dia)
    choveu, mm_chuva = sensor_chuva.simular_leitura_18h(chuva_real_mm)

    # 2. Leitura de solo às 18h
    # O solo às 18h reflecte: ontem + secagem do dia + (sem irrigação ainda)
    # NOTA: a chuva de HOJE só entra na leitura de amanhã (delay físico real)
    # Por isso passamos irrigou_ontem e chuva_ontem, não os de hoje
    theta_18h  = sensor_solo.ler_theta(
        irrigou_mm=irrigou_ontem_mm,
        chuva_mm=chuva_ontem_mm,
        dap=dap,
        adicionar_ruido=True
    )
    tensao_18h = umidade_para_tensao_kpa(theta_18h)

    # 3. Lógica de dupla confirmação
    volumes_mm = {0: 0.0, 1: 3.0, 2: 7.0}

    if choveu and tensao_18h <= TENSAO_OPTIMA_MAX:
        # Choveu E solo recuperou → cancelar irrigação
        classe_final = 0
        irrigou_mm   = 0.0
        motivo       = "cancelado_chuva_solo_ok"

    elif choveu and tensao_18h > TENSAO_OPTIMA_MAX:
        # Choveu MAS solo ainda em stress → manter decisão da manhã
        classe_final = classe_manha
        irrigou_mm   = volumes_mm[classe_manha]
        motivo       = "mantido_stress_persistente"

    else:
        # Não choveu → executar normalmente
        classe_final = classe_manha
        irrigou_mm   = volumes_mm[classe_manha]
        motivo       = "executado_normal"

    return {
        'classe_final' : classe_final,
        'irrigou_mm'   : irrigou_mm,
        'motivo'       : motivo,
        'choveu'       : choveu,
        'mm_chuva'     : mm_chuva,
        'tensao_18h'   : round(tensao_18h, 1),
        'theta_18h'    : round(theta_18h, 4),
    }
