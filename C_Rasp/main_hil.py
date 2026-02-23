# main_hil.py — Loop principal do Protocolo de Validação Hardware-in-the-Loop
#
# Executa os 4 cenários de validação sequencialmente.
# Decisão simulada às 06h00, acção às 18h com dupla confirmação
# (sensor de chuva + tensão do solo).
#
# Uso:
#   python3 main_hil.py                  — todos os cenários
#   python3 main_hil.py --cenario 3      — apenas o cenário 3
#   python3 main_hil.py --sem-api        — sem chamar Open-Meteo (offline)
#
# Outputs:
#   resultados_hil/cenario_N_nome.csv    — dados diários por cenário
#   resultados_hil/relatorio_hil.md      — relatório gerado automaticamente

import os
import sys
import csv
import time
import json
import pickle
import argparse
import numpy as np
from datetime import datetime, timedelta

from config_hil import (
    PKL_INICIAL, PKL_CAMPO, RESULTADOS_DIR,
    TENSAO_OPTIMA_MAX, TENSAO_STRESS_MOD, TENSAO_STRESS_SEV,
    TENSAO_RANGE_MAX, TENSAO_ENCHARCADO,
    VOLUMES_MM, NORM_N_INICIAL,
    FEEDBACK_JANELA_DIAS, FEEDBACK_MIN_OCORRENCIAS,
    LATITUDE, LONGITUDE
)
from almmo0 import ALMMo0, carregar_modelo
from simulador_sensor import (
    SimuladorSensor, SimuladorChuva,
    umidade_para_tensao_kpa, decidir_accao_18h
)

# Importações opcionais (não disponíveis em todos os ambientes)
try:
    import matplotlib
    matplotlib.use('Agg')  # sem display — gera ficheiros PNG
    import matplotlib.pyplot as plt
    MATPLOTLIB_OK = True
except ImportError:
    MATPLOTLIB_OK = False
    print("[INFO] matplotlib não disponível — gráficos não serão gerados")


# ==============================================================================
# NORMALIZADOR ONLINE (Welford)
# ==============================================================================

class NormalizadorOnline:
    """
    Actualiza média e desvio padrão incrementalmente (algoritmo de Welford).
    Sem histórico em memória — compatível com edge computing.

    n_inicial=50: dados simulados têm peso moderado nas primeiras semanas.
    Parâmetro sensível documentado no TCC:
      - n alto (100+): estabilidade inicial, adaptação lenta
      - n baixo (10-20): adaptação rápida, risco de instabilidade
      - n=50: compromisso para campo sem dados históricos locais
    """

    def __init__(self, media_inicial, std_inicial, n_inicial=NORM_N_INICIAL):
        self.n     = n_inicial
        self.media = np.asarray(media_inicial, dtype=float).copy()
        self.M2    = (np.asarray(std_inicial, dtype=float) ** 2) * self.n

    def actualizar(self, x):
        self.n   += 1
        x         = np.asarray(x, dtype=float)
        delta     = x - self.media
        self.media += delta / self.n
        self.M2    += delta * (x - self.media)

    @property
    def std(self):
        if self.n < 2:
            return np.ones_like(self.media)
        s = np.sqrt(self.M2 / (self.n - 1))
        s[s < 1e-8] = 1.0
        return s


# ==============================================================================
# FEEDBACK DE STRESS HÍDRICO
# ==============================================================================

class FeedbackStressHidrico:
    """
    Mecanismo de aprendizado online sem label verdadeiro.
    Infere a classe correcta pela evolução da tensão do solo
    nos dias seguintes à decisão.

    Melhorias sobre o briefing original:
      - Ponderação 40/60 (último dia tem mais peso)
      - min_ocorrencias=2 (anti-degeneração)
      - Agregação de erros por classe_real (não por par pred→real)
    """

    def __init__(self, janela_dias=FEEDBACK_JANELA_DIAS,
                 min_ocorrencias=FEEDBACK_MIN_OCORRENCIAS):
        self.janela_dias      = janela_dias
        self.min_ocorrencias  = min_ocorrencias
        self.buffer           = []
        # Chave: classe_real → contagem de erros pendentes
        self.historico_erros  = {}

    def registar_decisao(self, x, classe, tensao):
        """Chamar no momento da decisão (06h00)."""
        self.buffer.append({
            'input'      : np.asarray(x, dtype=float).copy(),
            'classe'     : int(classe),
            'tensao_pre' : float(tensao),
            'tensao_apos': [],
        })

    def registar_tensao_diaria(self, tensao):
        """Chamar uma vez por dia com a tensão observada."""
        for entrada in self.buffer:
            if len(entrada['tensao_apos']) < self.janela_dias:
                entrada['tensao_apos'].append(float(tensao))

    def avaliar_e_retreinar(self, modelo):
        """
        Avalia decisões com janela completa. Retreina se erro consistente.
        Retorna lista de ajustes feitos.
        """
        ajustes = []
        prontas = [e for e in self.buffer
                   if len(e['tensao_apos']) >= self.janela_dias]

        for entrada in prontas:
            # Ponderação 40/60 — último dia tem mais peso (mais informativo)
            if len(entrada['tensao_apos']) == 2:
                tensao_pos = (0.4 * entrada['tensao_apos'][0]
                              + 0.6 * entrada['tensao_apos'][1])
            else:
                tensao_pos = entrada['tensao_apos'][-1]

            classe_real = self._inferir_classe(
                entrada['tensao_pre'], tensao_pos, entrada['classe']
            )

            if classe_real != entrada['classe']:
                # Anti-degeneração: agregar por classe_real
                chave = classe_real
                self.historico_erros[chave] = \
                    self.historico_erros.get(chave, 0) + 1

                if self.historico_erros[chave] >= self.min_ocorrencias:
                    modelo.learn(entrada['input'], classe_real)
                    self.historico_erros[chave] = 0  # reset após aprender
                    ajustes.append({
                        'classe_pred': entrada['classe'],
                        'classe_real': classe_real,
                        'tensao_pre' : round(entrada['tensao_pre'], 1),
                        'tensao_pos' : round(tensao_pos, 1),
                    })

        # Remover entradas processadas
        self.buffer = [e for e in self.buffer
                       if len(e['tensao_apos']) < self.janela_dias]
        return ajustes

    def _inferir_classe(self, tensao_pre, tensao_pos, classe_pred):
        """
        Infere classe correcta pela evolução da tensão.

        Dead zone documentada: tensão 40–90 kPa sem tendência clara
        → retorna classe_pred sem alteração (comportamento esperado).
        Esta limitação é documentada no relatório HIL (Secção 5).
        """
        delta = tensao_pos - tensao_pre

        # Solo encharcado após irrigação → irrigou demais
        if tensao_pos < TENSAO_ENCHARCADO and classe_pred > 0:
            return max(0, classe_pred - 1)

        # Stress severo e piorando → irrigou pouco
        if tensao_pos > TENSAO_STRESS_SEV and delta > 5.0:
            return min(2, classe_pred + 1)

        # Acima do range de treino → irrigação máxima
        if tensao_pos > TENSAO_RANGE_MAX:
            return 2

        # Dead zone (40–90 kPa) — não alterar
        # Erros nesta faixa serão corrigidos por dados de campo reais
        return classe_pred


# ==============================================================================
# DEFINIÇÃO DOS 4 CENÁRIOS
# ==============================================================================

CENARIOS = {
    1: {
        'nome'               : 'Seca Progressiva',
        'ficheiro'           : 'cenario_1_seca.csv',
        'descricao'          : 'Solo começa na capacidade de campo e seca sem chuva.',
        'dias'               : 14,
        'theta_inicial'      : 0.180,
        'precipitacao_diaria': [0.0] * 14,
        'tmax_diaria'        : [36.5, 37.0, 37.5, 37.8, 38.0,
                                 38.2, 38.0, 37.5, 38.5, 38.8,
                                 39.0, 38.5, 38.0, 37.5],
        'dap_inicial'        : 60,
        'comportamento_esperado': (
            'Dias 1-3: C0 (solo ainda húmido). '
            'Dias 4-8: C1 (stress moderado). '
            'Dias 9-14: C2 (stress severo). '
            'Escalamento monotónico esperado.'
        ),
    },
    2: {
        'nome'               : 'Evento de Chuva',
        'ficheiro'           : 'cenario_2_chuva.csv',
        'descricao'          : 'Solo seco recebe chuva intensa no dia 4.',
        'dias'               : 10,
        'theta_inicial'      : 0.120,
        'precipitacao_diaria': [0.0, 0.0, 0.0, 35.0, 12.0,
                                 5.0, 0.0, 0.0, 0.0, 0.0],
        'tmax_diaria'        : [37.0] * 10,
        'dap_inicial'        : 45,
        'comportamento_esperado': (
            'Dias 1-3: C1 ou C2 (solo seco). '
            'Dias 4-7: C0 (chuva acumulada alta). '
            'Dias 8-10: retorno gradual a C1.'
        ),
    },
    3: {
        'nome'               : 'Cegueira C1 e Dead Zone do Feedback',
        'ficheiro'           : 'cenario_3_feedback.csv',
        'descricao'          : (
            'Demonstra o comportamento real do sistema: o modelo tem cegueira '
            'para C1 (precision=12.5%) e oscila entre C0 e C2 na zona 40-90 kPa. '
            'O feedback nao actua nesta zona (dead zone por design). '
            'Este e um achado de investigacao para o TCC.'
        ),
        'dias'               : 21,
        # theta=0.175 (~43 kPa): zona de conforto hídrico, C0 dominante.
        # Sem chuva, seca ~0.008/dia → tensão sobe ~5-10 kPa/dia.
        # O modelo oscila C0↔C2 ignorando C1 (cegueira documentada).
        # O feedback nao dispara porque:
        #   - Dead zone: tensao_pos 40-90 kPa, nunca >90 por 2 dias consecutivos
        #   - Quando tensao sobe >80 kPa, modelo CORRIGE sozinho com C2
        #   - Correcao via feedback so actua em falhas persistentes (>90 kPa)
        # ACHADO para o TCC: o threshold do feedback precisa de calibracao
        # com dados reais de campo para cobrir o stress moderado (40-90 kPa).
        'theta_inicial'      : 0.175,  # ~43 kPa — C0 confirmado nesta tensao
        'precipitacao_diaria': [0.0] * 21,
        'tmax_diaria'        : [37.0] * 21,
        'dap_inicial'        : 55,
        'comportamento_esperado': (
            'Dias 1-3: C0 (tensao ~43-72 kPa — dead zone, feedback inactivo). '
            'Dias 4+: oscilacao C0-C2 (cegueira C1 — modelo nunca usa irrigacao moderada). '
            'C1 ausente ou raro em todos os 21 dias. '
            'Feedback: 0 eventos esperados — dead zone cobre a zona do vies C1. '
            'ACHADO TCC: feedback calibrado para stress critico (>90 kPa), '
            'nao para optimizacao de stress moderado. '
            'Solucao: calibrar threshold com dados de campo reais.'
        ),
    },
    4: {
        'nome'               : 'Stress Crítico e Recuperação',
        'ficheiro'           : 'cenario_4_stress.csv',
        'descricao'          : 'Solo próximo do PM. Irrigação intensa recupera.',
        'dias'               : 10,
        'theta_inicial'      : 0.095,   # próximo do PM — stress severo
        'precipitacao_diaria': [0.0] * 10,
        'tmax_diaria'        : [38.0] * 10,
        'dap_inicial'        : 80,
        'comportamento_esperado': (
            'Dias 1-3: C2 (tensão >90 kPa, acima do range de treino). '
            'tensao_acima_range=True no log. '
            'Após irrigação intensa: tensão cai, predições voltam a C1/C0.'
        ),
    },
}


# ==============================================================================
# METEOROLOGIA (Open-Meteo real ou mock offline)
# ==============================================================================

def obter_dados_meteorologicos(lat=LATITUDE, lon=LONGITUDE,
                               cache_path='cache_meteo.json',
                               usar_api=True):
    """
    Obtém dados reais de Imperatriz-MA via Open-Meteo (gratuito, sem chave).
    Fallback para cache se sem rede. Fallback final se cache corrompida.
    """
    if usar_api:
        try:
            import requests
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": lat, "longitude": lon,
                "daily": ["precipitation_sum", "temperature_2m_max"],
                "timezone": "America/Fortaleza",
                "past_days": 3, "forecast_days": 1,
            }
            t0   = time.time()
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            latencia_ms = int((time.time() - t0) * 1000)
            data        = resp.json()["daily"]
            chuva_3d    = float(sum(data["precipitation_sum"][-4:-1]))
            tmax_3d     = float(max(data["temperature_2m_max"][-4:-1]))
            resultado   = {
                "chuva_acum_3d_mm": chuva_3d,
                "tmax_max_3d_c"   : tmax_3d,
                "fonte"           : "api",
                "latencia_ms"     : latencia_ms,
                "timestamp"       : datetime.now().isoformat(),
            }
            with open(cache_path, 'w') as f:
                json.dump(resultado, f, indent=2)
            return resultado
        except Exception as e:
            print(f"  [METEO] Falha API ({e}). Usando cache.")

    # Tentar cache
    try:
        with open(cache_path) as f:
            dados           = json.load(f)
            dados["fonte"]  = "cache"
            dados["latencia_ms"] = 0
            return dados
    except Exception:
        pass

    # Fallback final
    return {
        "chuva_acum_3d_mm": 0.0,
        "tmax_max_3d_c"   : 35.0,
        "fonte"           : "fallback",
        "latencia_ms"     : 0,
        "timestamp"       : datetime.now().isoformat(),
    }


# ==============================================================================
# EXECUÇÃO DE UM CENÁRIO HIL
# ==============================================================================

def executar_cenario(cenario, modelo, feedback, normalizador,
                     dados_meteo, sensor_chuva, usar_api=True):
    """
    Executa um cenário HIL completo.

    Arquitectura temporal simulada:
      06h00 — leitura sensor + inferência (decisão da manhã)
      18h00 — dupla confirmação (sensor chuva + tensão actual)
      19h00 — acção (bomba ligada X segundos — simulada)

    Returns:
        list[dict] — resultados linha a linha (um por dia)
    """
    sensor          = SimuladorSensor(theta_inicial=cenario['theta_inicial'])
    resultados      = []
    irrigou_ontem   = 0.0
    chuva_ontem     = 0.0
    sensor_chuva.reset_diario()

    print(f"\n{'='*60}")
    print(f"  CENÁRIO {cenario.get('id', '?')} — {cenario['nome']}")
    print(f"  {cenario['descricao']}")
    print(f"{'='*60}")
    print(f"  {'Dia':>3} | {'kPa':>6} | {'Cls':>3} | {'Irr':>5} | "
          f"{'Motivo18h':<28} | {'Regras':>6} | Flags")
    print(f"  {'-'*80}")

    for dia in range(cenario['dias']):
        dap       = cenario['dap_inicial'] + dia
        chuva_dia = cenario['precipitacao_diaria'][dia]
        tmax_dia  = cenario['tmax_diaria'][dia]

        # ------ 06h00: LEITURA SENSOR ------
        # Nota: o sensor lê o estado do solo que reflecte
        # irrigação e chuva do dia ANTERIOR (delay físico real)
        theta_6h  = sensor.ler_theta(
            irrigou_mm=irrigou_ontem,
            chuva_mm=chuva_ontem,
            dap=dap,
            adicionar_ruido=True
        )
        tensao_6h = umidade_para_tensao_kpa(theta_6h)

        # ------ 06h00: FEATURE VECTOR ------
        # chuva_3d: usa dados da API (histórico real dos 3 dias anteriores)
        # No HIL, aproximamos com a janela do cenário
        chuva_3d = sum(cenario['precipitacao_diaria'][max(0, dia - 2):dia])
        tmax_3d  = max(cenario['tmax_diaria'][max(0, dia - 2):dia + 1])

        x = np.array([tensao_6h, chuva_3d, tmax_3d, float(dap)])

        # ------ 06h00: NORMALIZAR + INFERÊNCIA ------
        normalizador.actualizar(x)
        modelo.input_mean = normalizador.media.copy()
        modelo.input_std  = normalizador.std.copy()

        classe_manha, confianca = modelo.predict_com_confianca(x)

        # ------ 06h00: REGISTAR DECISÃO NO FEEDBACK ------
        feedback.registar_decisao(x, classe_manha, tensao_6h)
        feedback.registar_tensao_diaria(tensao_6h)

        # ------ 18h00: DUPLA CONFIRMAÇÃO ------
        # Usa um segundo SimuladorSensor internamente para a leitura das 18h
        # (o sensor principal continua com o estado do início do dia)
        sensor_18h = SimuladorSensor(theta_inicial=theta_6h)
        decisao_18h = decidir_accao_18h(
            classe_manha  = classe_manha,
            sensor_chuva  = sensor_chuva,
            sensor_solo   = sensor_18h,
            chuva_real_mm = chuva_dia,
            irrigou_ontem_mm = irrigou_ontem,
            chuva_ontem_mm   = chuva_ontem,
            dap           = dap
        )
        sensor_chuva.reset_diario()

        classe_final = decisao_18h['classe_final']
        irrigou_mm   = decisao_18h['irrigou_mm']
        motivo_18h   = decisao_18h['motivo']

        # ------ FEEDBACK: AVALIAR DECISÕES ANTERIORES ------
        ajustes = feedback.avaliar_e_retreinar(modelo)

        # ------ ESTADO DAS REGRAS ------
        dist_regras = modelo.distribuicao_regras()
        acima_range = tensao_6h > TENSAO_RANGE_MAX

        # ------ REGISTAR RESULTADO ------
        resultado = {
            'dia'                : dia + 1,
            'dap'                : dap,
            'theta_6h'           : round(theta_6h, 4),
            'tensao_6h_kpa'      : round(tensao_6h, 1),
            'theta_18h'          : round(decisao_18h['theta_18h'], 4),
            'tensao_18h_kpa'     : decisao_18h['tensao_18h'],
            'chuva_3d_mm'        : round(chuva_3d, 1),
            'tmax_3d_c'          : round(tmax_3d, 1),
            'chuva_dia_mm'       : chuva_dia,
            'choveu_sensor'      : decisao_18h['choveu'],
            'mm_chuva_sensor'    : decisao_18h['mm_chuva'],
            'classe_manha'       : classe_manha,
            'confianca_manha'    : round(confianca, 3),
            'classe_final'       : classe_final,
            'irrigou_mm'         : irrigou_mm,
            'motivo_18h'         : motivo_18h,
            'n_regras'           : len(modelo.rules),
            'regras_c0'          : dist_regras[0],
            'regras_c1'          : dist_regras[1],
            'regras_c2'          : dist_regras[2],
            'houve_feedback'     : len(ajustes) > 0,
            'n_ajustes'          : len(ajustes),
            'tensao_acima_range' : acima_range,
            'fonte_meteo'        : dados_meteo.get('fonte', 'n/a'),
        }
        resultados.append(resultado)

        # Preparar para próximo dia
        irrigou_ontem = irrigou_mm
        chuva_ontem   = chuva_dia

        # Print em tempo real
        flags = ""
        if acima_range:
            flags += "⚠️ "
        if len(ajustes) > 0:
            flags += "📚"
        if decisao_18h['choveu']:
            flags += "🌧️"

        print(f"  {dia+1:>3d} | {tensao_6h:>6.1f} | "
              f"C{classe_manha}→C{classe_final} | "
              f"{irrigou_mm:>4.1f}mm | "
              f"{motivo_18h:<28} | "
              f"{len(modelo.rules):>3}r C0:{dist_regras[0]} C1:{dist_regras[1]} C2:{dist_regras[2]} | "
              f"{flags}")

    return resultados


# ==============================================================================
# GUARDAR RESULTADOS EM CSV
# ==============================================================================

def salvar_csv(resultados, caminho):
    """Guarda lista de resultados em CSV."""
    if not resultados:
        return
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    with open(caminho, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=resultados[0].keys())
        writer.writeheader()
        writer.writerows(resultados)
    print(f"  → CSV guardado: {caminho}")


# ==============================================================================
# GRÁFICOS (se matplotlib disponível)
# ==============================================================================

def gerar_graficos(todos_resultados, output_dir):
    """Gera gráficos PNG para cada cenário e o gráfico chave do Cenário 3."""
    if not MATPLOTLIB_OK:
        return

    graficos_dir = os.path.join(output_dir, 'graficos_hil')
    os.makedirs(graficos_dir, exist_ok=True)

    cores_classe = {0: '#2196F3', 1: '#FF9800', 2: '#F44336'}

    for id_cenario, resultados in todos_resultados.items():
        if not resultados:
            continue

        cenario = CENARIOS[id_cenario]
        dias     = [r['dia'] for r in resultados]
        tensoes  = [r['tensao_6h_kpa'] for r in resultados]
        classes  = [r['classe_final'] for r in resultados]
        irr      = [r['irrigou_mm'] for r in resultados]

        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        fig.suptitle(f"Cenário {id_cenario} — {cenario['nome']}", fontsize=13)

        # Subplot 1: Tensão do solo
        ax1 = axes[0]
        ax1.plot(dias, tensoes, 'b-o', markersize=4, linewidth=1.5, label='Tensão 06h')
        ax1.axhline(TENSAO_OPTIMA_MAX, color='green', linestyle='--',
                    alpha=0.7, label=f'Ótimo ({TENSAO_OPTIMA_MAX} kPa)')
        ax1.axhline(TENSAO_STRESS_MOD, color='orange', linestyle='--',
                    alpha=0.7, label=f'Stress mod ({TENSAO_STRESS_MOD} kPa)')
        ax1.axhline(TENSAO_STRESS_SEV, color='red', linestyle='--',
                    alpha=0.7, label=f'Stress sev ({TENSAO_STRESS_SEV} kPa)')
        ax1.axhline(TENSAO_RANGE_MAX, color='darkred', linestyle=':',
                    alpha=0.5, label=f'Range max ({TENSAO_RANGE_MAX} kPa)')
        ax1.set_ylabel('Tensão do solo (kPa)')
        ax1.legend(fontsize=7, loc='upper right')
        ax1.grid(True, alpha=0.3)

        # Subplot 2: Classe predita + irrigação
        ax2 = axes[1]
        cores_barras = [cores_classe[c] for c in classes]
        ax2.bar(dias, classes, color=cores_barras, alpha=0.8, label='Classe final')
        ax2_twin = ax2.twinx()
        ax2_twin.plot(dias, irr, 'k--', alpha=0.5, linewidth=1, label='Irrigação (mm)')
        ax2_twin.set_ylabel('Irrigação (mm)', fontsize=9)
        ax2.set_ylabel('Classe (0=Nenhuma, 1=Mod, 2=Int)')
        ax2.set_ylim(-0.2, 2.5)
        ax2.set_yticks([0, 1, 2])
        ax2.grid(True, alpha=0.3)

        # Subplot 3: Distribuição de regras (mais relevante no C3)
        ax3 = axes[2]
        r0 = [r['regras_c0'] for r in resultados]
        r1 = [r['regras_c1'] for r in resultados]
        r2 = [r['regras_c2'] for r in resultados]
        ax3.stackplot(dias, r0, r1, r2,
                      labels=['Regras C0', 'Regras C1', 'Regras C2'],
                      colors=['#2196F3', '#FF9800', '#F44336'], alpha=0.7)
        ax3.set_ylabel('Nº de regras')
        ax3.set_xlabel('Dia do cenário')
        ax3.legend(fontsize=8, loc='upper right')
        ax3.grid(True, alpha=0.3)

        # Marcar eventos de feedback
        fb_dias = [r['dia'] for r in resultados if r['houve_feedback']]
        for fd in fb_dias:
            for ax in axes:
                ax.axvline(fd, color='purple', alpha=0.3, linewidth=1)

        plt.tight_layout()
        nome_ficheiro = os.path.join(
            graficos_dir, f"cenario_{id_cenario}_{cenario['nome'].replace(' ', '_').lower()}.png"
        )
        plt.savefig(nome_ficheiro, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"  → Gráfico: {nome_ficheiro}")


# ==============================================================================
# RELATÓRIO HIL AUTOMÁTICO
# ==============================================================================

def gerar_relatorio(todos_resultados, dados_meteo, tempos, modelo_source,
                    output_dir):
    """Gera relatorio_hil.md com todas as secções especificadas no briefing."""

    caminho = os.path.join(output_dir, 'relatorio_hil.md')
    linhas  = []
    W       = lambda s: linhas.append(s)

    W("# Relatório de Validação Hardware-in-the-Loop (HIL)")
    W(f"**Data de execução:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    W(f"**Modelo carregado de:** `{modelo_source}`")
    W("")

    # ------ Secção 1: Validação de Hardware ------
    W("## Secção 1 — Validação de Hardware")
    W("")
    for id_c, (t_load, t_inf) in tempos.items():
        cenario = CENARIOS[id_c]
        W(f"**Cenário {id_c} ({cenario['nome']}):**")
        W(f"- Tempo carregamento pkl: `{t_load:.1f} ms`")
        W(f"- Tempo médio de inferência: `{t_inf:.2f} ms/ciclo`")
    W(f"- Fonte do modelo: `{modelo_source}`")
    W("")

    # ------ Secção 2: Validação de Integração ------
    W("## Secção 2 — Validação de Integração")
    W("")
    W(f"- API Open-Meteo: `{dados_meteo.get('fonte', 'n/a')}`")
    if dados_meteo.get('fonte') == 'api':
        W(f"- Latência: `{dados_meteo.get('latencia_ms', '?')} ms` ✅")
    W(f"- Chuva acum. 3d (Imperatriz-MA): `{dados_meteo.get('chuva_acum_3d_mm', '?')} mm`")
    W(f"- Tmax 3d: `{dados_meteo.get('tmax_max_3d_c', '?')} °C`")
    W(f"- Cache JSON: `cache_meteo.json` — {'✅' if os.path.exists('cache_meteo.json') else '❌'}")
    W("")

    # ------ Secção 3: Resultados por Cenário ------
    W("## Secção 3 — Resultados por Cenário")
    W("")

    for id_cenario, resultados in todos_resultados.items():
        cenario = CENARIOS[id_cenario]
        W(f"### Cenário {id_cenario} — {cenario['nome']}")
        W(f"*{cenario['descricao']}*")
        W("")
        W(f"**Comportamento esperado:** {cenario['comportamento_esperado']}")
        W("")

        if resultados:
            W("| Dia | kPa | θ | C manhã | C final | Irr mm | Motivo 18h | Regras | Flags |")
            W("|-----|-----|---|---------|---------|--------|------------|--------|-------|")
            for r in resultados:
                flags = ""
                if r['tensao_acima_range']:
                    flags += "⚠️range "
                if r['houve_feedback']:
                    flags += "📚fb "
                if r['choveu_sensor']:
                    flags += "🌧️"
                W(f"| {r['dia']} | {r['tensao_6h_kpa']} | {r['theta_6h']} "
                  f"| C{r['classe_manha']} | C{r['classe_final']} "
                  f"| {r['irrigou_mm']} | {r['motivo_18h']} "
                  f"| {r['n_regras']} ({r['regras_c0']}/{r['regras_c1']}/{r['regras_c2']}) "
                  f"| {flags} |")
        W("")

    # ------ Secção 4: Análise do Aprendizado Online (Cenário 3) ------
    W("## Secção 4 — Análise do Aprendizado Online (Cenário 3)")
    W("")

    r3 = todos_resultados.get(3, [])
    if r3:
        semanas = [r3[0:7], r3[7:14], r3[14:21]]
        W("### % de dias C1+C2 por semana")
        W("")
        W("| Semana | Dias | C0 | C1+C2 | % C1+C2 |")
        W("|--------|------|----|-------|---------|")

        pcts = []
        for i, semana in enumerate(semanas):
            if not semana:
                continue
            c0    = sum(1 for r in semana if r['classe_final'] == 0)
            c12   = sum(1 for r in semana if r['classe_final'] > 0)
            pct   = c12 / len(semana) * 100
            pcts.append(pct)
            W(f"| Semana {i+1} | {len(semana)} | {c0} | {c12} | {pct:.0f}% |")

        W("")
        if len(pcts) >= 2:
            aprendeu = pcts[-1] > pcts[0]
            W(f"**MÉTRICA CHAVE — % C1+C2 aumentou da semana 1 para semana 3?** "
              f"{'✅ PASS' if aprendeu else '❌ FAIL'} "
              f"({pcts[0]:.0f}% → {pcts[-1]:.0f}%)")

        n_fb = sum(1 for r in r3 if r['houve_feedback'])
        W(f"\n**Total de eventos de feedback:** {n_fb} em {len(r3)} dias")
        W("")

    # ------ Secção 5: Dead Zone ------
    W("## Secção 5 — Dead Zone Documentada")
    W("")
    W("A dead zone corresponde a tensões entre 40–90 kPa onde o mecanismo de "
      "feedback não actua (tensão dentro de limites aceitáveis, sem tendência clara).")
    W("Esta é uma limitação conhecida e intencional — erros nesta faixa serão "
      "corrigidos por dados de campo reais com o sensor físico calibrado.")
    W("")

    for id_cenario, resultados in todos_resultados.items():
        dead = [r for r in resultados
                if 40.0 < r['tensao_6h_kpa'] <= 90.0 and not r['houve_feedback']]
        W(f"**Cenário {id_cenario}:** {len(dead)} dias na dead zone sem feedback")

    W("")
    W("---")
    W("*Relatório gerado automaticamente por `main_hil.py`.*")
    W(f"*Sistema: ALMMo-0 + Saxton & Rawls + Open-Meteo + Dupla Confirmação 18h*")

    with open(caminho, 'w', encoding='utf-8') as f:
        f.write('\n'.join(linhas))
    print(f"\n  → Relatório HIL: {caminho}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Validação HIL — Sistema ALMMo-0')
    parser.add_argument('--cenario', type=int, choices=[1, 2, 3, 4],
                        help='Executar apenas este cenário (omitir = todos)')
    parser.add_argument('--sem-api', action='store_true',
                        help='Não chamar Open-Meteo (usar cache ou fallback)')
    args = parser.parse_args()

    usar_api = not args.sem_api
    os.makedirs(RESULTADOS_DIR, exist_ok=True)

    print("\n" + "="*60)
    print("  PROTOCOLO DE VALIDAÇÃO HARDWARE-IN-THE-LOOP (HIL)")
    print("  Sistema ALMMo-0 — Irrigação Preditiva")
    print("  Cultura: Tomate | Solo: Franco-Arenoso | Imperatriz-MA")
    print("="*60)

    # ------ Carregar modelo ------
    t0_load = time.time()
    modelo, modelo_source = carregar_modelo(PKL_CAMPO, PKL_INICIAL)
    t_load_ms = (time.time() - t0_load) * 1000
    print(f"\n[Sistema] {modelo.info()}")
    print(f"[Sistema] Carregamento: {t_load_ms:.1f} ms")

    # ------ Obter dados meteorológicos reais ------
    print(f"\n[METEO] Obtendo dados de Imperatriz-MA (Open-Meteo)...")
    dados_meteo = obter_dados_meteorologicos(usar_api=usar_api)
    print(f"[METEO] Fonte: {dados_meteo['fonte']} | "
          f"Chuva 3d: {dados_meteo['chuva_acum_3d_mm']} mm | "
          f"Tmax 3d: {dados_meteo['tmax_max_3d_c']} °C")

    # ------ Inicializar componentes partilhados ------
    normalizador = NormalizadorOnline(
        media_inicial=modelo.input_mean.copy(),
        std_inicial=modelo.input_std.copy(),
    )
    sensor_chuva = SimuladorChuva(seed=42)

    # ------ Seleccionar cenários a executar ------
    ids_cenarios = [args.cenario] if args.cenario else [1, 2, 3, 4]

    todos_resultados = {}
    tempos           = {}

    for id_cenario in ids_cenarios:
        cenario        = CENARIOS[id_cenario].copy()
        cenario['id']  = id_cenario

        # Cada cenário parte de um cold start limpo — garante independência
        # e reprodutibilidade. Sem este reset, aprendizado acumulado dos
        # cenários anteriores contamina os resultados seguintes.
        modelo_cenario, _ = carregar_modelo(PKL_CAMPO, PKL_INICIAL)
        normalizador_cenario = NormalizadorOnline(
            media_inicial=modelo_cenario.input_mean.copy(),
            std_inicial=modelo_cenario.input_std.copy(),
        )

        # Feedback reiniciado por cenário
        feedback = FeedbackStressHidrico()

        # Medir tempo de inferência
        t0_cenario = time.time()
        resultados = executar_cenario(
            cenario, modelo_cenario, feedback, normalizador_cenario,
            dados_meteo, sensor_chuva, usar_api=usar_api
        )
        t_total    = time.time() - t0_cenario
        t_inf_ms   = (t_total / len(resultados)) * 1000 if resultados else 0

        todos_resultados[id_cenario] = resultados
        tempos[id_cenario]           = (t_load_ms, t_inf_ms)

        # Guardar CSV do cenário
        caminho_csv = os.path.join(RESULTADOS_DIR, cenario['ficheiro'])
        salvar_csv(resultados, caminho_csv)

        print(f"\n  Tempo total: {t_total:.2f}s | "
              f"Média/ciclo: {t_inf_ms:.1f}ms")

    # ------ Gráficos ------
    if MATPLOTLIB_OK:
        print("\n[Gráficos] Gerando...")
        gerar_graficos(todos_resultados, RESULTADOS_DIR)
    else:
        print("\n[Gráficos] matplotlib não disponível — omitidos.")

    # ------ Relatório HIL ------
    print("\n[Relatório] Gerando relatorio_hil.md...")
    gerar_relatorio(todos_resultados, dados_meteo, tempos,
                    modelo_source, RESULTADOS_DIR)

    # ------ Salvar modelo actualizado ------
    modelo.salvar(PKL_CAMPO)
    print(f"\n[Modelo] Estado final guardado em: {PKL_CAMPO}")
    print(f"[Modelo] {modelo.info()}")

    print("\n" + "="*60)
    print("  VALIDAÇÃO HIL CONCLUÍDA")
    print(f"  Resultados em: {RESULTADOS_DIR}/")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
