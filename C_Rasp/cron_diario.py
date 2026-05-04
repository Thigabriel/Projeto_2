#!/usr/bin/env python3
"""
============================================================================
cron_diario.py — Ciclo diario do sistema de irrigacao ALMMo-0 v12
============================================================================

Roda automaticamente as 7h via cron (ou manualmente para testes).
Local: Imperatriz-MA | Cultura: Tomate | Ensemble: 14 modelos ALMMo-0 v12

Fluxo:
  1. Le tensao do Arduino (serial USB) ou do dashboard (tensao_input.json)
  2. Busca T2M_MAX e PRECTOTCORR da NASA POWER (ultimos 3 dias)
  3. Calcula as 5 features [tensao, chuva_3d, tmax_3d, dap, delta]
  4. Carrega ensemble v12 (14 modelos ALMMo-0) — normalizacao embutida
  5. Votacao simples — classe mais votada entre os 14 modelos
  6. Calcula tempo de acionamento da bomba e envia ao Arduino
  7. Salva estado.json (consultado pela API web)
  8. Acrescenta linha em log_decisoes.csv

IMPORTANTE — sem shift de tensao:
  No campo com sensor real, a leitura ja representa o estado atual do solo.
  O shift que existe nas simulacoes AquaCrop NAO se aplica aqui.

Protocolo Arduino (serial):
  Rasp envia:     "LEITURA\n"     → solicita leitura do sensor
  Arduino responde: "TENSAO:52.5\n" → tensao em kPa
  Rasp envia:     "BOMBA:120\n"   → acionar bomba por 120 segundos
  Arduino responde: "OK:BOMBA:120\n" → confirmacao

Uso:
  python3 cron_diario.py           # producao (Arduino + NASA POWER)
  python3 cron_diario.py --teste   # dados dummy, sem hardware, sem API

Cron (7h todo dia):
  crontab -e
  0 7 * * * cd /home/pi/irrigacao && python3 cron_diario.py >> logs/cron.log 2>&1
============================================================================
"""

import sys
import os
import json
import csv
import pickle
import argparse
import requests
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta, date
from collections import Counter


# ============================================================================
# PATHS
# ============================================================================
BASE_DIR     = Path(__file__).parent.resolve()
PARENT_DIR   = BASE_DIR.parent
CONFIG_FILE  = BASE_DIR / 'config.json'
ESTADO_FILE  = BASE_DIR / 'estado.json'
LOG_FILE     = BASE_DIR / 'log_decisoes.csv'
TENSAO_INPUT = BASE_DIR / 'tensao_input.json'
LOGS_DIR     = BASE_DIR / 'logs'
LOGS_DIR.mkdir(exist_ok=True)


# ============================================================================
# CONSTANTES
# ============================================================================
API_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

NOMES_CLASSE = {
    0: 'Sem Irrigacao (C0)',
    1: 'Manutencao (C1)',
    2: 'Irrigacao Intensiva (C2)',
}
COR_CLASSE = {
    0: '#1a7a4a',
    1: '#d97706',
    2: '#dc2626',
}


# ============================================================================
# ALMMo-0 (embutido para deploy autonomo no Rasp)
# ============================================================================
class ALMMo0:
    def __init__(self, n_inputs=5, r_threshold=0.5, max_rules=50,
                 age_limit=100, epsilon=1e-8, n_classes=3,
                 min_rules_per_class=3):
        self.n_inputs            = n_inputs
        self.r_threshold         = r_threshold
        self.max_rules           = max_rules
        self.age_limit           = age_limit
        self.epsilon             = epsilon
        self.n_classes           = n_classes
        self.min_rules_per_class = min_rules_per_class
        self.rules               = []
        self.input_mean          = np.zeros(n_inputs)
        self.input_std           = np.ones(n_inputs)
        self.n_samples_seen      = 0
        self.n_rules_created     = 0
        self.n_rules_pruned      = 0

    def normalize(self, x):
        return (x - self.input_mean) / (self.input_std + 1e-10)

    def _dists(self, x_n):
        return np.array([
            np.sqrt(np.sum((x_n - np.array(r['center']))**2))
            for r in self.rules
        ])

    def predict(self, x):
        x_n = self.normalize(x)
        if not self.rules:
            return 0, np.ones(self.n_classes) / self.n_classes
        d = self._dists(x_n)
        w = 1.0 / (d**2 + self.epsilon)
        v = np.zeros(self.n_classes)
        for i, r in enumerate(self.rules):
            v[r['consequent']] += w[i]
        s = v.sum()
        return int(np.argmax(v)), v / s if s > 0 else v

    def n_rules_por_classe(self):
        return {c: sum(1 for r in self.rules if r['consequent'] == c)
                for c in range(self.n_classes)}

    @classmethod
    def from_dict(cls, d):
        m = cls(
            n_inputs         = d['n_inputs'],
            r_threshold      = d['r_threshold'],
            max_rules        = d['max_rules'],
            age_limit        = d['age_limit'],
            n_classes        = d.get('n_classes', 3),
            min_rules_per_class = d.get('min_rules_per_class', 3)
        )
        m.rules = [
            {**r, 'center': np.array(r['center'])}
            for r in d['rules']
        ]
        m.input_mean     = np.array(d['input_mean'])
        m.input_std      = np.array(d['input_std'])
        m.n_samples_seen = d.get('n_samples_seen', 0)
        m.n_rules_created = d.get('n_rules_created', 0)
        m.n_rules_pruned  = d.get('n_rules_pruned', 0)
        return m


# ============================================================================
# ARDUINO — LEITURA DO SENSOR E CONTROLE DA BOMBA
# ============================================================================

def ler_tensao_arduino(cfg_arduino):
    """
    Le tensao do solo via Arduino conectado por USB serial.

    Protocolo:
      Rasp envia:      "LEITURA\n"
      Arduino responde: "TENSAO:52.5\n"

    Retorna tensao em kPa (float) ou None em caso de falha.
    """
    import time
    porta   = cfg_arduino.get('porta', '/dev/ttyUSB0')
    baud    = cfg_arduino.get('baudrate', 9600)
    timeout = cfg_arduino.get('timeout_s', 5)
    prefixo = cfg_arduino['protocolo'].get('prefixo_tensao', 'TENSAO:')
    cmd_lei = cfg_arduino['protocolo'].get('solicitar_leitura', 'LEITURA')

    try:
        import serial
    except ImportError:
        print("[Arduino] pyserial nao instalado. "
              "Execute: pip3 install pyserial --break-system-packages")
        return None

    try:
        with serial.Serial(porta, baud, timeout=timeout) as ser:
            time.sleep(1.5)           # aguarda Arduino inicializar
            ser.reset_input_buffer()
            ser.write(f"{cmd_lei}\n".encode('utf-8'))
            time.sleep(0.5)

            linha = ser.readline().decode('utf-8').strip()
            print(f"[Arduino] Raw: '{linha}'")

            if linha.startswith(prefixo):
                tensao = float(linha[len(prefixo):])
                if 0 < tensao < 2000:
                    print(f"[Arduino] Tensao lida: {tensao:.1f} kPa")
                    return tensao
                else:
                    print(f"[Arduino] Tensao fora do range: {tensao}")
                    return None
            else:
                print(f"[Arduino] Resposta inesperada: '{linha}'")
                return None

    except Exception as e:
        print(f"[Arduino] Falha na leitura: {e}")
        return None


def acionar_bomba_arduino(segundos, cfg_arduino):
    """
    Envia comando de acionamento da bomba para o Arduino.

    Protocolo:
      Rasp envia:      "BOMBA:120\n"     (segundos como inteiro)
      Arduino responde: "OK:BOMBA:120\n"

    Retorna True se confirmado, False em caso de falha.
    """
    import time
    if segundos <= 0:
        print("[Arduino] Bomba: 0s — sem irrigacao, comando nao enviado")
        return True

    porta   = cfg_arduino.get('porta', '/dev/ttyUSB0')
    baud    = cfg_arduino.get('baudrate', 9600)
    timeout = cfg_arduino.get('timeout_s', 5)
    cmd_bom = cfg_arduino['protocolo'].get('comando_bomba', 'BOMBA:')

    try:
        import serial
    except ImportError:
        print("[Arduino] pyserial nao instalado.")
        return False

    try:
        with serial.Serial(porta, baud, timeout=timeout) as ser:
            time.sleep(1.5)
            ser.reset_input_buffer()
            comando = f"{cmd_bom}{int(segundos)}\n".encode('utf-8')
            ser.write(comando)
            print(f"[Arduino] Enviado: {comando.decode().strip()}")
            time.sleep(0.5)

            resposta = ser.readline().decode('utf-8').strip()
            print(f"[Arduino] Resposta bomba: '{resposta}'")

            ok = resposta.startswith('OK:') and str(int(segundos)) in resposta
            if ok:
                print(f"[Arduino] Bomba confirmada: {segundos}s ({segundos/60:.1f} min)")
            else:
                print(f"[Arduino] AVISO: confirmacao inesperada — verificar fisicamente")
            return ok

    except Exception as e:
        print(f"[Arduino] Falha ao acionar bomba: {e}")
        return False


def calcular_tempo_bomba(mm_irrigar, cfg_irrigacao):
    """
    Converte mm de irrigacao em segundos de acionamento da bomba.

    Formula:
      volume_L = mm * area_m2          (1mm = 1L/m2)
      tempo_s  = volume_L / (vazao_lpm / 60)

    Exemplo: 15mm, 10m2, 2 L/min → 150L / (2/60) = 4500s = 75 min
    """
    if mm_irrigar <= 0:
        return 0
    vazao  = cfg_irrigacao.get('bomba_vazao_lpm', 2.0)
    area   = cfg_irrigacao.get('bomba_area_m2', 10.0)
    volume = mm_irrigar * area
    tempo  = volume / (vazao / 60.0)
    return round(tempo)


# ============================================================================
# TENSAO — LEITURA COM FALLBACK
# ============================================================================

def ler_tensao_atual(cfg):
    """
    Prioridade 1: Arduino (se habilitado em config).
    Prioridade 2: tensao_input.json (inserida via dashboard web).
    Prioridade 3: ultimo valor do log (fallback de emergencia).

    Retorna (tensao_kpa, fonte_str, via_dashboard:bool)
    """
    cfg_arduino = cfg.get('arduino', {})
    arduino_hab = cfg_arduino.get('habilitado', False)

    # --- Arduino ---
    if arduino_hab:
        tensao = ler_tensao_arduino(cfg_arduino)
        if tensao is not None:
            return tensao, 'arduino (sensor fisico)', False

    # --- Dashboard manual ---
    if TENSAO_INPUT.exists():
        try:
            with open(TENSAO_INPUT) as f:
                data = json.load(f)
            tensao = float(data.get('tensao_kpa', 0))
            if tensao > 0:
                ts = data.get('timestamp', '—')[:16]
                return tensao, f'dashboard ({ts})', True
        except Exception as e:
            print(f"[AVISO] tensao_input.json invalido: {e}")

    # --- Fallback: ultimo log ---
    if LOG_FILE.exists():
        try:
            with open(LOG_FILE) as f:
                rows = list(csv.DictReader(f))
            if rows:
                ultima = rows[-1]
                print(f"[AVISO] Usando tensao do log como fallback")
                return float(ultima['tensao_kpa']), f"fallback/log ({ultima['data']})", False
        except Exception as e:
            print(f"[AVISO] Falha lendo log: {e}")

    return None, None, False


def ler_tensao_ontem():
    """Retorna tensao do ultimo registro do log, ou None."""
    if not LOG_FILE.exists():
        return None
    try:
        with open(LOG_FILE) as f:
            rows = list(csv.DictReader(f))
        if rows:
            return float(rows[-1]['tensao_kpa'])
    except Exception:
        pass
    return None


# ============================================================================
# METEOROLOGIA — NASA POWER
# ============================================================================

def calcular_meteo_3d(lat, lon, hoje, timeout=60):
    """
    Busca T2M_MAX e PRECTOTCORR dos ultimos 3 dias via NASA POWER.
    Retorna (chuva_acum_3d_mm, tmax_max_3d_c, fonte_str).
    Fallback automatico se API falhar.
    """
    d_inicio = hoje - timedelta(days=3)
    d_fim    = hoje - timedelta(days=1)

    try:
        params = {
            'parameters': 'T2M_MAX,PRECTOTCORR',
            'community':  'AG',
            'longitude':  lon,
            'latitude':   lat,
            'start':      d_inicio.strftime('%Y%m%d'),
            'end':        d_fim.strftime('%Y%m%d'),
            'format':     'JSON',
        }
        r = requests.get(API_URL, params=params, timeout=timeout)
        r.raise_for_status()
        meteo = r.json()['properties']['parameter']

        tmax_vals = [v for v in meteo['T2M_MAX'].values()    if v > -900]
        prec_vals = [max(v, 0.0) for v in meteo['PRECTOTCORR'].values() if v > -900]

        if not tmax_vals or not prec_vals:
            raise ValueError("Dados vazios na resposta da API")

        return (
            round(sum(prec_vals), 2),
            round(max(tmax_vals), 1),
            f'NASA POWER ({d_inicio} — {d_fim})',
        )
    except Exception as e:
        print(f"[AVISO] NASA POWER falhou: {e}. Usando fallback (chuva=0, tmax=33°C)")
        return 0.0, 33.0, 'fallback (sem NASA POWER)'


# ============================================================================
# ENSEMBLE
# ============================================================================

def localizar_pkl(cfg):
    """Encontra o PKL do ensemble tentando varios caminhos."""
    nome = cfg['modelo']['pkl_ensemble']
    candidatos = [
        PARENT_DIR / nome,
        BASE_DIR   / nome,
        PARENT_DIR / 'memoria_cold_start_v12_ensemble.pkl',
    ]
    for p in candidatos:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"PKL nao encontrado. Candidatos:\n" +
        "\n".join(f"  {p}" for p in candidatos)
    )


def carregar_ensemble(cfg):
    pkl_path = localizar_pkl(cfg)
    with open(pkl_path, 'rb') as f:
        raw = pickle.load(f)
    modelos = [ALMMo0.from_dict(d) for d in raw['models']]
    n_regras = sum(len(m.rules) for m in modelos)
    print(f"[Ensemble] {len(modelos)} modelos | {n_regras} regras totais | {pkl_path.name}")
    return modelos


def predizer_ensemble(modelos, features_dict):
    """Votacao simples — classe mais votada pelos 14 modelos."""
    x = np.array([
        features_dict['tensao_solo_kpa'],
        features_dict['chuva_acum_3d_mm'],
        features_dict['tmax_max_3d_c'],
        features_dict['dap'],
        features_dict['delta_tensao_kpa'],
    ], dtype=float)

    votos = Counter(m.predict(x)[0] for m in modelos)
    classe   = votos.most_common(1)[0][0]
    consenso = votos[classe] / len(modelos) * 100
    return classe, round(consenso, 1), {int(k): int(v) for k, v in votos.items()}


# ============================================================================
# REGRAS AGRONOMICAS — TOMATE (Doorenbos & Pruitt, FAO-24)
# ============================================================================

def regras_agronomicas(tensao_kpa, delta_kpa, chuva_3d_mm, dap):
    """
    Limiares deterministas baseados em tensiometria para tomate.

    Classe base pela tensao atual:
      < 60 kPa  → C0 (zona otima)
      60-90 kPa → C1 (atencao, irrigacao moderada)
      >= 90 kPa → C2 (estresse, irrigacao intensiva)

    Ajustes secundarios:
      Delta > +8 kPa  → escala 1 classe (solo secando rapido)
      Delta < -8 kPa  → reduz 1 classe (solo umedecendo)
      Chuva 3d > 10mm → reduz 1 classe (chuva recente)
      DAP >= 56 + tensao >= 75 kPa → forca C2 (fase critica: frutificacao)
    """
    # Classe base
    if tensao_kpa >= 90:
        classe = 2
    elif tensao_kpa >= 60:
        classe = 1
    else:
        classe = 0

    # Tendencia de secagem/umedecimento
    if delta_kpa >= 8 and classe < 2:
        classe += 1
    elif delta_kpa <= -8 and classe > 0:
        classe -= 1

    # Chuva recente reduz necessidade
    if chuva_3d_mm >= 10 and classe > 0:
        classe -= 1

    # Fase critica de frutificacao: threshold mais rigido
    if dap >= 56 and tensao_kpa >= 75 and classe < 2:
        classe = min(classe + 1, 2)

    return int(classe)


def decidir_hibrido(classe_ensemble, consenso_pct, tensao_kpa, delta_kpa, chuva_3d_mm, dap):
    """
    Combina ensemble e regras agronomicas.

    Logica de confianca:
      Consenso >= 70%  : ensemble decide (alta confianca, aprendizado confiavel)
      Consenso 51-69%  : ensemble decide, mas regras corrigem casos extremos
                         (tensao >= 80 kPa com ensemble conservador,
                          ou tensao < 40 kPa com ensemble agressivo)
      Consenso <= 50%  : empate — regras agronomicas decidem

    Retorna (classe_final, origem_str, classe_agro)
    """
    classe_agro = regras_agronomicas(tensao_kpa, delta_kpa, chuva_3d_mm, dap)

    if consenso_pct >= 70:
        return classe_ensemble, 'ensemble', classe_agro

    if consenso_pct > 50:
        if tensao_kpa >= 80 and classe_ensemble < classe_agro:
            return classe_agro, 'agro (corrige extremo alto)', classe_agro
        if tensao_kpa < 40 and classe_ensemble > classe_agro:
            return classe_agro, 'agro (corrige extremo baixo)', classe_agro
        return classe_ensemble, 'ensemble (medio)', classe_agro

    # Empate ou incerteza: regras agronomicas
    return classe_agro, 'agro (desempate)', classe_agro


# ============================================================================
# ESTADO E LOG
# ============================================================================

def salvar_estado(estado):
    with open(ESTADO_FILE, 'w') as f:
        json.dump(estado, f, indent=2, ensure_ascii=False)


def carregar_estado():
    if ESTADO_FILE.exists():
        with open(ESTADO_FILE) as f:
            return json.load(f)
    return {}


def append_log(row):
    existe = LOG_FILE.exists()
    with open(LOG_FILE, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not existe:
            w.writeheader()
        w.writerow(row)


# ============================================================================
# MAIN
# ============================================================================

def main(modo_teste=False):
    sep = '=' * 65
    hoje  = date.today()
    agora = datetime.now()

    print(f"\n{sep}")
    print(f"CICLO DIARIO ALMMo-0 v12 | {hoje} | {agora.strftime('%H:%M:%S')}")
    print(sep)

    # ---- Configuracao ----
    cfg = json.loads(CONFIG_FILE.read_text())
    lat  = cfg['localizacao']['latitude']
    lon  = cfg['localizacao']['longitude']
    data_plantio = datetime.strptime(cfg['plantio']['data_plantio'], '%Y-%m-%d').date()
    dap_max      = cfg['plantio'].get('dap_maximo', 107)
    dap          = (hoje - data_plantio).days

    print(f"[Config] {cfg['localizacao']['cidade']} ({lat}, {lon}) | "
          f"Plantio: {data_plantio} | DAP: {dap}")

    # ---- Ciclo de cultivo ----
    if dap < 0 or dap > dap_max:
        msg = f"DAP={dap} fora do intervalo [0, {dap_max}]. Sistema em pausa."
        print(f"[INFO] {msg}")
        estado = carregar_estado()
        estado.update({'status': 'fora_ciclo', 'mensagem': msg,
                       'ultima_atualizacao': agora.strftime('%Y-%m-%d %H:%M:%S')})
        salvar_estado(estado)
        return

    # ---- Tensao ----
    if modo_teste:
        tensao_hoje, fonte_tensao, via_dashboard = 58.0, 'teste (dummy)', False
    else:
        tensao_hoje, fonte_tensao, via_dashboard = ler_tensao_atual(cfg)
        if tensao_hoje is None:
            msg = ("Nenhuma tensao disponivel. "
                   "Insira via dashboard ou conecte o Arduino.")
            print(f"[ERRO] {msg}")
            estado = carregar_estado()
            estado.update({'status': 'aguardando_tensao', 'mensagem': msg,
                           'ultima_atualizacao': agora.strftime('%Y-%m-%d %H:%M:%S')})
            salvar_estado(estado)
            return

    tensao_ontem = ler_tensao_ontem()
    delta = round(tensao_hoje - tensao_ontem, 3) if tensao_ontem is not None else 0.0
    print(f"[Tensao] Hoje={tensao_hoje:.1f} kPa | "
          f"Ontem={tensao_ontem if tensao_ontem else '—'} kPa | "
          f"Delta={delta:+.2f} kPa | Fonte={fonte_tensao}")

    # ---- Meteorologia ----
    if modo_teste:
        chuva_3d, tmax_3d, fonte_meteo = 0.0, 33.0, 'teste (dummy)'
    else:
        chuva_3d, tmax_3d, fonte_meteo = calcular_meteo_3d(
            lat, lon, hoje, cfg['nasa_power']['timeout_s']
        )
    print(f"[Meteo]  Chuva 3d={chuva_3d:.1f}mm | Tmax 3d={tmax_3d:.1f}°C | {fonte_meteo}")

    # ---- Features ----
    features = {
        'tensao_solo_kpa':  tensao_hoje,
        'chuva_acum_3d_mm': chuva_3d,
        'tmax_max_3d_c':    tmax_3d,
        'dap':              dap,
        'delta_tensao_kpa': delta,
    }

    # ---- Ensemble ----
    modelos  = carregar_ensemble(cfg)
    classe_ensemble, consenso, votos = predizer_ensemble(modelos, features)
    n_regras = sum(len(m.rules) for m in modelos)

    # ---- Decisao hibrida (ensemble + regras agronomicas) ----
    classe, decisao_origem, classe_agro = decidir_hibrido(
        classe_ensemble, consenso,
        tensao_hoje, delta, chuva_3d, dap
    )
    mm_irrigar    = cfg['irrigacao'][f'mm_C{classe}']
    tempo_bomba_s = calcular_tempo_bomba(mm_irrigar, cfg['irrigacao'])

    print(f"[Ensemble] Classe={classe_ensemble} | Consenso={consenso:.1f}%")
    print(f"[Agro]     Classe={classe_agro} (regras deterministas)")
    print(f"[Decisao]  Classe={classe} | {NOMES_CLASSE[classe]} | "
          f"Origem={decisao_origem} | Irrigar {mm_irrigar:.0f}mm | "
          f"Bomba={tempo_bomba_s}s ({tempo_bomba_s/60:.1f}min)")

    # ---- Aciona bomba via Arduino ----
    arduino_ok = None
    cfg_arduino = cfg.get('arduino', {})
    if cfg_arduino.get('habilitado', False) and not modo_teste:
        arduino_ok = acionar_bomba_arduino(tempo_bomba_s, cfg_arduino)
    else:
        motivo = 'modo_teste' if modo_teste else 'arduino desabilitado (config)'
        print(f"[Bomba]  Arduino nao acionado ({motivo})")

    # ---- Salva estado ----
    regras_por_classe = {}
    for m in modelos:
        for c, n in m.n_rules_por_classe().items():
            regras_por_classe[c] = regras_por_classe.get(c, 0) + n

    estado = {
        'status':             'ok',
        'data':               str(hoje),
        'dap':                dap,
        'tensao_kpa':         round(tensao_hoje, 2),
        'delta_kpa':          delta,
        'chuva_3d_mm':        chuva_3d,
        'tmax_3d_c':          tmax_3d,
        'classe':             classe,
        'nome_classe':        NOMES_CLASSE[classe],
        'cor_classe':         COR_CLASSE[classe],
        'mm_irrigar':         mm_irrigar,
        'tempo_bomba_s':      tempo_bomba_s,
        'consenso_pct':       consenso,
        'votos':              votos,
        'n_regras':           n_regras,
        'n_modelos':          len(modelos),
        'regras_por_classe':  regras_por_classe,
        'classe_ensemble':    classe_ensemble,
        'classe_agro':        classe_agro,
        'decisao_origem':     decisao_origem,
        'fonte_tensao':       fonte_tensao,
        'fonte_meteo':        fonte_meteo,
        'arduino_habilitado': cfg_arduino.get('habilitado', False),
        'arduino_bomba_ok':   arduino_ok,
        'modo_teste':         modo_teste,
        'ultima_atualizacao': agora.strftime('%Y-%m-%d %H:%M:%S'),
        'mensagem': (
            f"[TESTE] " if modo_teste else ""
        ) + f"[{decisao_origem}] Irrigar {mm_irrigar:.0f}mm ({tempo_bomba_s}s de bomba)",
    }
    salvar_estado(estado)

    # ---- Log ----
    append_log({
        'data':           str(hoje),
        'hora':           agora.strftime('%H:%M'),
        'dap':            dap,
        'tensao_kpa':     round(tensao_hoje, 2),
        'delta_kpa':      delta,
        'chuva_3d_mm':    chuva_3d,
        'tmax_3d_c':      tmax_3d,
        'classe':         classe,
        'nome_classe':    NOMES_CLASSE[classe],
        'consenso_pct':   consenso,
        'mm_irrigar':     mm_irrigar,
        'tempo_bomba_s':  tempo_bomba_s,
        'n_regras':        n_regras,
        'regras_c0':       regras_por_classe.get(0, 0),
        'regras_c1':       regras_por_classe.get(1, 0),
        'regras_c2':       regras_por_classe.get(2, 0),
        'classe_ensemble': classe_ensemble,
        'classe_agro':     classe_agro,
        'decisao_origem':  decisao_origem,
        'fonte_tensao':    fonte_tensao,
        'arduino_ok':      int(arduino_ok) if arduino_ok is not None else '',
        'modo_teste':      int(modo_teste),
    })

    # ---- Limpa tensao pendente do dashboard ----
    if via_dashboard and TENSAO_INPUT.exists():
        TENSAO_INPUT.unlink()

    print(f"\n{'✓' * 3} {NOMES_CLASSE[classe]} — {mm_irrigar:.0f}mm | Bomba: {tempo_bomba_s}s")
    print(f"{'✓' * 3} Estado: {ESTADO_FILE.name} | Log: {LOG_FILE.name}")
    print(sep)


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ciclo diario ALMMo-0 v12')
    parser.add_argument('--teste', action='store_true',
                        help='Dados dummy — sem Arduino, sem NASA POWER')
    args = parser.parse_args()
    main(modo_teste=args.teste)
