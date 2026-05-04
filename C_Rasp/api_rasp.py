#!/usr/bin/env python3
"""
============================================================================
api_rasp.py — Flask REST API + Dashboard Web
Sistema ALMMo-0 v12 — Irrigacao Inteligente de Tomate | Imperatriz-MA
============================================================================

Roda continuamente no Raspberry Pi.
Serve o dashboard web (dashboard.html) e a API REST consultada por ele.
Nao carrega o PKL — apenas serve os arquivos de estado gerados pelo cron.

Dashboard web:
  http://localhost:5000/              → abre no browser (PC ou celular)

Endpoints:
  GET  /api/health          → health check
  GET  /api/status          → estado completo (decisao + ensemble + meteo)
  GET  /api/decisao         → decisao de hoje (resumido)
  GET  /api/historico?n=30  → ultimas N linhas do log CSV
  GET  /api/tensao          → tensao pendente (aguardando cron)
  POST /api/tensao          → {"tensao_kpa": 45.2} — registra leitura manual
  POST /api/executar        → dispara cron_diario.py imediatamente

Iniciar:
  python3 api_rasp.py

Porta: 5000 (configuravel em config.json)

Instalar dependencias:
  pip3 install flask flask-cors --break-system-packages
============================================================================
"""

import os
import sys
import json
import csv
import subprocess
from pathlib import Path
from datetime import datetime

from flask import Flask, jsonify, request, abort, send_file
from flask_cors import CORS


# ============================================================================
# PATHS
# ============================================================================
BASE_DIR     = Path(__file__).parent.resolve()
CONFIG_FILE  = BASE_DIR / 'config.json'
ESTADO_FILE  = BASE_DIR / 'estado.json'
LOG_FILE     = BASE_DIR / 'log_decisoes.csv'
TENSAO_INPUT = BASE_DIR / 'tensao_input.json'
CRON_SCRIPT  = BASE_DIR / 'cron_diario.py'


# ============================================================================
# FLASK APP
# ============================================================================
app = Flask(__name__)
CORS(app)   # permite acesso cross-origin (Node-RED, browser externo)


# ============================================================================
# HELPERS
# ============================================================================
def _estado_vazio():
    return {
        'status':            'aguardando_primeiro_ciclo',
        'mensagem':          'Execute cron_diario.py para iniciar o sistema.',
        'data':              None,
        'dap':               None,
        'classe':            None,
        'nome_classe':       '—',
        'cor_classe':        '#7f8c8d',
        'icone_classe':      'help_outline',
        'mm_irrigar':        0,
        'consenso_pct':      None,
        'tensao_kpa':        None,
        'delta_kpa':         None,
        'chuva_3d_mm':       None,
        'tmax_3d_c':         None,
        'n_regras':          None,
        'n_modelos':         None,
        'votos':             {},
        'ultima_atualizacao': None,
    }


def _ler_estado():
    if ESTADO_FILE.exists():
        try:
            with open(ESTADO_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return _estado_vazio()


def _ler_log(n=30):
    if not LOG_FILE.exists():
        return []
    try:
        with open(LOG_FILE, newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        # Remove chaves None (ocorre quando CSV tem colunas sem header — versoes antigas)
        clean = [{k: v for k, v in row.items() if k is not None} for row in rows]
        # Retorna os ultimos N, do mais recente para o mais antigo
        return list(reversed(clean[-n:]))
    except Exception:
        return []


def _cfg_porta():
    porta = 5000
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE) as f:
                cfg = json.load(f)
            porta = cfg.get('sistema', {}).get('porta_flask', 5000)
        except Exception:
            pass
    return porta


# ============================================================================
# ROTAS
# ============================================================================

@app.route('/')
def dashboard():
    """Serve o dashboard web — acessível em http://IP-DO-PI:5000/"""
    html = BASE_DIR / 'dashboard.html'
    if html.exists():
        return send_file(str(html))
    return '<h2>dashboard.html não encontrado. Verifique o diretório.</h2>', 404


@app.route('/api/health')
def health():
    """Health check — responde sempre, mesmo sem dados."""
    return jsonify({
        'status': 'ok',
        'ts':     datetime.now().isoformat(),
        'server': 'ALMMo-0 v12 API',
    })


@app.route('/api/status')
def status():
    """
    Estado completo do sistema.
    Inclui decisao de hoje, dados meterologicos, info do ensemble.
    """
    return jsonify(_ler_estado())


@app.route('/api/decisao')
def decisao():
    """Resumo da decisao de hoje — para o card principal do dashboard."""
    e = _ler_estado()
    return jsonify({
        'data':         e.get('data'),
        'dap':          e.get('dap'),
        'classe':       e.get('classe'),
        'nome_classe':  e.get('nome_classe', '—'),
        'cor_classe':   e.get('cor_classe', '#7f8c8d'),
        'icone_classe': e.get('icone_classe', 'help_outline'),
        'mm_irrigar':   e.get('mm_irrigar', 0),
        'consenso_pct': e.get('consenso_pct'),
        'tensao_kpa':   e.get('tensao_kpa'),
        'status':       e.get('status'),
        'mensagem':     e.get('mensagem', ''),
    })


@app.route('/api/historico')
def historico():
    """
    Ultimas N decisoes do log.
    Query param: n (default 30, max 365)
    """
    n = min(request.args.get('n', 30, type=int), 365)
    rows = _ler_log(n)
    return jsonify(rows)


@app.route('/api/tensao', methods=['GET'])
def get_tensao():
    """Retorna tensao pendente aguardando o cron."""
    if TENSAO_INPUT.exists():
        try:
            with open(TENSAO_INPUT) as f:
                return jsonify(json.load(f))
        except Exception as e:
            return jsonify({'erro': str(e)}), 500
    return jsonify({'tensao_kpa': None, 'status': 'nenhuma_leitura_pendente'})


@app.route('/api/tensao', methods=['POST'])
def set_tensao():
    """
    Registra tensao inserida manualmente via dashboard.
    Sera consumida pelo cron_diario.py no proximo ciclo.

    Body: {"tensao_kpa": 45.2}
    """
    dados = request.get_json(force=True, silent=True)
    if not dados or 'tensao_kpa' not in dados:
        abort(400, description='Campo tensao_kpa e obrigatorio. Ex: {"tensao_kpa": 45.2}')

    try:
        tensao = float(dados['tensao_kpa'])
    except (ValueError, TypeError):
        abort(400, description='tensao_kpa deve ser um numero.')

    if tensao < 0 or tensao > 2000:
        abort(400, description=f'Tensao {tensao} kPa fora do intervalo valido [0, 2000].')

    payload = {
        'tensao_kpa': tensao,
        'timestamp':  datetime.now().isoformat(),
        'fonte':      'dashboard_nodered',
    }
    try:
        with open(TENSAO_INPUT, 'w') as f:
            json.dump(payload, f, indent=2)
    except Exception as e:
        abort(500, description=f'Falha ao salvar tensao: {e}')

    return jsonify({
        'status':    'ok',
        'tensao_kpa': tensao,
        'mensagem':  f'Tensao {tensao:.1f} kPa registrada. Sera usada no proximo ciclo (7h).',
    })


@app.route('/api/executar', methods=['POST'])
def executar():
    """
    Dispara cron_diario.py de forma nao-bloqueante.
    Util para testes ou quando se quer executar fora do horario do cron.
    """
    teste = request.get_json(force=True, silent=True) or {}
    args = [sys.executable, str(CRON_SCRIPT)]
    if teste.get('modo_teste'):
        args.append('--teste')

    try:
        proc = subprocess.Popen(
            args,
            stdout=open(BASE_DIR / 'logs' / 'cron_manual.log', 'a'),
            stderr=subprocess.STDOUT,
            cwd=str(BASE_DIR),
        )
        return jsonify({
            'status':   'iniciado',
            'pid':      proc.pid,
            'mensagem': 'Ciclo iniciado. Aguarde ~30s e atualize o dashboard.',
            'log':      str(BASE_DIR / 'logs' / 'cron_manual.log'),
        })
    except Exception as e:
        return jsonify({'status': 'erro', 'mensagem': str(e)}), 500


# ============================================================================
# TRATAMENTO DE ERROS
# ============================================================================
@app.errorhandler(400)
def bad_request(e):
    return jsonify({'erro': str(e.description)}), 400


@app.errorhandler(404)
def not_found(e):
    rotas = [
        'GET  /api/health',
        'GET  /api/status',
        'GET  /api/decisao',
        'GET  /api/historico?n=30',
        'GET  /api/tensao',
        'POST /api/tensao    {"tensao_kpa": float}',
        'POST /api/executar',
    ]
    return jsonify({'erro': 'Rota nao encontrada', 'rotas_validas': rotas}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({'erro': 'Erro interno', 'detalhe': str(e.description)}), 500


# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    porta = _cfg_porta()

    print("=" * 60)
    print("  ALMMo-0 v12 — API REST para Node-RED")
    print("=" * 60)
    print(f"  URL: http://0.0.0.0:{porta}")
    print(f"  Dashboard Node-RED: http://localhost:{porta}/api/status")
    print(f"  Health check:       http://localhost:{porta}/api/health")
    print()
    print("  Endpoints principais:")
    print(f"    GET  http://localhost:{porta}/api/status")
    print(f"    GET  http://localhost:{porta}/api/historico?n=30")
    print(f"    POST http://localhost:{porta}/api/tensao")
    print(f"    POST http://localhost:{porta}/api/executar")
    print("=" * 60)
    print()

    app.run(host='0.0.0.0', port=porta, debug=False)
