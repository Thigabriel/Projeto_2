#!/bin/bash
# ============================================================================
# instalar.sh — Setup do sistema ALMMo-0 v12 no Raspberry Pi
# ============================================================================
#
# O que este script faz:
#   1. Instala dependencias Python (Flask, requests, numpy)
#   2. Instala Node.js e Node-RED (se nao estiver instalado)
#   3. Instala node-red-dashboard (interface visual)
#   4. Configura cron diario as 7h
#   5. Cria servico systemd para a Flask API (auto-start no boot)
#   6. Instruções para importar o flow no Node-RED
#
# Uso:
#   chmod +x instalar.sh
#   bash instalar.sh
#
# Pre-requisitos:
#   - Raspberry Pi OS (Bookworm ou Bullseye)
#   - Python 3.10+
#   - Acesso a internet
#   - Arquivo ../memoria_cold_start_v12_ensemble.pkl deve existir
# ============================================================================

set -e  # Para em caso de erro

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log()  { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[AVISO]${NC} $1"; }
info() { echo -e "${BLUE}[INFO]${NC} $1"; }
err()  { echo -e "${RED}[ERRO]${NC} $1"; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
PKL_FILE="$PARENT_DIR/memoria_cold_start_v12_ensemble.pkl"
CRON_LOG="$SCRIPT_DIR/logs/cron.log"
SERVICE_NAME="almmo-api"
PYTHON_CMD="python3"

echo ""
echo "============================================================"
echo "  Setup — Sistema de Irrigação ALMMo-0 v12"
echo "  Diretório: $SCRIPT_DIR"
echo "============================================================"
echo ""

# ============================================================
# 0. VERIFICAÇÕES INICIAIS
# ============================================================
info "Verificando pré-requisitos..."

# Python
if ! command -v $PYTHON_CMD &>/dev/null; then
    err "Python 3 não encontrado. Instale: sudo apt install python3"
fi
PY_VER=$($PYTHON_CMD --version 2>&1)
log "Python: $PY_VER"

# PKL
if [ ! -f "$PKL_FILE" ]; then
    warn "PKL não encontrado em: $PKL_FILE"
    warn "Certifique-se de que memoria_cold_start_v12_ensemble.pkl está em $PARENT_DIR"
    warn "O sistema instalará mas não funcionará sem o modelo."
else
    log "Modelo PKL: $PKL_FILE"
fi

# Criar diretório de logs
mkdir -p "$SCRIPT_DIR/logs"
log "Diretório logs/ criado"

# ============================================================
# 1. DEPENDÊNCIAS PYTHON
# ============================================================
echo ""
info "Instalando dependências Python..."

pip3 install flask flask-cors requests numpy --break-system-packages 2>/dev/null || \
pip3 install flask flask-cors requests numpy 2>/dev/null || \
err "Falha ao instalar dependências Python. Tente manualmente:
     pip3 install flask flask-cors requests numpy --break-system-packages"

log "Flask, Flask-CORS, requests, numpy instalados"

# Verificação
$PYTHON_CMD -c "import flask, requests, numpy; print('Imports OK')" 2>/dev/null && \
    log "Imports Python verificados" || \
    warn "Verificação de imports falhou — verifique manualmente"

# ============================================================
# 2. PYSERIAL (COMUNICAÇÃO ARDUINO)
# ============================================================
echo ""
info "Instalando pyserial (comunicação Arduino)..."

pip3 install pyserial --break-system-packages 2>/dev/null || \
pip3 install pyserial 2>/dev/null || \
warn "Falha ao instalar pyserial. Instale manualmente:
     pip3 install pyserial --break-system-packages"

$PYTHON_CMD -c "import serial; print('pyserial OK')" 2>/dev/null && \
    log "pyserial instalado" || warn "pyserial não disponível — instale manualmente"

# ============================================================
# 3. CRON DIÁRIO (7h)
# ============================================================
echo ""
info "Configurando cron diário às 7h00..."

CRON_LINE="0 7 * * * cd $SCRIPT_DIR && $PYTHON_CMD cron_diario.py >> $CRON_LOG 2>&1"

# Verifica se ja existe
EXISTING=$(crontab -l 2>/dev/null | grep "cron_diario.py" || true)
if [ -n "$EXISTING" ]; then
    warn "Cron já configurado: $EXISTING"
    echo "  Substituir? [s/N]"
    read -r SUBST
    if [[ "$SUBST" =~ ^[Ss]$ ]]; then
        (crontab -l 2>/dev/null | grep -v "cron_diario.py"; echo "$CRON_LINE") | crontab -
        log "Cron atualizado"
    else
        info "Cron mantido sem alteração"
    fi
else
    (crontab -l 2>/dev/null; echo "$CRON_LINE") | crontab -
    log "Cron configurado: 0 7 * * *"
fi

# Verificar cron
crontab -l | grep "cron_diario" && log "Cron ativo" || warn "Verifique o cron manualmente: crontab -l"

# ============================================================
# 4. SERVIÇO SYSTEMD PARA A FLASK API
# ============================================================
echo ""
info "Criando serviço systemd para a Flask API..."

SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

sudo tee "$SERVICE_FILE" > /dev/null <<EOF
[Unit]
Description=ALMMo-0 Irrigation API (Flask)
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$SCRIPT_DIR
ExecStart=$PYTHON_CMD $SCRIPT_DIR/api_rasp.py
Restart=always
RestartSec=10
StandardOutput=append:$SCRIPT_DIR/logs/api.log
StandardError=append:$SCRIPT_DIR/logs/api.log
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable $SERVICE_NAME
sudo systemctl start $SERVICE_NAME

sleep 2

if systemctl is-active --quiet $SERVICE_NAME; then
    log "Serviço $SERVICE_NAME ativo e rodando"
else
    warn "Serviço pode não ter iniciado. Verifique: sudo systemctl status $SERVICE_NAME"
fi

# ============================================================
# 5. VERIFICAÇÃO FINAL
# ============================================================
echo ""
echo "============================================================"
echo -e "${GREEN}  SETUP CONCLUÍDO${NC}"
echo "============================================================"
echo ""
echo "  📁 Diretório:     $SCRIPT_DIR"
echo "  🧠 Modelo PKL:    $PKL_FILE"
echo "  🌐 Flask API:     http://localhost:5000"
echo "  ⏰ Cron:          todos os dias às 7h00"
echo "  🔧 Serviço:       sudo systemctl {status|stop|restart} $SERVICE_NAME"
echo ""

  echo "  ─────────────────────────────────────────────────────"
  echo "  📊 DASHBOARD WEB:"
  echo ""
  echo "  Abra no browser (qualquer dispositivo na mesma rede):"
  echo "     http://$(hostname -I | awk '{print $1}'):5000"
  echo ""
  echo "  No próprio Raspberry Pi:"
  echo "     http://localhost:5000"
  echo ""
  echo "  ─────────────────────────────────────────────────────"
  echo "  🔌 ARDUINO:"
  echo ""
  echo "  Quando conectar o Arduino, edite config.json:"
  echo "     \"habilitado\": true"
  echo "     \"porta\": \"/dev/ttyUSB0\"   (ou /dev/ttyACM0)"
  echo "  ─────────────────────────────────────────────────────"

echo ""
echo "  🧪 Testar o sistema agora (sem NASA POWER):"
echo "     cd $SCRIPT_DIR"
echo "     python3 cron_diario.py --teste"
echo ""
echo "  💧 Inserir leitura de tensão via API:"
echo "     curl -X POST http://localhost:5000/api/tensao \\"
echo "          -H 'Content-Type: application/json' \\"
echo "          -d '{\"tensao_kpa\": 52.5}'"
echo ""
echo "  📋 Ver estado atual:"
echo "     curl http://localhost:5000/api/status | python3 -m json.tool"
echo ""
echo "============================================================"
