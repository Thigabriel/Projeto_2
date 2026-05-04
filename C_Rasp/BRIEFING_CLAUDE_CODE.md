# Briefing — Sistema de Irrigação ALMMo-0 v12
**Para uso em sessão Claude Code via SSH no Raspberry Pi**

---

## O que é este projeto

Sistema de irrigação inteligente para tomate baseado no modelo neuro-fuzzy ALMMo-0.
O Raspberry Pi roda o modelo de IA, lê o sensor de tensão do solo via Arduino (serial USB),
decide se e quanto irrigar, aciona a bomba e disponibiliza um dashboard web na rede local.

**Local:** Imperatriz-MA | **Cultura:** Tomate | **Solo:** Franco-Arenoso

---

## Estrutura de arquivos esperada no Rasp

```
/home/ifmarobotica/Documents/Python Gabriel/ALMMo Irrigacao
│
├── memoria_cold_start_v12_ensemble.pkl   ← modelo treinado (OBRIGATÓRIO)
│
├── config.json          ← parâmetros do sistema (editar antes de usar)
├── cron_diario.py       ← ciclo diário: lê sensor, decide, aciona bomba
├── api_rasp.py          ← servidor Flask: API REST + dashboard web
├── dashboard.html       ← interface web (servida pelo Flask)
├── instalar.sh          ← script de setup (rodar uma vez)
│
├── venv/                ← ambiente virtual Python (criar aqui)
├── logs/                ← logs do cron e da API (criado automaticamente)
├── estado.json          ← estado atual (gerado pelo cron)
└── log_decisoes.csv     ← histórico diário (gerado pelo cron)
```

---

## Tarefas para fazer no Rasp

### 1. Criar ambiente virtual Python
```bash
cd /home/ifmarobotica/Documents/Python Gabriel/ALMMo Irrigacao
python3 -m venv venv
source venv/bin/activate
pip install flask flask-cors requests numpy pyserial
```

Verificar:
```bash
python3 -c "import flask, requests, numpy, serial; print('Tudo OK')"
```

### 2. Confirmar que o PKL está presente
```bash
ls -lh /home/pi/irrigacao/memoria_cold_start_v12_ensemble.pkl
```
Se não estiver, transferir do computador via SCP:
```bash
scp usuario@ip-do-computador:/caminho/Projeto_2/memoria_cold_start_v12_ensemble.pkl \
    /home/pi/irrigacao/
```

### 3. Atualizar config.json
Campos obrigatórios para revisar:
- `plantio.data_plantio` — data real de plantio no campo (formato YYYY-MM-DD)
- `irrigacao.bomba_vazao_lpm` — vazão real da bomba em L/min
- `irrigacao.bomba_area_m2` — área real do canteiro em m²
- `arduino.habilitado` — `false` por enquanto, `true` quando Arduino estiver conectado
- `arduino.porta` — verificar com `ls /dev/ttyUSB*` ou `ls /dev/ttyACM*`

### 4. Testar o sistema sem hardware (modo dummy)
```bash
source venv/bin/activate
python3 cron_diario.py --teste
```
Deve imprimir: configuração lida, 14 modelos carregados, decisão tomada, arquivos salvos.

### 5. Configurar cron diário às 7h
```bash
crontab -e
```
Adicionar linha:
```
0 7 * * * cd /home/pi/irrigacao && /home/pi/irrigacao/venv/bin/python3 cron_diario.py >> logs/cron.log 2>&1
```
Verificar:
```bash
crontab -l
```

### 6. Criar serviço systemd para a API Flask (auto-start no boot)
Criar `/etc/systemd/system/irrigacao-api.service`:
```ini
[Unit]
Description=Irrigação ALMMo-0 — API Flask
After=network.target

[Service]
User=pi
WorkingDirectory=/home/pi/irrigacao
ExecStart=/home/pi/irrigacao/venv/bin/python3 api_rasp.py
Restart=always
RestartSec=10
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```
Ativar:
```bash
sudo systemctl daemon-reload
sudo systemctl enable irrigacao-api
sudo systemctl start irrigacao-api
sudo systemctl status irrigacao-api
```

Dashboard acessível em: `http://IP-DO-PI:5000/`

### 7. Verificar portas do Arduino (quando conectado)
```bash
ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null
```
Testar comunicação serial:
```bash
python3 -c "
import serial, time
s = serial.Serial('/dev/ttyUSB0', 9600, timeout=5)
time.sleep(1.5)
s.write(b'LEITURA\n')
time.sleep(0.5)
print(s.readline().decode().strip())
s.close()
"
```
Resposta esperada: `TENSAO:52.5`

Quando confirmado, editar config.json:
```json
"arduino": { "habilitado": true, "porta": "/dev/ttyUSB0" }
```

### 8. Checar logs após primeiro ciclo real
```bash
cat logs/cron.log
cat log_decisoes.csv
curl http://localhost:5000/api/status | python3 -m json.tool
```

---

## Protocolo de comunicação com o Arduino

| Direção         | Mensagem           | Significado                        |
|-----------------|--------------------|------------------------------------|
| Rasp → Arduino  | `LEITURA\n`        | Solicitar leitura do sensor        |
| Arduino → Rasp  | `TENSAO:52.5\n`    | Tensão do solo em kPa              |
| Rasp → Arduino  | `BOMBA:120\n`      | Acionar bomba por 120 segundos     |
| Arduino → Rasp  | `OK:BOMBA:120\n`   | Confirmação de acionamento         |

---

## Features do modelo (ordem obrigatória)

```python
x = [tensao_solo_kpa, chuva_acum_3d_mm, tmax_max_3d_c, dap, delta_tensao_kpa]
```

- `tensao_solo_kpa` — leitura direta do sensor (SEM shift — isso era só para simulação AquaCrop)
- `chuva_acum_3d_mm` — soma dos últimos 3 dias (NASA POWER API)
- `tmax_max_3d_c` — máxima dos últimos 3 dias (NASA POWER API)
- `dap` — dias após plantio (calculado automaticamente)
- `delta_tensao_kpa` — tensão_hoje − tensão_ontem (do log)

---

## Classes de decisão

| Classe | Nome              | Ação padrão | Cor       |
|--------|-------------------|-------------|-----------|
| C0     | Sem Irrigação     | 0 mm        | Verde     |
| C1     | Manutenção        | 5 mm        | Amarelo   |
| C2     | Irrigação Intensiva | 15 mm     | Vermelho  |

Volumes em `config.json → irrigacao`. Tempo de bomba calculado automaticamente:
`tempo_s = (mm × área_m²) / (vazão_lpm / 60)`

---

## Dependências Python

```
flask
flask-cors
requests
numpy
pyserial
```

Todas instaladas no venv. O cron deve usar o Python do venv, não o do sistema.

---

## O que NÃO fazer

- Não aplicar shift na tensão lida do sensor real (shift existe só nas simulações AquaCrop)
- Não sobrescrever `memoria_cold_start_v12_ensemble.pkl` sem backup
- Não rodar `cron_diario.py` sem `--teste` antes de confirmar que o PKL carrega corretamente
- Não mudar `arduino.habilitado` para `true` sem antes testar a comunicação serial manualmente
