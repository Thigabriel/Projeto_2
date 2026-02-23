# Sistema de Irrigação Preditiva com ALMMo-0
## Guia de Instalação e Uso — Raspberry Pi 4B

**Projecto:** Controlo Preditivo de Irrigação com Edge AI  
**Cultura:** Tomate | **Solo:** Franco-Arenoso | **Local:** Imperatriz-MA  
**Fase actual:** Validação Hardware-in-the-Loop (HIL) com sensor simulado

---

## Índice

1. [O que está neste projecto](#1-o-que-está-neste-projecto)
2. [Requisitos](#2-requisitos)
3. [Transferir os ficheiros para o Raspberry Pi](#3-transferir-os-ficheiros-para-o-raspberry-pi)
4. [Instalar dependências](#4-instalar-dependências)
5. [Configuração inicial](#5-configuração-inicial)
6. [Executar a validação HIL](#6-executar-a-validação-hil)
7. [Interpretar os resultados](#7-interpretar-os-resultados)
8. [Estrutura dos ficheiros](#8-estrutura-dos-ficheiros)
9. [Erros comuns e soluções](#9-erros-comuns-e-soluções)
10. [Próximos passos — sensor físico](#10-próximos-passos--sensor-físico)

---

## 1. O que está neste projecto

Este sistema toma decisões diárias de irrigação usando um modelo de inteligência artificial (ALMMo-0) que aprende com o campo ao longo do tempo.

**Fase HIL (actual):** o Raspberry Pi corre o código e o modelo real. O sensor de solo é simulado por software. Os dados meteorológicos são **reais** (Open-Meteo, Imperatriz-MA).

**Fase Campo (próxima):** substitui apenas o módulo do sensor — o resto não muda.

---

## 2. Requisitos

### Hardware
- Raspberry Pi 4B (qualquer quantidade de RAM)
- Cartão SD com Raspberry Pi OS (Bookworm ou Bullseye)
- Acesso à internet (WiFi ou Ethernet) — para Open-Meteo

### Software
- Python 3.10 ou superior (já incluído no Raspberry Pi OS)
- pip3

### Verificar versão do Python
```bash
python3 --version
# Deve mostrar Python 3.10.x ou superior
```

---

## 3. Transferir os ficheiros para o Raspberry Pi

### Opção A — Via SCP (do computador para o Pi)
No seu computador (não no Pi), abra o terminal e execute:

```bash
# Criar pasta no Pi
ssh pi@<IP-DO-PI> "mkdir -p /home/pi/irrigacao"

# Copiar todos os ficheiros de uma vez
scp config_hil.py almmo0.py simulador_sensor.py main_hil.py \
    memoria_cold_start_v7.pkl \
    pi@<IP-DO-PI>:/home/pi/irrigacao/
```

> Substitua `<IP-DO-PI>` pelo endereço IP do seu Raspberry Pi.  
> Para descobrir o IP: no Pi, execute `hostname -I`

### Opção B — Via pendrive
Copie os ficheiros para um pendrive, ligue ao Pi e execute:

```bash
# Ver onde o pendrive foi montado (normalmente /media/pi/NOME)
ls /media/pi/

# Copiar para o projecto
cp /media/pi/<NOME-PENDRIVE>/*.py /home/pi/irrigacao/
cp /media/pi/<NOME-PENDRIVE>/memoria_cold_start_v7.pkl /home/pi/irrigacao/
```

### Ficheiros obrigatórios
Confirme que estes ficheiros estão em `/home/pi/irrigacao/`:

```
memoria_cold_start_v7.pkl   ← modelo treinado (obrigatório)
config_hil.py
almmo0.py
simulador_sensor.py
main_hil.py
```

---

## 4. Instalar dependências

Abra um terminal no Raspberry Pi (via SSH ou directamente) e execute:

```bash
# Actualizar o sistema primeiro
sudo apt update && sudo apt upgrade -y

# Instalar numpy e requests
pip3 install numpy requests --break-system-packages

# Instalar matplotlib (para gráficos — opcional mas recomendado)
pip3 install matplotlib --break-system-packages
```

### Verificar instalação
```bash
python3 -c "import numpy, requests, matplotlib; print('OK')"
# Deve imprimir: OK
```

---

## 5. Configuração inicial

### Entrar na pasta do projecto
```bash
cd /home/pi/irrigacao
```

### Editar a data de plantio em config_hil.py
```bash
nano config_hil.py
```

Altere a linha:
```python
DATA_PLANTIO = "2025-06-01"   # ← mudar para a data real de plantio
```

Guardar e sair: `Ctrl+X` → `Y` → `Enter`

### Verificar que o modelo carrega correctamente
```bash
python3 -c "
from almmo0 import carregar_modelo
m, s = carregar_modelo('memoria_campo.pkl', 'memoria_cold_start_v7.pkl')
print('Modelo OK:', m.info())
"
```

Deve aparecer algo como:
```
[ALMMo0] Carregado (inicial): ALMMo-0 | 41 regras | C0:21 C1:10 C2:10 | r=0.25 | seen=2186
Modelo OK: ALMMo-0 | 41 regras | C0:21 C1:10 C2:10 | r=0.25 | seen=2186
```

---

## 6. Executar a validação HIL

### 6.1 — Executar todos os cenários (recomendado)
```bash
cd /home/pi/irrigacao
python3 main_hil.py
```

A execução completa demora cerca de **30–60 segundos**.  
Verá o progresso de cada cenário em tempo real no terminal.

### 6.2 — Executar apenas um cenário específico
```bash
# Cenário 1 — Seca Progressiva (14 dias)
python3 main_hil.py --cenario 1

# Cenário 2 — Evento de Chuva (10 dias)
python3 main_hil.py --cenario 2

# Cenário 3 — Aprendizado Online (21 dias) ← mais importante para o TCC
python3 main_hil.py --cenario 3

# Cenário 4 — Stress Crítico e Recuperação (10 dias)
python3 main_hil.py --cenario 4
```

### 6.3 — Executar sem internet (modo offline)
Se o Pi não tiver acesso à internet, use o modo offline.  
O sistema usará valores de fallback para a meteorologia (chuva=0, tmax=35°C).

```bash
python3 main_hil.py --sem-api
```

> **Nota:** a primeira execução com internet guarda um ficheiro `cache_meteo.json`.  
> Nas execuções seguintes sem internet, esse cache é usado automaticamente.

---

## 7. Interpretar os resultados

### 7.1 — Output no terminal
Durante a execução verá uma tabela como esta:

```
  Dia |   kPa | Cls | Irr  | Motivo18h                    | Regras         | Flags
    1 |   47.3 | C0→C0 |  0.0mm | executado_normal             |  41r C0:21 C1:10 C2:10 |
    2 |   61.9 | C0→C0 |  0.0mm | executado_normal             |  41r C0:21 C1:10 C2:10 | 📚
    3 |   75.2 | C1→C1 |  3.0mm | executado_normal             |  42r C0:21 C1:10 C2:11 |
    4 |   91.7 | C2→C0 |  0.0mm | cancelado_chuva_solo_ok      |  42r ...               | 🌧️
```

**Legenda das colunas:**
| Coluna | Significado |
|--------|-------------|
| `kPa` | Tensão do solo às 06h (leitura do sensor) |
| `C0→C1` | Decisão da manhã → decisão final (após dupla confirmação 18h) |
| `Irr mm` | Volume de irrigação aplicado |
| `Motivo18h` | Por que a decisão final foi essa |
| `Regras` | Total de regras e distribuição por classe |

**Legenda dos motivos:**
| Motivo | Significado |
|--------|-------------|
| `executado_normal` | Não choveu — decisão da manhã executada |
| `cancelado_chuva_solo_ok` | Choveu E solo recuperou — irrigação cancelada |
| `mantido_stress_persistente` | Choveu mas solo ainda em stress — irrigação mantida |

**Legenda das flags:**
| Flag | Significado |
|------|-------------|
| ⚠️ | Tensão acima de 136 kPa (fora do range de treino do modelo) |
| 📚 | Feedback activado — modelo aprendeu neste ciclo |
| 🌧️ | Sensor de chuva detectou precipitação |

---

### 7.2 — Ficheiros gerados

Após a execução, encontrará em `/home/pi/irrigacao/resultados_hil/`:

```
resultados_hil/
├── cenario_1_seca.csv           ← dados diários do cenário 1
├── cenario_2_chuva.csv          ← dados diários do cenário 2
├── cenario_3_feedback.csv       ← dados diários do cenário 3
├── cenario_4_stress.csv         ← dados diários do cenário 4
├── relatorio_hil.md             ← relatório completo (para o TCC)
└── graficos_hil/
    ├── cenario_1_seca_progressiva.png
    ├── cenario_2_evento_de_chuva.png
    ├── cenario_3_aprendizado_online_por_feedback.png
    └── cenario_4_stress_crítico_e_recuperação.png
```

### 7.3 — Ver o relatório
```bash
cat /home/pi/irrigacao/resultados_hil/relatorio_hil.md
```

Ou copie para o seu computador para abrir num editor:
```bash
# No computador:
scp pi@<IP-DO-PI>:/home/pi/irrigacao/resultados_hil/relatorio_hil.md ./
scp -r pi@<IP-DO-PI>:/home/pi/irrigacao/resultados_hil/graficos_hil/ ./
```

### 7.4 — O que procurar no Cenário 3 (métrica-chave do TCC)
O Cenário 3 demonstra que o modelo aprende em campo. No relatório, procure a tabela:

```
| Semana | Dias | C0 | C1+C2 | % C1+C2 |
|--------|------|----|-------|---------|
| Semana 1 |  7 |  6 |     1 |     14% |
| Semana 2 |  7 |  4 |     3 |     43% |
| Semana 3 |  7 |  2 |     5 |     71% |
```

**PASS** = % C1+C2 aumentou da semana 1 para a semana 3.  
Isto demonstra que o mecanismo de feedback está a corrigir o viés C0 do cold start.

---

## 8. Estrutura dos ficheiros

```
/home/pi/irrigacao/
│
├── memoria_cold_start_v7.pkl   ← modelo inicial (NÃO apagar)
├── memoria_campo.pkl           ← modelo actualizado (gerado após execução)
│
├── config_hil.py               ← parâmetros editáveis
├── almmo0.py                   ← classe do modelo ALMMo-0
├── simulador_sensor.py         ← sensor simulado + dupla confirmação 18h
├── main_hil.py                 ← script principal
│
├── cache_meteo.json            ← cache da API meteorológica (gerado automaticamente)
│
└── resultados_hil/             ← gerado após execução
    ├── *.csv
    ├── relatorio_hil.md
    └── graficos_hil/
```

> **Importante:** `memoria_cold_start_v7.pkl` é o modelo original de treino.  
> Nunca é sobrescrito. O sistema actualizado em campo é guardado em `memoria_campo.pkl`.

---

## 9. Erros comuns e soluções

### ❌ `ModuleNotFoundError: No module named 'numpy'`
```bash
pip3 install numpy --break-system-packages
```

### ❌ `RuntimeError: Nenhum pkl válido encontrado`
O ficheiro `memoria_cold_start_v7.pkl` não está na pasta do projecto.
```bash
ls /home/pi/irrigacao/*.pkl
# Se não aparecer nada, copiar o pkl novamente (ver Passo 3)
```

### ❌ `[METEO] Falha API` durante a execução
Normal se não houver internet. O sistema usa o cache ou valores de fallback automaticamente. Não afecta a validação HIL.

### ❌ Execução muito lenta (>5 min)
Verifique se há outros processos a consumir CPU:
```bash
top
# Pressionar Q para sair
```
Se necessário, reiniciar o Pi e executar novamente.

### ❌ Gráficos não gerados (`matplotlib não disponível`)
```bash
pip3 install matplotlib --break-system-packages
```
Se o erro persistir no Pi com pouca memória, execute no computador:
```bash
# Copiar CSVs para o computador e gerar gráficos lá
scp -r pi@<IP-DO-PI>:/home/pi/irrigacao/resultados_hil/ ./
```

### ❌ Permissão negada ao executar o script
```bash
chmod +x /home/pi/irrigacao/main_hil.py
```

---

## 10. Próximos passos — sensor físico

Quando o sensor capacitivo físico for adquirido e calibrado com tensiômetro de referência, **apenas um ficheiro muda**: `simulador_sensor.py` é substituído por `sensores.py` com a leitura GPIO real. O `main_hil.py`, `almmo0.py`, `config_hil.py` e o modelo pkl ficam intactos.

O processo de calibração do sensor está documentado no `relatorio_hil.md` gerado após a execução HIL.

---

## Referência rápida — comandos mais usados

```bash
# Ir para a pasta do projecto
cd /home/pi/irrigacao

# Executar validação completa
python3 main_hil.py

# Executar apenas o cenário 3 (aprendizado online)
python3 main_hil.py --cenario 3

# Executar sem internet
python3 main_hil.py --sem-api

# Ver relatório
cat resultados_hil/relatorio_hil.md

# Ver estado actual do modelo
python3 -c "
from almmo0 import carregar_modelo
m, s = carregar_modelo('memoria_campo.pkl', 'memoria_cold_start_v7.pkl')
print(m.info())
"

# Copiar resultados para o computador (executar NO computador)
scp -r pi@<IP-DO-PI>:/home/pi/irrigacao/resultados_hil/ ./resultados_hil/
```

---

*Documentação gerada para o projecto de TCC — Sistema de Controlo Preditivo de Irrigação com Edge AI (ALMMo-0)*
