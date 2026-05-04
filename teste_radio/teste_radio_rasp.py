#!/usr/bin/env python3
"""
============================================================
teste_radio_rasp.py — Teste de comunicacao via radio telemetria
Ground Module conectado ao Raspberry Pi via USB

Uso:
  python3 teste_radio_rasp.py
  python3 teste_radio_rasp.py --porta /dev/ttyUSB1
  python3 teste_radio_rasp.py --baud 9600

Fluxo:
  1. Conecta ao Ground Module
  2. Roda teste automatico (3 PINGs + 1 LEITURA)
  3. Entra em modo interativo — voce digita, o Arduino responde
============================================================
"""

import serial
import serial.tools.list_ports
import threading
import time
import argparse

# ============================================================
# CONFIGURACAO
# ============================================================
PORTA_PADRAO = '/dev/ttyUSB0'
BAUD_PADRAO  = 57600    # ajuste conforme o seu modulo
TIMEOUT      = 2.0

# ============================================================
# RECEPCAO EM BACKGROUND
# ============================================================

recebendo = True

def thread_recepcao(ser):
    """Imprime tudo que chega do Arduino em background."""
    global recebendo
    while recebendo:
        try:
            if ser.in_waiting:
                linha = ser.readline().decode('utf-8', errors='replace').strip()
                if linha:
                    print(f"\n  [ARDUINO] {linha}")
        except serial.SerialException:
            break
        except Exception:
            pass
        time.sleep(0.01)


def enviar(ser, msg):
    """Envia mensagem e imprime confirmacao."""
    payload = (msg + '\n').encode()
    ser.write(payload)
    print(f"  [RASP]    {msg}")


def listar_portas():
    """Lista portas seriais disponiveis."""
    portas = list(serial.tools.list_ports.comports())
    if not portas:
        print("  Nenhuma porta serial encontrada.")
    else:
        print("  Portas disponiveis:")
        for p in portas:
            print(f"    {p.device}  —  {p.description}")


# ============================================================
# MAIN
# ============================================================

def main():
    global recebendo

    parser = argparse.ArgumentParser(description='Teste radio telemetria Raspberry Pi')
    parser.add_argument('--porta', default=PORTA_PADRAO, help=f'Porta serial (padrao: {PORTA_PADRAO})')
    parser.add_argument('--baud',  type=int, default=BAUD_PADRAO, help=f'Baud rate (padrao: {BAUD_PADRAO})')
    args = parser.parse_args()

    print()
    print("=" * 54)
    print("  Teste — Radio Telemetria FPV")
    print("=" * 54)
    print(f"  Porta : {args.porta}")
    print(f"  Baud  : {args.baud}")
    print()

    # Tenta conectar
    try:
        ser = serial.Serial(args.porta, args.baud, timeout=TIMEOUT)
        time.sleep(1.5)   # aguarda estabilizar
        ser.reset_input_buffer()
    except serial.SerialException as e:
        print(f"  ERRO ao conectar: {e}")
        print()
        listar_portas()
        print()
        print("  Dica: verifique se o Ground Module esta plugado e use --porta /dev/ttyUSB1 se necessario.")
        return

    print(f"  Conectado em {args.porta} @ {args.baud} baud")
    print()

    # Inicia thread de recepcao
    t = threading.Thread(target=thread_recepcao, args=(ser,), daemon=True)
    t.start()

    # --------------------------------------------------------
    # 1. TESTE AUTOMATICO
    # --------------------------------------------------------
    print("-" * 54)
    print("  FASE 1 — Teste automatico")
    print("-" * 54)
    print()

    # 3 PINGs com intervalo
    for i in range(1, 4):
        enviar(ser, "PING")
        time.sleep(2.5)

    # Solicita leitura de tensao simulada
    print()
    enviar(ser, "LEITURA")
    time.sleep(2.5)

    # Mensagem personalizada
    enviar(ser, "OLA ARDUINO")
    time.sleep(2.5)

    # --------------------------------------------------------
    # 2. MODO INTERATIVO
    # --------------------------------------------------------
    print()
    print("-" * 54)
    print("  FASE 2 — Modo interativo")
    print("  Digite mensagens para enviar ao Arduino.")
    print("  Comandos especiais:")
    print("    PING     → espera PONG")
    print("    LEITURA  → Arduino simula leitura de tensao")
    print("    SAIR     → encerra o teste")
    print("-" * 54)
    print()

    try:
        while True:
            try:
                msg = input("  > ").strip()
            except EOFError:
                break

            if not msg:
                continue

            if msg.upper() == 'SAIR':
                break

            enviar(ser, msg)
            time.sleep(0.3)   # pequena pausa para recepcao

    except KeyboardInterrupt:
        pass

    # Encerra
    recebendo = False
    time.sleep(0.2)
    ser.close()

    print()
    print("=" * 54)
    print("  Teste encerrado.")
    print("=" * 54)
    print()


if __name__ == '__main__':
    main()
