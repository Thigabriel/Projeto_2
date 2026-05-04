// ============================================================
// teste_radio_arduino.ino
// Teste de comunicacao via modulo FPV de radio telemetria
//
// Conexao do Air Module ao Arduino:
//   VCC  → 5V
//   GND  → GND
//   TX   → pino 10 (RX do SoftwareSerial)
//   RX   → pino 11 (TX do SoftwareSerial)
//
// Ajuste BAUD_RADIO para o baud rate do seu modulo.
// Modulos FPV comuns: 57600 (padrao), 9600, 115200
// ============================================================

#include <SoftwareSerial.h>

// Pinos do Air Module
#define PIN_RX  10
#define PIN_TX  11

// Baud rate do modulo de radio — ajuste conforme o seu modulo
#define BAUD_RADIO  57600

// Baud rate do monitor serial (debug via USB, nao muda)
#define BAUD_USB    9600

SoftwareSerial radio(PIN_RX, PIN_TX);

unsigned long ultimo_heartbeat = 0;
int contador = 0;

void setup() {
  Serial.begin(BAUD_USB);
  radio.begin(BAUD_RADIO);

  Serial.println("==============================================");
  Serial.println("  Arduino — Teste Radio Telemetria");
  Serial.println("==============================================");
  Serial.print("  Air Module: pinos RX=");
  Serial.print(PIN_RX);
  Serial.print(" TX=");
  Serial.println(PIN_TX);
  Serial.print("  Baud radio: ");
  Serial.println(BAUD_RADIO);
  Serial.println("----------------------------------------------");
  Serial.println("  Enviando heartbeat a cada 3s...");
  Serial.println("  Responde PONG ao receber PING");
  Serial.println("==============================================");
}

void loop() {

  // ---- Recebe mensagem via radio ----
  if (radio.available()) {
    String msg = lerLinha();

    Serial.print("[RECEBIDO] ");
    Serial.println(msg);

    // Resposta ao PING
    if (msg.equalsIgnoreCase("PING")) {
      radio.println("PONG");
      Serial.println("[ENVIADO]  PONG");

    // Resposta ao TENSAO? (simula leitura do sensor)
    } else if (msg.equalsIgnoreCase("LEITURA")) {
      float tensao_simulada = 58.0 + random(-50, 150) / 10.0;
      String resp = "TENSAO:" + String(tensao_simulada, 1);
      radio.println(resp);
      Serial.print("[ENVIADO]  ");
      Serial.println(resp);

    // Eco generico
    } else {
      String eco = "ECO:" + msg;
      radio.println(eco);
      Serial.print("[ENVIADO]  ");
      Serial.println(eco);
    }
  }

  // ---- Heartbeat a cada 3 segundos ----
  unsigned long agora = millis();
  if (agora - ultimo_heartbeat >= 3000) {
    ultimo_heartbeat = agora;
    contador++;
    String hb = "ALIVE:" + String(contador);
    radio.println(hb);
    Serial.print("[HEARTBEAT] ");
    Serial.println(hb);
  }

  // ---- Passthrough: Monitor Serial → Radio ----
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    if (cmd.length() > 0) {
      radio.println(cmd);
      Serial.print("[USB->RADIO] ");
      Serial.println(cmd);
    }
  }
}

// Le uma linha do radio com timeout de 200ms
String lerLinha() {
  String resultado = "";
  unsigned long inicio = millis();
  while (millis() - inicio < 200) {
    if (radio.available()) {
      char c = radio.read();
      if (c == '\n') break;
      if (c != '\r') resultado += c;
    }
  }
  resultado.trim();
  return resultado;
}
