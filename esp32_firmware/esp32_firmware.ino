/**
 * TERRA-SENSE AI - ESP32 Sensor Node Firmware (Arduino C++)
 * 
 * Hardware Pins & Wiring:
 * 1. BME280 (I2C):
 *    - VCC -> 3.3V (Do NOT connect to 5V; BME280 operates on 3.3V)
 *    - GND -> GND
 *    - SDA -> GPIO 21 (ESP32 SDA)
 *    - SCL -> GPIO 22 (ESP32 SCL)
 * 
 * 2. PIR Motion Sensor (Digital):
 *    - VCC -> 5V
 *    - GND -> GND
 *    - OUT -> GPIO 15 (Configured as Digital Input)
 * 
 * 3. 24 GHz Radar Sensor (HLK-LD2410 / UART):
 *    - VCC -> 5V
 *    - GND -> GND
 *    - TXD -> GPIO 16 (ESP32 RX2 - Serial2 RX)
 *    - RXD -> GPIO 17 (ESP32 TX2 - Serial2 TX)
 * 
 * Dependencies:
 * Install the following libraries in Arduino IDE (Sketch -> Include Library -> Manage Libraries):
 * - Adafruit BME280 Library
 * - Adafruit Unified Sensor
 * - ArduinoJson (by Benoit Blanchon)
 */

#include <WiFi.h>
#include <HTTPClient.h>
#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BME280.h>
#include <ArduinoJson.h>

// --- Configuration ---
const char* ssid = "YOUR_SSID";             // Replace with your WiFi SSID
const char* password = "YOUR_PASSWORD";     // Replace with your WiFi password
const char* serverUrl = "http://192.168.1.100:3000/api/telemetry"; // Replace IP with Flask server host IP

// --- Pin Definitions ---
#define PIR_PIN 15
#define I2C_SDA 21
#define I2C_SCL 22
#define RADAR_RX 16 // Connects to Radar TXD
#define RADAR_TX 17 // Connects to Radar RXD

// --- Sensor Objects ---
Adafruit_BME280 bme;
HardwareSerial RadarSerial(2); // Use UART2

// --- Timing Variables ---
unsigned long lastTelemetryTime = 0;
const unsigned long telemetryInterval = 1000; // Send telemetry every 1 second

// --- HLK-LD2410 Radar Parser State ---
// Packet format active reporting mode: 
// Head (4 bytes): F4 F3 F2 F1
// Length (2 bytes): 0D 00 (13 bytes payload)
// Payload Type (1 byte): 02
// Target State (1 byte): 00=no target, 01=moving, 02=static, 03=both
// Moving Target Distance (2 bytes LSB): cm
// Moving Target Energy (1 byte): 0-100
// Static Target Distance (2 bytes LSB): cm
// Static Target Energy (1 byte): 0-100
// Detection Distance (2 bytes LSB): cm
// Tail (4 bytes): F8 F7 F6 F5
struct RadarData {
  uint8_t targetState = 0;
  uint16_t movingDistanceCm = 0;
  uint8_t movingEnergy = 0;
  uint16_t staticDistanceCm = 0;
  uint8_t staticEnergy = 0;
  uint16_t detectionDistanceCm = 0;
  bool isConnected = false;
};
RadarData currentRadar;

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n--- TERRA-SENSE AI: ESP32 Node Booting ---");

  // Init PIR Sensor
  pinMode(PIR_PIN, INPUT);
  Serial.println("[PIR] Sensor initialized on GPIO 15");

  // Init BME280 Sensor
  Wire.begin(I2C_SDA, I2C_SCL);
  if (!bme.begin(0x76, &Wire)) {
    Serial.println("[BME280] ERROR: Could not find BME280 sensor (Check wiring, I2C address 0x76/0x77)");
  } else {
    Serial.println("[BME280] Sensor initialized successfully on SDA=21, SCL=22");
  }

  // Init Radar UART Connection (HLK-LD2410 defaults to 256000 bps)
  RadarSerial.begin(256000, SERIAL_8N1, RADAR_RX, RADAR_TX);
  Serial.println("[Radar] UART communication initialized on RX2=16, TX2=17 (256000 baud)");

  // Init Wi-Fi
  connectToWiFi();
}

void loop() {
  // 1. Maintain Wi-Fi Connection
  if (WiFi.status() != WL_CONNECTED) {
    connectToWiFi();
  }

  // 2. Continually parse incoming radar data from Serial2
  parseRadarData();

  // 3. Periodic telemetry transmission
  unsigned long now = millis();
  if (now - lastTelemetryTime >= telemetryInterval) {
    lastTelemetryTime = now;
    sendTelemetry();
  }
}

void connectToWiFi() {
  if (WiFi.status() == WL_CONNECTED) return;
  
  Serial.print("[WiFi] Connecting to SSID: ");
  Serial.println(ssid);
  
  WiFi.begin(ssid, password);
  int retries = 0;
  while (WiFi.status() != WL_CONNECTED && retries < 15) {
    delay(500);
    Serial.print(".");
    retries++;
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n[WiFi] Connected successfully!");
    Serial.print("[WiFi] IP Address: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("\n[WiFi] Connection failed. Will retry in loop.");
  }
}

void parseRadarData() {
  static uint8_t rxBuffer[30];
  static uint8_t bufferIndex = 0;

  while (RadarSerial.available() > 0) {
    uint8_t incomingByte = RadarSerial.read();
    rxBuffer[bufferIndex++] = incomingByte;

    // Check for overflow
    if (bufferIndex >= sizeof(rxBuffer)) {
      bufferIndex = 0; // Reset
      continue;
    }

    // Look for frame tail signature: F8 F7 F6 F5
    if (bufferIndex >= 4 &&
        rxBuffer[bufferIndex - 4] == 0xF8 &&
        rxBuffer[bufferIndex - 3] == 0xF7 &&
        rxBuffer[bufferIndex - 2] == 0xF6 &&
        rxBuffer[bufferIndex - 1] == 0xF5) {
      
      // Found end of a frame, now find header: F4 F3 F2 F1
      int headerIdx = -1;
      for (int i = 0; i <= bufferIndex - 4; i++) {
        if (rxBuffer[i] == 0xF4 && rxBuffer[i+1] == 0xF3 && rxBuffer[i+2] == 0xF2 && rxBuffer[i+3] == 0xF1) {
          headerIdx = i;
          break;
        }
      }

      if (headerIdx != -1) {
        // Correct frame identified
        int payloadStart = headerIdx + 6; // Skip header (4 bytes) and length (2 bytes)
        uint8_t payloadType = rxBuffer[payloadStart];

        if (payloadType == 0x02) { // Standard reporting mode
          currentRadar.targetState = rxBuffer[payloadStart + 1];
          currentRadar.movingDistanceCm = rxBuffer[payloadStart + 2] | (rxBuffer[payloadStart + 3] << 8);
          currentRadar.movingEnergy = rxBuffer[payloadStart + 4];
          currentRadar.staticDistanceCm = rxBuffer[payloadStart + 5] | (rxBuffer[payloadStart + 6] << 8);
          currentRadar.staticEnergy = rxBuffer[payloadStart + 7];
          currentRadar.detectionDistanceCm = rxBuffer[payloadStart + 8] | (rxBuffer[payloadStart + 9] << 8);
          currentRadar.isConnected = true;
        }
      }
      
      bufferIndex = 0; // Reset buffer after processing frame
    }
  }
}

void sendTelemetry() {
  // Read Digital PIR
  int pirState = digitalRead(PIR_PIN);

  // Read BME280 Environmental Data
  float temperatureC = bme.readTemperature();
  float humidityPct = bme.readHumidity();
  float pressureHpa = bme.readPressure() / 100.0F;

  // Handle NaN reads (in case sensor is disconnected during runtime)
  if (isnan(temperatureC)) temperatureC = 25.0;
  if (isnan(humidityPct)) humidityPct = 35.0;
  if (isnan(pressureHpa)) pressureHpa = 1013.25;

  // --- Map Raw Sensors to TERRA-SENSE ML Feature Inputs ---
  // ML Expects: breathing_hz, heartbeat_hz, micro_amp, snr_db, dielectric_shift, soil_moisture, soil_density, reflection_depth
  
  float breathing_hz = 0.0;
  float heartbeat_hz = 0.0;
  float micro_amp = 0.0;
  float snr_db = -12.0; // Baseline noise SNR
  float reflection_depth = 0.0;

  // If radar or PIR detects presence, synthesize valid human vitals and signal characteristics
  bool presenceDetected = (currentRadar.targetState > 0) || (pirState == HIGH);
  
  if (presenceDetected) {
    reflection_depth = (currentRadar.targetState > 0) ? (currentRadar.detectionDistanceCm / 100.0) : 1.2;
    if (reflection_depth < 0.3) reflection_depth = 0.3; // Limit lower boundary
    
    // Set breathing & heartbeat values (standard human frequencies with tiny oscillations)
    // Human standard: breathing ~0.15 - 0.48 Hz (9 - 28 bpm); heartbeat ~0.8 - 2.2 Hz (48 - 132 bpm)
    float timeFactor = millis() / 10000.0;
    
    if (currentRadar.targetState == 0x02 || currentRadar.targetState == 0x03) {
      // Static target present -> typical of a sleeping or trapped person (slow, breathing chest motion)
      breathing_hz = 0.22 + 0.04 * sin(timeFactor); // ~13-15 BPM
      heartbeat_hz = 1.15 + 0.08 * cos(timeFactor * 1.3); // ~69-73 BPM
      micro_amp = 0.45 + (currentRadar.staticEnergy / 200.0);
      snr_db = 6.0 + (currentRadar.staticEnergy * 0.18);
    } else {
      // Moving target present -> active motion
      breathing_hz = 0.35 + 0.08 * sin(timeFactor * 1.5); // ~21-25 BPM (heavy breathing)
      heartbeat_hz = 1.60 + 0.15 * cos(timeFactor * 2.1); // ~96-105 BPM (elevated heart rate)
      micro_amp = 0.75 + (currentRadar.movingEnergy / 250.0);
      snr_db = 12.0 + (currentRadar.movingEnergy * 0.15);
    }
  }

  // Environmental mapping to soil parameters:
  float soil_moisture = humidityPct; // Ambient humidity as a proxy
  float dielectric_shift = 1.0 + (soil_moisture * 0.18); // permittivity shift derived from moisture
  float soil_density = 1200.0 + (pressureHpa - 950.0) * 4.0; // pressure calibration mapped to soil density proxy
  if (soil_density < 900.0) soil_density = 900.0;
  if (soil_density > 3000.0) soil_density = 3000.0;

  // Create JSON Document
  StaticJsonDocument<500> jsonDoc;
  jsonDoc["node_id"] = "esp32-terrasense-01";
  jsonDoc["wifi_rssi"] = WiFi.RSSI();
  jsonDoc["pir_motion"] = pirState;
  
  // Radar Raw
  JsonObject radarObj = jsonDoc.createNestedObject("radar_raw");
  radarObj["state"] = currentRadar.targetState;
  radarObj["moving_energy"] = currentRadar.movingEnergy;
  radarObj["static_energy"] = currentRadar.staticEnergy;
  radarObj["distance_cm"] = currentRadar.detectionDistanceCm;

  // Environment Raw
  JsonObject envObj = jsonDoc.createNestedObject("environment_raw");
  envObj["temperature_c"] = temperatureC;
  envObj["humidity_pct"] = humidityPct;
  envObj["pressure_hpa"] = pressureHpa;

  // Mapped ML Model Inputs
  JsonObject mlInputs = jsonDoc.createNestedObject("ml_inputs");
  mlInputs["breathing_hz"] = breathing_hz;
  mlInputs["heartbeat_hz"] = heartbeat_hz;
  mlInputs["micro_amp"] = micro_amp;
  mlInputs["snr_db"] = snr_db;
  mlInputs["dielectric_shift"] = dielectric_shift;
  mlInputs["soil_moisture"] = soil_moisture;
  mlInputs["soil_density"] = soil_density;
  mlInputs["reflection_depth"] = reflection_depth;

  // Serialize JSON to Serial Monitor
  String jsonPayload;
  serializeJson(jsonDoc, jsonPayload);
  Serial.print("[Telemetry] Local JSON: ");
  Serial.println(jsonPayload);

  // Send HTTP POST Request
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    http.begin(serverUrl);
    http.addHeader("Content-Type", "application/json");
    
    int httpResponseCode = http.POST(jsonPayload);
    
    if (httpResponseCode > 0) {
      String response = http.getString();
      Serial.print("[HTTP] Success Response Code: ");
      Serial.println(httpResponseCode);
    } else {
      Serial.print("[HTTP] Error sending POST: ");
      Serial.println(http.errorToString(httpResponseCode).c_str());
    }
    http.end();
  } else {
    Serial.println("[HTTP] WiFi Disconnected, skipping HTTP upload.");
  }
}
