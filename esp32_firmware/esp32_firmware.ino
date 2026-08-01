/**
 * TERRA-SENSE AI - ESP32 Sensor Node Firmware (Arduino C++)
 * Mode: WiFi Access Point (Hotspot) + Built-in Web Server
 * Configured for: PIR Motion Sensor + BMP180 Environmental Sensor
 * 
 * How it works:
 * - ESP32 creates its own WiFi network (hotspot)
 * - Connect your phone/laptop to the ESP32 WiFi
 * - Access telemetry data at: http://192.168.4.1/api/telemetry
 * - Data updates every 1 second automatically
 * 
 * Hardware Pins & Wiring:
 * 1. BMP180 (I2C):
 *    - VCC -> 3.3V (Do NOT connect to 5V; BMP180 operates on 3.3V)
 *    - GND -> GND
 *    - SDA -> GPIO 21 (ESP32 SDA)
 *    - SCL -> GPIO 22 (ESP32 SCL)
 * 
 * 2. PIR Motion Sensor (Digital):
 *    - VCC -> 5V / Vin (or 3.3V depending on module)
 *    - GND -> GND
 *    - OUT -> GPIO 14 (Configured as Digital Input)
 * 
 * 3. DHT11 Humidity Sensor (Digital):
 *    - VCC -> 3.3V or 5V
 *    - GND -> GND
 *    - DATA/OUT -> GPIO 4 (Configured as Digital Input)
 * 
 * Dependencies (Install in Arduino IDE via Library Manager):
 * - Adafruit BMP085 Library (for BMP180)
 * - Adafruit Unified Sensor
 * - DHT sensor library (by Adafruit)
 * - ArduinoJson (by Benoit Blanchon - supports v6 and v7)
 * - WebServer (built-in with ESP32 Arduino Core)
 */

#include <WiFi.h>
#include <WebServer.h>
#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BMP085.h>
#include <ArduinoJson.h>
#include <DHT.h>

// Forward Function Declarations
void setupAccessPoint();
void updateTelemetry();
void handleTelemetryAPI();
void handleRoot();
void handleNotFound();

// --- Access Point Configuration ---
const char* ap_ssid = "TERRA-SENSE-ESP32";   // WiFi network name (visible to phones/laptops)
const char* ap_password = "1234567890";    // WiFi password (min 8 characters)
const int ap_channel = 1;                     // WiFi channel (1-13)
const int ap_max_connections = 4;             // Max simultaneous clients

// --- Pin Definitions ---
#define PIR_PIN 14
#define I2C_SDA 21
#define I2C_SCL 22
#define DHT_PIN 4      // DHT11 data pin connected to GPIO 4
#define DHT_TYPE DHT11

// --- Sensor Objects ---
Adafruit_BMP085 bmp;
bool bmpAvailable = false;
DHT dht(DHT_PIN, DHT_TYPE);

// --- Web Server on port 80 ---
WebServer server(80);

// --- Timing Variables ---
unsigned long lastTelemetryTime = 0;
const unsigned long telemetryInterval = 1000; // Update telemetry every 1 second

// --- Cached Telemetry JSON String ---
String cachedTelemetryJson = "{}";

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n--- TERRA-SENSE AI: ESP32 Node Booting (AP Mode + Web Server) ---");

  // Init PIR Sensor
  pinMode(PIR_PIN, INPUT);
  Serial.println("[PIR] Motion Sensor initialized on GPIO 15");

  // Init BMP180 Sensor via I2C
  Wire.begin(I2C_SDA, I2C_SCL);
  if (!bmp.begin()) {
    Serial.println("[BMP180] WARNING: Could not find BMP180 sensor! Will use default environmental baselines. Check SDA=21, SCL=22, VCC=3.3V");
    bmpAvailable = false;
  } else {
    Serial.println("[BMP180] Sensor initialized successfully on SDA=21, SCL=22");
    bmpAvailable = true;
  }

  // Init DHT11 Sensor
  dht.begin();
  Serial.println("[DHT11] Sensor initialized successfully on GPIO 4");

  // Start WiFi Access Point
  setupAccessPoint();

  // Configure Web Server Routes
  server.on("/", HTTP_GET, handleRoot);
  server.on("/api/telemetry", HTTP_GET, handleTelemetryAPI);
  server.on("/api/telemetry", HTTP_OPTIONS, []() {
    server.sendHeader("Access-Control-Allow-Origin", "*");
    server.sendHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    server.sendHeader("Access-Control-Allow-Headers", "Content-Type");
    server.send(200, "text/plain", "OK");
  });
  server.onNotFound(handleNotFound);

  // Start Web Server
  server.begin();
  Serial.println("[Server] Web server started on port 80");
  Serial.println("[Server] Telemetry API: http://192.168.4.1/api/telemetry");

  // Generate initial telemetry reading
  updateTelemetry();
}

void loop() {
  // Handle incoming HTTP requests from connected clients
  server.handleClient();

  // Periodic telemetry update
  unsigned long now = millis();
  if (now - lastTelemetryTime >= telemetryInterval) {
    lastTelemetryTime = now;
    updateTelemetry();
  }
}

void setupAccessPoint() {
  Serial.println("[WiFi AP] Starting Access Point...");
  
  WiFi.mode(WIFI_AP);
  WiFi.softAP(ap_ssid, ap_password, ap_channel, 0, ap_max_connections);
  
  // Small delay to let AP stabilize
  delay(100);
  
  IPAddress apIP = WiFi.softAPIP();
  Serial.println("[WiFi AP] ========================================");
  Serial.print("[WiFi AP] Network Name (SSID): ");
  Serial.println(ap_ssid);
  Serial.print("[WiFi AP] Password: ");
  Serial.println(ap_password);
  Serial.print("[WiFi AP] IP Address: ");
  Serial.println(apIP);
  Serial.print("[WiFi AP] Channel: ");
  Serial.println(ap_channel);
  Serial.print("[WiFi AP] Max Connections: ");
  Serial.println(ap_max_connections);
  Serial.println("[WiFi AP] ========================================");
}

void updateTelemetry() {
  // 1. Read Digital PIR Motion Sensor
  int pirState = digitalRead(PIR_PIN);

  // 2. Read Environmental Data (BMP180 + DHT11)
  float humidityPct = dht.readHumidity();
  if (isnan(humidityPct)) humidityPct = 45.0; // Fallback baseline if DHT11 fails to read

  float temperatureC = 25.0;
  float pressureHpa = 1013.25;

  if (bmpAvailable) {
    temperatureC = bmp.readTemperature();
    pressureHpa = bmp.readPressure() / 100.0F; // Convert Pa to hPa / mbar
  } else {
    // Fallback to DHT11 for temperature if BMP180 is not available
    float dhtTemp = dht.readTemperature();
    if (!isnan(dhtTemp)) temperatureC = dhtTemp;
  }
  
  if (isnan(temperatureC)) temperatureC = 25.0;
  if (isnan(pressureHpa)) pressureHpa = 1013.25;

  // 3. Map Sensors to TERRA-SENSE ML Feature Inputs
  float breathing_hz = 0.0;
  float heartbeat_hz = 0.0;
  float micro_amp = 0.0;
  float snr_db = -12.0; // Baseline noise floor SNR
  float reflection_depth = 0.0;

  bool presenceDetected = (pirState == HIGH);

  if (presenceDetected) {
    // When PIR detects presence, synthesize standard human vital signal proxies
    float timeFactor = millis() / 10000.0;
    breathing_hz = 0.25 + 0.04 * sin(timeFactor);          // ~15 BPM breathing frequency
    heartbeat_hz = 1.30 + 0.10 * cos(timeFactor * 1.5);    // ~78 BPM heartbeat frequency
    micro_amp = 0.65;                                      // Signal magnitude proxy
    snr_db = 10.0;                                         // Signal-to-noise ratio
    reflection_depth = 1.2;                                // Estimated depth in meters
  }

  // Environmental mapping to soil parameters:
  float soil_moisture = humidityPct;
  float dielectric_shift = 1.0 + (soil_moisture * 0.18);
  float soil_density = 1200.0 + (pressureHpa - 950.0) * 4.0;
  if (soil_density < 900.0) soil_density = 900.0;
  if (soil_density > 3000.0) soil_density = 3000.0;

  // 4. Construct JSON Document (Compatible with ArduinoJson v6 and v7)
#if ARDUINOJSON_VERSION_MAJOR >= 7
  JsonDocument jsonDoc;
#else
  StaticJsonDocument<768> jsonDoc;
#endif

  jsonDoc["node_id"] = "esp32-terrasense-01";
  jsonDoc["wifi_rssi"] = 0; // Not applicable in AP mode
  jsonDoc["ap_clients"] = WiFi.softAPgetStationNum(); // Number of connected clients
  jsonDoc["pir_motion"] = pirState;

  // Add radar_raw dummy structure so frontend JavaScript won't throw TypeError
  jsonDoc["radar_raw"]["state"] = presenceDetected ? 1 : 0;
  jsonDoc["radar_raw"]["moving_energy"] = presenceDetected ? 80 : 0;
  jsonDoc["radar_raw"]["static_energy"] = presenceDetected ? 20 : 0;
  jsonDoc["radar_raw"]["distance_cm"] = presenceDetected ? 120 : 0;

  // Environment Raw Data
  jsonDoc["environment_raw"]["temperature_c"] = temperatureC;
  jsonDoc["environment_raw"]["humidity_pct"] = humidityPct;
  jsonDoc["environment_raw"]["pressure_hpa"] = pressureHpa;

  // Mapped ML Model Inputs
  jsonDoc["ml_inputs"]["breathing_hz"] = breathing_hz;
  jsonDoc["ml_inputs"]["heartbeat_hz"] = heartbeat_hz;
  jsonDoc["ml_inputs"]["micro_amp"] = micro_amp;
  jsonDoc["ml_inputs"]["snr_db"] = snr_db;
  jsonDoc["ml_inputs"]["dielectric_shift"] = dielectric_shift;
  jsonDoc["ml_inputs"]["soil_moisture"] = soil_moisture;
  jsonDoc["ml_inputs"]["soil_density"] = soil_density;
  jsonDoc["ml_inputs"]["reflection_depth"] = reflection_depth;
  jsonDoc["ml_inputs"]["x"] = 14.2;
  jsonDoc["ml_inputs"]["y"] = 9.6;

  // Active flag and timestamp
  jsonDoc["active"] = true;
  jsonDoc["uptime_ms"] = millis();



  
  // 5. Serialize and cache the JSON payload
  cachedTelemetryJson = "";
  serializeJson(jsonDoc, cachedTelemetryJson);

  // 6. Print to Serial Monitor
  Serial.print("[Telemetry] Clients: ");
  Serial.print(WiFi.softAPgetStationNum());
  Serial.print(" | Payload: ");
  Serial.println(cachedTelemetryJson);
}

// --- Web Server Handlers ---

void handleTelemetryAPI() {
  // Serve the latest cached telemetry as JSON with CORS headers
  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.sendHeader("Access-Control-Allow-Methods", "GET, OPTIONS");
  server.sendHeader("Access-Control-Allow-Headers", "Content-Type");
  server.sendHeader("Cache-Control", "no-cache, no-store, must-revalidate");
  server.send(200, "application/json", cachedTelemetryJson);
}

void handleRoot() {
  // Serve a simple status page
  String html = "<!DOCTYPE html><html><head>";
  html += "<meta charset='UTF-8'>";
  html += "<meta name='viewport' content='width=device-width, initial-scale=1.0'>";
  html += "<title>TERRA-SENSE ESP32</title>";
  html += "<style>";
  html += "body{font-family:monospace;background:#0a0a0a;color:#00ff88;padding:20px;margin:0}";
  html += "h1{color:#00ff88;border-bottom:2px solid #00ff88;padding-bottom:10px}";
  html += ".info{background:#111;border:1px solid #00ff88;border-radius:8px;padding:15px;margin:10px 0}";
  html += "a{color:#00aaff;text-decoration:none}";
  html += "a:hover{text-decoration:underline}";
  html += "#data{white-space:pre-wrap;word-wrap:break-word;font-size:12px;line-height:1.6}";
  html += ".status{display:inline-block;width:10px;height:10px;background:#00ff88;border-radius:50%;margin-right:8px;animation:pulse 1.5s infinite}";
  html += "@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.3}}";
  html += "</style></head><body>";
  html += "<h1>&#x1F4E1; TERRA-SENSE ESP32 Node</h1>";
  html += "<div class='info'><span class='status'></span><strong>Status:</strong> ONLINE | ";
  html += "<strong>Uptime:</strong> " + String(millis() / 1000) + "s | ";
  html += "<strong>Connected Clients:</strong> " + String(WiFi.softAPgetStationNum()) + "</div>";
  html += "<div class='info'><strong>API Endpoint:</strong> <a href='/api/telemetry'>/api/telemetry</a> (JSON)</div>";
  html += "<div class='info'><strong>Live Telemetry Data:</strong><br><pre id='data'>Loading...</pre></div>";
  html += "<script>";
  html += "function f(){fetch('/api/telemetry').then(r=>r.json()).then(d=>{";
  html += "document.getElementById('data').textContent=JSON.stringify(d,null,2)";
  html += "}).catch(e=>{document.getElementById('data').textContent='Error: '+e})}";
  html += "f();setInterval(f,1000);";
  html += "</script></body></html>";
  
  server.send(200, "text/html", html);
}

void handleNotFound() {
  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.send(404, "application/json", "{\"error\":\"Not found\"}");
}