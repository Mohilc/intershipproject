/**
 * TERRA-SENSE AI - ESP32 Sensor Node Firmware (Arduino C++)
 * Mode: WiFi Access Point (Hotspot) + Built-in Web Server
 * Configured for: PIR Motion Sensor + BME690 Multi-Environmental Sensor
 * 
 * How it works:
 * - ESP32 creates its own WiFi network (hotspot)
 * - Connect your phone/laptop to the ESP32 WiFi
 * - Access telemetry data at: http://192.168.4.1/api/telemetry
 * - Data updates every 1 second automatically
 * 
 * Hardware Pins & Wiring:
 * 1. BME690 (I2C):
 *    - VCC -> 3.3V (Do NOT connect to 5V; BME690 operates on 3.3V)
 *    - GND -> GND
 *    - SDA -> GPIO 21 (ESP32 SDA)
 *    - SCL -> GPIO 22 (ESP32 SCL)
 * 
 * 2. PIR Motion Sensor (Digital):
 *    - VCC -> 5V / Vin (or 3.3V depending on module)
 *    - GND -> GND
 *    - OUT -> GPIO 14 (Configured as Digital Input)
 * 
 * Dependencies (Install in Arduino IDE via Library Manager):
 * - 7Semi BME690 Library (by 7Semi-solutions)
 * - Adafruit Unified Sensor
 * - ArduinoJson (by Benoit Blanchon - supports v6 and v7)
 * - WebServer (built-in with ESP32 Arduino Core)
 */

#include <WiFi.h>
#include <WebServer.h>
#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <7semi_BME690.h>
#include <ArduinoJson.h>

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

// --- Sensor Objects ---
BME690_7semi bme;
bool bmeAvailable = false;

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
  Serial.println("[PIR] Motion Sensor initialized on GPIO 14");

  // Init BME690 Sensor via I2C
  Wire.begin(I2C_SDA, I2C_SCL);
  if (!bme.begin()) {
    Serial.println("[BME690] WARNING: Could not find BME690 sensor! Will use default environmental baselines. Check SDA=21, SCL=22, VCC=3.3V");
    bmeAvailable = false;
  } else {
    Serial.println("[BME690] Sensor initialized successfully on SDA=21, SCL=22");
    bmeAvailable = true;
  }

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

  // 2. Read Environmental Data from BME690
  float temperatureC = 25.0;
  float humidityPct = 45.0;
  float pressureHpa = 1013.25;
  float gasResistanceKOhms = 0.0;

  if (bmeAvailable) {
    temperatureC = bme.getTemperature();
    humidityPct = bme.getHumidity();
    
    // Safety check: Convert Pascals (Pa) to hectopascals (hPa) if raw pressure is returned
    float rawPressure = bme.getPressure();
    if (rawPressure > 2000.0) {
      pressureHpa = rawPressure / 100.0F;
    } else {
      pressureHpa = rawPressure;
    }
    
    gasResistanceKOhms = bme.getGasResistance() / 1000.0F; // Convert Ohms to kOhms
  }

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
  jsonDoc["environment_raw"]["gas_resistance_kiohms"] = gasResistanceKOhms;

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
  // Serve a clean, white-themed console and status dashboard
  String html = "<!DOCTYPE html><html><head>";
  html += "<meta charset='UTF-8'>";
  html += "<meta name='viewport' content='width=device-width, initial-scale=1.0'>";
  html += "<title>TERRA-SENSE ESP32 Console</title>";
  html += "<style>";
  html += "body{font-family:monospace;background:#050505;color:#ffffff;padding:20px;margin:0}";
  html += "h1{color:#ffffff;border-bottom:1px solid #333333;padding-bottom:10px;font-size:22px}";
  html += ".info{background:#111111;border:1px solid #333333;border-radius:6px;padding:12px;margin:10px 0;line-height:1.6;font-size:13px}";
  html += ".status-dot{display:inline-block;width:8px;height:8px;background:#00ff88;border-radius:50%;margin-right:6px;animation:pulse 1.5s infinite}";
  html += "@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.3}}";
  html += ".sensor-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;margin:15px 0}";
  html += ".sensor-card{background:#111111;border:1px solid #222222;padding:12px 15px;border-radius:6px}";
  html += ".sensor-lbl{color:#888888;font-size:10px;text-transform:uppercase;letter-spacing:0.5px}";
  html += ".sensor-val{font-size:18px;font-weight:bold;margin-top:4px}";
  html += ".console-container{background:#000000;border:1px solid #333333;border-radius:6px;padding:12px;margin-top:20px}";
  html += ".console-hdr{color:#888888;border-bottom:1px solid #222222;padding-bottom:6px;margin-bottom:8px;font-size:11px;display:flex;justify-content:space-between}";
  html += "#console{height:180px;overflow-y:auto;white-space:pre-wrap;word-wrap:break-word;font-size:11px;line-height:1.4;color:#ffffff}";
  html += "</style></head><body>";
  html += "<h1>&#x1F4E1; TERRA-SENSE ESP32 Node</h1>";
  html += "<div class='info'><span class='status-dot'></span><strong>Status:</strong> ONLINE | ";
  html += "<strong>Uptime:</strong> <span id='uptime'>--</span>s | ";
  html += "<strong>Clients:</strong> <span id='clients'>--</span></div>";
  
  html += "<div class='sensor-grid'>";
  html += "<div class='sensor-card'><div class='sensor-lbl'>Temperature</div><div class='sensor-val' id='temp'>-- &deg;C</div></div>";
  html += "<div class='sensor-card'><div class='sensor-lbl'>Humidity</div><div class='sensor-val' id='humid'>-- %</div></div>";
  html += "<div class='sensor-card'><div class='sensor-lbl'>Pressure</div><div class='sensor-val' id='pres'>-- hPa</div></div>";
  html += "<div class='sensor-card'><div class='sensor-lbl'>Gas Resistance</div><div class='sensor-val' id='gas'>-- k&Omega;</div></div>";
  html += "<div class='sensor-card'><div class='sensor-lbl'>Motion (PIR)</div><div class='sensor-val' id='motion'>--</div></div>";
  html += "</div>";

  html += "<div class='console-container'>";
  html += "<div class='console-hdr'><span>ESP32 SERIAL MONITOR OUTPUT</span><span>115200 baud</span></div>";
  html += "<div id='console'>Connecting log stream...</div>";
  html += "</div>";

  html += "<script>";
  html += "const c=document.getElementById('console');";
  html += "function log(msg){";
  html += "  const t=new Date().toLocaleTimeString();";
  html += "  if(c.textContent==='Connecting log stream...') c.textContent='';";
  html += "  c.textContent+=`[${t}] ${msg}\\n`;";
  html += "  c.scrollTop=c.scrollHeight;";
  html += "}";
  html += "function f(){";
  html += "  fetch('/api/telemetry').then(r=>r.json()).then(d=>{";
  html += "    document.getElementById('uptime').textContent=Math.round(d.uptime_ms/1000);";
  html += "    document.getElementById('clients').textContent=d.ap_clients;";
  html += "    if(d.environment_raw){";
  html += "      document.getElementById('temp').textContent=d.environment_raw.temperature_c.toFixed(1)+' &deg;C';";
  html += "      document.getElementById('humid').textContent=d.environment_raw.humidity_pct.toFixed(1)+' %';";
  html += "      document.getElementById('pres').textContent=d.environment_raw.pressure_hpa.toFixed(1)+' hPa';";
  html += "      const g=d.environment_raw.gas_resistance_kiohms;";
  html += "      document.getElementById('gas').textContent=g?g.toFixed(1)+' k&Omega;':'N/A';";
  html += "    }";
  html += "    document.getElementById('motion').textContent=d.pir_motion?'WARNING':'CLEAR';";
  html += "    document.getElementById('motion').style.color=d.pir_motion?'#ff4444':'#ffffff';";
  html += "    log('Serial Payload: '+JSON.stringify(d));";
  html += "  }).catch(e=>{log('Error: '+e)});";
  html += "}";
  html += "f();setInterval(f,1000);";
  html += "</script></body></html>";
  
  server.send(200, "text/html", html);
}

void handleNotFound() {
  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.send(404, "application/json", "{\"error\":\"Not found\"}");
}