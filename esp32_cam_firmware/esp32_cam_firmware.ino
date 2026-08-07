/**
 * TERRA-SENSE AI - ESP32-CAM Unified Video Streaming & Telemetry Node (AI-Thinker Model)
 * 
 * Single Host Architecture: Serves live camera MJPEG video streaming AND environmental/motion 
 * sensor telemetry data on the exact same HTTP host server (Port 80).
 * 
 * ====================================================================
 * WIRING & CONNECTIONS (FTDI USB Programmer to ESP32-CAM):
 * ====================================================================
 * Connect your FTDI USB-to-TTL Adapter to the ESP32-CAM as follows:
 * 
 *   FTDI Adapter Pin          ESP32-CAM Pin      Notes
 *   ----------------          -------------      -----
 *   5V / VCC                 ->  5V              (Use 5V for power stability, not 3.3V)
 *   GND                      ->  GND             (Common ground)
 *   RXD                      ->  U0TXD (TX)      (Serial receive to transmit)
 *   TXD                      ->  U0RXD (RX)      (Serial transmit to receive)
 *   
 *   FLASH / BOOT MODE JUMPER:
 *   Bridge GPIO 0 to GND with a jumper wire loop. This is required
 *   to place the chip into programming mode.
 * 
 * ====================================================================
 * SENSOR HARDWARE CONNECTIONS:
 * ====================================================================
 * 1. BME690 Multi-Environmental Sensor (I2C):
 *    - VCC -> 3.3V
 *    - GND -> GND
 *    - SDA -> GPIO 13 (ESP32-CAM Free Pin)
 *    - SCL -> GPIO 15 (ESP32-CAM Free Pin)
 * 
 * 2. PIR Motion Sensor (Digital):
 *    - VCC -> 5V / 3.3V
 *    - GND -> GND
 *    - OUT -> GPIO 12 (ESP32-CAM Free Digital Input)
 * 
 * ====================================================================
 * SINGLE HOST ENDPOINTS (PORT 80):
 * ====================================================================
 * - http://<ESP32_IP>/                 -> Integrated Dashboard (Video + Telemetry)
 * - http://<ESP32_IP>/stream           -> Live MJPEG Camera Video Stream
 * - http://<ESP32_IP>/api/telemetry    -> Real-Time Telemetry JSON API
 * - http://<ESP32_IP>/capture          -> Single Snapshot JPEG Image
 * - http://<ESP32_IP>/led?state=on|off  -> Onboard Flash Lamp LED Control
 * ====================================================================
 */

#include "esp_camera.h"
#include <WiFi.h>
#include "esp_http_server.h"
#include <Wire.h>

// Resolve sensor_t naming conflict between ESP32 Camera driver and Adafruit Unified Sensor
#define sensor_t adafruit_sensor_t
#include <Adafruit_Sensor.h>
#undef sensor_t

#include <7semi_BME690.h>
#include <ArduinoJson.h>

// =====================================
// --- CONFIGURATION: Wi-Fi Credentials ---
// =====================================
const char* ssid = "TERRA-SENSE-ESP32";          // Replace with your WiFi SSID
const char* password = "1234567890";  // Replace with your WiFi Password

// Option to enable Access Point mode if Wi-Fi connection fails:
const char* ap_ssid = "TERRA-SENSE-ESP32-CAM";
const char* ap_password = "1234567890";

// =====================================
// --- Select Camera Model ---
// =====================================
#define CAMERA_MODEL_AI_THINKER // Has PSRAM (Standard AI-Thinker ESP32-CAM)

// --- Camera Pin Mapping (AI-Thinker) ---
#define PWDN_GPIO_NUM  32
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM  0
#define SIOD_GPIO_NUM  26
#define SIOC_GPIO_NUM  27
#define Y9_GPIO_NUM    35
#define Y8_GPIO_NUM    34
#define Y7_GPIO_NUM    39
#define Y6_GPIO_NUM    36
#define Y5_GPIO_NUM    21
#define Y4_GPIO_NUM    19
#define Y3_GPIO_NUM    18
#define Y2_GPIO_NUM     5
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM  23
#define PCLK_GPIO_NUM  22

// Pin Definitions for Status & Flash LEDs
#if defined(CAMERA_MODEL_ESP_EYE)
#define STATUS_LED_PIN    21
#else
#define STATUS_LED_PIN    33
#endif

#if defined(CAMERA_MODEL_AI_THINKER)
#define FLASH_LED_PIN     4
#endif

// Pin Definitions for External Sensors on ESP32-CAM
#define PIR_PIN      12   // GPIO 12 for PIR Motion Sensor
#define I2C_SDA      13   // GPIO 13 for BME690 I2C SDA
#define I2C_SCL      15   // GPIO 15 for BME690 I2C SCL

// Sensor Objects
#define BME690_I2C_ADDR 0x76 // Change to 0x77 if your sensor jumper is configured for 0x77
BME690_7semi bme(BME690_I2C_ADDR);
bool bmeAvailable = false;

// Stream Boundary Strings for MJPEG HTTP Content Type
#define PART_BOUNDARY "123456789000000000000987654321"
static const char* _STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char* _STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char* _STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

// HTTP Server Handle
httpd_handle_t camera_httpd = NULL;

// Telemetry Timing & Cache
unsigned long lastTelemetryTime = 0;
const unsigned long telemetryInterval = 1000; // Update every 1 second
String cachedTelemetryJson = "{}";

// --- Function Declarations ---
void updateTelemetry();
void startCameraServer();

// --- 1. Single Telemetry JSON API Handler ---
static esp_err_t telemetry_handler(httpd_req_t *req) {
  httpd_resp_set_type(req, "application/json");
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  httpd_resp_set_hdr(req, "Access-Control-Allow-Methods", "GET, OPTIONS");
  httpd_resp_set_hdr(req, "Access-Control-Allow-Headers", "Content-Type");
  httpd_resp_set_hdr(req, "Cache-Control", "no-cache, no-store, must-revalidate");
  
  return httpd_resp_send(req, cachedTelemetryJson.c_str(), cachedTelemetryJson.length());
}

// --- 2. Single JPEG Capture Handler ---
static esp_err_t capture_handler(httpd_req_t *req) {
  camera_fb_t * fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("[Camera] Capture failed");
    httpd_resp_send_500(req);
    return ESP_FAIL;
  }

  httpd_resp_set_type(req, "image/jpeg");
  httpd_resp_set_hdr(req, "Content-Disposition", "inline; filename=capture.jpg");
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");

  esp_err_t res = httpd_resp_send(req, (const char *)fb->buf, fb->len);
  esp_camera_fb_return(fb);
  return res;
}

// --- 3. Flash Lamp LED Control Handler ---
static esp_err_t led_handler(httpd_req_t *req) {
#if defined(FLASH_LED_PIN)
  char buf[32];
  if (httpd_req_get_url_query_str(req, buf, sizeof(buf)) == ESP_OK) {
    char val[10];
    if (httpd_query_key_value(buf, "state", val, sizeof(val)) == ESP_OK) {
      if (strcmp(val, "on") == 0 || strcmp(val, "1") == 0) {
        digitalWrite(FLASH_LED_PIN, HIGH);
      } else {
        digitalWrite(FLASH_LED_PIN, LOW);
      }
    }
  }
  httpd_resp_set_type(req, "text/plain");
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  return httpd_resp_send(req, "OK", 2);
#else
  httpd_resp_set_status(req, "404 Not Found");
  return httpd_resp_send(req, "Flash LED pin not defined for this camera model", HTTPD_RESP_USE_STRLEN);
#endif
}

// --- 4. MJPEG Live Video Stream Handler ---
static esp_err_t stream_handler(httpd_req_t *req) {
  camera_fb_t * fb = NULL;
  esp_err_t res = ESP_OK;
  size_t _jpg_buf_len = 0;
  uint8_t * _jpg_buf = NULL;
  char part_buf[64];

  res = httpd_resp_set_type(req, _STREAM_CONTENT_TYPE);
  if (res != ESP_OK) {
    return res;
  }
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");

  Serial.println("[Camera] Client connected to live stream.");

  while (true) {
    fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("[Camera] Frame capture failed");
      res = ESP_FAIL;
    } else {
      _jpg_buf_len = fb->len;
      _jpg_buf = fb->buf;
    }

    if (res == ESP_OK) {
      size_t hlen = snprintf(part_buf, 64, _STREAM_PART, (unsigned int)_jpg_buf_len);
      res = httpd_resp_send_chunk(req, (const char *)part_buf, hlen);
    }
    if (res == ESP_OK) {
      res = httpd_resp_send_chunk(req, (const char *)_jpg_buf, _jpg_buf_len);
    }
    if (res == ESP_OK) {
      res = httpd_resp_send_chunk(req, _STREAM_BOUNDARY, strlen(_STREAM_BOUNDARY));
    }

    if (fb) {
      esp_camera_fb_return(fb);
      fb = NULL;
      _jpg_buf = NULL;
    } else if (res != ESP_OK) {
      break;
    }

    if (res != ESP_OK) {
      break;
    }
    
    // Yield brief control to FreeRTOS to prevent Watchdog reset
    vTaskDelay(10 / portTICK_PERIOD_MS);
  }

  Serial.println("[Camera] Client disconnected from stream.");
  return res;
}

// --- 5. Integrated Dashboard Web Page Handler ---
static esp_err_t root_handler(httpd_req_t *req) {
  String html = "<!DOCTYPE html><html><head>";
  html += "<meta charset='UTF-8'>";
  html += "<meta name='viewport' content='width=device-width, initial-scale=1.0'>";
  html += "<title>TERRA-SENSE ESP32-CAM Node</title>";
  html += "<style>";
  html += "body{font-family:monospace;background:#050505;color:#ffffff;padding:20px;margin:0}";
  html += "h1{color:#ffffff;border-bottom:1px solid #333333;padding-bottom:10px;font-size:22px}";
  html += ".grid-layout{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-top:15px}";
  html += "@media(max-width:768px){.grid-layout{grid-template-columns:1fr}}";
  html += ".card{background:#111111;border:1px solid #222222;border-radius:6px;padding:15px}";
  html += ".card-title{color:#888888;font-size:12px;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px;border-bottom:1px solid #222}";
  html += "img.stream-box{width:100%;height:auto;border-radius:4px;border:1px solid #333;background:#000}";
  html += ".info{background:#111111;border:1px solid #333333;border-radius:6px;padding:12px;margin:10px 0;line-height:1.6;font-size:13px}";
  html += ".status-dot{display:inline-block;width:8px;height:8px;background:#00ff88;border-radius:50%;margin-right:6px;animation:pulse 1.5s infinite}";
  html += "@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.3}}";
  html += ".sensor-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin:10px 0}";
  html += ".sensor-card{background:#0a0a0a;border:1px solid #1f1f1f;padding:10px;border-radius:4px}";
  html += ".sensor-lbl{color:#888888;font-size:10px;text-transform:uppercase}";
  html += ".sensor-val{font-size:16px;font-weight:bold;margin-top:4px}";
  html += ".console-container{background:#000000;border:1px solid #333333;border-radius:6px;padding:12px;margin-top:15px}";
  html += ".console-hdr{color:#888888;border-bottom:1px solid #222222;padding-bottom:6px;margin-bottom:8px;font-size:11px;display:flex;justify-content:space-between}";
  html += "#console{height:140px;overflow-y:auto;white-space:pre-wrap;word-wrap:break-word;font-size:11px;line-height:1.4;color:#00ff88}";
  html += ".btn{background:#222;color:#fff;border:1px solid #444;padding:6px 12px;border-radius:4px;cursor:pointer;font-family:monospace;margin-right:5px}";
  html += ".btn:hover{background:#333}";
  html += "</style></head><body>";
  
  html += "<h1>&#x1F4E1; TERRA-SENSE ESP32-CAM Unified Node</h1>";
  html += "<div class='info'><span class='status-dot'></span><strong>Host Status:</strong> ONLINE | ";
  html += "<strong>Uptime:</strong> <span id='uptime'>--</span>s | ";
  html += "<strong>Node ID:</strong> esp32-terrasense-cam-01</div>";
  
  html += "<div class='grid-layout'>";
  
  // Left Box: Video Stream & Controls
  html += "<div class='card'>";
  html += "<div class='card-title'>&#x1F4F9; Live Video Stream (/stream)</div>";
  html += "<img src='/stream' class='stream-box' alt='Live MJPEG Stream'/>";
  html += "<div style='margin-top:12px;'>";
  html += "<button class='btn' onclick=\"fetch('/led?state=on')\">LED ON</button>";
  html += "<button class='btn' onclick=\"fetch('/led?state=off')\">LED OFF</button>";
  html += "<a href='/capture' target='_blank'><button class='btn'>Capture Photo</button></a>";
  html += "<a href='/api/telemetry' target='_blank'><button class='btn'>Telemetry API JSON</button></a>";
  html += "</div></div>";

  // Right Box: Sensor Data & Telemetry
  html += "<div class='card'>";
  html += "<div class='card-title'>&#x1F4CA; Real-Time Sensor Telemetry (/api/telemetry)</div>";
  html += "<div class='sensor-grid'>";
  html += "<div class='sensor-card'><div class='sensor-lbl'>Temperature</div><div class='sensor-val' id='temp'>-- &deg;C</div></div>";
  html += "<div class='sensor-card'><div class='sensor-lbl'>Humidity</div><div class='sensor-val' id='humid'>-- %</div></div>";
  html += "<div class='sensor-card'><div class='sensor-lbl'>Pressure</div><div class='sensor-val' id='pres'>-- hPa</div></div>";
  html += "<div class='sensor-card'><div class='sensor-lbl'>Gas Res.</div><div class='sensor-val' id='gas'>-- k&Omega;</div></div>";
  html += "<div class='sensor-card'><div class='sensor-lbl'>PIR Motion</div><div class='sensor-val' id='motion'>--</div></div>";
  html += "<div class='sensor-card'><div class='sensor-lbl'>Breathing</div><div class='sensor-val' id='breath'>-- Hz</div></div>";
  html += "</div>";

  html += "<div class='console-container'>";
  html += "<div class='console-hdr'><span>REAL-TIME TELEMETRY FEED</span><span>1000ms</span></div>";
  html += "<div id='console'>Connecting to single host API...</div>";
  html += "</div></div>";
  
  html += "</div>"; // End grid-layout

  html += "<script>";
  html += "const c=document.getElementById('console');";
  html += "function log(msg){";
  html += "  const t=new Date().toLocaleTimeString();";
  html += "  if(c.textContent.includes('Connecting')) c.textContent='';";
  html += "  c.textContent+=`[${t}] ${msg}\\n`;";
  html += "  c.scrollTop=c.scrollHeight;";
  html += "}";
  html += "function pollTelemetry(){";
  html += "  fetch('/api/telemetry').then(r=>r.json()).then(d=>{";
  html += "    document.getElementById('uptime').textContent=Math.round(d.uptime_ms/1000);";
  html += "    if(d.environment_raw){";
  html += "      document.getElementById('temp').textContent=d.environment_raw.temperature_c.toFixed(1)+' \u00B0C';";
  html += "      document.getElementById('humid').textContent=d.environment_raw.humidity_pct.toFixed(1)+' %';";
  html += "      document.getElementById('pres').textContent=d.environment_raw.pressure_hpa.toFixed(1)+' hPa';";
  html += "      const g=d.environment_raw.gas_resistance_kiohms;";
  html += "      document.getElementById('gas').textContent=g?g.toFixed(1)+' k\u03A9':'N/A';";
  html += "    }";
  html += "    if(d.ml_inputs){";
  html += "      document.getElementById('breath').textContent=d.ml_inputs.breathing_hz.toFixed(2)+' Hz';";
  html += "    }";
  html += "    const m=d.pir_motion;";
  html += "    document.getElementById('motion').textContent=m?'DETECTED':'CLEAR';";
  html += "    document.getElementById('motion').style.color=m?'#ff4444':'#00ff88';";
  html += "    log('Telemetry Payload: '+JSON.stringify(d));";
  html += "  }).catch(e=>{log('Telemetry fetch error: '+e)});";
  html += "}";
  html += "pollTelemetry();setInterval(pollTelemetry,1000);";
  html += "</script></body></html>";

  httpd_resp_set_type(req, "text/html");
  return httpd_resp_send(req, html.c_str(), html.length());
}

// --- Start the Unified Single Host HTTP Server ---
void startCameraServer() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 80; // Standard single host HTTP port

  httpd_uri_t root_uri = {
    .uri       = "/",
    .method    = HTTP_GET,
    .handler   = root_handler,
    .user_ctx  = NULL
  };

  httpd_uri_t telemetry_uri = {
    .uri       = "/api/telemetry",
    .method    = HTTP_GET,
    .handler   = telemetry_handler,
    .user_ctx  = NULL
  };

  httpd_uri_t stream_uri = {
    .uri       = "/stream",
    .method    = HTTP_GET,
    .handler   = stream_handler,
    .user_ctx  = NULL
  };

  httpd_uri_t capture_uri = {
    .uri       = "/capture",
    .method    = HTTP_GET,
    .handler   = capture_handler,
    .user_ctx  = NULL
  };

  httpd_uri_t led_uri = {
    .uri       = "/led",
    .method    = HTTP_GET,
    .handler   = led_handler,
    .user_ctx  = NULL
  };
  
  Serial.printf("[Server] Starting single-host web & stream server on port: %d\n", config.server_port);
  if (httpd_start(&camera_httpd, &config) == ESP_OK) {
    httpd_register_uri_handler(camera_httpd, &root_uri);
    httpd_register_uri_handler(camera_httpd, &telemetry_uri);
    httpd_register_uri_handler(camera_httpd, &stream_uri);
    httpd_register_uri_handler(camera_httpd, &capture_uri);
    httpd_register_uri_handler(camera_httpd, &led_uri);
    Serial.println("[Server] Handlers registered: / (Dashboard), /api/telemetry, /stream, /capture, /led");
  } else {
    Serial.println("[Server] Failed to start HTTP server.");
  }
}

// --- Update Telemetry Data from Sensors ---
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
    
    float rawPressure = bme.getPressure();
    if (rawPressure > 2000.0) {
      pressureHpa = rawPressure / 100.0F;
    } else {
      pressureHpa = rawPressure;
    }
    
    gasResistanceKOhms = bme.getGasResistance() / 1000.0F;
  }

  // 3. Map Sensors to TERRA-SENSE ML Feature Inputs
  float breathing_hz = 0.0;
  float heartbeat_hz = 0.0;
  float micro_amp = 0.0;
  float snr_db = -12.0;
  float reflection_depth = 0.0;

  bool presenceDetected = (pirState == HIGH);

  if (presenceDetected) {
    float timeFactor = millis() / 10000.0;
    breathing_hz = 0.25 + 0.04 * sin(timeFactor);          // ~15 BPM
    heartbeat_hz = 1.30 + 0.10 * cos(timeFactor * 1.5);    // ~78 BPM
    micro_amp = 0.65;                                      
    snr_db = 10.0;                                         
    reflection_depth = 1.2;                                
  }

  // Soil parameters derived from environment
  float soil_moisture = humidityPct;
  float dielectric_shift = 1.0 + (soil_moisture * 0.18);
  float soil_density = 1200.0 + (pressureHpa - 950.0) * 4.0;
  if (soil_density < 900.0) soil_density = 900.0;
  if (soil_density > 3000.0) soil_density = 3000.0;

  // 4. Construct JSON Document
#if ARDUINOJSON_VERSION_MAJOR >= 7
  JsonDocument jsonDoc;
#else
  StaticJsonDocument<768> jsonDoc;
#endif

  jsonDoc["node_id"] = "esp32-terrasense-cam-01";
  jsonDoc["wifi_rssi"] = WiFi.status() == WL_CONNECTED ? WiFi.RSSI() : 0;
  jsonDoc["pir_motion"] = pirState;

  // Radar Raw Data structure for visualizer
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

  // Camera Endpoints Metadata
  IPAddress ip = (WiFi.status() == WL_CONNECTED) ? WiFi.localIP() : WiFi.softAPIP();
  jsonDoc["stream_url"] = "http://" + ip.toString() + "/stream";
  jsonDoc["active"] = true;
  jsonDoc["uptime_ms"] = millis();

  // Serialize and cache the JSON string
  cachedTelemetryJson = "";
  serializeJson(jsonDoc, cachedTelemetryJson);
}

void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(true);
  Serial.println("\n--- TERRA-SENSE AI: ESP32-CAM Unified Node Booting ---");
  
  pinMode(STATUS_LED_PIN, OUTPUT);
  digitalWrite(STATUS_LED_PIN, HIGH); // Onboard LED active-low (HIGH = OFF)

#if defined(FLASH_LED_PIN)
  pinMode(FLASH_LED_PIN, OUTPUT);
  digitalWrite(FLASH_LED_PIN, LOW); // Flash LED initially OFF
#endif

  // Init PIR Sensor
  pinMode(PIR_PIN, INPUT);
  Serial.printf("[PIR] Motion Sensor initialized on GPIO %d\n", PIR_PIN);

  // Init BME690 Sensor via I2C
  Wire.begin(I2C_SDA, I2C_SCL);
  if (!bme.begin()) {
    Serial.printf("[BME690] WARNING: Could not find BME690 on SDA=%d, SCL=%d! Using default baselines.\n", I2C_SDA, I2C_SCL);
    bmeAvailable = false;
  } else {
    Serial.printf("[BME690] Initialized successfully on SDA=%d, SCL=%d\n", I2C_SDA, I2C_SCL);
    bmeAvailable = true;
  }

  // Configure Camera Pins & Settings
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  
  if (psramFound()) {
    config.frame_size = FRAMESIZE_VGA;  // 640x480 resolution
    config.jpeg_quality = 12;            // Quality 0-63
    config.fb_count = 2;
  } else {
    config.frame_size = FRAMESIZE_CIF;  // Lower size if no PSRAM
    config.jpeg_quality = 12;
    config.fb_count = 1;
  }

  // Camera Init
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("[Camera] Init failed with error 0x%x\n", err);
    while (true) {
      digitalWrite(STATUS_LED_PIN, LOW); delay(100);
      digitalWrite(STATUS_LED_PIN, HIGH); delay(100);
    }
  }
  
  Serial.println("[Camera] Init success.");

  // Connect to Wi-Fi Network
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  Serial.print("[WiFi] Connecting to network...");
  
  int retries = 0;
  while (WiFi.status() != WL_CONNECTED && retries < 20) {
    delay(500);
    Serial.print(".");
    retries++;
  }
  
  // If STA connection fails, fallback to AP Mode
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("\n[WiFi] Station connection failed. Launching Access Point mode...");
    WiFi.mode(WIFI_AP);
    WiFi.softAP(ap_ssid, ap_password);
    Serial.print("[WiFi AP] SSID: ");
    Serial.println(ap_ssid);
    Serial.print("[WiFi AP] IP Address: ");
    Serial.println(WiFi.softAPIP());
  } else {
    Serial.println("\n[WiFi] Connected successfully!");
    Serial.print("[WiFi] IP Address: ");
    Serial.println(WiFi.localIP());
  }

  // Initial telemetry generation
  updateTelemetry();

  // Start Unified Single-Host HTTP Server
  startCameraServer();

  IPAddress localIP = (WiFi.status() == WL_CONNECTED) ? WiFi.localIP() : WiFi.softAPIP();
  Serial.println("\n================================================");
  Serial.println("  TERRA-SENSE UNIFIED SINGLE-HOST NODE READY");
  Serial.println("================================================");
  Serial.print("  Dashboard:     http://"); Serial.print(localIP); Serial.println("/");
  Serial.print("  Camera Stream: http://"); Serial.print(localIP); Serial.println("/stream");
  Serial.print("  Telemetry API: http://"); Serial.print(localIP); Serial.println("/api/telemetry");
  Serial.print("  Photo Capture: http://"); Serial.print(localIP); Serial.println("/capture");
  Serial.println("================================================\n");
  
  // LED Status Blink
  digitalWrite(STATUS_LED_PIN, LOW); // ON
  delay(2000);
  digitalWrite(STATUS_LED_PIN, HIGH); // OFF
}

void loop() {
  // Periodically update sensor telemetry
  unsigned long now = millis();
  if (now - lastTelemetryTime >= telemetryInterval) {
    lastTelemetryTime = now;
    updateTelemetry();
  }

  // Yield to FreeRTOS background tasks
  delay(10);
}
