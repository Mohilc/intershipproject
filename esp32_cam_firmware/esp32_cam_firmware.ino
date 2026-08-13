/**
 * TERRA-SENSE AI - ESP32-CAM Video Stream Node
 * Connects to Sensor ESP32's WiFi hotspot as a client (STA mode)
 * Serves: /stream (MJPEG live), /capture (single JPEG), /led (flash control)
 * 
 * IMPORTANT: Power ON the Sensor ESP32 FIRST so its hotspot is running
 * before booting this ESP32-CAM.
 * 
 * Hardware: AI-Thinker ESP32-CAM module
 * Status LED (GPIO 33): Slow blink = connecting, Solid ON = connected, Fast blink = camera error
 * Flash LED (GPIO 4): Controllable via /led?state=on|off
 */

#include "esp_camera.h"
#include <WiFi.h>
#include "esp_http_server.h"
#include "soc/soc.h"            // Brownout detector disable
#include "soc/rtc_cntl_reg.h"   // Brownout detector register

// =====================================
// --- Wi-Fi: Connect to Sensor ESP32's Hotspot ---
// =====================================
const char* ssid = "TERRA-SENSE-ESP32";
const char* password = "1234567890";

// WiFi connection settings
#define WIFI_CONNECT_TIMEOUT_MS  15000   // 15 seconds per attempt
#define WIFI_MAX_RETRIES         10      // Max connection attempts before reboot
#define WIFI_RECONNECT_DELAY_MS  3000    // Wait between retry attempts
#define WIFI_CHECK_INTERVAL_MS   5000    // Check WiFi health every 5s in loop

// =====================================
// --- Camera Pin Mapping (AI-Thinker ESP32-CAM) ---
// =====================================
#define PWDN_GPIO_NUM    32
#define RESET_GPIO_NUM   -1
#define XCLK_GPIO_NUM     0
#define SIOD_GPIO_NUM    26
#define SIOC_GPIO_NUM    27
#define Y9_GPIO_NUM      35
#define Y8_GPIO_NUM      34
#define Y7_GPIO_NUM      39
#define Y6_GPIO_NUM      36
#define Y5_GPIO_NUM      21
#define Y4_GPIO_NUM      19
#define Y3_GPIO_NUM      18
#define Y2_GPIO_NUM       5
#define VSYNC_GPIO_NUM   25
#define HREF_GPIO_NUM    23
#define PCLK_GPIO_NUM    22

// Pin Definitions for Status & Flash LEDs
#define STATUS_LED_PIN   33   // Onboard LED (active-LOW: LOW = ON, HIGH = OFF)
#define FLASH_LED_PIN     4

// Stream Boundary Strings for MJPEG HTTP Content Type
#define PART_BOUNDARY "123456789000000000000987654321"
static const char* _STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char* _STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char* _STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

// HTTP Server Handle
httpd_handle_t camera_httpd = NULL;

// WiFi health tracking
unsigned long lastWifiCheckTime = 0;

// --- Function Declarations ---
void startCameraServer();
bool connectToWiFi();
void statusLedBlink(int onMs, int offMs, int count);

// =============================================
// --- WiFi Connection with Retries & Timeout ---
// =============================================
bool connectToWiFi() {
  int retryCount = 0;

  while (retryCount < WIFI_MAX_RETRIES) {
    retryCount++;
    Serial.printf("\n[WiFi] Attempt %d/%d: Connecting to '%s'...\n", retryCount, WIFI_MAX_RETRIES, ssid);

    WiFi.disconnect(true);   // Clear any stale connection state
    delay(200);
    WiFi.mode(WIFI_STA);     // Station mode (client)
    WiFi.setAutoReconnect(true);
    WiFi.begin(ssid, password);

    // Wait up to WIFI_CONNECT_TIMEOUT_MS for connection
    unsigned long startAttempt = millis();
    while (WiFi.status() != WL_CONNECTED && (millis() - startAttempt) < WIFI_CONNECT_TIMEOUT_MS) {
      delay(500);
      Serial.print(".");
      // Blink status LED while connecting (slow blink)
      digitalWrite(STATUS_LED_PIN, LOW);  delay(100);
      digitalWrite(STATUS_LED_PIN, HIGH); delay(400);
    }

    if (WiFi.status() == WL_CONNECTED) {
      Serial.println("\n[WiFi] *** CONNECTED! ***");
      Serial.print("[WiFi] IP Address: ");
      Serial.println(WiFi.localIP());
      Serial.print("[WiFi] Gateway (Sensor ESP32): ");
      Serial.println(WiFi.gatewayIP());
      Serial.print("[WiFi] Signal Strength (RSSI): ");
      Serial.print(WiFi.RSSI());
      Serial.println(" dBm");
      
      // Solid LED ON to indicate connected
      digitalWrite(STATUS_LED_PIN, LOW);
      return true;
    }

    // Connection failed — show error reason
    wl_status_t status = WiFi.status();
    Serial.printf("\n[WiFi] Attempt %d failed. Status code: %d ", retryCount, status);
    switch (status) {
      case WL_NO_SSID_AVAIL: Serial.println("(Hotspot not found! Is Sensor ESP32 ON?)"); break;
      case WL_CONNECT_FAILED: Serial.println("(Wrong password or connection refused)"); break;
      case WL_IDLE_STATUS: Serial.println("(WiFi idle — retrying)"); break;
      default: Serial.println("(Unknown reason)"); break;
    }

    // Blink fast to indicate error
    statusLedBlink(100, 100, 5);
    delay(WIFI_RECONNECT_DELAY_MS);
  }

  Serial.println("[WiFi] !!! ALL RETRIES EXHAUSTED — REBOOTING IN 5 SECONDS !!!");
  delay(5000);
  ESP.restart();
  return false; // Never reached, but keeps compiler happy
}

// --- Status LED Helper ---
void statusLedBlink(int onMs, int offMs, int count) {
  for (int i = 0; i < count; i++) {
    digitalWrite(STATUS_LED_PIN, LOW);  delay(onMs);
    digitalWrite(STATUS_LED_PIN, HIGH); delay(offMs);
  }
}

// --- 1. Single JPEG Capture Handler ---
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

// --- 2. Flash Lamp LED Control Handler ---
static esp_err_t led_handler(httpd_req_t *req) {
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
}

// --- 3. MJPEG Live Video Stream Handler ---
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
    vTaskDelay(2 / portTICK_PERIOD_MS);
  }

  Serial.println("[Camera] Client disconnected from stream.");
  return res;
}

// --- Start the HTTP Stream Server ---
void startCameraServer() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 80;

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
  
  Serial.printf("[Server] Starting stream server on port: %d\n", config.server_port);
  if (httpd_start(&camera_httpd, &config) == ESP_OK) {
    httpd_register_uri_handler(camera_httpd, &stream_uri);
    httpd_register_uri_handler(camera_httpd, &capture_uri);
    httpd_register_uri_handler(camera_httpd, &led_uri);
    Serial.println("[Server] Handlers registered: /stream, /capture, /led");
  } else {
    Serial.println("[Server] Failed to start HTTP server.");
  }
}

void setup() {
  // Disable brownout detector — prevents boot-loop on weak external power (Vin from another ESP32)
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);

  Serial.begin(115200);
  delay(2000); // Wait 2s for external power supply to stabilize
  Serial.setDebugOutput(true);
  Serial.println("\n--- TERRA-SENSE AI: ESP32-CAM Video Stream Node Booting ---");
  
  // Init LEDs first for visual feedback
  pinMode(STATUS_LED_PIN, OUTPUT);
  digitalWrite(STATUS_LED_PIN, HIGH); // OFF initially (active-low)

  pinMode(FLASH_LED_PIN, OUTPUT);
  digitalWrite(FLASH_LED_PIN, LOW);   // Flash LED OFF

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
  config.pin_sccb_sda = SIOD_GPIO_NUM;   // Fixed: was pin_sscb_sda (deprecated typo)
  config.pin_sccb_scl = SIOC_GPIO_NUM;   // Fixed: was pin_sscb_scl (deprecated typo)
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;  // 20MHz (high framerate)
  config.pixel_format = PIXFORMAT_JPEG;
  config.grab_mode = CAMERA_GRAB_LATEST; // Always grab the latest frame
  
  if (psramFound()) {
    Serial.println("[Camera] PSRAM found — using VGA (640x480) at high rate");
    config.frame_size = FRAMESIZE_VGA;
    config.jpeg_quality = 14;      // Optimized for high speed transmission (quality 10-63, lower is better)
    config.fb_count = 2;           // Double buffering
  } else {
    Serial.println("[Camera] No PSRAM — using CIF (352x288)");
    config.frame_size = FRAMESIZE_CIF;
    config.jpeg_quality = 14;
    config.fb_count = 1;
  }

  // Camera Init
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("[Camera] Init FAILED with error 0x%x\n", err);
    Serial.println("[Camera] Check: Is the camera ribbon cable seated properly?");
    // Fast blink = camera hardware error
    while (true) {
      statusLedBlink(100, 100, 1);
    }
  }
  
  Serial.println("[Camera] Init success.");

  // *** Connect to Sensor ESP32's WiFi Hotspot ***
  // This will retry multiple times and reboot if it fails
  connectToWiFi();

  // Start HTTP Web & Stream Server
  startCameraServer();

  Serial.println("\n================================================");
  Serial.println("  TERRA-SENSE VIDEO STREAM NODE READY");
  Serial.println("================================================");
  Serial.print("  Dashboard:     http://"); Serial.print(WiFi.localIP()); Serial.println("/");
  Serial.print("  Camera Stream: http://"); Serial.print(WiFi.localIP()); Serial.println("/stream");
  Serial.print("  Photo Capture: http://"); Serial.print(WiFi.localIP()); Serial.println("/capture");
  Serial.print("  Flash LED:     http://"); Serial.print(WiFi.localIP()); Serial.println("/led?state=on");
  Serial.println("================================================\n");
  
  // Solid LED ON for 2 seconds = success
  digitalWrite(STATUS_LED_PIN, LOW);
  delay(2000);
  digitalWrite(STATUS_LED_PIN, HIGH);
}

void loop() {
  // Periodically check WiFi connection health and reconnect if lost
  unsigned long now = millis();
  if (now - lastWifiCheckTime >= WIFI_CHECK_INTERVAL_MS) {
    lastWifiCheckTime = now;
    
    if (WiFi.status() != WL_CONNECTED) {
      Serial.println("[WiFi] CONNECTION LOST — Reconnecting...");
      digitalWrite(STATUS_LED_PIN, HIGH); // LED OFF = disconnected
      connectToWiFi();
      // Restart HTTP server after reconnect
      if (camera_httpd != NULL) {
        httpd_stop(camera_httpd);
        camera_httpd = NULL;
      }
      startCameraServer();
      Serial.print("[WiFi] Reconnected! New IP: ");
      Serial.println(WiFi.localIP());
    }
  }

  delay(10);
}