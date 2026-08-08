

#include "esp_camera.h"
#include <WiFi.h>
#include "esp_http_server.h"

// =====================================
// --- Wi-Fi: Connect to Sensor ESP32's Hotspot ---
// =====================================
// The sensor ESP32 creates WiFi hotspot "TERRA-SENSE-ESP32".
// This ESP32-CAM joins that same network as a client.
const char* ssid = "TERRA-SENSE-ESP32";
const char* password = "1234567890";

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
#define STATUS_LED_PIN   33
#define FLASH_LED_PIN     4

// Stream Boundary Strings for MJPEG HTTP Content Type
#define PART_BOUNDARY "123456789000000000000987654321"
static const char* _STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char* _STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char* _STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

// HTTP Server Handle
httpd_handle_t camera_httpd = NULL;

// --- Function Declarations ---
void startCameraServer();

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
    vTaskDelay(10 / portTICK_PERIOD_MS);
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
  Serial.begin(115200);
  Serial.setDebugOutput(true);
  Serial.println("\n--- TERRA-SENSE AI: ESP32-CAM Video Stream Node Booting ---");
  
  pinMode(STATUS_LED_PIN, OUTPUT);
  digitalWrite(STATUS_LED_PIN, HIGH); // Onboard LED active-low (HIGH = OFF)

  pinMode(FLASH_LED_PIN, OUTPUT);
  digitalWrite(FLASH_LED_PIN, LOW); // Flash LED initially OFF

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

  // Connect to Sensor ESP32's WiFi Hotspot
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  Serial.printf("[WiFi] Connecting to '%s'...", ssid);
  
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  
  Serial.println("\n[WiFi] Connected!");
  Serial.print("[WiFi] IP Address: ");
  Serial.println(WiFi.localIP());

  // Start HTTP Web & Stream Server
  startCameraServer();

  Serial.println("\n================================================");
  Serial.println("  TERRA-SENSE VIDEO STREAM NODE READY");
  Serial.println("================================================");
  Serial.print("  Dashboard:     http://"); Serial.print(WiFi.localIP()); Serial.println("/");
  Serial.print("  Camera Stream: http://"); Serial.print(WiFi.localIP()); Serial.println("/stream");
  Serial.print("  Photo Capture: http://"); Serial.print(WiFi.localIP()); Serial.println("/capture");
  Serial.println("================================================\n");
  
  // LED Status Blink
  digitalWrite(STATUS_LED_PIN, LOW); // ON
  delay(2000);
  digitalWrite(STATUS_LED_PIN, HIGH); // OFF
}

void loop() {
  // Yield to FreeRTOS background tasks
  delay(10);
}
