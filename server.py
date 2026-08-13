"""
TERRA-SENSE AI - Flask Web Application & Prediction API Server
Serves static files and runs subsurface detection predictions with CORS support.
Performance optimized: LRU prediction caching, gzip compression, static asset caching.
Includes: NVIDIA StreamPETR 3D human localizer for obstacle-proximity analysis.
"""

import requests
import cv2
import numpy as np
import base64
import time
import math
from flask import Flask, request, jsonify, send_from_directory, Response
from ml_model import SubsurfacePythonMLEngine
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import os
import gzip
from io import BytesIO

# Load .env file manually
def load_env():
    if os.path.exists('.env'):
        with open('.env', 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, val = line.split('=', 1)
                    os.environ[key.strip()] = val.strip()

load_env()

executor = ThreadPoolExecutor(max_workers=8)

app = Flask(__name__, static_folder='.')

# Instantiate AI Engine (trains on startup)
ml_engine = SubsurfacePythonMLEngine()

# ── StreamPETR 3D Analyzer (lazy-loaded on first /api/streampetr call) ────────
_streampetr_analyzer = None

def get_streampetr():
    global _streampetr_analyzer
    if _streampetr_analyzer is None:
        try:
            from streampetr_3d import StreamPETRAnalyzer
            _streampetr_analyzer = StreamPETRAnalyzer()
            print("[Server] StreamPETR 3D Analyzer loaded.")
        except Exception as e:
            print(f"[Server] StreamPETR load warning: {e}")
    return _streampetr_analyzer

# --- LRU Prediction Cache ---
@lru_cache(maxsize=128)
def cached_predict(breathing_hz, heartbeat_hz, pir_motion, radar_state, radar_energy,
                   micro_amp, snr_db, bme_temp_c, bme_humidity_pct, bme_pressure_hpa,
                   dielectric_shift, soil_density, reflection_depth, grid_x=12.5, grid_y=8.2):
    """Cache predictions by rounding inputs to avoid near-duplicate computations."""
    feature_input = {
        'breathing_hz': breathing_hz,
        'heartbeat_hz': heartbeat_hz,
        'pir_motion': pir_motion,
        'radar_state': radar_state,
        'radar_energy': radar_energy,
        'micro_amp': micro_amp,
        'snr_db': snr_db,
        'bme_temp_c': bme_temp_c,
        'bme_humidity_pct': bme_humidity_pct,
        'bme_pressure_hpa': bme_pressure_hpa,
        'dielectric_shift': dielectric_shift,
        'soil_density': soil_density,
        'reflection_depth': reflection_depth,
        'x': grid_x,
        'y': grid_y
    }
    return ml_engine.predict(feature_input)

@app.after_request
def add_headers(response):
    # CORS headers
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'

    # Cache-Control for static assets (CSS, JS, HTML)
    if request.path.endswith(('.css', '.js', '.html', '.svg', '.png', '.jpg', '.ico')):
        response.headers['Cache-Control'] = 'public, max-age=3600'  # 1 hour cache
    elif request.path == '/':
        response.headers['Cache-Control'] = 'public, max-age=300'  # 5 min for index

    # Gzip compression for text/json responses when supported by client
    accept_enc = request.headers.get('Accept-Encoding', '')
    if ('gzip' in accept_enc and
        response.content_type and
        any(ct in response.content_type for ct in ['text/', 'application/json', 'application/javascript']) and
        'Content-Encoding' not in response.headers and
        not response.direct_passthrough and
        len(response.get_data()) > 500):
        data = response.get_data()
        buf = BytesIO()
        with gzip.GzipFile(fileobj=buf, mode='wb', compresslevel=6) as f:
            f.write(data)
        response.set_data(buf.getvalue())
        response.headers['Content-Encoding'] = 'gzip'
        response.headers['Content-Length'] = len(response.get_data())
        response.headers['Vary'] = 'Accept-Encoding'

    return response

# --- Global Live Telemetry Cache ---
import time
latest_telemetry = {
    "node_id": "none",
    "wifi_rssi": 0,
    "pir_motion": 0,
    "radar_raw": {
        "state": 0,
        "moving_energy": 0,
        "static_energy": 0,
        "distance_cm": 0
    },
    "environment_raw": {
        "temperature_c": 20.0,
        "humidity_pct": 30.0,
        "pressure_hpa": 1013.25
    },
    "ml_inputs": {
        "breathing_hz": 0.0,
        "heartbeat_hz": 0.0,
        "micro_amp": 0.0,
        "snr_db": -12.0,
        "dielectric_shift": 1.0,
        "soil_moisture": 35.0,
        "soil_density": 1600.0,
        "reflection_depth": 0.0
    },
    "active": False,
    "last_update": 0
}

# --- Global Latest Camera / YOLO Detection Cache ---
# Updated by the live YOLO stream on every detected frame and by snapshot analysis.
# Consumed by /api/predict_fused to apply vision-sensor fusion without an extra fetch.
latest_camera_result = {
    "human_detected": False,
    "confidence": 0.0,        # 0.0 – 1.0
    "box_count": 0,
    "last_update": 0.0,       # Unix timestamp
    "source": "none"          # 'stream' | 'snapshot' | 'none'
}

@app.route('/api/telemetry', methods=['GET', 'POST', 'OPTIONS'])
def handle_telemetry():
    global latest_telemetry
    if request.method == 'OPTIONS':
        return jsonify({"status": "ok"}), 200

    if request.method == 'POST':
        try:
            data = request.get_json(force=True) or {}
            latest_telemetry = data
            latest_telemetry['active'] = True
            latest_telemetry['last_update'] = time.time()
            return jsonify({"status": "success", "message": "Telemetry received"}), 200
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 400

    # GET logic
    # Telemetry is marked stale and inactive if no updates have arrived in the last 7 seconds
    current_time = time.time()
    if latest_telemetry['active'] and (current_time - latest_telemetry.get('last_update', 0) > 7.0):
        latest_telemetry['active'] = False

    return jsonify(latest_telemetry), 200

@app.route('/api/camera/proxy', methods=['GET', 'POST', 'OPTIONS'])
def camera_proxy():
    if request.method == 'OPTIONS':
        return jsonify({"status": "ok"}), 200

    target_url = request.args.get('url')
    if not target_url:
        return jsonify({"status": "error", "message": "Missing target 'url' parameter"}), 400

    target_url = target_url.strip()
    if not target_url.startswith(('http://', 'https://')):
        target_url = 'http://' + target_url

    try:
        if request.method == 'POST':
            data = request.get_json(silent=True) or {}
            resp = requests.post(target_url, json=data, timeout=5)
        else:
            resp = requests.get(target_url, timeout=5)

        content_type = resp.headers.get('Content-Type', 'text/plain')
        return (resp.content, resp.status_code, {'Content-Type': content_type, 'Access-Control-Allow-Origin': '*'})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Camera node offline or unreachable: {str(e)}"}), 502

# --- YOLOv8 / OpenCV Human Detection Integration ---
import queue
import threading

_yolo_model = None
_hog_detector = None
_face_cascade = None

class YOLOBackgroundWorker:
    def __init__(self):
        self.frame_queue = queue.Queue(maxsize=1)
        self.latest_boxes = []
        self.latest_detected = False
        self.latest_conf = 0.0
        self.lock = threading.Lock()
        self.thread = None
        self.running = False

    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._worker_loop, daemon=True)
            self.thread.start()

    def stop(self):
        self.running = False
        try:
            self.frame_queue.put_nowait(None)
        except Exception:
            pass
        if self.thread is not None:
            try:
                self.thread.join(timeout=1.0)
            except Exception:
                pass

    def _worker_loop(self):
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1.0)
                if frame is None or not self.running:
                    continue
                
                # Run YOLO/OpenCV raw detection
                detected, conf, boxes = detect_humans_raw(frame)
                
                with self.lock:
                    self.latest_detected = detected
                    self.latest_conf = conf
                    self.latest_boxes = boxes
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[YOLO Worker] Error: {e}")

    def update_frame(self, frame):
        if not self.running or frame is None:
            return
        # Keep queue size at most 1 to avoid latency lag
        try:
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put_nowait(frame.copy())
        except Exception:
            pass

    def get_results(self):
        with self.lock:
            return self.latest_detected, self.latest_conf, self.latest_boxes

def get_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        try:
            from ultralytics import YOLO
            # Load the pre-trained nano model (fastest inference, automatically cached locally)
            _yolo_model = YOLO('yolov8n.pt')
            print("[Server] YOLOv8 model loaded successfully.")
        except Exception as e:
            print(f"[Server] YOLOv8 load warning: {e}")
    return _yolo_model

def get_opencv_fallback():
    global _hog_detector, _face_cascade
    if _hog_detector is None:
        try:
            _hog_detector = cv2.HOGDescriptor()
            _hog_detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        except Exception as e:
            print(f"[Server] OpenCV HOG init error: {e}")
    if _face_cascade is None:
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            _face_cascade = cv2.CascadeClassifier(cascade_path)
        except Exception as e:
            print(f"[Server] OpenCV Haar Cascade init error: {e}")
    return _hog_detector, _face_cascade

def draw_hud_box(img, x1, y1, x2, y2, label, color=(16, 185, 129)):
    """Draws a high-visibility futuristic HUD bounding box with corner notches and label."""
    h_img, w_img = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img - 1, x2), min(h_img - 1, y2)
    w_box = x2 - x1
    h_box = y2 - y1
    
    # Main rectangle
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    
    # Corner notches
    notch_len = min(15, max(5, int(w_box * 0.15)), max(5, int(h_box * 0.15)))
    cv2.line(img, (x1, y1), (x1 + notch_len, y1), (255, 255, 255), 3)
    cv2.line(img, (x1, y1), (x1, y1 + notch_len), (255, 255, 255), 3)
    cv2.line(img, (x2, y1), (x2 - notch_len, y1), (255, 255, 255), 3)
    cv2.line(img, (x2, y1), (x2, y1 + notch_len), (255, 255, 255), 3)
    cv2.line(img, (x1, y2), (x1 + notch_len, y2), (255, 255, 255), 3)
    cv2.line(img, (x1, y2), (x1, y2 - notch_len), (255, 255, 255), 3)
    cv2.line(img, (x2, y2), (x2 - notch_len, y2), (255, 255, 255), 3)
    cv2.line(img, (x2, y2), (x2, y2 - notch_len), (255, 255, 255), 3)
    
    # Center crosshair
    cx, cy = x1 + w_box // 2, y1 + h_box // 2
    cv2.line(img, (cx - 6, cy), (cx + 6, cy), color, 1)
    cv2.line(img, (cx, cy - 6), (cx, cy + 6), color, 1)
    
    # Label banner with solid dark background
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.48
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    
    tag_y = max(y1 - 6, text_h + 8)
    cv2.rectangle(img, (x1, tag_y - text_h - 6), (x1 + text_w + 10, tag_y + 4), (10, 15, 25), -1)
    cv2.rectangle(img, (x1, tag_y - text_h - 6), (x1 + text_w + 10, tag_y + 4), color, 1)
    cv2.putText(img, label, (x1 + 5, tag_y - 2), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

def detect_humans_raw(img):
    """
    Runs YOLOv8 human detection (or OpenCV fallback) on a frame.
    Returns: (human_detected, highest_confidence, boxes_info)
    """
    if img is None:
        return False, 0.0, []

    human_detected = False
    highest_conf = 0.0
    boxes_info = []

    model = get_yolo_model()
    yolo_success = False

    if model is not None:
        try:
            # classes=[0] selects only 'person' class in COCO; imgsz=320 ensures 20+ FPS
            results = model.predict(img, classes=[0], conf=0.25, imgsz=320, verbose=False)
            yolo_success = True
            
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = map(int, xyxy)
                    w_box = max(1, x2 - x1)
                    h_box = max(1, y2 - y1)
                    
                    highest_conf = max(highest_conf, conf)
                    human_detected = True
                    boxes_info.append({"x": x1, "y": y1, "w": w_box, "h": h_box, "confidence": conf})
        except Exception as e:
            print(f"[Server] YOLOv8 inference warning: {e}")
            yolo_success = False

    # Fallback to OpenCV HOG / Haar Cascade if YOLO failed or unavailable
    if not yolo_success or (not human_detected and model is None):
        hog, cascade = get_opencv_fallback()
        if hog is not None:
            try:
                h_orig, w_orig = img.shape[:2]
                scale = 400.0 / max(1, w_orig)
                resized = cv2.resize(img, (int(w_orig * scale), int(h_orig * scale)))
                (rects, weights) = hog.detectMultiScale(resized, winStride=(4, 4), padding=(8, 8), scale=1.05)
                
                for (x, y, w_b, h_b), weight in zip(rects, weights):
                    x1 = int(x / scale)
                    y1 = int(y / scale)
                    x2 = int((x + w_b) / scale)
                    y2 = int((y + h_b) / scale)
                    conf = float(min(0.95, max(0.40, weight * 0.4)))
                    highest_conf = max(highest_conf, conf)
                    human_detected = True
                    boxes_info.append({"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1, "confidence": conf})
            except Exception as e:
                print(f"[Server] OpenCV HOG detect failed: {e}")

    return human_detected, highest_conf, boxes_info

def draw_detection_hud(img, human_detected, highest_conf, boxes_info):
    """Draws boxes and the HUD status text on the frame."""
    if img is None:
        return img
    for box in boxes_info:
        x1, y1 = box["x"], box["y"]
        x2, y2 = x1 + box["w"], y1 + box["h"]
        conf = box["confidence"]
        
        color = (16, 185, 129) if conf > 0.60 else (251, 191, 36)
        label = f"HUMAN: {conf * 100:.1f}%"
        draw_hud_box(img, x1, y1, x2, y2, label, color)
        
    # Top HUD Status Tag
    hud_text = f"YOLO AI: {'HUMAN DETECTED' if human_detected else 'SCANNING — CLEAR'} ({len(boxes_info)} targets)"
    hud_color = (16, 185, 129) if human_detected else (0, 242, 254)
    cv2.putText(img, hud_text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 15, 25), 3, cv2.LINE_AA)
    cv2.putText(img, hud_text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, hud_color, 1, cv2.LINE_AA)
    return img

def detect_humans(img):
    """Backward compatibility wrapper that does both detection and drawing."""
    detected, conf, boxes = detect_humans_raw(img)
    img = draw_detection_hud(img, detected, conf, boxes)
    return img, detected, conf, boxes

def create_diagnostic_frame(title, subtitle="", details=None, frame_idx=0):
    """Generates an informative HUD diagnostic frame when camera connection is lost or offline."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Outer HUD grid and border
    cv2.rectangle(frame, (10, 10), (630, 470), (0, 242, 254), 1)
    cv2.rectangle(frame, (14, 14), (626, 466), (20, 30, 45), -1)
    
    # Corner brackets
    for (cx, cy, dx, dy) in [(20, 20, 20, 20), (620, 20, -20, 20), (20, 460, 20, -20), (620, 460, -20, -20)]:
        cv2.line(frame, (cx, cy), (cx + dx, cy), (0, 242, 254), 2)
        cv2.line(frame, (cx, cy), (cx, cy + dy), (0, 242, 254), 2)

    # Animated radar pulse dot
    pulse_radius = int(8 + 4 * math.sin(frame_idx * 0.2))
    cv2.circle(frame, (50, 50), pulse_radius, (244, 63, 94), -1)
    cv2.circle(frame, (50, 50), pulse_radius + 6, (244, 63, 94), 1)

    # Title & Subtitle
    cv2.putText(frame, title, (80, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    if subtitle:
        cv2.putText(frame, subtitle, (80, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 242, 254), 1, cv2.LINE_AA)
        
    cv2.line(frame, (30, 100), (610, 100), (0, 242, 254), 1)

    # Checklist / Instructions
    y_offset = 140
    if details:
        for idx, line in enumerate(details):
            color = (16, 185, 129) if line.startswith("✓") else (251, 191, 36) if line.startswith("⚠") else (200, 220, 240)
            cv2.putText(frame, line, (40, y_offset + idx * 32), cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1, cv2.LINE_AA)

    # Footer status
    status_bar = f"STATUS: RETRYING CONNECTION... [FRAME {frame_idx}]"
    cv2.putText(frame, status_bar, (40, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 160, 180), 1, cv2.LINE_AA)
    
    _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return buffer.tobytes()
@app.route('/api/camera/stream_yolo')
def stream_yolo():
    """
    Proxies ESP32-CAM MJPEG stream or PC Webcam (url=webcam/0), runs YOLOv8 human detection
    on every frame, and returns a live annotated MJPEG stream.
    Supports auto-probing common ESP32 / IP camera endpoints (/stream, :81/stream, /video, /capture).
    """
    target_url = (request.args.get('url') or '').strip()
    yolo_param = request.args.get('yolo', 'true').lower()
    enable_yolo = yolo_param not in ('false', '0', 'off', 'no')
    
    yolo_worker = None
    if enable_yolo:
        yolo_worker = YOLOBackgroundWorker()
        yolo_worker.start()
    
    def generate_frames():
        global latest_camera_result
        
        try:
            def generate_demo():
                last_switch = time.time()
                show_human = False
                frame_idx = 0
                while True:
                    time.sleep(0.06)
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.circle(frame, (320, 240), 180, (40, 40, 0), 1)
                    cv2.circle(frame, (320, 240), 120, (30, 30, 0), 1)
                    cv2.line(frame, (320, 40), (320, 440), (40, 40, 0), 1)
                    cv2.line(frame, (120, 240), (520, 240), (40, 40, 0), 1)
                    
                    angle = (frame_idx * 0.05) % (2 * math.pi)
                    x_sweep = int(320 + 180 * math.cos(angle))
                    y_sweep = int(240 + 180 * math.sin(angle))
                    cv2.line(frame, (320, 240), (x_sweep, y_sweep), (254, 242, 0), 2)
                    
                    if time.time() - last_switch > 6.0:
                        show_human = not show_human
                        last_switch = time.time()
                        
                    if show_human:
                        draw_hud_box(frame, 270, 150, 370, 350, "HUMAN: 94.7%", (16, 185, 129))
                        dot_size = int(5 + 3 * math.sin(frame_idx * 0.2))
                        cv2.circle(frame, (320, 250), dot_size, (16, 185, 129), -1)
                        
                        latest_camera_result = {
                            "human_detected": True,
                            "confidence": 0.947,
                            "box_count": 1,
                            "last_update": time.time(),
                            "source": "demo"
                        }
                    else:
                        latest_camera_result = {
                            "human_detected": False,
                            "confidence": 0.0,
                            "box_count": 0,
                            "last_update": time.time(),
                            "source": "demo"
                        }
                    
                    cv2.putText(frame, "AI DETECTOR DEMO (CONNECT CAM TO STREAM)", (130, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (254, 242, 0), 1, cv2.LINE_AA)
                    
                    frame_idx += 1
                    _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            # Outer auto-reconnection loop for ESP32-CAM and webcams
            while True:
                # 1. Handle Local PC Webcam
                if target_url.lower() in ('webcam', '0', 'camera', 'local', 'local_cam'):
                    cap = None
                    try:
                        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                        if not cap.isOpened():
                            cap = cv2.VideoCapture(0)
                        
                        if not cap.isOpened():
                            print("[Server] Local webcam could not be opened.")
                            yield from generate_demo()
                            return
                        
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        
                        consecutive_failures = 0
                        while True:
                            ret, frame = cap.read()
                            if not ret or frame is None:
                                consecutive_failures += 1
                                if consecutive_failures > 30:
                                    print("[Server] Webcam frame read failures exceeded limit. Breaking webcam loop.")
                                    break
                                time.sleep(0.05)
                                continue
                            
                            consecutive_failures = 0
                            if enable_yolo and yolo_worker:
                                yolo_worker.update_frame(frame)
                                detected, conf, boxes = yolo_worker.get_results()
                                frame = draw_detection_hud(frame, detected, conf, boxes)
                                latest_camera_result = {
                                    "human_detected": detected,
                                    "confidence": float(conf),
                                    "box_count": len(boxes),
                                    "last_update": time.time(),
                                    "source": "webcam"
                                }
                            else:
                                cv2.putText(frame, "OPTICAL FEED (RAW)", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 242, 254), 1, cv2.LINE_AA)
                                latest_camera_result = {
                                    "human_detected": False,
                                    "confidence": 0.0,
                                    "box_count": 0,
                                    "last_update": time.time(),
                                    "source": "webcam"
                                }
                            
                            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                    except Exception as e:
                        print(f"[Server] Webcam stream error: {e}")
                        yield from generate_demo()
                    finally:
                        if cap is not None and cap.isOpened():
                            cap.release()
                    return

                if not target_url or target_url.lower() in ('demo', 'synthetic', 'test'):
                    yield from generate_demo()
                    return

                # 2. Build candidate stream URLs for ESP32-CAM / IP Camera
                clean_url = target_url.strip()
                if not clean_url.startswith(('http://', 'https://', 'rtsp://')):
                    clean_url = 'http://' + clean_url

                # Generate candidates to probe
                candidates = [clean_url]
                
                # If user entered raw IP/host or just port, try common stream paths
                base_no_slash = clean_url.rstrip('/')
                if not any(base_no_slash.endswith(p) for p in ['/stream', '/video', '/capture', '/shot.jpg', '/mjpeg', '/live']):
                    candidates.append(base_no_slash + '/stream')
                    # Check port 81 (standard Arduino ESP32 CameraWebServer stream port)
                    if ':81' not in base_no_slash:
                        if ':80' in base_no_slash:
                            candidates.append(base_no_slash.replace(':80', ':81') + '/stream')
                        else:
                            try:
                                from urllib.parse import urlparse
                                parsed = urlparse(base_no_slash)
                                host_only = parsed.hostname or base_no_slash.replace('http://', '').replace('https://', '').split('/')[0]
                                candidates.append(f"http://{host_only}:81/stream")
                            except Exception:
                                pass
                    candidates.append(base_no_slash + '/video')
                    candidates.append(base_no_slash + '/capture')
                    candidates.append(base_no_slash + '/shot.jpg')

                # Try to connect to candidate stream
                headers = {
                    'User-Agent': 'TerraSense-AI-Proxy/2.0',
                    'Accept': '*/*'
                }
                
                stream_resp = None
                active_url = None
                is_snapshot_endpoint = False

                for cand in candidates:
                    try:
                        r = requests.get(cand, stream=True, timeout=(2.5, 8.0), headers=headers)
                        if r.status_code == 200:
                            c_type = r.headers.get('Content-Type', '').lower()
                            if 'multipart' in c_type or 'mixed-replace' in c_type or 'octet-stream' in c_type or 'video' in c_type:
                                stream_resp = r
                                active_url = cand
                                is_snapshot_endpoint = False
                                print(f"[Server] Connected to MJPEG stream at: {active_url}")
                                break
                            elif 'image/' in c_type:
                                # Single JPEG capture endpoint (e.g. /capture or /shot.jpg)
                                stream_resp = r
                                active_url = cand
                                is_snapshot_endpoint = True
                                print(f"[Server] Connected to snapshot endpoint at: {active_url}")
                                break
                    except Exception:
                        continue

                # Strategy 2A: Continuous Snapshot Polling Loop (for /capture endpoints)
                if is_snapshot_endpoint and active_url:
                    # Close the probing connection first to avoid socket leak
                    if stream_resp is not None:
                        try:
                            stream_resp.close()
                        except Exception:
                            pass
                    
                    consecutive_failures = 0
                    frame_idx = 0
                    while True:
                        try:
                            resp = requests.get(active_url, timeout=3.0, headers=headers)
                            try:
                                if resp.status_code == 200:
                                    np_arr = np.frombuffer(resp.content, dtype=np.uint8)
                                    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                                    if frame is not None:
                                        consecutive_failures = 0
                                        if enable_yolo and yolo_worker:
                                            yolo_worker.update_frame(frame)
                                            detected, conf, boxes = yolo_worker.get_results()
                                            frame = draw_detection_hud(frame, detected, conf, boxes)
                                            latest_camera_result = {
                                                "human_detected": detected,
                                                "confidence": float(conf),
                                                "box_count": len(boxes),
                                                "last_update": time.time(),
                                                "source": "esp32_cam"
                                            }
                                        else:
                                            cv2.putText(frame, "OPTICAL FEED (RAW)", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 242, 254), 1, cv2.LINE_AA)
                                            latest_camera_result = {
                                                "human_detected": False,
                                                "confidence": 0.0,
                                                "box_count": 0,
                                                "last_update": time.time(),
                                                "source": "esp32_cam"
                                            }

                                        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                                        yield (b'--frame\r\n'
                                               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                                    else:
                                        consecutive_failures += 1
                                else:
                                    consecutive_failures += 1
                            finally:
                                try:
                                    resp.close()
                                except Exception:
                                    pass
                            
                            if consecutive_failures > 15:
                                print("[Server] Snapshot poll consecutive failures exceeded limit. Breaking poll loop.")
                                break
                                
                            time.sleep(0.06)
                            frame_idx += 1
                        except Exception as snap_err:
                            print(f"[Server] Snapshot poll error: {snap_err}")
                            consecutive_failures += 1
                            if consecutive_failures > 15:
                                break
                            time.sleep(0.5)

                # Strategy 2B: MJPEG Streaming with Frame Buffering
                elif stream_resp is not None and not is_snapshot_endpoint:
                    byte_data = b''
                    try:
                        for chunk in stream_resp.iter_content(chunk_size=4096):
                            if not chunk:
                                continue
                            byte_data += chunk
                            
                            while True:
                                a = byte_data.find(b'\xff\xd8')
                                if a == -1:
                                    if len(byte_data) > 8192:
                                        byte_data = byte_data[-2048:]
                                    break
                                
                                b = byte_data.find(b'\xff\xd9', a + 2)
                                if b == -1:
                                    # If buffer is excessively large without EOI, drop corrupted data
                                    if len(byte_data) - a > 131072:
                                        next_a = byte_data.find(b'\xff\xd8', a + 2)
                                        if next_a != -1:
                                            byte_data = byte_data[next_a:]
                                        else:
                                            byte_data = byte_data[-4096:]
                                    break
                                
                                jpg = byte_data[a:b+2]
                                byte_data = byte_data[b+2:]
                                
                                # Frame-skipping: if buffer has accumulated backlogged frames, jump to latest
                                if len(byte_data) > 32768:
                                    last_a = byte_data.rfind(b'\xff\xd8')
                                    if last_a > 0:
                                        last_b = byte_data.find(b'\xff\xd9', last_a + 2)
                                        if last_b != -1:
                                            jpg = byte_data[last_a:last_b+2]
                                            byte_data = byte_data[last_b+2:]
                                
                                np_arr = np.frombuffer(jpg, dtype=np.uint8)
                                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                                
                                if frame is not None:
                                    if enable_yolo and yolo_worker:
                                        yolo_worker.update_frame(frame)
                                        detected, conf, boxes = yolo_worker.get_results()
                                        frame = draw_detection_hud(frame, detected, conf, boxes)
                                        latest_camera_result = {
                                            "human_detected": detected,
                                            "confidence": float(conf),
                                            "box_count": len(boxes),
                                            "last_update": time.time(),
                                            "source": "esp32_cam"
                                        }
                                    else:
                                        cv2.putText(frame, "OPTICAL FEED (RAW)", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 242, 254), 1, cv2.LINE_AA)
                                        latest_camera_result = {
                                            "human_detected": False,
                                            "confidence": 0.0,
                                            "box_count": 0,
                                            "last_update": time.time(),
                                            "source": "esp32_cam"
                                        }
                                    
                                    _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                                    yield (b'--frame\r\n'
                                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                    except Exception as stream_err:
                        print(f"[Server] MJPEG stream reading interrupted: {stream_err}")
                    finally:
                        try:
                            stream_resp.close()
                        except Exception:
                            pass

                # Strategy 2C: Fallback to OpenCV VideoCapture (handles RTSP, non-standard HTTP, etc.)
                else:
                    try:
                        print(f"[Server] Attempting OpenCV VideoCapture on: {clean_url}")
                        cap = cv2.VideoCapture(clean_url)
                        if cap.isOpened():
                            consecutive_failures = 0
                            while True:
                                ret, frame = cap.read()
                                if not ret or frame is None:
                                    consecutive_failures += 1
                                    if consecutive_failures > 30:
                                        print("[Server] VideoCapture frame read failures exceeded limit. Breaking fallback loop.")
                                        break
                                    time.sleep(0.05)
                                    continue
                                
                                consecutive_failures = 0
                                if enable_yolo and yolo_worker:
                                    yolo_worker.update_frame(frame)
                                    detected, conf, boxes = yolo_worker.get_results()
                                    frame = draw_detection_hud(frame, detected, conf, boxes)
                                    latest_camera_result = {
                                        "human_detected": detected,
                                        "confidence": float(conf),
                                        "box_count": len(boxes),
                                        "last_update": time.time(),
                                        "source": "esp32_cam"
                                    }
                                else:
                                    cv2.putText(frame, "OPTICAL FEED (RAW)", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 242, 254), 1, cv2.LINE_AA)
                                    latest_camera_result = {
                                        "human_detected": False,
                                        "confidence": 0.0,
                                        "box_count": 0,
                                        "last_update": time.time(),
                                        "source": "esp32_cam"
                                    }

                                _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                                yield (b'--frame\r\n'
                                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                    except Exception as cap_err:
                        print(f"[Server] VideoCapture fallback error: {cap_err}")

                # Strategy 2D: Camera is unreachable — yield informative diagnostic HUD frames for 5 seconds
                print(f"[Server] Camera node unreachable at: {clean_url}. Yielding diagnostics and retrying...")
                details = [
                    f"TARGET URL: {clean_url}",
                    "⚠ 1. Ensure ESP32-CAM is powered ON (5V / GND)",
                    "⚠ 2. Connect PC WiFi to 'TERRA-SENSE-ESP32' Hotspot",
                    "⚠ 3. Standard stream endpoint is: 192.168.4.2/stream",
                    "✓ Retrying connection automatically..."
                ]
                for diag_frame_idx in range(1, 6): # 5 seconds of diagnostics
                    diag_jpg = create_diagnostic_frame(
                        title="CAMERA NODE OFFLINE / UNREACHABLE",
                        subtitle="SEARCH & RESCUE OPTICAL SENSOR LINK FAILED",
                        details=details,
                        frame_idx=diag_frame_idx
                    )
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + diag_jpg + b'\r\n')
                    time.sleep(1.0)
        finally:
            if yolo_worker is not None:
                try:
                    yolo_worker.stop()
                except Exception:
                    pass

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/camera/analyze_snapshot', methods=['GET', 'POST', 'OPTIONS'])
def analyze_snapshot():
    """
    Runs human detection on a single snapshot frame from the ESP32-CAM or PC Webcam.
    """
    global latest_camera_result
    if request.method == 'OPTIONS':
        return jsonify({"status": "ok"}), 200

    target_url = (request.args.get('url') or '').strip()
    is_demo = request.args.get('demo', 'false').lower() == 'true' or not target_url
    
    try:
        if is_demo:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.circle(frame, (320, 240), 180, (40, 40, 0), 1)
            cv2.circle(frame, (320, 240), 120, (30, 30, 0), 1)
            cv2.line(frame, (320, 40), (320, 440), (40, 40, 0), 1)
            cv2.line(frame, (120, 240), (520, 240), (40, 40, 0), 1)
            
            draw_hud_box(frame, 270, 150, 370, 350, "HUMAN: 95.8%", (16, 185, 129))
            cv2.putText(frame, "AI DETECTOR SNAPSHOT (DEMO)", (170, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (254, 242, 0), 1, cv2.LINE_AA)
            
            _, buffer = cv2.imencode('.jpg', frame)
            b64_str = base64.b64encode(buffer).decode('utf-8')
            
            return jsonify({
                "status": "success",
                "human_detected": True,
                "confidence_pct": 95.8,
                "box_count": 1,
                "boxes": [{"x": 270, "y": 150, "w": 100, "h": 200, "confidence": 0.958}],
                "image": f"data:image/jpeg;base64,{b64_str}"
            }), 200
            
        # Check if local webcam requested
        if target_url.lower() in ('webcam', '0', 'camera', 'local'):
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                return jsonify({"status": "error", "message": "Failed to access local webcam"}), 500
            ret, frame = cap.read()
            cap.release()
            if not ret or frame is None:
                return jsonify({"status": "error", "message": "Failed to read frame from webcam"}), 500
        else:
            clean_url = target_url
            if not clean_url.startswith(('http://', 'https://')):
                clean_url = 'http://' + clean_url
            
            # Try candidates: clean_url, clean_url/capture, clean_url/shot.jpg
            candidates = [clean_url]
            base_no_slash = clean_url.rstrip('/')
            if not base_no_slash.endswith(('/capture', '/shot.jpg')):
                candidates.insert(0, base_no_slash + '/capture')
                candidates.insert(1, base_no_slash + '/shot.jpg')

            frame = None
            for cand in candidates:
                try:
                    resp = requests.get(cand, timeout=4, headers={'User-Agent': 'TerraSense-AI-Proxy/2.0'})
                    if resp.status_code == 200 and resp.content:
                        # Find JPEG in case of multipart or raw image
                        content = resp.content
                        a = content.find(b'\xff\xd8')
                        if a != -1:
                            b = content.find(b'\xff\xd9', a + 2)
                            if b != -1:
                                content = content[a:b+2]
                            else:
                                content = content[a:]
                        np_arr = np.frombuffer(content, dtype=np.uint8)
                        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                        if frame is not None:
                            break
                except Exception:
                    continue

            if frame is None:
                return jsonify({"status": "error", "message": f"Failed to retrieve/decode frame from camera ({clean_url})"}), 502
            
        frame, detected, conf, boxes = detect_humans(frame)
        
        # Update shared camera cache for ML fusion
        latest_camera_result = {
            "human_detected": detected,
            "confidence": float(conf),
            "box_count": len(boxes),
            "last_update": time.time(),
            "source": "snapshot"
        }
        
        _, buffer = cv2.imencode('.jpg', frame)
        b64_str = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            "status": "success",
            "human_detected": detected,
            "confidence_pct": round(conf * 100, 1),
            "box_count": len(boxes),
            "boxes": boxes,
            "image": f"data:image/jpeg;base64,{b64_str}"
        }), 200
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    if os.path.exists(path):
        return send_from_directory('.', path)
    return send_from_directory('.', 'index.html')

@app.route('/api/predict', methods=['POST', 'OPTIONS'])
def predict_subsurface():
    if request.method == 'OPTIONS':
        return jsonify({"status": "ok"}), 200

    try:
        data = request.get_json(force=True) or {}
        
        # Check if list of targets is sent
        if 'targets' in data:
            def process_target(target_data):
                breathing_hz = round(float(target_data.get('breathing_hz', 0.0)), 2)
                heartbeat_hz = round(float(target_data.get('heartbeat_hz', 0.0)), 2)
                pir_motion = float(target_data.get('pir_motion', 1 if (breathing_hz > 0.08 or heartbeat_hz > 0.4) else 0))
                radar_state = float(target_data.get('radar_state', 2 if (breathing_hz > 0.08 or heartbeat_hz > 0.4) else 0))
                radar_energy = round(float(target_data.get('radar_energy', 75.0 if (breathing_hz > 0.08 or heartbeat_hz > 0.4) else 0.0)), 1)
                micro_amp = round(float(target_data.get('micro_amp', 0.0)), 2)
                snr_db = round(float(target_data.get('snr_db', 0.0)), 2)
                bme_temp_c = round(float(target_data.get('bme_temp_c', target_data.get('temperature_c', 25.0))), 1)
                bme_humidity_pct = round(float(target_data.get('bme_humidity_pct', target_data.get('soil_moisture', 35.0))), 1)
                bme_pressure_hpa = round(float(target_data.get('bme_pressure_hpa', target_data.get('pressure_hpa', 1013.25))), 1)
                dielectric_shift = round(float(target_data.get('dielectric_shift', 0.0)), 2)
                soil_density = round(float(target_data.get('soil_density', 1600.0)), 0)
                reflection_depth = round(float(target_data.get('reflection_depth', 1.5)), 2)

                grid_x = round(float(target_data.get('x', target_data.get('grid_x', 12.5))), 1)
                grid_y = round(float(target_data.get('y', target_data.get('grid_y', 8.2))), 1)

                res = cached_predict(
                    breathing_hz, heartbeat_hz, pir_motion, radar_state, radar_energy,
                    micro_amp, snr_db, bme_temp_c, bme_humidity_pct, bme_pressure_hpa,
                    dielectric_shift, soil_density, reflection_depth, grid_x, grid_y
                )
                
                if 'x' in target_data and 'y' in target_data and 'z' in target_data:
                    res = dict(res)
                    res['coordinates'] = {
                        'x': float(target_data['x']),
                        'y': float(target_data['y']),
                        'z': float(target_data['z'])
                    }
                return res

            results = list(executor.map(process_target, data['targets']))
            human_count = sum(1 for r in results if r.get('human_detected') is True)
                
            return jsonify({
                "status": "success",
                "results": results,
                "human_count": human_count,
                "total_targets": len(results)
            })

        # Single target fallback
        breathing_hz = round(float(data.get('breathing_hz', 0.0)), 2)
        heartbeat_hz = round(float(data.get('heartbeat_hz', 0.0)), 2)
        pir_motion = float(data.get('pir_motion', 1 if (breathing_hz > 0.08 or heartbeat_hz > 0.4) else 0))
        radar_state = float(data.get('radar_state', 2 if (breathing_hz > 0.08 or heartbeat_hz > 0.4) else 0))
        radar_energy = round(float(data.get('radar_energy', 75.0 if (breathing_hz > 0.08 or heartbeat_hz > 0.4) else 0.0)), 1)
        micro_amp = round(float(data.get('micro_amp', 0.0)), 2)
        snr_db = round(float(data.get('snr_db', 0.0)), 2)
        bme_temp_c = round(float(data.get('bme_temp_c', data.get('temperature_c', 25.0))), 1)
        bme_humidity_pct = round(float(data.get('bme_humidity_pct', data.get('soil_moisture', 35.0))), 1)
        bme_pressure_hpa = round(float(data.get('bme_pressure_hpa', data.get('pressure_hpa', 1013.25))), 1)
        dielectric_shift = round(float(data.get('dielectric_shift', 0.0)), 2)
        soil_density = round(float(data.get('soil_density', 1600.0)), 0)
        reflection_depth = round(float(data.get('reflection_depth', 1.5)), 2)
        grid_x = round(float(data.get('x', data.get('grid_x', 12.5))), 1)
        grid_y = round(float(data.get('y', data.get('grid_y', 8.2))), 1)

        prediction_result = cached_predict(
            breathing_hz, heartbeat_hz, pir_motion, radar_state, radar_energy,
            micro_amp, snr_db, bme_temp_c, bme_humidity_pct, bme_pressure_hpa,
            dielectric_shift, soil_density, reflection_depth, grid_x, grid_y
        )
        
        return jsonify({
            "status": "success",
            "result": prediction_result
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400


@app.route('/api/camera/latest_detection', methods=['GET', 'OPTIONS'])
def get_latest_camera_detection():
    """Returns the most recent camera YOLO detection result for the frontend to query."""
    if request.method == 'OPTIONS':
        return jsonify({"status": "ok"}), 200
    age_s = time.time() - latest_camera_result.get('last_update', 0)
    return jsonify({
        "status": "success",
        "human_detected": latest_camera_result['human_detected'],
        "confidence_pct": round(latest_camera_result['confidence'] * 100, 1),
        "box_count": latest_camera_result['box_count'],
        "age_seconds": round(age_s, 1),
        "source": latest_camera_result['source'],
        "fresh": age_s < 30.0
    }), 200


@app.route('/api/predict_fused', methods=['POST', 'OPTIONS'])
def predict_fused():
    """
    Sensor-fusion prediction endpoint.
    Combines the 6-model ML subsurface ensemble (weight: 0.70) with the latest
    YOLOv8 camera visual detection (weight: 0.30) to produce a fused probability.

    Accepts same body as /api/predict plus optional fields:
      camera_confidence   : float  0.0 – 1.0  (override; uses cached value if absent)
      camera_human_detected : bool  (override; uses cached value if absent)
      camera_max_age_s    : int    max seconds since last camera reading to accept (default 30)

    Returns /api/predict result fields plus:
      fusion_applied        : bool
      camera_confidence_pct : float
      camera_human_detected : bool
      fused_probability_pct : float
      fused_human_detected  : bool
      fused_human_count     : int
    """
    if request.method == 'OPTIONS':
        return jsonify({"status": "ok"}), 200

    ML_WEIGHT     = 0.70
    CAMERA_WEIGHT = 0.30
    THRESHOLD     = 50.0  # fused probability threshold to declare human detected

    try:
        data = request.get_json(force=True) or {}
        max_age_s = float(data.get('camera_max_age_s', 30.0))

        # ── Determine camera signal ───────────────────────────────────────────
        cam_age = time.time() - latest_camera_result.get('last_update', 0)
        cam_fresh = cam_age < max_age_s and latest_camera_result['source'] != 'none'

        # Allow body override (e.g. snapshot result just taken by frontend)
        if 'camera_confidence' in data:
            cam_conf  = float(data['camera_confidence'])          # 0.0–1.0
            cam_human = bool(data.get('camera_human_detected', cam_conf >= 0.50))
            cam_fresh = True
        else:
            cam_conf  = latest_camera_result['confidence'] if cam_fresh else 0.0
            cam_human = latest_camera_result['human_detected'] if cam_fresh else False

        fusion_applied = cam_fresh

        # ── Process targets (batch or single) ────────────────────────────────
        def process_fused_target(target_data, cam_conf_override=cam_conf, cam_human_override=cam_human):
            breathing_hz  = round(float(target_data.get('breathing_hz', 0.0)), 2)
            heartbeat_hz  = round(float(target_data.get('heartbeat_hz', 0.0)), 2)
            pir_motion    = float(target_data.get('pir_motion', 1 if (breathing_hz > 0.08 or heartbeat_hz > 0.4) else 0))
            radar_state   = float(target_data.get('radar_state', 2 if (breathing_hz > 0.08 or heartbeat_hz > 0.4) else 0))
            radar_energy  = round(float(target_data.get('radar_energy', 75.0 if (breathing_hz > 0.08 or heartbeat_hz > 0.4) else 0.0)), 1)
            micro_amp     = round(float(target_data.get('micro_amp', 0.0)), 2)
            snr_db        = round(float(target_data.get('snr_db', 0.0)), 2)
            bme_temp_c    = round(float(target_data.get('bme_temp_c', target_data.get('temperature_c', 25.0))), 1)
            bme_humidity  = round(float(target_data.get('bme_humidity_pct', target_data.get('soil_moisture', 35.0))), 1)
            bme_pressure  = round(float(target_data.get('bme_pressure_hpa', target_data.get('pressure_hpa', 1013.25))), 1)
            dielectric    = round(float(target_data.get('dielectric_shift', 0.0)), 2)
            soil_density  = round(float(target_data.get('soil_density', 1600.0)), 0)
            refl_depth    = round(float(target_data.get('reflection_depth', 1.5)), 2)
            grid_x        = round(float(target_data.get('x', target_data.get('grid_x', 12.5))), 1)
            grid_y        = round(float(target_data.get('y', target_data.get('grid_y', 8.2))), 1)

            ml_result = cached_predict(
                breathing_hz, heartbeat_hz, pir_motion, radar_state, radar_energy,
                micro_amp, snr_db, bme_temp_c, bme_humidity, bme_pressure,
                dielectric, soil_density, refl_depth, grid_x, grid_y
            )

            ml_prob = float(ml_result.get('probability_percentage', 0.0))

            # Weighted fusion
            if fusion_applied:
                if cam_human_override:
                    # Camera seeing a person above ground raises/boosts the fused score
                    fused_prob = (ml_prob * ML_WEIGHT) + (cam_conf_override * 100.0 * CAMERA_WEIGHT)
                    if cam_conf_override >= 0.50:
                        # Ensure fused probability is at least above the threshold if camera is confident
                        fused_prob = max(fused_prob, cam_conf_override * 100.0)
                    fused_prob = round(min(99.8, max(0.2, fused_prob)), 1)
                else:
                    # Camera clear (does not see anyone above ground).
                    # This should NOT suppress a subsurface radar detection.
                    fused_prob = ml_prob
            else:
                fused_prob = ml_prob

            fused_human = fused_prob >= THRESHOLD

            result = dict(ml_result)
            result['fusion_applied']          = fusion_applied
            result['camera_confidence_pct']   = round(cam_conf_override * 100, 1)
            result['camera_human_detected']   = cam_human_override
            result['fused_probability_pct']   = fused_prob
            result['fused_human_detected']    = fused_human
            result['probability_percentage']  = fused_prob   # override for UI consistency
            result['human_detected']          = fused_human
            return result

        if 'targets' in data:
            results = list(executor.map(process_fused_target, data['targets']))
            human_count      = sum(1 for r in results if r.get('fused_human_detected') is True)
            fused_human_count = human_count
            return jsonify({
                "status": "success",
                "results": results,
                "human_count": human_count,
                "fused_human_count": fused_human_count,
                "total_targets": len(results),
                "fusion_applied": fusion_applied,
                "camera_confidence_pct": round(cam_conf * 100, 1),
                "camera_human_detected": cam_human
            })

        # Single target
        fused_result = process_fused_target(data)
        return jsonify({
            "status": "success",
            "result": fused_result,
            "fusion_applied": fusion_applied,
            "camera_confidence_pct": round(cam_conf * 100, 1),
            "camera_human_detected": cam_human
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route('/api/streampetr', methods=['POST', 'OPTIONS'])
def streampetr_analyze():
    """
    NVIDIA StreamPETR 3D Human Localizer endpoint.
    Accepts sensor feature dict + optional grid_x / grid_y position.
    Returns 3D human position, obstacle proximity, entrapment posture,
    rescue access direction, and top-down GPR heatmap preview (base64).

    Request body (JSON):
      Same sensor fields as /api/predict, plus optional:
        grid_x  (float) — horizontal X position in metres
        grid_y  (float) — horizontal Y position in metres

    Response (JSON):
      {
        status: 'success',
        streampetr: {
          human_detected: bool,
          confidence_pct: float,
          position_3d: { x_m, y_m, depth_m },
          obstacle_proximity_m: float,
          obstacle_type: string,
          entrapment_posture: string,
          rescue_access_direction: string,
          near_obstacle: bool,
          under_obstacle: bool,
          heatmap_preview_b64: string (JPEG base64 top-down view),
          api_status: string
        }
      }
    """
    if request.method == 'OPTIONS':
        return jsonify({"status": "ok"}), 200

    try:
        data = request.get_json(force=True) or {}

        # Extract sensor features
        features = {
            "breathing_hz":     float(data.get("breathing_hz",    0.0)),
            "heartbeat_hz":     float(data.get("heartbeat_hz",    0.0)),
            "micro_amp":        float(data.get("micro_amp",       0.0)),
            "snr_db":           float(data.get("snr_db",          0.0)),
            "pir_motion":       float(data.get("pir_motion",      0.0)),
            "radar_state":      float(data.get("radar_state",     0.0)),
            "radar_energy":     float(data.get("radar_energy",    0.0)),
            "bme_temp_c":       float(data.get("bme_temp_c",     25.0)),
            "bme_humidity_pct": float(data.get("bme_humidity_pct", 35.0)),
            "bme_pressure_hpa": float(data.get("bme_pressure_hpa", 1013.25)),
            "dielectric_shift": float(data.get("dielectric_shift",  0.0)),
            "reflection_depth": float(data.get("reflection_depth",  1.5)),
        }
        grid_x = float(data.get("grid_x", data.get("x", 0.0)))
        grid_y = float(data.get("grid_y", data.get("y", 0.0)))

        analyzer = get_streampetr()
        if analyzer is None:
            return jsonify({"status": "error", "message": "StreamPETR module unavailable. Run: pip install Pillow"}), 503

        result = analyzer.analyze(features, grid_x=grid_x, grid_y=grid_y)
        return jsonify({"status": "success", "streampetr": result}), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


if __name__ == '__main__':
    print("\n=======================================================")
    print(" TERRA-SENSE AI - Subsurface Detection Server")
    print(" Running at: http://localhost:3000")
    print(f" AI Engine Accuracy: {ml_engine.accuracy_score:.2f}%")
    print(f" Prediction Cache: LRU (128 entries)")
    print(" StreamPETR 3D: /api/streampetr  (lazy-loaded)")
    print("=======================================================\n")
    app.run(host='0.0.0.0', port=3000, debug=False)
