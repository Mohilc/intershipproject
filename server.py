"""
TERRA-SENSE AI - Flask Web Application & Prediction API Server
Serves static files and runs subsurface detection predictions with CORS support.
Performance optimized: LRU prediction caching, gzip compression, static asset caching.
Includes: NVIDIA StreamPETR 3D human localizer for obstacle-proximity analysis.
"""

from flask import Flask, request, jsonify, send_from_directory
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

    # Gzip compression for text responses
    if (response.content_type and
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
