"""
TERRA-SENSE AI — NVIDIA GPT-OSS-20B 3D Human-Under-Obstacle Analyzer
=====================================================================
Uses NVIDIA NIM (openai/gpt-oss-20b) as the AI reasoning engine for
Subsurface Human Bio-Detection & Search-and-Rescue operations.

Pipeline:
  Sensor Features → Rich Text Prompt → GPT-OSS-20B NIM API → JSON 3D Analysis
                 → Radar Heatmap Renderer (Pillow)           → Dashboard Preview

GPT-OSS-20B (20 billion parameters, open-source GPT architecture) interprets
12-channel sensor telemetry (GPR, Doppler Radar, PIR, BMP180, DHT11) and
returns structured 3D bounding boxes, obstacle proximity, entrapment posture,
and rescue access guidance — adapted from autonomous-driving 3D detection
to SAR human localisation.

Usage:
    from streampetr_3d import StreamPETRAnalyzer
    analyzer = StreamPETRAnalyzer()
    result = analyzer.analyze(sensor_features, grid_x=2.5, grid_y=-1.2)
"""

import os
import io
import math
import base64
import json
import requests
import numpy as np

# ─── PIL / Pillow for heatmap rendering ───────────────────────────────────────
try:
    from PIL import Image, ImageDraw, ImageFilter, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("[StreamPETR] Warning: Pillow not installed. Run: pip install Pillow")

# ─── NVIDIA NIM Configuration — GPT-OSS-20B ──────────────────────────────────
NVIDIA_API_KEY   = os.environ.get("NVIDIA_API_KEY", "")
NIM_URL          = "https://integrate.api.nvidia.com/v1/chat/completions"
NIM_MODEL        = "openai/gpt-oss-20b"

# Image dimensions for the synthesised GPR heatmap views (dashboard preview only)
IMG_W, IMG_H = 640, 640

# ─── Confidence threshold: detections below this are ignored ──────────────────
CONFIDENCE_THRESHOLD = 0.30

# ─── Rescue access direction options ──────────────────────────────────────────
ACCESS_OPTIONS = [
    "surface_extraction",
    "top_drill_with_shield",
    "lateral_micro_tunnel",
    "hydraulic_deep_bore",
    "vertical_extraction",
]

# ─── Obstacle type vocabulary ──────────────────────────────────────────────────
OBSTACLE_TYPES = [
    "soil_compression",
    "concrete_slab",
    "rubble_pile",
    "steel_beam",
    "wooden_debris",
    "loose_earth_void",
]


class StreamPETRAnalyzer:
    """
    NVIDIA GPT-OSS-20B powered 3D SAR Human Localizer.

    Workflow
    --------
    1. render_gpr_heatmap()    — Render 3 synthetic GPR views as JPEG images
                                 (top-down, front cross-section, side cross-section)
                                 → used for dashboard heatmap preview
    2. call_nim_api()          — Build a rich structured-text prompt from the
                                 12-channel sensor telemetry and POST to
                                 NVIDIA NIM openai/gpt-oss-20b for JSON analysis
    3. interpret_result()      — Parse JSON → human XYZ + obstacle proximity
    4. analyze()               — Full pipeline, returns SAR result dict
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or NVIDIA_API_KEY
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
            "Accept":        "application/json",
        }
        print(f"[GPT-OSS-20B] Initialized — NVIDIA NIM: {NIM_URL} | model: {NIM_MODEL}")

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 1: Render Sensor Data → Synthetic GPR Heatmap Images
    # ──────────────────────────────────────────────────────────────────────────

    def _sensor_to_intensity(self, features: dict) -> float:
        """Normalise sensor features into a [0, 1] signal intensity value."""
        breathing  = float(features.get("breathing_hz", 0.0))
        heartbeat  = float(features.get("heartbeat_hz", 0.0))
        micro_amp  = float(features.get("micro_amp", 0.0))
        snr_db     = float(features.get("snr_db", 0.0))
        pir        = float(features.get("pir_motion", 0.0))
        radar_e    = float(features.get("radar_energy", 0.0))
        dielectric = float(features.get("dielectric_shift", 0.0))

        # Weighted composite score from all sensor channels
        score = (
            min(breathing  / 0.5,  1.0) * 0.25 +
            min(heartbeat  / 2.2,  1.0) * 0.25 +
            min(micro_amp  / 1.0,  1.0) * 0.15 +
            min(snr_db     / 32.0, 1.0) * 0.15 +
            pir                          * 0.10 +
            min(radar_e    / 100,  1.0) * 0.05 +
            min(dielectric / 15.0, 1.0) * 0.05
        )
        return max(0.0, min(1.0, score))

    def render_gpr_heatmap(self, features: dict, grid_x: float = 0.0, grid_y: float = 0.0) -> list:
        """
        Render THREE synthetic sensor views as RGB images.
        Returns list of 3 PIL Image objects:
          [0] top_down  — bird's-eye radar heatmap (XY plane)
          [1] front     — cross-section front view (XZ depth plane)
          [2] side      — cross-section side view  (YZ depth plane)
        """
        if not PIL_AVAILABLE:
            return []

        intensity     = self._sensor_to_intensity(features)
        depth         = float(features.get("reflection_depth", 1.5))
        temp          = float(features.get("bme_temp_c", 25.0))
        humidity      = float(features.get("bme_humidity_pct", 40.0))
        pressure      = float(features.get("bme_pressure_hpa", 1013.25))

        images = []

        # ── View 0: Top-Down GPR Heatmap ──────────────────────────────────────
        img_td = Image.new("RGB", (IMG_W, IMG_H), color=(10, 12, 20))
        draw   = ImageDraw.Draw(img_td)

        # Background soil texture gradient
        for row in range(IMG_H):
            ratio = row / IMG_H
            soil_r = int(20  + ratio * 15)
            soil_g = int(25  + ratio * 10)
            soil_b = int(35  + ratio * 20)
            draw.line([(0, row), (IMG_W, row)], fill=(soil_r, soil_g, soil_b))

        # Grid lines (scan sweep pattern)
        for xi in range(0, IMG_W, 40):
            draw.line([(xi, 0), (xi, IMG_H)], fill=(20, 30, 45), width=1)
        for yi in range(0, IMG_H, 40):
            draw.line([(0, yi), (IMG_W, yi)], fill=(20, 30, 45), width=1)

        # Obstacle simulation: random rectangular debris blocks
        np.random.seed(int(abs(grid_x * 10 + grid_y * 10)) % 2**31)
        for _ in range(np.random.randint(2, 6)):
            ox = np.random.randint(30, IMG_W - 80)
            oy = np.random.randint(30, IMG_H - 80)
            ow = np.random.randint(40, 120)
            oh = np.random.randint(30, 80)
            c  = np.random.randint(35, 65)
            draw.rectangle([ox, oy, ox + ow, oy + oh], fill=(c, c - 5, c - 10), outline=(90, 90, 100))

        # Human signal blob — position mapped from grid coordinates
        cx = int(IMG_W * 0.5 + (grid_x / 20.0) * IMG_W * 0.35)
        cy = int(IMG_H * 0.5 + (grid_y / 20.0) * IMG_H * 0.35)
        cx = max(40, min(IMG_W - 40, cx))
        cy = max(40, min(IMG_H - 40, cy))

        if intensity > 0.15:
            # Outer glow — thermal anomaly halo
            for r in range(90, 10, -8):
                alpha = int(intensity * 180 * (1 - r / 90))
                col_r = min(255, int(alpha * 1.8))
                col_g = min(255, int(alpha * 0.6))
                col_b = min(255, int(alpha * 0.1))
                draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                             fill=(col_r, col_g, col_b))

            # Core signal peak — bright white-yellow centre
            core = int(intensity * 30)
            draw.ellipse([cx - core, cy - core, cx + core, cy + core],
                         fill=(255, 240, 180))

            # Depth ring — reflection depth indicator
            dr = int(depth * 18)
            draw.ellipse([cx - dr, cy - dr, cx + dr, cy + dr],
                         outline=(100, 200, 255), width=2)

            # Vital-sync pulse ring
            vr = int(float(features.get("heartbeat_hz", 1.0)) * 25)
            draw.ellipse([cx - vr, cy - vr, cx + vr, cy + vr],
                         outline=(255, 120, 80), width=1)

        # Scan beam sweep lines from centre
        for angle_deg in range(0, 360, 30):
            ang = math.radians(angle_deg)
            ex  = int(cx + math.cos(ang) * 200)
            ey  = int(cy + math.sin(ang) * 200)
            draw.line([(cx, cy), (ex, ey)], fill=(0, 60, 80), width=1)

        # Labels
        draw.text((8, 8),   "TERRA-SENSE :: TOP-DOWN GPR HEATMAP", fill=(0, 200, 255))
        draw.text((8, 24),  f"Grid X:{grid_x:.1f}m  Y:{grid_y:.1f}m  Signal:{intensity:.0%}", fill=(150, 200, 150))
        draw.text((8, IMG_H - 20), f"Depth:{depth:.2f}m  Temp:{temp:.1f}°C  RH:{humidity:.0f}%  P:{pressure:.0f}hPa", fill=(100, 150, 100))

        img_td = img_td.filter(ImageFilter.GaussianBlur(radius=0.8))
        images.append(img_td)

        # ── View 1: Front Cross-Section (X vs Depth Z) ───────────────────────
        img_fv = Image.new("RGB", (IMG_W, IMG_H), color=(8, 10, 18))
        draw2  = ImageDraw.Draw(img_fv)

        # Soil strata bands
        strata = [
            (0.00, 0.20, (40, 35, 28), "Topsoil"),
            (0.20, 0.42, (55, 45, 30), "Clay"),
            (0.42, 0.65, (65, 55, 38), "Sandy Loam"),
            (0.65, 0.85, (75, 65, 45), "Dense Clay"),
            (0.85, 1.00, (85, 75, 50), "Bedrock"),
        ]
        for (y0r, y1r, col, label) in strata:
            y0px = int(y0r * IMG_H)
            y1px = int(y1r * IMG_H)
            draw2.rectangle([0, y0px, IMG_W, y1px], fill=col)
            draw2.text((6, y0px + 4), label, fill=(200, 190, 170))

        # Victim position in cross-section
        px = int(IMG_W * 0.5 + (grid_x / 20.0) * IMG_W * 0.4)
        py = int((depth / 5.0) * IMG_H * 0.85 + IMG_H * 0.05)
        px = max(30, min(IMG_W - 30, px))
        py = max(30, min(IMG_H - 30, py))

        if intensity > 0.15:
            hr  = int(intensity * 28 + 12)
            # Human capsule silhouette
            draw2.ellipse([px - hr, py - hr * 2, px + hr, py + hr * 2], fill=(220, 120, 60), outline=(255, 200, 100), width=2)
            draw2.ellipse([px - hr // 2, py - hr * 2 - hr, px + hr // 2, py - hr], fill=(240, 170, 120), outline=(255, 220, 150), width=1)
            # Vertical probe line
            draw2.line([(px, 0), (px, py - hr * 2)], fill=(0, 200, 255), width=1)
            draw2.text((px + 5, py - hr * 2 - 24), f"Z={depth:.2f}m", fill=(0, 200, 255))

        # Obstacle blocks above victim
        if intensity > 0.2:
            obx0 = px - np.random.randint(20, 60)
            obx1 = px + np.random.randint(20, 60)
            oby0 = max(5, py - int(depth * 35) - 40)
            oby1 = max(10, py - int(depth * 35))
            draw2.rectangle([obx0, oby0, obx1, oby1], fill=(80, 80, 90), outline=(120, 120, 130))
            draw2.text((obx0, oby0 - 14), "OBSTACLE", fill=(255, 120, 40))

        draw2.text((8, 8), "TERRA-SENSE :: FRONT CROSS-SECTION (X–Z)", fill=(0, 200, 255))
        draw2.line([(0, IMG_H - 1), (IMG_W, IMG_H - 1)], fill=(60, 60, 70), width=2)
        images.append(img_fv)

        # ── View 2: Side Cross-Section (Y vs Depth Z) ────────────────────────
        img_sv = Image.new("RGB", (IMG_W, IMG_H), color=(8, 10, 18))
        draw3  = ImageDraw.Draw(img_sv)

        for (y0r, y1r, col, label) in strata:
            y0px = int(y0r * IMG_H)
            y1px = int(y1r * IMG_H)
            draw3.rectangle([0, y0px, IMG_W, y1px], fill=col)
            draw3.text((6, y0px + 4), label, fill=(200, 190, 170))

        sy  = int(IMG_W * 0.5 + (grid_y / 20.0) * IMG_W * 0.4)
        sz  = int((depth / 5.0) * IMG_H * 0.85 + IMG_H * 0.05)
        sy  = max(30, min(IMG_W - 30, sy))
        sz  = max(30, min(IMG_H - 30, sz))

        if intensity > 0.15:
            hr  = int(intensity * 28 + 12)
            draw3.ellipse([sy - hr, sz - hr * 2, sy + hr, sz + hr * 2], fill=(200, 100, 60), outline=(255, 180, 80), width=2)
            draw3.ellipse([sy - hr // 2, sz - hr * 2 - hr, sy + hr // 2, sz - hr], fill=(230, 160, 110), outline=(255, 200, 130), width=1)
            draw3.line([(sy, 0), (sy, sz - hr * 2)], fill=(0, 200, 255), width=1)
            draw3.text((sy + 5, sz - hr * 2 - 24), f"Z={depth:.2f}m", fill=(0, 200, 255))

        draw3.text((8, 8), "TERRA-SENSE :: SIDE CROSS-SECTION (Y–Z)", fill=(0, 200, 255))
        images.append(img_sv)

        return images

    def _image_to_base64(self, img: "Image.Image") -> str:
        """Convert PIL Image → JPEG base64 string for API payload."""
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 2: Call NVIDIA GPT-OSS-20B NIM via Structured Text Prompt
    # ──────────────────────────────────────────────────────────────────────────

    def call_nim_api(self, features: dict, grid_x: float, grid_y: float) -> dict:
        """
        Build a structured telemetry prompt from sensor features and POST to
        NVIDIA NIM openai/gpt-oss-20b for 3D human-under-obstacle analysis.
        Returns parsed JSON dict or a fallback result on error.
        """
        depth      = float(features.get("reflection_depth", 1.5))
        breathing  = float(features.get("breathing_hz",     0.0))
        heartbeat  = float(features.get("heartbeat_hz",     0.0))
        micro_amp  = float(features.get("micro_amp",        0.0))
        snr_db     = float(features.get("snr_db",           0.0))
        pir        = float(features.get("pir_motion",       0.0))
        radar_s    = float(features.get("radar_state",      0.0))
        radar_e    = float(features.get("radar_energy",     0.0))
        temp_c     = float(features.get("bme_temp_c",      25.0))
        humidity   = float(features.get("bme_humidity_pct",35.0))
        pressure   = float(features.get("bme_pressure_hpa",1013.25))
        dielectric = float(features.get("dielectric_shift", 0.0))
        intensity  = self._sensor_to_intensity(features)

        radar_state_map = {0: "Clear (no target)", 1: "Moving target",
                           2: "Static target", 3: "Both moving+static"}

        # ── System prompt: SAR expert persona ─────────────────────────────────
        system_prompt = (
            "You are an expert AI system for Search and Rescue (SAR) operations, "
            "specialised in 3D subsurface human detection using multi-sensor fusion. "
            "You analyse Ground Penetrating Radar (GPR), 24 GHz Doppler Radar, PIR, "
            "barometric, and humidity sensor telemetry to determine whether a human "
            "is trapped beneath the surface — particularly near or under physical "
            "obstacles such as concrete slabs, rubble piles, or soil compression layers. "
            "Your task: analyse the provided sensor readings and return a precise "
            "JSON object. Respond ONLY with valid JSON — no markdown, no explanation."
        )

        # ── User prompt: full sensor telemetry ────────────────────────────────
        user_prompt = f"""SAR SENSOR TELEMETRY REPORT — TerraSense AI

Grid Position: X={grid_x:.2f}m, Y={grid_y:.2f}m
Scan Signal Intensity (normalised): {intensity:.3f} ({intensity*100:.1f}%)

=== CHANNEL 1: 24 GHz Doppler Vital Radar ===
  Respiration Rate  : {breathing*60:.1f} bpm  ({breathing:.4f} Hz)
  Heartbeat Rate    : {heartbeat*60:.1f} bpm  ({heartbeat:.4f} Hz)
  Chest Displacement: {micro_amp:.4f} µV (micro-amplitude)
  Signal-to-Noise   : {snr_db:.2f} dB
  Radar State       : {int(radar_s)} — {radar_state_map.get(int(radar_s), 'Unknown')}
  Radar Energy      : {radar_e:.1f}%

=== CHANNEL 2: Ground Penetrating Radar (GPR) ===
  Subsurface Reflection Depth : {depth:.3f} metres
  Dielectric Permittivity Shift: {dielectric:.3f} ε (F/m)
  (High dielectric shift > 5 indicates human tissue/air void boundary)

=== CHANNEL 3: PIR Motion Sensor HC-SR501 ===
  PIR Motion State  : {'ACTIVE (body heat detected)' if pir > 0.5 else 'INACTIVE'} ({pir:.1f})

=== CHANNEL 4: BMP180 Barometric Sensor ===
  Temperature       : {temp_c:.2f} °C
  Atmospheric Pressure: {pressure:.2f} hPa

=== CHANNEL 5: DHT11 Humidity Sensor ===
  Relative Humidity : {humidity:.1f}%
  (Humidity > 45% in enclosed space may indicate human respiration moisture)

=== PHYSICAL CONTEXT ===
  Estimated soil stratum at {depth:.2f}m depth:
    {'Topsoil (0–0.3m)' if depth < 0.3 else 'Clay layer (0.3–1.2m)' if depth < 1.2 else 'Sandy loam (1.2–2.5m)' if depth < 2.5 else 'Dense clay / near-bedrock (>2.5m)'}
  Obstacle likelihood: {'HIGH — dielectric shift and SNR pattern suggest obstacle overhead' if dielectric > 5.0 and snr_db < 15 else 'MODERATE' if dielectric > 3.0 else 'LOW'}

Analyse all sensor channels holistically. Consider that humans trapped under obstacles
show attenuated respiration signals, elevated dielectric shift due to body tissue, and
often have low SNR due to signal scatter from obstacle material above them.

Return ONLY this JSON structure (no other text):
{{
  "human_detected": <bool>,
  "confidence": <float 0.0–1.0>,
  "position_3d": {{"x": <float>, "y": <float>, "z": <float>}},
  "obstacle_proximity_m": <float>,
  "obstacle_type": <string from: soil_compression | concrete_slab | rubble_pile | steel_beam | wooden_debris | loose_earth_void>,
  "entrapment_posture": <string from: supine_in_void | foetal_compressed | prone_flat | seated_upright | unknown>,
  "bounding_box_3d": {{"cx": <float>, "cy": <float>, "cz": <float>, "w": 0.55, "h": 1.75, "d": 0.40}},
  "rescue_access_direction": <string from: surface_extraction | top_drill_with_shield | lateral_micro_tunnel | hydraulic_deep_bore | vertical_extraction>,
  "reasoning": <string, brief 1-sentence explanation>
}}"""

        payload = {
            "model":       NIM_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            "max_tokens":  600,
            "temperature": 0.05,
            "top_p":       0.95,
        }

        try:
            print(f"[GPT-OSS-20B] Sending sensor telemetry to NVIDIA NIM ({NIM_MODEL})...")
            response = requests.post(
                NIM_URL,
                headers=self.headers,
                json=payload,
                timeout=40
            )
            response.raise_for_status()
            raw     = response.json()
            content = (raw.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()
            print(f"[GPT-OSS-20B] Response received — {len(content)} chars")

            # Strip markdown code fences if present
            if "```" in content:
                parts = content.split("```")
                content = parts[1] if len(parts) > 1 else parts[0]
                if content.startswith("json"):
                    content = content[4:]

            parsed = json.loads(content.strip())
            print(f"[GPT-OSS-20B] human_detected={parsed.get('human_detected')}, "
                  f"confidence={parsed.get('confidence', 0)*100:.1f}%")
            return {"status": "success", "streampetr_result": parsed, "raw_response": raw}

        except requests.exceptions.HTTPError as e:
            code = e.response.status_code if e.response is not None else '?'
            body = e.response.text[:300] if e.response is not None else ''
            print(f"[GPT-OSS-20B] HTTP {code}: {body}")
            return self._fallback_result(features, reason=f"HTTP {code}")
        except requests.exceptions.Timeout:
            print("[GPT-OSS-20B] Request timed out (40s) — physics fallback active")
            return self._fallback_result(features, reason="API timeout")
        except json.JSONDecodeError as e:
            print(f"[GPT-OSS-20B] JSON parse error: {e}")
            return self._fallback_result(features, reason=f"JSON parse error: {e}")
        except Exception as e:
            print(f"[GPT-OSS-20B] Error: {e}")
            return self._fallback_result(features, reason=str(e))

    # Keep old name as alias for backward compatibility
    def call_streampetr_api(self, images: list, features: dict) -> dict:
        """Backward-compatible alias — now delegates to call_nim_api."""
        return self.call_nim_api(features, 0.0, 0.0)

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 3: Interpret 3D Bounding Boxes → SAR Result
    # ──────────────────────────────────────────────────────────────────────────

    def interpret_boxes(self, api_result: dict, features: dict,
                        grid_x: float, grid_y: float) -> dict:
        """
        Convert raw StreamPETR API result → structured SAR position report.
        Falls back to physics-based estimation if API confidence is low.
        """
        depth     = float(features.get("reflection_depth", 1.5))
        intensity = self._sensor_to_intensity(features)

        # If API returned valid structured result
        sp = api_result.get("streampetr_result", {})
        if sp and isinstance(sp, dict):
            confidence = float(sp.get("confidence", 0.0))
            human_det  = bool(sp.get("human_detected", False))

            if human_det and confidence >= CONFIDENCE_THRESHOLD:
                pos = sp.get("position_3d", {})
                bbox = sp.get("bounding_box_3d", {})
                return {
                    "source":                  "nvidia_streampetr",
                    "human_detected":          True,
                    "confidence_pct":          round(confidence * 100, 1),
                    "position_3d": {
                        "x_m":   float(pos.get("x", grid_x)),
                        "y_m":   float(pos.get("y", grid_y)),
                        "depth_m": float(pos.get("z", depth)),
                    },
                    "bounding_box_3d": bbox,
                    "obstacle_proximity_m":    float(sp.get("obstacle_proximity_m", 0.3)),
                    "obstacle_type":           sp.get("obstacle_type", "unknown"),
                    "entrapment_posture":       sp.get("entrapment_posture", "unknown"),
                    "rescue_access_direction": sp.get("rescue_access_direction", "top_drill"),
                    "near_obstacle":           float(sp.get("obstacle_proximity_m", 0.3)) < 0.5,
                    "under_obstacle":          float(pos.get("z", depth)) > 0.5,
                }

        # ── Physics-based fallback (when API is unavailable or low confidence) ─
        return self._physics_fallback(features, grid_x, grid_y, depth, intensity,
                                      reason=api_result.get("reason", "low_confidence"))

    def _physics_fallback(self, features: dict, grid_x: float, grid_y: float,
                          depth: float, intensity: float, reason: str = "") -> dict:
        """Compute 3D position analytically from sensor readings when API unavailable."""
        breathing  = float(features.get("breathing_hz", 0.0))
        heartbeat  = float(features.get("heartbeat_hz", 0.0))
        dielectric = float(features.get("dielectric_shift", 0.0))

        human_detected = (intensity > 0.35) and (breathing > 0.08 or heartbeat > 0.4)

        # Obstacle proximity heuristic: high dielectric + low SNR → buried under obstacle
        snr = float(features.get("snr_db", 0.0))
        obstacle_proximity = max(0.05, 2.0 - (dielectric / 8.0) - max(0, (snr / 20.0)))
        under_obstacle     = depth > 0.5 and dielectric > 5.0

        # Rescue access direction heuristic
        if depth < 0.8:
            access = "surface_extraction"
        elif obstacle_proximity < 0.3:
            access = "lateral_micro_tunnel"
        elif depth < 2.0:
            access = "top_drill_with_shield"
        else:
            access = "hydraulic_deep_bore"

        # Entrapment posture from aspect ratio of signal ellipse (approximated)
        micro_amp = float(features.get("micro_amp", 0.5))
        if micro_amp > 0.7:
            posture = "supine_in_void"
        elif micro_amp > 0.4:
            posture = "foetal_compressed"
        elif micro_amp > 0.2:
            posture = "prone_flat"
        else:
            posture = "unknown"

        return {
            "source":                  f"physics_fallback ({reason})",
            "human_detected":          human_detected,
            "confidence_pct":          round(intensity * 95, 1),
            "position_3d": {
                "x_m":     grid_x,
                "y_m":     grid_y,
                "depth_m": depth,
            },
            "bounding_box_3d": {
                "cx": grid_x, "cy": grid_y, "cz": -depth,
                "w": 0.55, "h": 1.75, "d": 0.40,
            },
            "obstacle_proximity_m":    round(obstacle_proximity, 2),
            "obstacle_type":           "soil_compression" if under_obstacle else "surface_debris",
            "entrapment_posture":      posture,
            "rescue_access_direction": access,
            "near_obstacle":           obstacle_proximity < 0.5,
            "under_obstacle":          under_obstacle,
        }

    def _fallback_result(self, features: dict, reason: str = "") -> dict:
        """Return a status-only dict that routes to physics fallback."""
        return {"status": "fallback", "reason": reason, "streampetr_result": None}

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 4: Full Analysis Pipeline
    # ──────────────────────────────────────────────────────────────────────────

    def analyze(self, features: dict, grid_x: float = 0.0, grid_y: float = 0.0) -> dict:
        """
        Full GPT-OSS-20B SAR analysis pipeline:
          features  — dict of 12 sensor readings (same keys as ml_model.py)
          grid_x/y  — horizontal grid position in metres
        Returns a comprehensive position + obstacle analysis dict.
        """
        print(f"[GPT-OSS-20B] Analysing position ({grid_x:.1f}m, {grid_y:.1f}m) "
              f"depth={features.get('reflection_depth', '?')}m ...")

        # 1. Render GPR heatmap views (for dashboard preview)
        images = self.render_gpr_heatmap(features, grid_x, grid_y)

        # 2. Call NVIDIA GPT-OSS-20B NIM with structured text prompt
        api_result = self.call_nim_api(features, grid_x, grid_y)

        # 3. Interpret result → SAR position report
        sar_result = self.interpret_boxes(api_result, features, grid_x, grid_y)

        # 4. Attach top-down heatmap preview as base64 for frontend
        if images:
            sar_result["heatmap_preview_b64"] = self._image_to_base64(images[0])

        sar_result["api_status"] = api_result.get("status", "unknown")
        sar_result["model_used"] = NIM_MODEL

        print(f"[GPT-OSS-20B] Done — human_detected={sar_result['human_detected']}, "
              f"confidence={sar_result['confidence_pct']}%, "
              f"depth={sar_result['position_3d']['depth_m']:.2f}m, "
              f"under_obstacle={sar_result['under_obstacle']}")

        return sar_result


# ─── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    analyzer = StreamPETRAnalyzer()
    test_features = {
        "breathing_hz":    0.31,
        "heartbeat_hz":    1.15,
        "micro_amp":       0.85,
        "snr_db":          18.2,
        "pir_motion":      1.0,
        "radar_state":     2.0,
        "radar_energy":    72.0,
        "bme_temp_c":      28.5,
        "bme_humidity_pct": 42.0,
        "bme_pressure_hpa": 1013.5,
        "dielectric_shift":  8.4,
        "reflection_depth":  1.85,
    }
    result = analyzer.analyze(test_features, grid_x=2.5, grid_y=-1.2)
    print("\n[StreamPETR] Final SAR Result:")
    print(json.dumps({k: v for k, v in result.items() if k != "heatmap_preview_b64"}, indent=2))
