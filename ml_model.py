"""
TERRA-SENSE AI - Multi-Sensor Feature Fusion & Ensemble Prediction Engine
Integrates hardware data from BME280, PIR, and 24 GHz Radar sensors.
Uses 6 diverse classifiers internally + Gemini AI multi-modal reasoning.
"""

import numpy as np
import os
import requests
import json
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class SubsurfacePythonMLEngine:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.accuracy_score = 0.0
        
        # 12 Multi-Sensor Fusion Features
        self.feature_names = [
            'breathing_hz',       # 0: Respiration Doppler Frequency (Hz)
            'heartbeat_hz',       # 1: Pulse Doppler Frequency (Hz)
            'pir_motion',         # 2: PIR Infrared Motion State (0 or 1)
            'radar_state',        # 3: 24GHz Radar Target State (0=None, 1=Moving, 2=Static, 3=Both)
            'radar_energy',       # 4: 24GHz Radar Target Signal Energy (0 - 100)
            'micro_amp',          # 5: Micro-Doppler Amplitude (Chest Wall Motion)
            'snr_db',             # 6: Signal-to-Noise Ratio (dB)
            'bme_temp_c',         # 7: Ambient / Body Heat Anomaly Temp (°C)
            'bme_humidity_pct',   # 8: Relative Humidity / Moisture (%)
            'bme_pressure_hpa',   # 9: Atmospheric Pressure (hPa)
            'dielectric_shift',   # 10: Soil Dielectric Constant Shift
            'reflection_depth'    # 11: Subsurface Reflection Depth (meters)
        ]
        self.train_models()

    def generate_synthetic_dataset(self, n_samples=2500):
        """Generates realistic synthetic dataset combining 3 physical sensors (BME280, PIR, 24GHz Radar)."""
        np.random.seed(42)
        X, y = [], []

        for _ in range(n_samples):
            is_human = np.random.rand() > 0.5

            if is_human:
                # Trapped victim scenario: active vitals & sensor indicators
                breathing_hz = np.random.uniform(0.15, 0.48)
                heartbeat_hz = np.random.uniform(0.80, 2.20)
                
                # PIR: 85% chance motion detected, 15% static trapped under heavy rubble
                pir_motion = 1.0 if np.random.rand() < 0.85 else 0.0
                
                # Radar: Target present (1=moving, 2=static, 3=both)
                radar_state = float(np.random.choice([1, 2, 3], p=[0.25, 0.45, 0.30]))
                radar_energy = np.random.uniform(35.0, 98.0)
                micro_amp = np.random.uniform(0.35, 0.99)
                snr_db = np.random.uniform(5.0, 32.0)
                
                # BME280: Thermal anomaly (body heat reflection + ambient)
                bme_temp_c = np.random.uniform(24.5, 32.0)
                bme_humidity_pct = np.random.uniform(25.0, 75.0)
                bme_pressure_hpa = np.random.uniform(990.0, 1025.0)
                
                dielectric_shift = np.random.uniform(3.5, 12.0)
                reflection_depth = np.random.uniform(0.3, 5.0)
                label = 1
            else:
                # Clear matrix / mineral clutter scenario
                is_mineral = np.random.rand() > 0.60
                breathing_hz = np.random.uniform(0.0, 0.03)
                heartbeat_hz = np.random.uniform(0.0, 0.04)
                
                # PIR & Radar: No target (rare noise jitter < 5%)
                pir_motion = 1.0 if np.random.rand() < 0.04 else 0.0
                radar_state = 1.0 if np.random.rand() < 0.05 else 0.0
                radar_energy = np.random.uniform(0.0, 12.0) if radar_state > 0 else 0.0
                micro_amp = np.random.uniform(0.005, 0.08)

                if is_mineral:
                    snr_db = np.random.uniform(8.0, 24.0) # High reflection but zero vitals
                    dielectric_shift = np.random.uniform(8.5, 18.0)
                else:
                    snr_db = np.random.uniform(-12.0, 4.0)
                    dielectric_shift = np.random.uniform(0.1, 2.5)

                bme_temp_c = np.random.uniform(18.0, 24.0) # Standard cool background soil
                bme_humidity_pct = np.random.uniform(5.0, 80.0)
                bme_pressure_hpa = np.random.uniform(980.0, 1030.0)
                reflection_depth = np.random.uniform(0.0, 5.5)
                label = 0

            X.append([
                breathing_hz, heartbeat_hz, pir_motion, radar_state, radar_energy,
                micro_amp, snr_db, bme_temp_c, bme_humidity_pct, bme_pressure_hpa,
                dielectric_shift, reflection_depth
            ])
            y.append(label)

        return np.array(X), np.array(y)

    def train_models(self):
        print("[AI Engine] Training multi-sensor fusion models (BME280, PIR, 24GHz Radar)...")
        X, y = self.generate_synthetic_dataset()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train 6 diverse classifiers on 12 multi-sensor features
        self.models['m1'] = GradientBoostingClassifier(
            n_estimators=120, learning_rate=0.06, max_depth=7, random_state=42
        )
        self.models['m1'].fit(X_train, y_train)

        self.models['m2'] = RandomForestClassifier(
            n_estimators=150, max_depth=16, random_state=42
        )
        self.models['m2'].fit(X_train, y_train)

        self.models['m3'] = ExtraTreesClassifier(
            n_estimators=120, max_depth=14, random_state=42
        )
        self.models['m3'].fit(X_train, y_train)

        self.models['m4'] = AdaBoostClassifier(
            n_estimators=100, learning_rate=0.08, random_state=42
        )
        self.models['m4'].fit(X_train, y_train)

        self.models['m5'] = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32), max_iter=350, random_state=42
        )
        self.models['m5'].fit(X_train_scaled, y_train)

        self.models['m6'] = KNeighborsClassifier(n_neighbors=7)
        self.models['m6'].fit(X_train_scaled, y_train)

        # Weighted consensus evaluation
        weights = [0.25, 0.20, 0.18, 0.12, 0.15, 0.10]
        scaled_keys = {'m5', 'm6'}

        probs = []
        for key, w in zip(self.models.keys(), weights):
            X_eval = X_test_scaled if key in scaled_keys else X_test
            p = self.models[key].predict_proba(X_eval)[:, 1] * w
            probs.append(p)

        ensemble_probs = sum(probs)
        ensemble_preds = ensemble_probs >= 0.50

        self.accuracy_score = float(accuracy_score(y_test, ensemble_preds) * 100)
        print(f"[AI Engine] Multi-sensor training complete. Consensus Accuracy: {self.accuracy_score:.2f}%")

    def _predict_with_gemini(self, feature_dict, api_key):
        """Runs multi-sensor reasoning using Gemini AI Flash model."""
        breathing_hz = feature_dict.get('breathing_hz', 0.0)
        heartbeat_hz = feature_dict.get('heartbeat_hz', 0.0)
        pir_motion = feature_dict.get('pir_motion', 0)
        radar_state = feature_dict.get('radar_state', 0)
        radar_energy = feature_dict.get('radar_energy', 0.0)
        micro_amp = feature_dict.get('micro_amp', 0.0)
        snr_db = feature_dict.get('snr_db', 0.0)
        bme_temp_c = feature_dict.get('bme_temp_c', 25.0)
        bme_humidity_pct = feature_dict.get('bme_humidity_pct', 35.0)
        bme_pressure_hpa = feature_dict.get('bme_pressure_hpa', 1013.25)
        dielectric_shift = feature_dict.get('dielectric_shift', 1.0)
        soil_density = feature_dict.get('soil_density', 1600.0)
        reflection_depth = feature_dict.get('reflection_depth', 1.5)

        breathing_bpm = round(breathing_hz * 60, 1)
        heartbeat_bpm = round(heartbeat_hz * 60, 1)

        radar_state_desc = {
            0: "No target detected",
            1: "Moving target detected",
            2: "Static target detected (respiration / chest wall micro-motion)",
            3: "Moving & static target detected"
        }.get(int(radar_state), "Unknown state")

        prompt = f"""
You are the onboard TERRA-SENSE AI engine for subsurface bio-radar search and rescue.
We have captured live readings across 3 physical hardware sensors:

1. 24 GHz FMCW Radar Sensor (HLK-LD2410):
   - Target State: {radar_state_desc} (State ID: {radar_state})
   - Target Signal Energy: {radar_energy} / 100
   - Respiration Doppler Frequency: {breathing_hz} Hz ({breathing_bpm} breaths/min)
   - Pulse Doppler Frequency: {heartbeat_hz} Hz ({heartbeat_bpm} beats/min)
   - Signal Micro-amplitude: {micro_amp} (Chest wall displacement index)
   - SNR (Signal-to-Noise Ratio): {snr_db} dB
   - Target Reflection Depth: {reflection_depth} meters

2. PIR Motion Sensor (Infrared):
   - Motion State: {"ACTIVE MOTION DETECTED (HIGH)" if pir_motion == 1 else "CLEAR / NO MOTION (LOW)"}

3. BME280 Environmental Sensor:
   - Ambient / Thermal Anomaly Temp: {bme_temp_c} °C
   - Relative Humidity / Soil Moisture Proxy: {bme_humidity_pct}%
   - Atmospheric Pressure: {bme_pressure_hpa} hPa
   - Soil Dielectric Shift: {dielectric_shift}
   - Soil Density: {soil_density} kg/m^3

Analyze this multi-sensor matrix and evaluate:
1. Is a live human victim detected? Perform cross-sensor fusion (e.g. verify if Doppler vitals, Radar target energy, PIR motion, and thermal/moisture signatures align).
2. A confidence probability percentage (0.0 to 100.0) of human presence.
3. Urgency level: "STANDBY" (if not human), "HIGH PRIORITY - Rapid Manual Extraction" (depth <= 1.5m), "CRITICAL - Structural Support Required" (depth 1.5m to 3.0m), or "EXTREME - Heavy Machinery & Oxygen Probe" (depth > 3.0m).
4. An expert tactical rescue strategy tailored to the depth, soil moisture, and density.
5. Estimated oxygen supply in hours for a trapped person.
6. Soil air permeability percentage.
7. A detailed tactical field assessment report citing specific sensor readings (BME280 temp/humidity, PIR status, and 24 GHz Radar energy/vitals).

Output MUST be a valid JSON object matching this structure:
{{
  "human_detected": true/false,
  "probability_percentage": float,
  "urgency_level": "string",
  "rescue_strategy": "string",
  "estimated_oxygen_hours": float,
  "air_permeability_pct": int,
  "tactical_assessment": "string"
}}
Do NOT wrap the response in markdown formatting or write any other text. Output ONLY the raw JSON block.
"""

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json"
            }
        }
        
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=10)
            if r.status_code == 200:
                res_data = r.json()
                text = res_data['candidates'][0]['content']['parts'][0]['text'].strip()
                pred = json.loads(text)
                return {
                    "accuracy_score": round(self.accuracy_score, 2),
                    "human_detected": bool(pred.get("human_detected", False)),
                    "probability_percentage": float(pred.get("probability_percentage", 0.0)),
                    "tactical_assessment": str(pred.get("tactical_assessment", "")),
                    "rescue_guidance": {
                        "urgency_level": str(pred.get("urgency_level", "STANDBY")),
                        "rescue_strategy": str(pred.get("rescue_strategy", "No action needed")),
                        "estimated_oxygen_hours": float(pred.get("estimated_oxygen_hours", 0.0)),
                        "air_permeability_pct": int(pred.get("air_permeability_pct", 0))
                    }
                }
            else:
                print(f"[Gemini API Error] Status {r.status_code}: {r.text}")
        except Exception as e:
            print("[Gemini API Exception]", e)
        
        return None

    def predict(self, feature_dict):
        """Run multi-sensor inference. Uses Gemini API if configured, or falls back to local 6-model ML ensemble."""
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            print("[AI Engine] Running multi-sensor inference via Gemini AI...")
            gemini_res = self._predict_with_gemini(feature_dict, api_key)
            if gemini_res:
                return gemini_res
            print("[AI Engine] Gemini API unavailable. Using 6-model ML ensemble fallback.")

        # Extract 12 Multi-Sensor features
        b_hz = feature_dict.get('breathing_hz', 0.0)
        h_hz = feature_dict.get('heartbeat_hz', 0.0)
        pir_m = float(feature_dict.get('pir_motion', 1 if (b_hz > 0.08 or h_hz > 0.4) else 0))
        rad_st = float(feature_dict.get('radar_state', 2 if (b_hz > 0.08 or h_hz > 0.4) else 0))
        rad_en = float(feature_dict.get('radar_energy', 75.0 if (b_hz > 0.08 or h_hz > 0.4) else 0.0))
        m_amp = feature_dict.get('micro_amp', 0.0)
        snr = feature_dict.get('snr_db', -12.0)
        temp_c = feature_dict.get('bme_temp_c', feature_dict.get('temperature_c', 25.0))
        humid = feature_dict.get('bme_humidity_pct', feature_dict.get('soil_moisture', 35.0))
        press = feature_dict.get('bme_pressure_hpa', feature_dict.get('pressure_hpa', 1013.25))
        diel = feature_dict.get('dielectric_shift', 1.0)
        depth = feature_dict.get('reflection_depth', 1.5)

        raw_vector = np.array([[
            b_hz, h_hz, pir_m, rad_st, rad_en, m_amp, snr, temp_c, humid, press, diel, depth
        ]])

        scaled_vector = self.scaler.transform(raw_vector)
        weights = [0.25, 0.20, 0.18, 0.12, 0.15, 0.10]
        scaled_keys = {'m5', 'm6'}

        weighted_prob = 0.0
        for key, w in zip(self.models.keys(), weights):
            X_in = scaled_vector if key in scaled_keys else raw_vector
            p = float(self.models[key].predict_proba(X_in)[0][1])
            weighted_prob += p * w

        consensus_prob = min(99.6, max(0.4, weighted_prob * 100))
        is_human = consensus_prob >= 50.0

        # Rescue Strategy Calculation
        density = feature_dict.get('soil_density', 1600.0)
        air_perm = max(12, min(95, round(100 - (density / 28) - (depth * 9))))
        oxy_hours = round(max(2.5, 38.0 - (depth * 7.0) - (density / 180)), 1)

        if depth <= 1.5:
            strategy = "Shallow Trench: Manual Shoring & Vacuum Excavation"
            urgency = "HIGH PRIORITY - Rapid Manual Extraction"
        elif depth <= 3.0:
            strategy = "Medium Rubble: Hydraulic Trench Shield & Micro-Tunnel Probe"
            urgency = "CRITICAL - Structural Support Required"
        else:
            strategy = "Deep Collapse: Heavy Excavator + Directional Oxygen Injection"
            urgency = "EXTREME - Heavy Machinery & Oxygen Probe"

        assessment = f"Multi-Sensor Analysis Active. PIR: {'ACTIVE' if pir_m == 1 else 'CLEAR'} | Radar State: {int(rad_st)} (Energy: {rad_en:.0f}) | BME280: {temp_c:.1f}°C / {humid:.1f}% RH. Consensus verified by 6 ensemble ML classifiers."

        return {
            "accuracy_score": round(self.accuracy_score, 2),
            "human_detected": is_human,
            "probability_percentage": round(consensus_prob, 1),
            "tactical_assessment": assessment,
            "rescue_guidance": {
                "urgency_level": urgency,
                "rescue_strategy": strategy,
                "estimated_oxygen_hours": oxy_hours,
                "air_permeability_pct": air_perm
            }
        }


if __name__ == "__main__":
    engine = SubsurfacePythonMLEngine()
    test_sample = {
        'breathing_hz': 0.32, 'heartbeat_hz': 1.18, 'pir_motion': 1,
        'radar_state': 2, 'radar_energy': 82.0, 'micro_amp': 0.84,
        'snr_db': 18.4, 'bme_temp_c': 27.2, 'bme_humidity_pct': 42.0,
        'bme_pressure_hpa': 1009.5, 'dielectric_shift': 8.4, 'reflection_depth': 1.45
    }
    print("Test Multi-Sensor Inference:", engine.predict(test_sample))
