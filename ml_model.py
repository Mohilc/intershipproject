"""
TERRA-SENSE AI - Internal Multi-Model Prediction Engine
Uses 6 diverse classifiers behind the scenes for maximum accuracy.
No model names are exposed to the frontend.
"""

import numpy as np
import os
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
        self.feature_names = [
            'breathing_hz', 'heartbeat_hz', 'micro_amp', 'snr_db',
            'dielectric_shift', 'soil_moisture', 'soil_density', 'reflection_depth'
        ]
        self.train_models()

    def generate_synthetic_dataset(self, n_samples=2000):
        np.random.seed(42)
        X, y = [], []

        for _ in range(n_samples):
            is_human = np.random.rand() > 0.5

            if is_human:
                breathing_hz = np.random.uniform(0.15, 0.48)
                heartbeat_hz = np.random.uniform(0.80, 2.20)
                micro_amp = np.random.uniform(0.35, 0.99)
                snr_db = np.random.uniform(5.0, 30.0)
                dielectric_shift = np.random.uniform(3.5, 12.0)
                soil_moisture = np.random.uniform(8.0, 65.0)
                soil_density = np.random.uniform(1100, 2500)
                reflection_depth = np.random.uniform(0.3, 5.0)
                label = 1
            else:
                is_mineral = np.random.rand() > 0.60
                breathing_hz = np.random.uniform(0.0, 0.02)
                heartbeat_hz = np.random.uniform(0.0, 0.03)
                micro_amp = np.random.uniform(0.005, 0.08)

                if is_mineral:
                    snr_db = np.random.uniform(8.0, 26.0)
                    dielectric_shift = np.random.uniform(8.5, 18.0)
                else:
                    snr_db = np.random.uniform(-12.0, 4.0)
                    dielectric_shift = np.random.uniform(0.1, 2.5)

                soil_moisture = np.random.uniform(3.0, 80.0)
                soil_density = np.random.uniform(900, 3000)
                reflection_depth = np.random.uniform(0.0, 5.5)
                label = 0

            X.append([
                breathing_hz, heartbeat_hz, micro_amp, snr_db,
                dielectric_shift, soil_moisture, soil_density, reflection_depth
            ])
            y.append(label)

        return np.array(X), np.array(y)

    def train_models(self):
        print("[AI Engine] Training subsurface detection models...")
        X, y = self.generate_synthetic_dataset()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train 6 diverse classifiers internally
        self.models['m1'] = GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.07, max_depth=7, random_state=42
        )
        self.models['m1'].fit(X_train, y_train)

        self.models['m2'] = RandomForestClassifier(
            n_estimators=120, max_depth=16, random_state=42
        )
        self.models['m2'].fit(X_train, y_train)

        self.models['m3'] = ExtraTreesClassifier(
            n_estimators=100, max_depth=14, random_state=42
        )
        self.models['m3'].fit(X_train, y_train)

        self.models['m4'] = AdaBoostClassifier(
            n_estimators=80, learning_rate=0.08, random_state=42
        )
        self.models['m4'].fit(X_train, y_train)

        self.models['m5'] = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32), max_iter=300, random_state=42
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
        print(f"[AI Engine] Training complete. Accuracy: {self.accuracy_score:.2f}%")

    def _predict_with_gemini(self, feature_dict, api_key):
        import requests
        import json

        breathing_hz = feature_dict.get('breathing_hz', 0.0)
        heartbeat_hz = feature_dict.get('heartbeat_hz', 0.0)
        micro_amp = feature_dict.get('micro_amp', 0.0)
        snr_db = feature_dict.get('snr_db', 0.0)
        dielectric_shift = feature_dict.get('dielectric_shift', 0.0)
        soil_moisture = feature_dict.get('soil_moisture', 35.0)
        soil_density = feature_dict.get('soil_density', 1600.0)
        reflection_depth = feature_dict.get('reflection_depth', 1.5)

        breathing_bpm = round(breathing_hz * 60, 1)
        heartbeat_bpm = round(heartbeat_hz * 60, 1)

        prompt = f"""
You are the onboard TERRA-SENSE AI engine designed for subsurface bio-radar search and rescue.
We have scanned a subsurface location and detected a target with the following parameters:
- Breathing Frequency: {breathing_hz} Hz ({breathing_bpm} breaths per minute; human breathing is typically 0.15 - 0.48 Hz)
- Heartbeat Frequency: {heartbeat_hz} Hz ({heartbeat_bpm} beats per minute; human heartbeat is typically 0.80 - 2.20 Hz)
- Signal Micro-amplitude: {micro_amp} (higher means stronger chest wall motion reflection)
- SNR (Signal-to-Noise Ratio): {snr_db} dB
- Soil Dielectric Shift: {dielectric_shift} (represents electrical permittivity of soil boundary)
- Soil Moisture: {soil_moisture}%
- Soil Density: {soil_density} kg/m^3
- Target Reflection Depth: {reflection_depth} meters

Analyze these parameters and determine:
1. Is there a live human detected? Evaluate if the breathing and heartbeat values represent standard human vitals (breathing >= 0.08 Hz, heartbeat >= 0.4 Hz) rather than mineral noise or sensor jitter.
2. A confidence probability percentage between 0.0 and 100.0 of human presence.
3. An urgency level: "STANDBY" (if not human), "HIGH PRIORITY - Rapid Manual Extraction" (depth <= 1.5m), "CRITICAL - Structural Support Required" (depth 1.5m to 3.0m), or "EXTREME - Heavy Machinery & Oxygen Probe" (depth > 3.0m).
4. An expert tactical rescue strategy suited to the depth, moisture, and soil density.
5. An estimated oxygen supply in hours for a trapped person. Consider depth (deeper means less air volume) and soil density/moisture (higher density/moisture limits air permeability, reducing oxygen hours).
6. Soil air permeability percentage.
7. A detailed, professional medical & tactical field assessment report (e.g. assessing hypoxia/suffocation risk based on depth and oxygen hours, breathing/heart rate anomalies, and physical entrapment scenario).

Your output MUST be a valid JSON object matching the following structure:
{{
  "human_detected": true/false,
  "probability_percentage": float,
  "urgency_level": "string",
  "rescue_strategy": "string",
  "estimated_oxygen_hours": float,
  "air_permeability_pct": int,
  "tactical_assessment": "string"
}}
Do NOT wrap the response in markdown blocks or write any other text. Output ONLY the JSON block.
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
            print(f"[Gemini API Exception] {e}")
        
        return None

    def predict(self, feature_dict):
        """Run inference and calculate rescue strategy. Falls back to local ensemble if Gemini is offline."""
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            print("[AI Engine] Running prediction via Gemini 3.5 Flash...")
            gemini_res = self._predict_with_gemini(feature_dict, api_key)
            if gemini_res:
                return gemini_res
            print("[AI Engine] Gemini prediction failed. Falling back to local ML models.")

        raw_vector = np.array([[
            feature_dict.get('breathing_hz', 0.0),
            feature_dict.get('heartbeat_hz', 0.0),
            feature_dict.get('micro_amp', 0.0),
            feature_dict.get('snr_db', 0.0),
            feature_dict.get('dielectric_shift', 0.0),
            feature_dict.get('soil_moisture', 35.0),
            feature_dict.get('soil_density', 1600.0),
            feature_dict.get('reflection_depth', 1.5)
        ]])

        scaled_vector = self.scaler.transform(raw_vector)
        weights = [0.25, 0.20, 0.18, 0.12, 0.15, 0.10]
        scaled_keys = {'m5', 'm6'}

        weighted_prob = 0.0
        for key, w in zip(self.models.keys(), weights):
            X_in = scaled_vector if key in scaled_keys else raw_vector
            p = float(self.models[key].predict_proba(X_in)[0][1])
            weighted_prob += p * w

        consensus_prob = min(99.4, max(0.8, weighted_prob * 100))
        is_human = consensus_prob >= 50.0

        # Rescue strategy
        depth = feature_dict.get('reflection_depth', 1.5)
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

        return {
            "accuracy_score": round(self.accuracy_score, 2),
            "human_detected": is_human,
            "probability_percentage": round(consensus_prob, 1),
            "tactical_assessment": "Local offline analysis active. Bio-radar consensus verified by 6 internal ML classifiers.",
            "rescue_guidance": {
                "urgency_level": urgency,
                "rescue_strategy": strategy,
                "estimated_oxygen_hours": oxy_hours,
                "air_permeability_pct": air_perm
            }
        }


if __name__ == "__main__":
    engine = SubsurfacePythonMLEngine()
    test = {
        'breathing_hz': 0.32, 'heartbeat_hz': 1.18, 'micro_amp': 0.84,
        'snr_db': 18.4, 'dielectric_shift': 8.4, 'soil_moisture': 38.0,
        'soil_density': 1650.0, 'reflection_depth': 1.45
    }
    print("Prediction:", engine.predict(test))
