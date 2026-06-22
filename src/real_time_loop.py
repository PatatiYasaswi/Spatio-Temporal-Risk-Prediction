import time
import numpy as np
import pandas as pd
from pathlib import Path
LOG_PATH = Path("output/realtime_predictions_log.csv")

import joblib
from alert_system import send_high_risk_alert

MODEL_PATH = Path("models/rf_risk_level.pkl")

# We will use the columns from the processed training data
PROCESSED_DATA_PATH = Path("data/processed/safety_events_features.csv")

def load_model(model_path=MODEL_PATH):
    model = joblib.load(model_path)
    return model

def load_feature_template(path=PROCESSED_DATA_PATH):
    """
    Load processed data once to get the list of feature columns (X columns).
    We will use this to ensure our real-time features have the same columns.
    """
    df = pd.read_csv(path)
    feature_cols = [col for col in df.columns if col not in ["risk_score", "risk_level"]]
    return feature_cols

def simulate_event():
    """
    Generate one fake event (like in data_generator),
    but only for a single row.
    """
    area_types = ["market", "residential", "industrial", "park", "road_main", "road_small"]
    purposes = ["shopping", "commuting", "walking_home", "studying", "leisure", "work"]

    hour = np.random.randint(0, 24)
    is_night = 1 if (hour >= 18 or hour <= 5) else 0
    area_type = np.random.choice(area_types)
    purpose = np.random.choice(purposes)

    crowdedness = np.random.uniform(0, 1)
    weather_risk = np.random.uniform(0, 1)
    lighting = np.random.uniform(0, 1)
    duration_in_area = np.random.uniform(1, 60)

    event = {
        "hour": hour,
        "is_night": is_night,
        "area_type": area_type,
        "purpose": purpose,
        "crowdedness": crowdedness,
        "weather_risk": weather_risk,
        "lighting": lighting,
        "duration_in_area": duration_in_area,
    }

    return event

def event_to_features(event: dict, feature_cols: list) -> pd.DataFrame:
    """
    Convert one event dict to a DataFrame with the same columns as training features.
    We need to one-hot encode area_type and purpose, then align columns.
    """
    df = pd.DataFrame([event])

    # One-hot encode
    df_enc = pd.get_dummies(df, columns=["area_type", "purpose"], drop_first=True)

    # Add missing columns with 0, and ensure column order matches feature_cols
    for col in feature_cols:
        if col not in df_enc.columns:
            df_enc[col] = 0

    # Extra columns (if any) are dropped
    df_enc = df_enc[feature_cols]

    return df_enc

def main_loop():
    print("Loading model and feature template...")
    model = load_model()
    feature_cols = load_feature_template()
    print(f"Number of feature columns: {len(feature_cols)}")

    print("Starting real-time simulation. Press Ctrl+C to stop.\n")

    while True:
        event = simulate_event()
        features_df = event_to_features(event, feature_cols)

        pred_level = model.predict(features_df)[0]
        # predict_proba gives probabilities for each class in model.classes_
        proba = model.predict_proba(features_df)[0]

        # map class -> prob
        class_probs = dict(zip(model.classes_, proba))
        risk_prob = class_probs.get("high", 0.0)

        # Print event and prediction
        print("-" * 50)
        print(f"Event: {event}")
        print(f"Predicted risk level: {pred_level}")
        print(f"Probability of HIGH risk: {risk_prob:.3f}")
        
        log_prediction(event, pred_level, risk_prob)
        
        if pred_level == "high":
            send_high_risk_alert(event, risk_prob)
        else:
            print("Status: Normal / manageable risk.\n")

        # Wait a few seconds before next event
        time.sleep(3)
    
def log_prediction(event: dict, pred_level: str, risk_prob: float):
    """
    Append the event + prediction to a CSV log for the dashboard.
    """
    row = {
        **event,
        "predicted_risk_level": pred_level,
        "prob_high_risk": risk_prob,
        "timestamp": pd.Timestamp.now().isoformat(),
    }

    df_row = pd.DataFrame([row])

    # If file exists, append without header; else create with header
    if LOG_PATH.exists():
        df_row.to_csv(LOG_PATH, mode="a", header=False, index=False)
    else:
        df_row.to_csv(LOG_PATH, mode="w", header=True, index=False)

if __name__ == "__main__":
    main_loop()