import pandas as pd
from pathlib import Path

RAW_DATA_PATH = Path("data/raw/safety_events.csv")
PROCESSED_DATA_PATH = Path("data/processed/safety_events_features.csv")

def load_raw_data(path=RAW_DATA_PATH):
    df = pd.read_csv(path)
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # Select base features
    feature_cols = [
        "hour",
        "is_night",
        "area_type",
        "purpose",
        "crowdedness",
        "weather_risk",
        "lighting",
        "duration_in_area",
    ]

    df_feat = df[feature_cols].copy()

    # One-hot encode categorical variables
    df_feat = pd.get_dummies(df_feat, columns=["area_type", "purpose"], drop_first=True)

    # Add target columns
    df_feat["risk_score"] = df["risk_score"]
    df_feat["risk_level"] = df["risk_level"]

    return df_feat

def main():
    print("Loading raw data...")
    df_raw = load_raw_data()
    print(f"Raw shape: {df_raw.shape}")

    print("Engineering features...")
    df_feat = engineer_features(df_raw)
    print(f"Processed shape: {df_feat.shape}")

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_feat.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Saved processed data to {PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    main()