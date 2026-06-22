import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

DATA_PATH = Path("data/processed/safety_events_features.csv")
MODEL_PATH = Path("models/rf_risk_level.pkl")

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    return df

def split_features_target(df: pd.DataFrame):
    # All columns except targets
    feature_cols = [col for col in df.columns if col not in ["risk_score", "risk_level"]]
    X = df[feature_cols]
    y = df["risk_level"]
    return X, y

def train_random_forest(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return clf

def save_model(model, path=MODEL_PATH):
    import joblib
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to {path}")

def main():
    print("Loading processed data...")
    df = load_data()
    print(f"Data shape: {df.shape}")

    X, y = split_features_target(df)
    print(f"Feature shape: {X.shape}, Target shape: {y.shape}")

    print("Training Random Forest model...")
    model = train_random_forest(X, y)

    save_model(model)

if __name__ == "__main__":
    main()