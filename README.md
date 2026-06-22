# Spatio-Temporal Risk Prediction for Girls’ Safety

A real-time machine learning system that predicts safety risk for women in areas **outside CCTV coverage** using context such as time, location type, purpose (shopping, walking home, etc.), crowdedness, weather, and lighting.

Instead of trying to “see” outside cameras, this system **estimates risk** based on where and when a person is moving, and what they’re doing. It outputs a risk level (low / medium / high), logs predictions, triggers alerts for high risk, and visualizes everything through a Streamlit dashboard.

---

## What This Project Solves

CCTV cameras only monitor fixed zones. Outside these zones (“blind spots”), there is no visual data and no safety system. This project addresses that gap by:

- Predicting **spatio-temporal risk** for women in unmonitored areas.
- Using context parameters instead of video: time, place, purpose, crowdedness, weather, lighting.
- Providing a **real-time simulation** with alerts and a live dashboard.

---

## Key Features

- **Simulated dataset**: 30,000+ synthetic safety events with rule-based risk labels.
- **Feature engineering**: One-hot encoding of `area_type` and `purpose`, plus numeric features (`is_night`, `crowdedness`, `weather_risk`, etc.).
- **ML model**: Random Forest classifier predicting `risk_level` (low/medium/high), saved for real-time inference.
- **Real-time loop**: Generates events one by one, scores risk, logs predictions, and triggers structured high-risk alerts.
- **Alert system**: Modular alert module that prints detailed alerts (time, area type, purpose, risk probability).
- **Logging**: Each prediction saved to `output/realtime_predictions_log.csv` with full event details and timestamp.
- **Dashboard (Streamlit)**:
  - Shows total events and high-risk counts.
  - Displays recent events table.
  - Bar chart of risk-level distribution.
  - Filters by `area_type` and `purpose`.

---

## Tech Stack

- **Language**: Python
- **ML Libraries**: pandas, numpy, scikit-learn (RandomForestClassifier)
- **Visualization**: Streamlit
- **Model Storage**: joblib (`.pkl` file)
- **Logging**: CSV-based log file

---

## Project Structure

```text
spatio_temporal_risk_prediction/
  project_docs/
    problem_statement.md
  data/
    raw/
      safety_events.csv
    processed/
      safety_events_features.csv
  src/
    data_generator.py
    feature_engineering.py
    train_model.py
    real_time_loop.py
    alert_system.py
  models/
    rf_risk_level.pkl
  dashboard/
    app.py
  output/
    realtime_predictions_log.csv
  requirements.txt
  README.md
```

---

## How to Run

### 1. Setup Environment

```bash
python -m venv env
.\env\Scripts\Activate.ps1  # Windows PowerShell
# or: source env/bin/activate  # Linux/Mac

pip install pandas numpy scikit-learn joblib streamlit
```

### 2. Generate Dataset

```bash
python src\data_generator.py
```

This creates `data/raw/safety_events.csv`.

### 3. Engineer Features

```bash
python src\feature_engineering.py
```

This creates `data/processed/safety_events_features.csv`.

### 4. Train the Model

```bash
python src\train_model.py
```

This trains a Random Forest classifier and saves `models/rf_risk_level.pkl`.

### 5. Run Real-Time Simulation (Console)

```bash
python src\real_time_loop.py
```

This:
- Generates events one by one.
- Predicts risk level.
- Logs to `output/realtime_predictions_log.csv`.
- Triggers alerts for high risk.

Press `Ctrl + C` to stop.

### 6. Run Streamlit Dashboard

In a new terminal:

```bash
streamlit run dashboard\app.py
```

Open the shown URL (e.g., `http://localhost:8501`) in your browser to see live risk data.

---

## Resume-Ready Summary

Built an end-to-end ML system that predicts safety risk for women in CCTV-blind areas using spatio-temporal context (time, location, purpose, crowdedness, weather, lighting). The system simulates real-time events, scores risk with a Random Forest model, logs predictions, triggers high-risk alerts, and visualizes results on an interactive Streamlit dashboard.

Key highlights:
- 30,000+ synthetic safety events with rule-based risk labels.
- Feature engineering with one-hot encoding and numeric features.
- Real-time simulation loop and modular alert system.
- Production-like Streamlit dashboard for monitoring and filtering.

---

## Future Extensions (Optional)

- Feature importance visualization (top 10 features driving risk).
- Model performance metrics in dashboard (accuracy, F1-score).
- Configurable risk thresholds.
- Safe route suggestion module (graph-based pathfinding with risk as cost).
- Sequence-based future risk prediction (e.g., risk over next 5–10 minutes using LSTM).

---

## License

This project is for educational and research purposes.
