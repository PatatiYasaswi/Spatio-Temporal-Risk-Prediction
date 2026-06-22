import pandas as pd
import numpy as np

def generate_safety_dataset(n_events=30000, seed=42):
    np.random.seed(seed)

    area_types = ["market", "residential", "industrial", "park", "road_main", "road_small"]
    purposes = ["shopping", "commuting", "walking_home", "studying", "leisure", "work"]

    base_risk_map = {
        "market": 0.4,
        "residential": 0.3,
        "industrial": 0.6,
        "park": 0.5,
        "road_main": 0.2,
        "road_small": 0.4,
    }

    purpose_risk_map = {
        "shopping": 0.0,
        "commuting": 0.05,
        "walking_home": 0.15,
        "studying": 0.05,
        "leisure": 0.05,
        "work": 0.05,
    }

    data = []

    for i in range(n_events):
        user_id = f"user_{np.random.randint(1, 1001)}"
        hour = np.random.randint(0, 24)
        area_type = np.random.choice(area_types)
        purpose = np.random.choice(purposes)

        crowdedness = np.random.uniform(0, 1)
        weather_risk = np.random.uniform(0, 1)
        lighting = np.random.uniform(0, 1)
        duration_in_area = np.random.uniform(1, 60)

        # time feature
        is_night = 1 if (hour >= 18 or hour <= 5) else 0

        base_risk = base_risk_map[area_type]
        purpose_adj = purpose_risk_map[purpose]

        night_adj = 0.20 * is_night
        crowdedness_adj = 0.15 if crowdedness < 0.3 else 0.0
        weather_adj = 0.10 * weather_risk
        lighting_adj = 0.15 if lighting < 0.3 else 0.0

        # extra risk if walking home at night
        if purpose == "walking_home" and is_night:
            purpose_adj += 0.10

        risk_score = base_risk + purpose_adj + night_adj + crowdedness_adj + weather_adj + lighting_adj

        # clamp to [0, 1]
        risk_score = max(0.0, min(1.0, risk_score))

        if risk_score < 0.3:
            risk_level = "low"
        elif risk_score <= 0.6:
            risk_level = "medium"
        else:
            risk_level = "high"

        data.append({
            "event_id": i,
            "user_id": user_id,
            "hour": hour,
            "is_night": is_night,
            "area_type": area_type,
            "purpose": purpose,
            "crowdedness": crowdedness,
            "weather_risk": weather_risk,
            "lighting": lighting,
            "duration_in_area": duration_in_area,
            "risk_score": risk_score,
            "risk_level": risk_level,
        })

    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    df = generate_safety_dataset(n_events=30000)
    df.to_csv("data/raw/safety_events.csv", index=False)
    print("Dataset saved to data/raw/safety_events.csv")
    print(f"Total events: {len(df)}")
    print(df["risk_level"].value_counts())