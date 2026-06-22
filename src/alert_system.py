from datetime import datetime

def send_high_risk_alert(event: dict, risk_prob: float):
    """
    Simulate sending an alert when high risk is detected.
    Right now, it just prints details nicely.
    Later, you can extend this to send email/SMS/WhatsApp, etc.
    """
    timestamp = datetime.now().isoformat(timespec="seconds")
    location = event.get("area_type", "unknown_area")
    purpose = event.get("purpose", "unknown_purpose")

    print("========== HIGH RISK ALERT ==========")
    print(f"Time: {timestamp}")
    print(f"Location type: {location}")
    print(f"Purpose: {purpose}")
    print(f"Full event: {event}")
    print(f"Estimated probability of HIGH risk: {risk_prob:.3f}")
    print("Action: Notify guardian / emergency contact (simulated).")
    print("=====================================\n")