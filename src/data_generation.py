# Synthetic data generation
import numpy as np
import pandas as pd

def generate_synthetic_data(n_samples=2000, defect_rate=0.05, random_state=42):
    np.random.seed(random_state)
    
    df = pd.DataFrame({
        "oven_temperature_c": np.random.normal(220, 15, n_samples),
        "molding_pressure_bar": np.random.normal(150, 20, n_samples),
        "line_speed_mpm": np.random.normal(45, 5, n_samples),
        "ambient_humidity_pct": np.random.normal(45, 10, n_samples),
        "material_thickness_mm": np.random.normal(2.5, 0.3, n_samples),
        "material_strength_mpa": np.random.normal(350, 40, n_samples),
        "cycle_time_sec": np.random.normal(12, 2, n_samples),
        "machine_vibration_hz": np.random.uniform(0.5, 3.5, n_samples),
        "tool_age_hours": np.random.uniform(0, 500, n_samples),
        "shift": np.random.choice([1, 2, 3], n_samples),
        "operator_experience_years": np.random.uniform(1, 20, n_samples),
        "days_since_maintenance": np.random.uniform(1, 30, n_samples),
    })

    defect_score = (
        0.15 * np.abs(df["oven_temperature_c"] - 220)
        + 0.10 * np.abs(df["molding_pressure_bar"] - 150)
        + 0.12 * df["machine_vibration_hz"]
        + 0.08 * (df["tool_age_hours"] / 100)
        + 0.05 * df["days_since_maintenance"]
        - 0.03 * df["operator_experience_years"]
        + np.random.normal(0, 0.5, n_samples)
    )

    threshold = np.quantile(defect_score, 1 - defect_rate)
    df["defective"] = (defect_score > threshold).astype(int)
    return df


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    df = generate_synthetic_data()
    df.to_csv("data/synthetic_production_data.csv", index=False)
    print("Synthetic dataset created at data/synthetic_production_data.csv")
