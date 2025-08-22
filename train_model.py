import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import sys

# --- Import build_features from your pipeline ---
sys.path.append(os.path.join(os.path.dirname(__file__), "data_pipeline"))
from features import build_features

# Paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
features_file = os.path.join(BASE_DIR, "features_dataset.csv")
model_file = os.path.join(BASE_DIR, "credit_model.pkl")

def compute_rule_based_score(df):
    """Reuse your existing rule-based logic to compute today's score."""
    scores = []
    for _, row in df.iterrows():
        score = 100
        # Stock impact
        if row["stock_close"] > df["stock_close"].mean() * 1.05:
            score -= 10
        # Yield impact
        if row["10y_yield"] > df["10y_yield"].mean():
            score -= 15
        # Sentiment impact
        if row["sentiment_score"] < 0:
            score -= 10
        # Event impact
        if row["event_impact"] < 0:
            score += row["event_impact"]
        scores.append(score)
    return scores

def train_model():
    # --- Ensure dataset exists ---
    if not os.path.exists(features_file):
        print("⚠️ features_dataset.csv not found, generating it...")
        build_features()

    # --- Load features ---
    df = pd.read_csv(features_file)
    df.columns = df.columns.str.strip().str.lower()
    df = df.dropna()

    # ✅ Force numeric conversion
    num_cols = ["stock_close", "10y_yield", "cpi", "sentiment_score", "event_impact"]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()

    # --- Compute target: tomorrow's score ---
    df["score"] = compute_rule_based_score(df)
    df["target_score"] = df["score"].shift(-1)  # next-day score
    df = df.dropna()

    # Features
    X = df[num_cols]
    y = df["target_score"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Model
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print(f"✅ Model trained. RMSE={rmse:.2f}, R²={r2:.2f}")

    # Save model
    joblib.dump(model, model_file)
    print(f"💾 Model saved to {model_file}")

if __name__ == "__main__":
    train_model()
