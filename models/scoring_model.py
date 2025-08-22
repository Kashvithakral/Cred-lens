import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
import joblib

def train_scoring_model():
    # Load features
    df = pd.read_csv("features.csv")
    df = df.dropna()

    # Features (X)
    X = df[["Stock_Close", "10Y_Yield", "CPI", "sentiment_score"]]

    # Define target
    df["volatility"] = df["Stock_Close"].pct_change().abs()
    df["target"] = ((df["volatility"] > 0.02) | (df["sentiment_score"] < 0)).astype(int)
    y = df["target"]

    try:
        # Train logistic regression if both classes exist
        if len(y.unique()) > 1:
            model = LogisticRegression()
            model.fit(X, y)
            joblib.dump(model, "models/scoring_model.pkl")
            print("✅ Logistic model trained and saved as scoring_model.pkl")

            latest_features = X.iloc[-1:].values
            score = model.predict_proba(latest_features)[0][1] * 100
        else:
            raise ValueError("Only one class present in target")
    except Exception as e:
        print(f"⚠️ Model training failed: {e}")
        print("➡️ Using simple rule-based scoring instead.")

        # Simple rule-based scoring
        latest = df.iloc[-1]
        score = 100
        if latest["volatility"] > 0.02:
            score -= 20
        if latest["10Y_Yield"] > df["10Y_Yield"].mean():
            score -= 15
        if latest["sentiment_score"] < 0:
            score -= 10

    print(f"Latest score: {score:.2f}/100")

if __name__ == "__main__":
    train_scoring_model()


