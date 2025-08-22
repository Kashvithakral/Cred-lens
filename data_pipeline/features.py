import os
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

macro_file = os.path.join(BASE_DIR, "macro_data.csv")
news_file = os.path.join(BASE_DIR, "news_data.csv")
stock_file = os.path.join(BASE_DIR, "reliance_stock.csv")
output_file = os.path.join(BASE_DIR, "features_dataset.csv")  # define output file

def build_features():
    """
    Merge stock, macroeconomic, and news sentiment data into one features dataset.
    Handles column name variations and missing data gracefully.
    """

    # --- Load stock data ---
    stock_df = pd.read_csv(stock_file)
    stock_df.columns = stock_df.columns.str.strip().str.lower()
    if "date" not in stock_df.columns or "close" not in stock_df.columns:
        raise ValueError("Stock CSV must have 'Date' and 'Close' columns (case-insensitive)")

    stock_df["date"] = pd.to_datetime(stock_df["date"], errors="coerce")
    stock_df = stock_df.rename(columns={"close": "stock_close"})
    stock_df = stock_df[["date", "stock_close"]]

    # --- Load macro data ---
    macro_df = pd.read_csv(macro_file)
    macro_df.columns = macro_df.columns.str.strip().str.lower()
    for col in ["date", "10y_yield", "cpi"]:
        if col not in macro_df.columns:
            raise ValueError(f"Macro CSV must have '{col}' column")

    macro_df["date"] = pd.to_datetime(macro_df["date"], errors="coerce")
    macro_df = macro_df[["date", "10y_yield", "cpi"]]

    # --- Load news data ---
    news_df = pd.read_csv(news_file)
    news_df.columns = news_df.columns.str.strip().str.lower()
    if "timestamp" not in news_df.columns or "sentiment_score" not in news_df.columns or "event_impact" not in news_df.columns:
        raise ValueError("News CSV must have 'timestamp', 'sentiment_score', 'event_impact'")

    news_df["timestamp"] = pd.to_datetime(news_df["timestamp"], errors="coerce")
    news_df["date"] = news_df["timestamp"].dt.date
    news_df["date"] = pd.to_datetime(news_df["date"])

    # Aggregate: avg sentiment score + worst (min) event impact per day
    news_daily = news_df.groupby("date").agg({
        "sentiment_score": "mean",
        "event_impact": "min"
    }).reset_index()

    # --- Merge all sources ---
    df = pd.merge(stock_df, macro_df, on="date", how="outer")
    df = pd.merge(df, news_daily, on="date", how="outer")

    # --- Sort & clean ---
    df = df.sort_values("date").reset_index(drop=True)

    # Fill missing values
    df[["stock_close", "10y_yield", "cpi"]] = df[["stock_close", "10y_yield", "cpi"]].fillna(method="ffill")
    df["sentiment_score"] = df["sentiment_score"].fillna(0)
    df["event_impact"] = df["event_impact"].fillna(0)

    # --- Save ---
    df.to_csv(output_file, index=False)
    print(f"✅ Features dataset saved to {output_file}")
    print(df.tail())

if __name__ == "__main__":
    build_features()
