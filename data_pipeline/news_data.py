import yfinance as yf
import pandas as pd
from textblob import TextBlob
import re

# --- Event Classifier ---
def classify_event(title: str):
    t = title.lower()

    rules = [
        ("Financial Distress", -15, [r"\bbankrupt", r"\binsolv", r"\bdefault\b", r"debt restructur", r"liquidat"]),
        ("Earnings/Guidance", -10, [r"profit warning", r"guidance (cut|lower)", r"weak demand", r"miss(es|ed) estimates"]),
        ("Regulatory/Legal", -8,  [r"lawsuit", r"\bfine\b", r"\bban\b", r"probe", r"antitrust"]),
        ("Management/Leadership", -8, [r"ceo (resign|steps down|quits)", r"cfo (resign|steps down|quits)"]),
        ("Credit/Leverage", -12, [r"raises? debt", r"bond issue", r"refinanc", r"rating (downgrade|cut)"]),
        ("Growth/Positive", +5,   [r"\bacquisit(ion|ions)\b", r"\bmerger\b", r"partnership", r"record (revenue|profit)", r"beats? estimates"]),
    ]

    for cat, impact, patterns in rules:
        for pat in patterns:
            if re.search(pat, t):
                return cat, impact
    return "Other/Neutral", 0


def fetch_company_news(ticker="RELIANCE.NS"):
    stock = yf.Ticker(ticker)
    news_items = stock.news
    rows = []
    for item in news_items:
        title = item.get("title", "")
        ts = item.get("providerPublishTime", 0)
        published = pd.to_datetime(ts, unit="s", errors="coerce")
        link = item.get("link", "")

        # Sentiment Analysis
        polarity = TextBlob(title).sentiment.polarity
        if polarity > 0.1:
            sentiment, score = "positive", 1
        elif polarity < -0.1:
            sentiment, score = "negative", -1
        else:
            sentiment, score = "neutral", 0

        # Event Classification
        event, event_impact = classify_event(title)

        rows.append([published, title, sentiment, score, event, event_impact, link])

    return pd.DataFrame(rows, columns=[
        "timestamp", "headline", "sentiment", "sentiment_score", "event", "event_impact", "link"
    ])


if __name__ == "__main__":
    df = fetch_company_news("RELIANCE.NS")
    df.to_csv("data_pipeline/news_data.csv", index=False)
    print("✅ Saved news data with events and sentiment to news_data.csv")
    print(df.head())

