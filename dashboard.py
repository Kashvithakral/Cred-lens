import os
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import yfinance as yf
from textblob import TextBlob
import joblib
import re

# Optional SHAP import (we'll fall back gracefully if not installed)
try:
    import shap  # type: ignore
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
HERE = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
MODEL_PATH = os.path.join(HERE, "credit_model.pkl")
FEATURES_PATH = os.path.join(HERE, "features_dataset.csv")


# ---------------------------------------------------------------------
# Streamlit Page Config
# ---------------------------------------------------------------------
st.set_page_config(page_title="Explainable Credit Score", layout="wide")

# --- Auto Refresh every 2 min ---
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=120_000, key="auto_refresh")
except ImportError:
    st.error("This app requires the 'streamlit-autorefresh' component. Please install it.")
    st.code("pip install streamlit-autorefresh")
    st.stop()

st.title("📊 Explainable Credit Score Dashboard")
st.caption(
    "Score = Base 100 minus penalties from stock, yield, and news sentiment. "
    "Tune factor weights in the sidebar and refresh stock prices in real-time."
)


# ---------------------------------------------------------------------
# Cached Loaders
# ---------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_features(path=FEATURES_PATH):
    if not os.path.exists(path):
        st.error(f"features_dataset.csv not found at: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)

    # Normalize column names (strip spaces + lowercase)
    df.columns = df.columns.str.strip().str.lower()

    # Rename into consistent schema (mixed case for display columns)
    rename_map = {
        "date": "date",
        "stock_close": "Stock_Close",
        "close": "Stock_Close",         # sometimes only "Close" exists
        "10y_yield": "10Y_Yield",
        "cpi": "CPI",
        "sentiment_score": "sentiment_score",
        "event_impact": "event_impact",
    }
    df = df.rename(columns=rename_map)

    if "date" not in df.columns:
        st.error("❌ 'date' column missing in features_dataset.csv")
        return pd.DataFrame()

    # Ensure dtypes
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["Stock_Close", "10Y_Yield", "CPI", "sentiment_score", "event_impact"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def fetch_company_news(ticker_symbol: str) -> pd.DataFrame:
    """
    Returns columns: timestamp, headline, sentiment, sentiment_score, event, event_impact, link
    """
    try:
        stock = yf.Ticker(ticker_symbol)
        news_items = stock.news or []
    except Exception:
        news_items = []

    rows = []
    for item in news_items:
        title = item.get("title", "") or ""
        ts = item.get("providerPublishTime", 0)
        published = pd.to_datetime(ts, unit="s", errors="coerce")
        link = item.get("link", "")

        # Sentiment (headline-level)
        polarity = TextBlob(title).sentiment.polarity
        if polarity > 0.1:
            sentiment, sent_score = "positive", 1
        elif polarity < -0.1:
            sentiment, sent_score = "negative", -1
        else:
            sentiment, sent_score = "neutral", 0

        # Event classification
        event, event_impact = classify_event(title)

        rows.append([published, title, sentiment, sent_score, event, event_impact, link])

    return pd.DataFrame(rows, columns=[
        "timestamp", "headline", "sentiment", "sentiment_score", "event", "event_impact", "link"
    ])


# ---------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------
def classify_event(title: str):
    t = (title or "").lower()
    rules = [
        ("Financial Distress", -15, [r"\bbankrupt", r"\binsolv", r"\bdefault\b",
                                     r"debt restructur", r"liquidat", r"downgraded? to junk"]),
        ("Earnings/Guidance", -10, [r"profit warning", r"guidance (cut|lower)", r"weak demand",
                                    r"miss(es|ed) estimates", r"slower growth"]),
        ("Regulatory/Legal", -8,  [r"lawsuit", r"\bfine\b", r"\bban\b", r"probe",
                                   r"regulator", r"antitrust", r"\bsebi\b"]),
        ("Management/Leadership", -8, [r"ceo (steps down|resign|quits)", r"cfo (steps down|resign|quits)"]),
        ("Credit/Leverage", -12, [r"raises? debt", r"bond issue", r"refinanc", r"rating (downgrade|cut)"]),
        ("Growth/Positive", +5,   [r"\bacquisit(ion|ions)\b", r"\bmerger\b", r"partnership",
                                   r"record (revenue|profit)", r"beats? estimates", r"\bexpansion\b"]),
    ]
    for cat, impact, patterns in rules:
        for pat in patterns:
            if re.search(pat, t):
                return cat, impact
    return "Other/Neutral", 0


def compute_rule_score(view_df: pd.DataFrame, w_stock: int, w_yield: int, w_sent: int, w_event_cap: int,
                       recent_news: pd.DataFrame | None = None):
    """
    Compute rule-based score for the *latest* row using the same rules as the dashboard summary.
    `recent_news` is a news dataframe (already filtered) used to compute sentiment & event impacts.
    """
    if view_df.empty:
        return 100, 0, 0, 0, 0

    base = 100
    latest = view_df.iloc[-1]
    stock_mean = view_df["Stock_Close"].mean() if "Stock_Close" in view_df else np.nan
    yield_mean = view_df["10Y_Yield"].mean() if "10Y_Yield" in view_df else np.nan

    # Stock
    stock_impact = w_stock if (pd.notna(latest.get("Stock_Close")) and pd.notna(stock_mean) and
                               latest["Stock_Close"] > 1.05 * stock_mean) else 0
    # Yield
    yield_impact = w_yield if (pd.notna(latest.get("10Y_Yield")) and pd.notna(yield_mean) and
                               latest["10Y_Yield"] > yield_mean) else 0
    # News sentiment (use provided news DF if available)
    if recent_news is not None and not recent_news.empty:
        avg_news_sent = recent_news["sentiment_score"].head(10).mean()
    else:
        avg_news_sent = view_df.get("sentiment_score", pd.Series([0])).tail(10).mean()
    sent_impact = w_sent if (pd.notna(avg_news_sent) and avg_news_sent < 0) else 0

    # Event impact (cap)
    if recent_news is not None and not recent_news.empty:
        ev_imp = recent_news["event_impact"].head(5).sum()
    else:
        ev_imp = view_df.get("event_impact", pd.Series([0])).tail(5).sum()
    event_impact = int(np.clip(ev_imp, w_event_cap, 10))

    score = base + stock_impact + yield_impact + sent_impact + event_impact
    return int(score), int(stock_impact), int(yield_impact), int(sent_impact), int(event_impact)


def build_model_features_from_view(view_row: pd.Series) -> pd.DataFrame:
    """
    The trained model expects lowercase feature names:
    ['stock_close','10y_yield','cpi','sentiment_score','event_impact'].
    Convert the latest view row into that schema.
    """
    mapping = {
        "Stock_Close": "stock_close",
        "10Y_Yield": "10y_yield",
        "CPI": "cpi",
        "sentiment_score": "sentiment_score",
        "event_impact": "event_impact",
    }
    out = {}
    for k_display, k_model in mapping.items():
        val = view_row.get(k_display if k_display in view_row.index else k_model, np.nan)
        out[k_model] = pd.to_numeric(val, errors="coerce")
    return pd.DataFrame([out])


# ---------------------------------------------------------------------
# Company Selector + Data Load
# ---------------------------------------------------------------------
company_map = {
    "Reliance": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
}
st.sidebar.header("Filters")
company_choice = st.sidebar.selectbox("Choose Company", list(company_map.keys()))
ticker = company_map[company_choice]

df = load_features()
if df.empty:
    st.stop()

# Date range (handle 1-tuple vs 2-tuple gracefully)
min_d = df["date"].min().date()
max_d = df["date"].max().date()
date_input = st.sidebar.date_input("Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d)
if isinstance(date_input, tuple) and len(date_input) == 2:
    start_d, end_d = date_input
else:
    start_d = date_input
    end_d = date_input

# Factor weights
st.sidebar.subheader("⚖️ Adjust Factor Weights")
w_stock = st.sidebar.slider("Stock Weight", -20, 0, -10)
w_yield = st.sidebar.slider("Yield Weight", -20, 0, -15)
w_sent  = st.sidebar.slider("News Sentiment Weight", -20, 0, -10)
w_event_cap = st.sidebar.slider("Event Impact Cap (min)", -30, 0, -20)

# Sudden change alert threshold
st.sidebar.subheader("🚨 Alert Settings")
alert_drop = st.sidebar.slider("Alert if score drops by ≥", 5, 30, 15)

# --- Sidebar Refresh Button ---
if st.sidebar.button("🔄 Refresh Data"):
    try:
        # Clear yfinance cache and news cache
        yf.Ticker(ticker)._history = None
        fetch_company_news.clear()  # invalidate cached news
        load_features.clear()       # invalidate cached features

        # Update live last price if possible (non-persistent, but visible in session)
        hist = yf.Ticker(ticker).history(period="1d")
        if not hist.empty and "Stock_Close" in df.columns:
            latest_price = float(hist["Close"].iloc[-1])
            df.loc[df.index[-1], "Stock_Close"] = latest_price
            st.sidebar.success(f"Stock updated: {company_choice} → {latest_price:.2f}")

        st.sidebar.success("Data refreshed ✅")
    except Exception as e:
        st.sidebar.error(f"Refresh failed: {e}")


# ---------------------------------------------------------------------
# Filtered Views (Structured) + News
# ---------------------------------------------------------------------
mask = (df["date"].dt.date >= start_d) & (df["date"].dt.date <= end_d)
view = df.loc[mask].copy()
if view.empty:
    st.warning("No data in the selected date range. Try widening the filter.")
    st.stop()

# Fetch news and apply same date filter
news_df = fetch_company_news(ticker)
if not news_df.empty:
    news_df["date"] = news_df["timestamp"].dt.date
    news_filtered = news_df[(news_df["date"] >= start_d) & (news_df["date"] <= end_d)].copy()
else:
    news_filtered = pd.DataFrame()

# Compute rule-based score & components (using filtered news)
score, stock_imp, yield_imp, sent_imp, event_imp = compute_rule_score(
    view, w_stock, w_yield, w_sent, w_event_cap, recent_news=news_filtered if not news_filtered.empty else None
)

# For metrics:
avg_news_sent = (news_filtered["sentiment_score"].head(10).mean()
                 if not news_filtered.empty else 0.0)
latest_row = view.iloc[-1]


# ---------------------------------------------------------------------
# ML Prediction (if model exists)
# ---------------------------------------------------------------------
predicted_score = None
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        latest_features = build_model_features_from_view(latest_row)
        predicted_score = float(model.predict(latest_features)[0])
    except Exception as e:
        st.warning(f"⚠️ Could not run ML prediction: {e}")


# ---------------------------------------------------------------------
# Alerts: sudden score change (rule-based), compare last two rule-scores
# ---------------------------------------------------------------------
view_for_trend = view.copy()
# Compute rule score per day for trend (using same weights but without per-day news; use sentiment/event columns if present)
def row_score(row, ref_df):
    base = 100
    s_imp = w_stock if ("Stock_Close" in row and row["Stock_Close"] > 1.05 * ref_df["Stock_Close"].mean()) else 0
    y_imp = w_yield if ("10Y_Yield" in row and row["10Y_Yield"] > ref_df["10Y_Yield"].mean()) else 0
    sn = row.get("sentiment_score", 0)
    n_imp = w_sent if (pd.notna(sn) and sn < 0) else 0
    ev = row.get("event_impact", 0)
    e_imp = int(np.clip(ev, w_event_cap, 10)) if pd.notna(ev) else 0
    return base + s_imp + y_imp + n_imp + e_imp

if not view_for_trend.empty:
    view_for_trend["rule_score"] = view_for_trend.apply(lambda r: row_score(r, view_for_trend), axis=1)
    if len(view_for_trend) >= 2:
        last, prev = view_for_trend["rule_score"].iloc[-1], view_for_trend["rule_score"].iloc[-2]
        drop = prev - last
        if drop >= alert_drop:
            st.error(f"🚨 Sudden score drop detected: −{int(drop)} (from {int(prev)} to {int(last)})")
        elif drop > 0:
            st.warning(f"⚠️ Score dipped by {int(drop)} vs previous day")
        else:
            st.success("✅ No sudden score deterioration today")


# ---------------------------------------------------------------------
# Risk Summary Badges
# ---------------------------------------------------------------------
summary = []
if stock_imp < 0: summary.append("📉 Stock overvalued vs average")
if yield_imp < 0: summary.append("💹 Yields elevated vs average")
if sent_imp  < 0: summary.append("📰 Negative news sentiment")
if event_imp < 0: summary.append("⚠️ Risk event detected")
if event_imp > 0: summary.append("📈 Positive growth event")
if not summary: summary.append("✅ All indicators stable")

st.markdown("### Risk Summary")
for s in summary:
    st.markdown(f"- {s}")


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------
st.markdown("### 📊 Credit Scores")
m1, m2 = st.columns(2)
m1.metric("Today's Credit Score (Rule-based)", f"{int(score)} / 100")
if predicted_score is not None:
    m2.metric("Predicted Tomorrow's Score (ML)", f"{int(predicted_score)} / 100")
else:
    m2.warning("⚠️ ML model not available. Run `python train_model.py` after generating features.")

st.markdown("### 📌 Market Indicators")
i1, i2, i3 = st.columns(3)
i1.metric("Latest Stock Close", f"{latest_row.get('Stock_Close', float('nan')):.2f}")
i2.metric("10Y Yield (latest)", f"{latest_row.get('10Y_Yield', float('nan')):.2f}")
i3.metric("Recent News Sentiment (avg of last 10)", f"{avg_news_sent:.2f}")


# ---------------------------------------------------------------------
# Charts: Score Trend (Rule vs ML), Stock, Yield, Sentiment
# ---------------------------------------------------------------------
st.subheader("📈 Score Trend")
trend_cols = ["date", "rule_score"]
plot_trend = view_for_trend[trend_cols].copy() if "rule_score" in view_for_trend.columns else view[["date"]].copy()
if "rule_score" not in plot_trend.columns:
    # If for some reason rule_score was not computed, add current score only for the last point
    plot_trend = view[["date"]].copy()
    plot_trend["rule_score"] = np.nan
    plot_trend.loc[plot_trend.index[-1], "rule_score"] = score

fig_score = px.line(plot_trend, x="date", y="rule_score", markers=True, labels={"rule_score": "Rule-based Score"})
fig_score.update_yaxes(range=[0, 100])
st.plotly_chart(fig_score, use_container_width=True)

c1, c2 = st.columns(2)
with c1:
    st.subheader("📉 Stock Close Over Time")
    if "Stock_Close" in view.columns:
        fig_stock = px.line(view, x="date", y="Stock_Close", markers=True)
        st.plotly_chart(fig_stock, use_container_width=True)
    else:
        st.info("Stock_Close not available in features.")

with c2:
    st.subheader("💹 10Y Yield Over Time")
    if "10Y_Yield" in view.columns:
        fig_yield = px.line(view, x="date", y="10Y_Yield", markers=True)
        st.plotly_chart(fig_yield, use_container_width=True)
    else:
        st.info("10Y_Yield not available in features.")

st.subheader("📰 News Sentiment Trend (Filtered by Date Range)")
if not news_filtered.empty:
    nf = news_filtered.sort_values("timestamp").copy()
    fig_trend = px.area(
        nf,
        x="timestamp",
        y="sentiment_score",
        color="sentiment",
        line_shape="spline",
        color_discrete_map={"positive": "green", "negative": "red", "neutral": "gray"},
    )
    fig_trend.update_traces(opacity=0.5)
    fig_trend.update_layout(yaxis_title="Sentiment Score", xaxis_title="Date", height=400, showlegend=True)
    st.plotly_chart(fig_trend, use_container_width=True)
else:
    st.info("No news in the selected date range.")


# ---------------------------------------------------------------------
# Explainability – Feature Importance (SHAP or RF importances)
# ---------------------------------------------------------------------
st.subheader("🧠 Explainability: Feature Importance")
if model is not None:
    # Prepare a small sample window for SHAP (last N rows)
    feat_names = ["stock_close", "10y_yield", "cpi", "sentiment_score", "event_impact"]
    # Build a feature matrix from the *view* using lowercase names (safe conversions)
    sample = pd.DataFrame({
        "stock_close": pd.to_numeric(view.get("Stock_Close", pd.Series(np.nan)), errors="coerce"),
        "10y_yield": pd.to_numeric(view.get("10Y_Yield", pd.Series(np.nan)), errors="coerce"),
        "cpi": pd.to_numeric(view.get("CPI", pd.Series(np.nan)), errors="coerce"),
        "sentiment_score": pd.to_numeric(view.get("sentiment_score", pd.Series(np.nan)), errors="coerce"),
        "event_impact": pd.to_numeric(view.get("event_impact", pd.Series(np.nan)), errors="coerce"),
    }).dropna().tail(200)

    if not sample.empty:
        try:
            if SHAP_AVAILABLE:
                explainer = shap.Explainer(model)
                shap_values = explainer(sample)
                # Mean absolute SHAP per feature
                mean_abs = np.abs(shap_values.values).mean(axis=0)
                fi_df = pd.DataFrame({"Feature": feat_names, "Importance": mean_abs})
                fi_df = fi_df.sort_values("Importance", ascending=False)
                fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h", title="SHAP Mean |Feature Impact|")
                st.plotly_chart(fig_fi, use_container_width=True)
            else:
                raise ImportError("SHAP not available; falling back to model.feature_importances_.")
        except Exception:
            # Fallback to native feature_importances_ if available
            if hasattr(model, "feature_importances_"):
                # Map feature_importances_ in the same order used for training
                # We assume training used the same feature order as feat_names
                importances = getattr(model, "feature_importances_", None)
                if importances is not None and len(importances) == len(feat_names):
                    fi_df = pd.DataFrame({"Feature": feat_names, "Importance": importances})
                    fi_df = fi_df.sort_values("Importance", ascending=False)
                    fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                                    title="Model Feature Importances (Fallback)")
                    st.plotly_chart(fig_fi, use_container_width=True)
                else:
                    st.info("Feature importances not available on the model.")
            else:
                st.info("SHAP not installed and model has no feature_importances_.")
    else:
        st.info("Not enough clean rows to compute feature importance on this date range.")
else:
    st.info("ML model not loaded. Train it with `python train_model.py` to see feature importance.")


# ---------------------------------------------------------------------
# Event Timeline
# ---------------------------------------------------------------------
st.subheader("🗓️ Event Timeline")
if not news_filtered.empty:
    timeline_df = news_filtered[news_filtered["event_impact"] != 0].copy()
    if not timeline_df.empty:
        timeline_df["Impact_Label"] = timeline_df["event_impact"].apply(lambda x: "Positive" if x > 0 else "Negative")
        fig_timeline = px.scatter(
            timeline_df,
            x="timestamp",
            y="event_impact",
            color="Impact_Label",
            size=timeline_df["event_impact"].abs(),
            hover_data=["headline", "event"],
            color_discrete_map={"Positive": "#2ECC71", "Negative": "#E74C3C"},
        )
        fig_timeline.update_layout(title="Key Risk/Opportunity Events", height=420)
        st.plotly_chart(fig_timeline, use_container_width=True)
    else:
        st.info("No significant events detected in selected range.")
else:
    st.warning("No news available in the selected date range.")


# ---------------------------------------------------------------------
# Data (Filtered) + Download
# ---------------------------------------------------------------------
st.subheader("📄 Data (filtered)")
st.dataframe(view.tail(20), use_container_width=True)
csv_bytes = view.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download filtered CSV", data=csv_bytes,
                   file_name="features_filtered.csv", mime="text/csv")


# ---------------------------------------------------------------------
# News (Filtered)
# ---------------------------------------------------------------------
st.subheader(f"📰 Latest {company_choice} News (Filtered)")
if not news_filtered.empty:
    show_all = st.button("📖 Show More News")
    max_rows = 5 if not show_all else len(news_filtered)
    for _, row in news_filtered.sort_values("timestamp", ascending=False).head(max_rows).iterrows():
        color = "🟢" if row["sentiment"] == "positive" else "🔴" if row["sentiment"] == "negative" else "⚪"
        head = row.get("headline", "")
        link = row.get("link", "")
        st.markdown(f"- {color} **{head}** ({row['sentiment']})  " + (f"[Read more]({link})" if link else ""))
else:
    st.warning("No news available in the selected date range.")

