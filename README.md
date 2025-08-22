# Cred-lens
An interactive Streamlit dashboard that combines stock market data, company news, sentiment analysis, and an event timeline to support quick credit risk insights and a judge-friendly demo.

## 🌟 Features

- 📈 **Stock Trends** – Visualize stock prices over any custom date range.  
- 📰 **Company News Feed** – Pulls recent Yahoo Finance news related to the company.  
- 😃 **Sentiment Analysis** – Performs sentiment scoring of news headlines (positive, neutral, negative).  
- 📌 **Event Timeline** – Shows key news events overlaid on price charts.  
- ⚡ **Fast & Interactive** – Built using **Streamlit**, with interactive charts powered by **Plotly**.  

---

## 🛠️ Tech Stack

- **Frontend & UI**: [Streamlit](https://streamlit.io)  
- **Data**: [Yahoo Finance API (`yfinance`)](https://pypi.org/project/yfinance/)  
- **Sentiment**: [TextBlob](https://textblob.readthedocs.io/en/latest/)  
- **ML Support**: scikit-learn, SHAP, joblib (for credit risk model integration)  
- **Visualization**: Plotly  

---

## 📂 Project Structure
cred-hackathon/
├── dashboard.py # Main Streamlit app
├── requirements.txt # Python dependencies
├── Dockerfile # Optional: container build (not required for Streamlit Cloud)
├── credit_model.pkl # Optional ML model (do not include large files in GitHub)
├── features_dataset.csv # Optional supporting data
├── README.md # This file


## Live Demo /Public URL
https://cred-lens.streamlit.app/