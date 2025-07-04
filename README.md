# 🕵️‍♂️ Fake News Detector

A Streamlit-based web app that detects whether a news article is real or fake using machine learning, NLP, and web scraping.

## 🚀 Features

- Paste a **URL** or **raw news text**
- Get instant prediction (Real or Fake)
- Shows **confidence bar chart**
- Beautiful **dark mode**
- Powered by `joblib`, `nltk`, `sklearn`, `BeautifulSoup`, and `Streamlit`

## 🧠 Model

- Algorithm: Logistic Regression
- Trained on: TF-IDF features
- Serialized using `joblib`

## 📦 Installation

```bash
pip install -r requirements.txt
streamlit run Apps.py