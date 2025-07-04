import joblib
import re
import requests
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
import matplotlib.pyplot as plt

nltk.download('stopwords')
ps = PorterStemmer()

# Load model and vectorizer
model = joblib.load("model/fake_news_model(2).pkl")
vectorizer = joblib.load("model/tfidf_vectorizer(2).pkl")

def stemming(content):
    content = re.sub('[^a-zA-Z]', ' ', content)
    content = content.lower().split()
    content = [ps.stem(word) for word in content if word not in stopwords.words('english')]
    return ' '.join(content)

def preprocess_url(url):
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url
    return url

def fetch_news_content(url):
    try:
        url = preprocess_url(url)
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        news_content = ' '.join([para.get_text() for para in paragraphs])
        return news_content.strip()
    except Exception:
        return ""

def classify_news(news_text):
    if not news_text.strip():
        return "❌ No content provided.", None

    processed_text = stemming(news_text)
    input_data = vectorizer.transform([processed_text])
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    result = "📰 Real News" if prediction == 0 else "🚨 Fake News"

    labels = ['Real News', 'Fake News']
    colors = ['#28a745', '#dc3545']
    bar_colors = [colors[i] for i in range(len(probabilities))]

    fig, ax = plt.subplots(figsize=(6, 2))
    bars = ax.barh(labels, probabilities * 100, color=bar_colors)
    ax.set_xlim(0, 100)
    ax.set_xlabel('Confidence (%)')
    ax.set_title('Prediction Confidence')
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', va='center', color='white', fontweight='bold')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.xaxis.label.set_color('white')
    ax.title.set_color('white')
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')

    return result, fig