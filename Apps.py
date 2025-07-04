import streamlit as st
import matplotlib.pyplot as plt
from Utils import fetch_news_content, classify_news

# -------------------- Streamlit UI --------------------
st.set_page_config(layout="centered", page_title="🕵️‍♂️ Fake News Detector")

# Theme Toggle
dark_mode = st.toggle("🌙 Enable Dark Mode")

# Dark/Light Mode Styles
if dark_mode:
    st.markdown("""
        <style>
        body { background-color: #121212; color: #ffffff; }
        .stApp { background-color: #121212; color: #ffffff; }
        h1, h2, h3, h4, h5, h6, p, small, span, label, div {
            color: #ffffff !important;
        }
        .stTextInput>div>div>input,
        .stTextArea>div>textarea {
            background-color: #2b2b2b;
            color: #ffffff;
        }
        .stButton>button {
            background-color: #28a745;
            color: #ffffff;
            border: none;
        }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        .stButton>button {
            background-color: #2c3e50;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)

# Header
st.markdown("""
    <h1 style='text-align: center;'>🕵️‍♂️ Fake News Detector</h1>
    <p style='text-align: center; font-size: 18px;'>Check if a news article is real or fake using machine learning</p>
    <hr style='border: 1px solid #ccc;'/>
""", unsafe_allow_html=True)

# Choose Input Method
st.markdown("### 🧾 Choose Input Method")
input_method = st.radio("How would you like to provide the news?", ("Paste URL", "Paste Raw Text"))

news_text = ""

if input_method == "Paste URL":
    url = st.text_input("🔗 Enter News URL", placeholder="Paste the news article URL here...")
    if url:
        news_text = fetch_news_content(url)
elif input_method == "Paste Raw Text":
    news_text = st.text_area("📝 Paste News Text", placeholder="Paste the news content here...", height=250)

# Analyze Button
if st.button("🚀 Analyze"):
    with st.spinner("Analyzing... Please wait ⏳"):
        result, fig = classify_news(news_text)
        st.success("Analysis Complete ✅")

        st.markdown(f"### 🧠 Prediction Result: {result}")
        if fig:
            st.markdown("### 📊 Confidence Bar Chart")
            st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown(
    "<small style='color:gray;'>Developed as a Corporate-Grade Project. Includes machine learning, NLP, web scraping, and a modern interactive UI.</small>",
    unsafe_allow_html=True
)
