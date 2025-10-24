# 🧠 Fake News Detection App - Professional Version
import os
import streamlit as st
from joblib import load

# --- Load model and vectorizer ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = load(os.path.join(BASE_DIR, "model", "fake_news_model.pkl"))
vectorizer = load(os.path.join(BASE_DIR, "model", "vectorizer.pkl"))

# --- Page Config ---
st.set_page_config(page_title="Fake News Detection", layout="wide")

# --- Custom CSS ---
st.markdown("""
<style>
/* Page setup */
.stApp {
    background-color: #fff9e6; /* light yellow */
    font-family: 'Segoe UI', sans-serif;
    color: #1a1a1a;
}

/* Title */
.title {
    text-align: center;
    font-size: 40px;
    font-weight: 700;
    color: #333333;
    margin-top: 0;
    margin-bottom: 15px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #555555;
    font-size: 16px;
    margin-bottom: 25px;
}

/* Input box */
textarea {
    border: 1.5px solid #ccc !important;
    border-radius: 8px !important;
    padding: 12px !important;
    font-size: 14px !important;
    background-color: #fff !important;
    transition: all 0.2s ease;
    min-height: 150px !important;
}

textarea:focus {
    border: 1.5px solid #ffa500 !important;
    box-shadow: 0 0 8px rgba(255,165,0,0.2);
}

/* Predict Button */
.stButton>button {
    width: 100%;
    background-color: #ffa500;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 14px 0;
    font-size: 16px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s ease;
}

.stButton>button:hover {
    transform: scale(1.03);
    box-shadow: 0px 4px 12px rgba(255,165,0,0.3);
}

/* Result box */
.result-box {
    background-color: #ffffff;
    border-left: 5px solid #ffa500;
    color: #1a1a1a;
    padding: 18px;
    border-radius: 8px;
    font-size: 18px;
    font-weight: 600;
    margin-top: 20px;
    text-align: center;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.08);
    transition: all 0.2s ease;
}

.result-box:hover {
    transform: scale(1.01);
    border-left-color: #ff7f00;
}

/* Footer */
.footer {
    text-align: center;
    color: #555555;
    font-size: 13px;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

# --- App Header ---
st.markdown('<div class="title">Fake News Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Detect misleading or false news articles using a trained ML model</div>', unsafe_allow_html=True)

# --- User Input ---
news_article = st.text_area("Enter the News Article Below:", placeholder="Paste or type the article text here...")

# --- Prediction ---
if st.button("Analyze Article"):
    if news_article.strip() == "":
        st.warning("Please enter a news article to analyze.")
    else:
        input_vector = vectorizer.transform([news_article])
        prediction = model.predict(input_vector)[0]
        result_text = "✅ Real News - This article appears trustworthy." if prediction == 1 else "🚨 Fake News - This article may contain misleading information."
        color = "#00b300" if prediction == 1 else "#ff4b4b"
        st.markdown(f'<div class="result-box" style="border-left-color: {color};">{result_text}</div>', unsafe_allow_html=True)
