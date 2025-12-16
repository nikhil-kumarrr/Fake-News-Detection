# INCOMPLETE

# Email Spam Detection System
A machine learning–based web application that classifies emails as Spam or Safe (Not Spam) using natural language processing techniques.
Built with Scikit-Learn + Streamlit, this app provides fast and accurate email classification with a clean, modern UI.

## 🚀 Features
* ✔️ Detects Spam vs Not Spam emails
* ✔️ Uses trained ML model (Email Spam model.pkl)
* ✔️ Text vectorization using saved feature extractor
* ✔️ Supports real-time text input
* ✔️ Elegant glassmorphism UI
* ✔️ Clear visual feedback for predictions

## How It Works
### 1. Dataset
#### Uses a labeled email dataset (mail_data.csv) containing:
* Email content
* Spam / Not Spam labels

### 2. Text Processing
* Text cleaning and preprocessing
* Feature extraction using TF-IDF / CountVectorizer
* Vocabulary saved as feature_extraction.pkl

### 3. Machine Learning Model
* Model type: Binary Classification
* Algorithms used: Naive Bayes / Logistic Regression / SVM
* Trained and saved using Pickle

### 4. Prediction Pipeline
#### User input → Vectorization → Model prediction →
#### Result displayed as:
* 🚨 Spam Email
* ✅ Safe Email

## Tech Stack
* Python
* Streamlit
* Scikit-Learn
* Pickle
* NumPy
* NLP (Text Vectorization)

## 📦 Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2️⃣ Create virtual environment
```bash
python -m venv venv
```

### 3️⃣ Activate environment
#### Windows:
```bash
venv\Scripts\activate
```
#### Mac/Linux:
```bash
source venv/bin/activate
```

### 4️⃣ Install required libraries
```bash
pip install -r requirements.txt
```

### 5️⃣ Run the app
```bash
streamlit run main.py
```

## 📁 Project Structure
```bash
│── main.py                         # Streamlit app
│── Email Spam Detection.ipynb      # Model training notebook
│── mail_data.csv                   # Dataset
│── Email Spam model.pkl            # Trained ML model
│── feature_extraction.pkl          # Vectorizer
│── requirements.txt
└── README.md
```

## Dataset Info :
* Source: Kaggle – https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
* Contains real and fake news articles with text and labels.
* Used for supervised binary classification (Real = 1, Fake = 0).

## 🌐 Live Demo
https://newscheckapp.streamlit.app/

## 📸 Screenshots
![img alt](https://github.com/nikhil-kumarrr/images/blob/main/Screenshot%202025-12-16%20120431.png?raw=true)
![img alt](https://github.com/nikhil-kumarrr/images/blob/main/Screenshot%202025-12-15%20133744.png?raw=true)
