# Fake News Detection System
A machine learning–based web application that detects whether a news article is Real or Fake using natural language processing techniques.
Built with Scikit-Learn and Streamlit, the app analyzes raw news text and provides instant classification results.

## 🚀Features
* Real-time Fake vs Real news classification
* Trained ML model loaded using joblib
* Text vectorization using saved NLP vectorizer
* Clean, professional UI for article analysis
* Instant prediction with clear visual feedback

## How It Works
### 1. Dataset
#### The model is trained on a labeled fake news dataset containing:
* News article text
* Target label (Real / Fake)

### 2. Text Processing 
* Text cleaning and preprocessing
* Feature extraction using TF-IDF / CountVectorizer
* Vectorizer saved separately for inference

### 3.Machine Learning Model
* Binary classification model
* Trained in Jupyter Notebook
* Stored as fake_news_model.pkl

### 4. Prediction Pipeline
#### Prediction Pipeline
User inputs news article →
Text is vectorized →
Model predicts class →
Result displayed as:
* ✅ Real News
* 🚨 Fake News

## Tech Stack
* Python
* Streamlit
* Scikit-Learn
* Joblib
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
│── app.py                        # Streamlit application
│── Fake News Detection.ipynb     # Model training notebook
│
├── model/
│   ├── fake_news_model.pkl       # Trained ML model
│   └── vectorizer.pkl            # Text vectorizer
│
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
