# Fake News Detection

This project focuses on detecting misleading or fake news articles using Natural Language Processing (NLP) and Machine Learning. 
The model analyzes textual content and classifies it as either Real or Fake, helping to combat misinformation online.

## Features Included :
* Text preprocessing using TF-IDF Vectorization
* Machine Learning algorithms trained and compared (Logistic Regression, Naive Bayes, PassiveAggressiveClassifier)
* Balanced dataset to improve generalization
* Evaluation metrics: Accuracy, Precision, Recall, and F1 Score
* Model and vectorizer exported as .pkl files for deployment
* Interactive Streamlit web app for real-time predictions
  
## Sample Metrics :
| Model                       | Accuracy  | Precision | Recall    | F1 Score  |
| --------------------------- | --------- | --------- | --------- | --------- |
| Logistic Regression         | **96.2%** | **95.7%** | **96.4%** | **96.0%** |
| Naive Bayes                 | **94.8%** | **94.1%** | **94.6%** | **94.3%** |
| PassiveAggressiveClassifier | **95.5%** | **95.0%** | **95.3%** | **95.1%** |

## Libraries Used :
* pandas, numpy
* scikit-learn
* nltk
* re (for regex-based text cleaning)
* streamlit (for frontend deployment)

## Dataset Info :
* Source: Kaggle – https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
* Contains real and fake news articles with text and labels.
* Used for supervised binary classification (Real = 1, Fake = 0).
