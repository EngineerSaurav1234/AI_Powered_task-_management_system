from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from transformers import BertTokenizer, BertModel
import torch

app = Flask(__name__)
CORS(app)

# Load models
models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
try:
    nb_model = joblib.load(os.path.join(models_dir, 'naive_bayes_model.pkl'))
except Exception as e:
    print(f"Failed to load Naive Bayes model: {e}")
    nb_model = None
try:
    svm_model = joblib.load(os.path.join(models_dir, 'svm_model.pkl'))
except Exception as e:
    print(f"Failed to load SVM model: {e}")
    svm_model = None
try:
    rf_model = joblib.load(os.path.join(models_dir, 'random_forest_model.pkl'))
except Exception as e:
    print(f"Failed to load Random Forest model: {e}")
    rf_model = None
try:
    xgb_model = joblib.load(os.path.join(models_dir, 'xgboost_model.pkl'))
except Exception as e:
    print(f"Failed to load XGBoost model: {e}")
    xgb_model = None
try:
    tfidf_vectorizer = joblib.load(os.path.join(models_dir, 'tfidf_vectorizer.pkl'))
except Exception as e:
    print(f"Failed to load TF-IDF vectorizer: {e}")
    tfidf_vectorizer = None
try:
    label_encoder = joblib.load(os.path.join(models_dir, 'label_encoder.pkl'))
except Exception as e:
    print(f"Failed to load label encoder: {e}")
    label_encoder = None
try:
    priority_encoder = joblib.load(os.path.join(models_dir, 'priority_encoder.pkl'))
except Exception as e:
    print(f"Failed to load priority encoder: {e}")
    priority_encoder = None

# Data path
data_path = os.path.join(os.path.dirname(__file__), '..', 'Data', 'tasks.csv')

def preprocess_text(text):
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    import re

    nltk.download('stopwords', quiet=True)
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))

    text = re.sub(r'\W', ' ', text.lower())
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

@app.route('/classify', methods=['POST'])
def classify_task():
    data = request.get_json()
    if not data or 'description' not in data:
        return jsonify({'error': 'Missing description field'}), 400

    description = data['description']
    if nb_model is None or svm_model is None or tfidf_vectorizer is None or label_encoder is None:
        return jsonify({'error': 'Models not loaded'}), 500

    # Preprocess and classify with embeddings
    processed_text = preprocess_text(description)
    tfidf_features = tfidf_vectorizer.transform([processed_text])
    desc_length = len(description)
    word_count = len(description.split())

    # Add Word2Vec embeddings (retrain models for inference)
    try:
        df_large = pd.read_csv(data_path)  # Use tasks.csv instead of large_tasks.csv to avoid memory issues
        sentences = [str(desc).split() for desc in df_large['task_description']]
    except Exception as e:
        print(f"Failed to load data for Word2Vec training: {e}. Using default sentences.")
        sentences = [["default", "sentence", "for", "training"]]
    word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
    def get_word2vec_embedding(text):
        words = text.split()
        word_vectors = [word2vec_model.wv[word] for word in words if word in word2vec_model.wv]
        if word_vectors:
            return np.mean(word_vectors, axis=0)
        else:
            return np.zeros(100)
    word2vec_emb = get_word2vec_embedding(processed_text)

    # Add BERT embeddings (retrain models for inference)
    try:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert_model = BertModel.from_pretrained('bert-base-uncased')
        def get_bert_embedding(text):
            inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = bert_model(**inputs)
            return outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        bert_emb = get_bert_embedding(processed_text)
    except Exception as e:
        print(f"Failed to load BERT: {e}. Using zeros.")
        bert_emb = np.zeros(768)

    X_numeric = np.array([[desc_length, word_count]])
    X = np.hstack([tfidf_features.toarray(), word2vec_emb.reshape(1, -1), bert_emb.reshape(1, -1), X_numeric])

    nb_pred = nb_model.predict(X)[0]
    svm_pred = svm_model.predict(X)[0]

    nb_category = label_encoder.inverse_transform([nb_pred])[0]
    svm_category = label_encoder.inverse_transform([svm_pred])[0]

    return jsonify({
        'naive_bayes': nb_category,
        'svm': svm_category
    })

@app.route('/predict_priority', methods=['POST'])
def predict_priority():
    data = request.get_json()
    if not data or 'description' not in data:
        return jsonify({'error': 'Missing description field'}), 400

    description = data['description']
    if rf_model is None or xgb_model is None or tfidf_vectorizer is None or priority_encoder is None:
        return jsonify({'error': 'Models not loaded'}), 500

    # Preprocess and predict with embeddings
    processed_text = preprocess_text(description)
    tfidf_features = tfidf_vectorizer.transform([processed_text])
    desc_length = len(description)
    word_count = len(description.split())

    # Add Word2Vec embeddings (retrain models for inference)
    try:
        df_large = pd.read_csv(data_path)  # Use tasks.csv instead of large_tasks.csv to avoid memory issues
        sentences = [str(desc).split() for desc in df_large['task_description']]
    except Exception as e:
        print(f"Failed to load data for Word2Vec training: {e}. Using default sentences.")
        sentences = [["default", "sentence", "for", "training"]]
    word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
    def get_word2vec_embedding(text):
        words = text.split()
        word_vectors = [word2vec_model.wv[word] for word in words if word in word2vec_model.wv]
        if word_vectors:
            return np.mean(word_vectors, axis=0)
        else:
            return np.zeros(100)
    word2vec_emb = get_word2vec_embedding(processed_text)

    # Add BERT embeddings (retrain models for inference)
    try:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert_model = BertModel.from_pretrained('bert-base-uncased')
        def get_bert_embedding(text):
            inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = bert_model(**inputs)
            return outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        bert_emb = get_bert_embedding(processed_text)
    except Exception as e:
        print(f"Failed to load BERT: {e}. Using zeros.")
        bert_emb = np.zeros(768)

    X_numeric = np.array([[desc_length, word_count]])
    X = np.hstack([tfidf_features.toarray(), word2vec_emb.reshape(1, -1), bert_emb.reshape(1, -1), X_numeric])

    rf_pred = rf_model.predict(X)[0]
    xgb_pred = xgb_model.predict(X)[0]
    predicted_priority_num = int((rf_pred + xgb_pred) / 2)  # Average prediction
    predicted_priority = priority_encoder.inverse_transform([predicted_priority_num])[0]

    return jsonify({
        'priority': predicted_priority
    })

if __name__ == '__main__':
    app.run(debug=True)
