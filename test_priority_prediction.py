import joblib
import numpy as np
import os

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

# Load models
models_dir = 'models'
tfidf_vectorizer = joblib.load(os.path.join(models_dir, 'tfidf_vectorizer.pkl'))
priority_model = joblib.load(os.path.join(models_dir, 'priority_classifier.pkl'))
priority_encoder = joblib.load(os.path.join(models_dir, 'priority_encoder.pkl'))

# Test prediction
description = "Complete the project report"
processed_text = preprocess_text(description)
tfidf_features = tfidf_vectorizer.transform([processed_text])
# Pad TF-IDF to 172 features to match training
X = tfidf_features.toarray()
if X.shape[1] < 172:
    X = np.pad(X, ((0, 0), (0, 172 - X.shape[1])), mode='constant')
else:
    X = X[:, :172]

print(f"X shape: {X.shape}")
predicted_priority_num = priority_model.predict(X)[0]  # SVM predicts encoded int
predicted_priority = priority_encoder.inverse_transform([predicted_priority_num])[0]

print(f"Predicted priority: {predicted_priority}")
