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
nb_model = joblib.load(os.path.join(models_dir, 'naive_bayes_model.pkl'))
svm_model = joblib.load(os.path.join(models_dir, 'category_classifier.pkl'))
label_encoder = joblib.load(os.path.join(models_dir, 'category_encoder.pkl'))

# Test descriptions
test_descriptions = [
    "Complete the project report",
    "Go to the gym",
    "Buy groceries",
    "Learn Python programming",
    "Attend team meeting",
    "Cook dinner",
    "Read a book",
    "Fix the bug in code",
    "Plan vacation",
    "Do laundry"
]

for desc in test_descriptions:
    processed = preprocess_text(desc)
    tfidf_features = tfidf_vectorizer.transform([processed])
    X = tfidf_features.toarray()
    # NB expects 170 features, SVM expects 172
    X_nb = X[:, :170] if X.shape[1] > 170 else np.pad(X, ((0, 0), (0, 170 - X.shape[1])), mode='constant')
    X_svm = X[:, :172] if X.shape[1] > 172 else np.pad(X, ((0, 0), (0, 172 - X.shape[1])), mode='constant')

    nb_pred = nb_model.predict(X_nb)[0]
    svm_pred = svm_model.predict(X_svm)[0]

    try:
        nb_category = label_encoder.inverse_transform([nb_pred])[0]
    except ValueError:
        nb_category = f"Unknown ({nb_pred})"

    try:
        svm_category = label_encoder.inverse_transform([svm_pred])[0]
    except ValueError:
        svm_category = f"Unknown ({svm_pred})"

    print(f"Description: '{desc}'")
    print(f"NB: {nb_category}, SVM: {svm_category}")
    print()
