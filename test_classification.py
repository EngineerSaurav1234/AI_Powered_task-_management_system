import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Test loading models
models_dir = 'models'
tfidf_vectorizer = joblib.load(models_dir + '/tfidf_vectorizer.pkl')
nb_model = joblib.load(models_dir + '/naive_bayes_model.pkl')
svm_model = joblib.load(models_dir + '/category_classifier.pkl')
label_encoder = joblib.load(models_dir + '/category_encoder.pkl')

# Test preprocessing
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

# Test classification
test_text = 'Complete the project report'
processed = preprocess_text(test_text)
tfidf_features = tfidf_vectorizer.transform([processed])
X = tfidf_features.toarray()
if X.shape[1] < 1000:
    X = np.pad(X, ((0, 0), (0, 1000 - X.shape[1])), mode='constant')
else:
    X = X[:, :1000]

# NB expects 170 features, SVM expects 172
X_nb = X[:, :170] if X.shape[1] > 170 else np.pad(X, ((0, 0), (0, 170 - X.shape[1])), mode='constant')
X_svm = X[:, :172] if X.shape[1] > 172 else np.pad(X, ((0, 0), (0, 172 - X.shape[1])), mode='constant')

nb_pred = nb_model.predict(X_nb)[0]
svm_pred = svm_model.predict(X_svm)[0]

print('NB prediction:', nb_pred)
print('SVM prediction:', svm_pred)

try:
    nb_category = label_encoder.inverse_transform([nb_pred])[0]
    print('NB category:', nb_category)
except ValueError as e:
    print('NB error:', e)

try:
    svm_category = label_encoder.inverse_transform([svm_pred])[0]
    print('SVM category:', svm_category)
except ValueError as e:
    print('SVM error:', e)
