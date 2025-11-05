import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import joblib
import os
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from gensim.models import Word2Vec
from transformers import BertTokenizer, BertModel
import torch

# Load large dataset
df = pd.read_csv('Data/large_tasks.csv')

# Preprocess text (simple version)
def preprocess_text(text):
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    import re

    nltk.download('stopwords', quiet=True)
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))

    text = re.sub(r'\W', ' ', str(text).lower())
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Check if 'processed_description' column exists, if not, create it
if 'processed_description' not in df.columns:
    df['processed_description'] = df['task_description'].apply(preprocess_text)

df['processed_description'] = df['task_description'].apply(preprocess_text)
df['desc_length'] = df['task_description'].str.len()
df['word_count'] = df['task_description'].str.split().str.len()

# Prepare features
tfidf = TfidfVectorizer(max_features=1000)
X_text = tfidf.fit_transform(df['processed_description'])
X_numeric = df[['desc_length', 'word_count']].values

# Word2Vec embeddings
sentences = [desc.split() for desc in df['processed_description']]
word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
def get_word2vec_embedding(text):
    words = text.split()
    word_vectors = [word2vec_model.wv[word] for word in words if word in word2vec_model.wv]
    if word_vectors:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(100)
X_word2vec = np.array([get_word2vec_embedding(desc) for desc in df['processed_description']])

# BERT embeddings
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()
X_bert = np.array([get_bert_embedding(desc) for desc in df['processed_description']])

# Concatenate all features
X = np.hstack([X_text.toarray(), X_word2vec, X_bert, X_numeric])

# Encode targets
le_category = LabelEncoder()
le_priority = LabelEncoder()
y_category = le_category.fit_transform(df['category'])
y_priority = le_priority.fit_transform(df['priority'])

# Split data
X_train, X_test, y_cat_train, y_cat_test, y_pri_train, y_pri_test = train_test_split(
    X, y_category, y_priority, test_size=0.2, random_state=42
)

# Train final models with hyperparameter tuning
nb_model = MultinomialNB()
nb_model.fit(X_train, y_cat_train)

svm_param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
svm_grid = GridSearchCV(SVC(probability=True), svm_param_grid, cv=3, scoring='accuracy')
svm_grid.fit(X_train, y_cat_train)
svm_model = svm_grid.best_estimator_

rf_param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [10, 20, None]}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, cv=3, scoring='accuracy')
rf_grid.fit(X_train, y_pri_train)
rf_model = rf_grid.best_estimator_

xgb_param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [3, 6, 9]}
xgb_grid = GridSearchCV(xgb.XGBClassifier(random_state=42), xgb_param_grid, cv=3, scoring='accuracy')
xgb_grid.fit(X_train, y_pri_train)
xgb_model = xgb_grid.best_estimator_

# Evaluate
nb_pred = nb_model.predict(X_test)
svm_pred = svm_model.predict(X_test)
rf_pred = rf_model.predict(X_test)
xgb_pred = xgb_model.predict(X_test)

print("Final Model Performance:")
print(f"Naive Bayes Accuracy: {accuracy_score(y_cat_test, nb_pred):.2f}")
print(f"SVM Accuracy: {accuracy_score(y_cat_test, svm_pred):.2f}")
print(f"Random Forest Priority Accuracy: {accuracy_score(y_pri_test, rf_pred):.2f}")
print(f"XGBoost Priority Accuracy: {accuracy_score(y_pri_test, xgb_pred):.2f}")

# Save models
os.makedirs('models', exist_ok=True)
joblib.dump(nb_model, 'models/naive_bayes_model.pkl')
joblib.dump(svm_model, 'models/svm_model.pkl')
joblib.dump(rf_model, 'models/random_forest_model.pkl')
joblib.dump(xgb_model, 'models/xgboost_model.pkl')
joblib.dump(tfidf, 'models/tfidf_vectorizer.pkl')
# Note: Word2Vec and BERT models cannot be saved with joblib due to serialization issues
# They will be retrained during inference if needed
# joblib.dump(word2vec_model, 'models/word2vec_model.pkl')
# joblib.dump(tokenizer, 'models/bert_tokenizer.pkl')
# joblib.dump(bert_model, 'models/bert_model.pkl')
joblib.dump(le_category, 'models/label_encoder.pkl')
joblib.dump(le_priority, 'models/priority_encoder.pkl')

print("Final models saved (Word2Vec and BERT models not saved due to serialization limitations).")
