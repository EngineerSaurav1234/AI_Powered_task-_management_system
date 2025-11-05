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

# Load cleaned data
df = pd.read_csv('Data/cleaned_tasks.csv')

# Check if 'processed_description' column exists, if not, create it
if 'processed_description' not in df.columns:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    import re
    nltk.download('stopwords', quiet=True)
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))

    def preprocess_text(text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = nltk.word_tokenize(text)
        filtered_tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
        return ' '.join(filtered_tokens)

    df['processed_description'] = df['task_description'].apply(preprocess_text)

# Prepare features
df['desc_length'] = df['processed_description'].apply(len)
df['word_count'] = df['processed_description'].apply(lambda x: len(x.split()))
tfidf = TfidfVectorizer(max_features=1000)
X_text = tfidf.fit_transform(df['processed_description'])
X_numeric = df[['desc_length', 'word_count']].values
X = np.hstack([X_text.toarray(), X_numeric])

# For classification (assuming category as target)
y_class = df['category']

# For priority prediction
y_priority = df['priority']

# Split data
X_train, X_test, y_class_train, y_class_test, y_priority_train, y_priority_test = train_test_split(
    X, y_class, y_priority, test_size=0.2, random_state=42
)

# Train Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train, y_class_train)
nb_pred = nb_model.predict(X_test)
print("Naive Bayes Classification Report:")
print(classification_report(y_class_test, nb_pred))

# Train SVM
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_class_train)
svm_pred = svm_model.predict(X_test)
print("SVM Classification Report:")
print(classification_report(y_class_test, svm_pred))

# Train Random Forest for priority
rf_model = RandomForestClassifier(random_state=42)
param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20]}
grid_rf = GridSearchCV(rf_model, param_grid, cv=3)
grid_rf.fit(X_train, y_priority_train)
rf_pred = grid_rf.predict(X_test)
print("Random Forest Priority Prediction Report:")
print(classification_report(y_priority_test, rf_pred))

# Save models
os.makedirs('models', exist_ok=True)
joblib.dump(nb_model, 'models/naive_bayes_model.pkl')
joblib.dump(svm_model, 'models/svm_model.pkl')
joblib.dump(grid_rf.best_estimator_, 'models/random_forest_model.pkl')
joblib.dump(tfidf, 'models/tfidf_vectorizer.pkl')

print("Models saved successfully.")
