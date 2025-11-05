import joblib
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Load models
loaded_svm_priority = joblib.load('models/priority_classifier.pkl')
loaded_svm_category = joblib.load('models/category_classifier.pkl')
loaded_tfidf = joblib.load('models/tfidf_vectorizer.pkl')
loaded_le_priority = joblib.load('models/priority_encoder.pkl')
loaded_le_category = joblib.load('models/category_encoder.pkl')

# Test predictions
test_texts = [
    'Complete the project proposal urgently',
    'Fix the bug in the login system',
    'Write documentation for the API',
    'Deploy the application to production',
    'Conduct data analysis on user behavior'
]

print('Testing task classification:')
for text in test_texts:
    processed = preprocess_text(text)
    vector = loaded_tfidf.transform([processed])
    priority_pred = loaded_svm_priority.predict(vector)
    category_pred = loaded_svm_category.predict(vector)
    priority_label = loaded_le_priority.inverse_transform(priority_pred)[0]
    category_label = loaded_le_category.inverse_transform(category_pred)[0]
    print(f'Task: {text}')
    print(f'Predicted Priority: {priority_label}, Category: {category_label}')
    print('---')
