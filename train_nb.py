import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
import joblib

# Load data
df = pd.read_csv('Data/tasks.csv')

# Preprocess
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

df['processed_description'] = df['task_description'].apply(preprocess_text)

# TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
X_tfidf = tfidf_vectorizer.fit_transform(df['processed_description'])

# Encode category
le_category = LabelEncoder()
y_category = le_category.fit_transform(df['category'])

# Train NB
nb_model = MultinomialNB()
nb_model.fit(X_tfidf, y_category)

# Save
joblib.dump(nb_model, 'models/naive_bayes_model.pkl')
joblib.dump(tfidf_vectorizer, 'models/tfidf_vectorizer.pkl')
joblib.dump(le_category, 'models/category_encoder.pkl')

print("NB model trained and saved.")
