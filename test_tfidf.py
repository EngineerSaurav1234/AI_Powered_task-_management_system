import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

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

df = pd.read_csv('Data/tasks.csv')
df['processed_description'] = df['task_description'].apply(preprocess_text)

tfidf_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
X_tfidf = tfidf_vectorizer.fit_transform(df['processed_description'])

print("TF-IDF feature shape:", X_tfidf.shape)
print("Top 20 features:")
feature_names = tfidf_vectorizer.get_feature_names_out()
print(feature_names[:20])

le_priority = LabelEncoder()
le_category = LabelEncoder()

y_priority = le_priority.fit_transform(df['priority'])
y_category = le_category.fit_transform(df['category'])

print("Priority classes:", le_priority.classes_)
print("Category classes:", le_category.classes_)
print("Sample y_priority:", y_priority[:5])
print("Sample y_category:", y_category[:5])
