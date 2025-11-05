import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

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

le_priority = LabelEncoder()
le_category = LabelEncoder()

y_priority = le_priority.fit_transform(df['priority'])
y_category = le_category.fit_transform(df['category'])

# Test SVM for priority
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y_priority, test_size=0.2, random_state=42, stratify=y_priority)

svm_priority = SVC(kernel='linear', random_state=42)
svm_priority.fit(X_train, y_train)
y_pred_svm = svm_priority.predict(X_test)

print('SVM Priority Classification Results:')
print(f'Accuracy: {accuracy_score(y_test, y_pred_svm):.4f}')
print(f'Precision: {precision_score(y_test, y_pred_svm, average="weighted"):.4f}')
print(f'Recall: {recall_score(y_test, y_pred_svm, average="weighted"):.4f}')
print(f'F1-Score: {f1_score(y_test, y_pred_svm, average="weighted"):.4f}')
print('Classification Report:')
print(classification_report(y_test, y_pred_svm, target_names=le_priority.classes_))

# Test SVM for category
X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(X_tfidf, y_category, test_size=0.2, random_state=42)

svm_category = SVC(kernel='linear', random_state=42)
svm_category.fit(X_train_cat, y_train_cat)
y_pred_svm_cat = svm_category.predict(X_test_cat)

print('\nSVM Category Classification Results:')
print(f'Accuracy: {accuracy_score(y_test_cat, y_pred_svm_cat):.4f}')
print(f'Precision: {precision_score(y_test_cat, y_pred_svm_cat, average="weighted"):.4f}')
print(f'Recall: {recall_score(y_test_cat, y_pred_svm_cat, average="weighted"):.4f}')
print(f'F1-Score: {f1_score(y_test_cat, y_pred_svm_cat, average="weighted"):.4f}')
print('Classification Report:')
print(classification_report(y_test_cat, y_pred_svm_cat))
