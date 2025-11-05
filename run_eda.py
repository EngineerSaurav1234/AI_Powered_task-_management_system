import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import re
from wordcloud import WordCloud

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette('husl')

# Load the dataset
df = pd.read_csv('Data/tasks.csv')
print('Dataset shape:', df.shape)
print('\nFirst 5 rows:')
print(df.head())

# Basic information about the dataset
print('Dataset info:')
print(df.info())

print('\nDescriptive statistics:')
print(df.describe(include='all'))

# Check for missing values
print('Missing values:')
print(df.isnull().sum())

# Data cleaning
# Convert due_date to datetime
df['due_date'] = pd.to_datetime(df['due_date'])

# Check for duplicates
print('Duplicate rows:', df.duplicated().sum())

# Handle any missing values (if any)
# For this dataset, we'll assume no missing values, but in real scenarios:
# df['estimated_hours'].fillna(df['estimated_hours'].median(), inplace=True)

print('Data types after cleaning:')
print(df.dtypes)

# Exploratory Data Analysis
# Distribution of priorities
plt.figure(figsize=(10, 6))
priority_counts = df['priority'].value_counts()
plt.pie(priority_counts.values, labels=priority_counts.index, autopct='%1.1f%%')
plt.title('Distribution of Task Priorities')
plt.savefig('results/priority_distribution.png')
plt.close()

# Distribution of statuses
plt.figure(figsize=(12, 6))
status_counts = df['status'].value_counts()
sns.barplot(x=status_counts.index, y=status_counts.values)
plt.title('Distribution of Task Statuses')
plt.xlabel('Status')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.savefig('results/status_distribution.png')
plt.close()

# Distribution of categories
plt.figure(figsize=(12, 6))
category_counts = df['category'].value_counts()
sns.barplot(x=category_counts.index, y=category_counts.values)
plt.title('Distribution of Task Categories')
plt.xlabel('Category')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.savefig('results/category_distribution.png')
plt.close()

# Estimated hours distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['estimated_hours'], bins=10, kde=True)
plt.title('Distribution of Estimated Hours')
plt.xlabel('Estimated Hours')
plt.ylabel('Frequency')
plt.savefig('results/estimated_hours_distribution.png')
plt.close()

# Tasks by assignee
plt.figure(figsize=(12, 6))
assignee_counts = df['assigned_to'].value_counts()
sns.barplot(x=assignee_counts.index, y=assignee_counts.values)
plt.title('Tasks Assigned to Each Person')
plt.xlabel('Assignee')
plt.ylabel('Number of Tasks')
plt.savefig('results/assignee_distribution.png')
plt.close()

# Correlation heatmap for numerical columns
plt.figure(figsize=(8, 6))
numerical_cols = df.select_dtypes(include=[np.number]).columns
correlation_matrix = df[numerical_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.savefig('results/correlation_heatmap.png')
plt.close()

# NLP Preprocessing on Task Descriptions
# Initialize NLTK tools
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords and stem
    filtered_tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(filtered_tokens)

# Apply preprocessing to task descriptions
df['processed_description'] = df['task_description'].apply(preprocess_text)

print('Original vs Processed descriptions:')
for i in range(5):
    print(f'Original: {df["task_description"][i]}')
    print(f'Processed: {df["processed_description"][i]}')
    print()

# Word cloud of task descriptions
all_descriptions = ' '.join(df['processed_description'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_descriptions)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Task Descriptions')
plt.savefig('results/wordcloud.png')
plt.close()

# Save the cleaned and preprocessed dataset
df.to_csv('Data/cleaned_tasks.csv', index=False)
print('Cleaned dataset saved to Data/cleaned_tasks.csv')
