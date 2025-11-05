import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

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

# Data path
data_path = os.path.join(os.path.dirname(__file__), '..', 'Data', 'tasks.csv')

# Load models
models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
try:
    nb_model = joblib.load(os.path.join(models_dir, 'naive_bayes_model.pkl'))
except Exception as e:
    st.error(f"Failed to load Naive Bayes model: {e}")
    nb_model = None
try:
    svm_model = joblib.load(os.path.join(models_dir, 'category_classifier.pkl'))
except Exception as e:
    st.error(f"Failed to load SVM model: {e}")
    svm_model = None
try:
    rf_model = joblib.load(os.path.join(models_dir, 'random_forest_model.pkl'))
except Exception as e:
    st.error(f"Failed to load Random Forest model: {e}")
    rf_model = None
try:
    xgb_model = joblib.load(os.path.join(models_dir, 'xgboost_model.pkl'))
except Exception as e:
    st.error(f"Failed to load XGBoost model: {e}")
    xgb_model = None
try:
    tfidf_vectorizer = joblib.load(os.path.join(models_dir, 'tfidf_vectorizer.pkl'))
    # Check if vectorizer is fitted, if not, fit it
    if tfidf_vectorizer is not None:
        try:
            tfidf_vectorizer.transform(["test"])
        except Exception as e:
            if "idf vector is not fitted" in str(e):
                df = pd.read_csv(data_path)
                processed_texts = [preprocess_text(desc) for desc in df['task_description']]
                tfidf_vectorizer.fit(processed_texts)
except Exception as e:
    st.error(f"Failed to load TF-IDF vectorizer: {e}")
    tfidf_vectorizer = None
try:
    label_encoder = joblib.load(os.path.join(models_dir, 'category_encoder.pkl'))
except Exception as e:
    st.error(f"Failed to load label encoder: {e}")
    label_encoder = None
try:
    priority_encoder = joblib.load(os.path.join(models_dir, 'priority_encoder.pkl'))
except Exception as e:
    st.error(f"Failed to load priority encoder: {e}")
    priority_encoder = None

st.title("AI-Powered Task Management System")

# Sidebar for navigation
menu = st.sidebar.selectbox("Menu", ["Dashboard", "Add Task", "View Tasks", "AI Suggestions"])

if menu == "Dashboard":
    st.header("Dashboard")
    # Load and display tasks
    df = pd.read_csv(data_path)
    st.dataframe(df)

    # Visualizations
    st.subheader("Task Priority Distribution")
    if 'priority' in df.columns:
        priority_counts = df['priority'].value_counts()
        # Limit to top 10 priorities to avoid memory issues
        if len(priority_counts) > 10:
            priority_counts = priority_counts.head(10)
        st.bar_chart(priority_counts)

elif menu == "Add Task":
    st.header("Add New Task")
    description = st.text_area("Task Description")
    category = st.selectbox("Category", ["Work", "Personal", "Health", "Learning", "Home", "Career"])
    due_date = st.date_input("Due Date")
    estimated_time = st.number_input("Estimated Time (hours)", min_value=0.5, step=0.5)

    if st.button("Add Task"):
        if priority_encoder is None or tfidf_vectorizer is None or priority_encoder is None:
            st.error("Models not loaded. Cannot predict priority.")
        else:
            # AI-driven priority prediction using TF-IDF features only
            processed_text = preprocess_text(description)
            tfidf_features = tfidf_vectorizer.transform([processed_text])
            # Pad TF-IDF to 172 features to match training
            X = tfidf_features.toarray()
            if X.shape[1] < 172:
                X = np.pad(X, ((0, 0), (0, 172 - X.shape[1])), mode='constant')
            else:
                X = X[:, :172]
            priority_model = joblib.load(os.path.join(models_dir, 'priority_classifier.pkl'))
            predicted_priority_num = priority_model.predict(X)[0]  # SVM predicts encoded int
            predicted_priority = priority_encoder.inverse_transform([predicted_priority_num])[0]

            # Load existing data
            df = pd.read_csv(data_path)
            new_task_id = df['task_id'].max() + 1 if not df.empty else 1
            new_row = {
                'task_id': new_task_id,
                'task_description': description,
                'priority': predicted_priority,
                'status': 'Pending',
                'category': category,
                'due_date': str(due_date),
                'assigned_to': 'Unassigned',
                'estimated_hours': estimated_time
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.to_csv(data_path, index=False)

            st.success(f"Task added! Predicted Priority: {predicted_priority}")

elif menu == "View Tasks":
    st.header("View Tasks")
    df = pd.read_csv(data_path)
    for _, row in df.iterrows():
        st.write(f"**{row['task_description']}** - Priority: {row['priority']} - Status: {row['status']}")

elif menu == "AI Suggestions":
    st.header("AI Suggestions")
    st.write("Task Classification and Priority Suggestions")

    # AI Suggestions for task classification
    st.subheader("Task Classification")
    user_input = st.text_area("Enter task description for classification:")
    if st.button("Classify Task"):
        if user_input:
            if nb_model is None or svm_model is None or tfidf_vectorizer is None or label_encoder is None:
                st.error("Models not loaded. Cannot classify task.")
            else:
                # Preprocess and classify using TF-IDF only (matching training)
                processed_text = preprocess_text(user_input)
                tfidf_features = tfidf_vectorizer.transform([processed_text])
                X = tfidf_features.toarray()
                # NB expects 170 features, SVM expects 172 features
                X_nb = X[:, :170] if X.shape[1] > 170 else np.pad(X, ((0, 0), (0, 170 - X.shape[1])), mode='constant')
                X_svm = X[:, :172] if X.shape[1] > 172 else np.pad(X, ((0, 0), (0, 172 - X.shape[1])), mode='constant')

                nb_pred = nb_model.predict(X_nb)[0]
                svm_pred = svm_model.predict(X_svm)[0]

                try:
                    nb_category = label_encoder.inverse_transform([nb_pred])[0]
                except ValueError as e:
                    nb_category = f"Unknown ({nb_pred})"
                    st.warning(f"Naive Bayes predicted unknown category: {nb_pred}")

                try:
                    svm_category = label_encoder.inverse_transform([svm_pred])[0]
                except ValueError as e:
                    svm_category = f"Unknown ({svm_pred})"
                    st.warning(f"SVM predicted unknown category: {svm_pred}")

                st.write(f"Naive Bayes Classification: {nb_category}")
                st.write(f"SVM Classification: {svm_category}")

    # Workload balancing suggestion
    st.subheader("Workload Balancing")
    df = pd.read_csv(data_path)
    pending_tasks = df[df['status'] == 'Pending']
    total_hours = pending_tasks['estimated_hours'].sum()
    st.write(f"Total pending hours: {total_hours}")
    if total_hours > 8:
        st.warning("High workload! Consider postponing low-priority tasks.")
    else:
        st.success("Workload manageable.")
