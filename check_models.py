import joblib
nb_model = joblib.load('models/naive_bayes_model.pkl')
print('NB model type:', type(nb_model))
print('NB model n_features_in_:', nb_model.n_features_in_)

svm_model = joblib.load('models/category_classifier.pkl')
print('SVM model type:', type(svm_model))
print('SVM model n_features_in_:', svm_model.n_features_in_)

tfidf = joblib.load('models/tfidf_vectorizer.pkl')
print('TF-IDF max_features:', tfidf.max_features)
