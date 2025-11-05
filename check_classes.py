import joblib
nb_model = joblib.load('models/naive_bayes_model.pkl')
print('NB model classes_:', nb_model.classes_)
svm_model = joblib.load('models/category_classifier.pkl')
print('SVM model classes_:', svm_model.classes_)
