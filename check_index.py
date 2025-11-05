import joblib
le = joblib.load('models/label_encoder.pkl')
print('Label encoder classes:', le.classes_)
print('Index of ML:', list(le.classes_).index('ML'))
