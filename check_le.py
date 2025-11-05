import joblib
le = joblib.load('models/label_encoder.pkl')
print('Label encoder classes:', le.classes_)
print('Length:', len(le.classes_))
