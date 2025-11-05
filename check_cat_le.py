import joblib
le = joblib.load('models/category_encoder.pkl')
print('Category encoder classes:', le.classes_)
print('Length:', len(le.classes_))
