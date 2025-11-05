import joblib
import os

models_dir = 'models'
try:
    xgb = joblib.load(os.path.join(models_dir, 'xgboost_model.pkl'))
    print('XGBoost loaded successfully')
except FileNotFoundError:
    print('XGBoost model not found')
