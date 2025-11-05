import joblib
import os

print('Checking for Word2Vec and BERT models...')
try:
    w2v = joblib.load(os.path.join('models', 'word2vec_model.pkl'))
    print('Word2Vec loaded')
except Exception as e:
    print(f'Word2Vec error: {e}')
try:
    bert_tok = joblib.load(os.path.join('models', 'bert_tokenizer.pkl'))
    print('BERT Tokenizer loaded')
except Exception as e:
    print(f'BERT Tokenizer error: {e}')
try:
    bert_mod = joblib.load(os.path.join('models', 'bert_model.pkl'))
    print('BERT Model loaded')
except Exception as e:
    print(f'BERT Model error: {e}')
