from flask import Flask, render_template, request, jsonify, url_for
import os
import keras
import tensorflow as tf
from joblib import load
# from keras.models import load_model
from tensorflow.keras.models import load_model
from gensim.models import Word2Vec
import numpy as np
import string
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
nltk.download('punkt')
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, TFBertModel
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words('english'))
from transformers import TFBertModel, PretrainedConfig
import random

from transformers import TFBertModel
from langdetect import detect
from googletrans import Translator

# Đường dẫn đến thư mục lưu cache
cache_dir = './my_model_cache/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594'
# Tải lại mô hình từ thư mục cache đã lưu
bert_model = TFBertModel.from_pretrained(cache_dir)

app = Flask(__name__)

models_folder = os.path.dirname(os.path.abspath(__file__))

vite_w2v_model_path = os.path.join(models_folder, 'model','vite_w2v.model')
vite_w2v_model = Word2Vec.load(vite_w2v_model_path)

villanos_w2v_model_path = os.path.join(models_folder, 'model','villanos_w2v.model')
villanos_w2v_model = Word2Vec.load(villanos_w2v_model_path)

combine_w2v_model_path = os.path.join(models_folder, 'model','combine_w2v.model')
combine_w2v_model = Word2Vec.load(combine_w2v_model_path)

hibert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

vite_vectorizer_path = os.path.join(models_folder, 'model','vite_vectorizer.pkl')
vite_vectorizer = joblib.load(vite_vectorizer_path)

villanos_vectorizer_path = os.path.join(models_folder, 'model','villanos_vectorizer.pkl')
villanos_vectorizer = joblib.load(villanos_vectorizer_path)

combine_vectorizer_path = os.path.join(models_folder, 'model','combine_vectorizer.pkl')
combine_vectorizer = joblib.load(combine_vectorizer_path)

# Load models of VITE dataset
vite_nb_path = os.path.join(models_folder, 'model','vite_nb.pkl')
vite_nb_model = joblib.load(vite_nb_path)

vite_bilstm_path = os.path.join(models_folder, 'model','vite_bilstm.h5')
vite_bilstm_model = load_model(vite_bilstm_path, compile=False)

vite_hibert_path = os.path.join(models_folder, 'model','vite_hibert.h5')
vite_hibert_model = tf.keras.models.load_model(vite_hibert_path, custom_objects={'TFBertModel': TFBertModel, 'KerasLayer': tf.keras.layers.Layer})

# Load models of VILLANOS dataset
villanos_nb_path = os.path.join(models_folder, 'model','villanos_nb.pkl')
villanos_nb_model = joblib.load(villanos_nb_path)

villanos_bilstm_path = os.path.join(models_folder, 'model','villanos_biLSTM')
villanos_bilstm_model = load_model(villanos_bilstm_path)

villanos_hibert_path = os.path.join(models_folder, 'model','villanos_hibert.h5')
villanos_hibert_model = load_model(villanos_hibert_path)


# Load models of Combine dataset
combine_nb_path = os.path.join(models_folder, 'model','combine_nb.pkl')
combine_nb_model = joblib.load(combine_nb_path)

combine_bilstm_path = os.path.join(models_folder, 'model','combine_bilstm.h5')
combine_bilstm_model = load_model(combine_bilstm_path)

combine_hibert_path = os.path.join(models_folder, 'model','combine_hibert.h5')
combine_hibert_model = load_model(combine_hibert_path)

punctuations=list(string.punctuation)

def preprocess_text(text):
    result = text.lower()
    result = re.sub(r'\d+', '', result)
    result = re.sub(r'[^\w\s_]', '', result)
    result = re.sub(r'<[^>]+>', '', result)
    tokens = word_tokenize(result)
    
    filtered_tokens = [w for w in tokens if not w in stop_words]

    lemmatizer = WordNetLemmatizer()
    lemmed_tokens = [lemmatizer.lemmatize(token, pos='v') for token in filtered_tokens]
    return ' '.join(lemmed_tokens)

def text_to_sequence(tokens, model):
    return [model.wv.key_to_index[word] for word in tokens if word in model.wv]

def encode_text(text, tokenizer, max_len=128):
    encoded = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_attention_mask=True
    )
    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask']
    return np.array(input_ids), np.array(attention_mask)

def predict_nb(text, vectorizer, cls_model):
    clean_text = preprocess_text(text)
    clean_text = [clean_text]
    input_text = vectorizer.transform(clean_text)
    prediction = cls_model.predict(input_text)
    result = prediction[0].lower().strip()
    print(result)
    if result == 'violent':
        result = 'violence'
    elif result == 'nonviolent':
        result = 'non-violence'
    return result

def predict_bilstm(text, w2v, cls_model):
    clean_text = preprocess_text(text)
    tokens = clean_text.split()
    seq = text_to_sequence(tokens, w2v)
    # Ensure the sequence is a 2D array
    seq = [seq]
    input_text = pad_sequences(seq, maxlen=100)
    prediction = cls_model.predict(input_text)
    labels = ['non-violence', 'violence']
    predicted = labels[np.argmax(prediction[0])]
    return predicted

def predict_hibert(text, tokenizer, cls_model):
    clean_text = preprocess_text(text)
    ids, masks = encode_text(clean_text, tokenizer)
    
    ids = np.expand_dims(ids, axis=0)
    masks = np.expand_dims(masks, axis=0)
    prediction = cls_model.predict({'input_ids': ids, 'attention_mask': masks})
    labels = ['non-violence', 'violence']
    predicted = labels[np.argmax(prediction[0])]
    return predicted

def translate_to_english(text):
    detected_language = detect(text)
    if detected_language == 'vi':
        translator = Translator()
        translated_text = translator.translate(text, dest='en')
        return translated_text.text
    else:
        return text
    
def get_random_image_path(directory):
    try:
        models_folder = os.path.dirname(os.path.abspath(__file__))
        # Lấy danh sách các tệp tin trong thư mục
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(models_folder, directory, f))]
        if not files:
            return None
        # Chọn ngẫu nhiên một tệp tin
        random_file = random.choice(files)
        return os.path.join(directory, random_file)

    except Exception as e:
        return str(e)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text_to_predict = request.form['text_to_predict']

        text_to_predict = translate_to_english(text_to_predict)

        vite_nb = 'non-violence'
        vite_bilstm = 'non-violence'
        vite_hibert = 'non-violence'

        villanos_nb = 'non-violence'
        villanos_bilstm = 'non-violence'
        villanos_hibert = 'non-violence'

        combine_nb = 'non-violence'
        combine_bilstm = 'non-violence'
        combine_hibert = 'non-violence'
        
        vite_nb = predict_nb(text_to_predict, vite_vectorizer, vite_nb_model)
        vite_bilstm = predict_bilstm(text_to_predict, vite_w2v_model, vite_bilstm_model)
        vite_hibert = predict_hibert(text_to_predict, hibert_tokenizer,vite_hibert_model)

        villanos_nb = predict_nb(text_to_predict, villanos_vectorizer, villanos_nb_model)
        villanos_bilstm = predict_bilstm(text_to_predict, villanos_w2v_model, villanos_bilstm_model)
        villanos_hibert = predict_hibert(text_to_predict, hibert_tokenizer, villanos_hibert_model)

        combine_nb = predict_nb(text_to_predict, combine_vectorizer, combine_nb_model)
        combine_bilstm = predict_bilstm(text_to_predict, combine_w2v_model, combine_bilstm_model)
        combine_hibert = predict_hibert(text_to_predict, hibert_tokenizer, combine_hibert_model)
        
        values = [
            vite_nb, vite_bilstm, vite_hibert,
            villanos_nb, villanos_bilstm, villanos_hibert,
            combine_nb, combine_bilstm, combine_hibert
        ]

        # Đếm số lượng 'violence' và 'non-violence'
        violence_count = values.count('violence')
        non_violence_count = values.count('non-violence')

        final_result = ''
        path = ''
        # Xác định giá trị của biến kq
        if violence_count > non_violence_count:
            final_result = 'violence'
            path = get_random_image_path(os.path.join('static','violence'))
        else:
            final_result = 'non-violence'
            path = get_random_image_path(os.path.join('static','non-violence'))
        print(path)

        return render_template('result.html',
                               text_to_predict = text_to_predict,
                               vite_nb = vite_nb,
                               vite_bilstm = vite_bilstm,
                               vite_hibert = vite_hibert,

                               villanos_nb = villanos_nb,
                               villanos_bilstm = villanos_bilstm,
                               villanos_hibert = villanos_hibert,

                               combine_nb = combine_nb,
                               combine_bilstm = combine_bilstm,
                               combine_hibert = combine_hibert,
                               final_result = final_result,
                               path = path
                               )

@app.errorhandler(KeyError)
def handle_key_error(e):
    error_message = f'KeyError: {str(e)}'
    return render_template('error.html', error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True)
