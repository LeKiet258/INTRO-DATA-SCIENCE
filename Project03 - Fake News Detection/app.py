import pandas as pd
import numpy as np
import re
import joblib
import requests

from underthesea import word_tokenize

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import streamlit as st
import base64

class TextReducer(BaseEstimator, TransformerMixin):
    stopwords_raw_url = "https://raw.githubusercontent.com/stopwords/vietnamese-stopwords/master/vietnamese-stopwords.txt"

    def __init__(self):
        self.stopwords = requests.get(self.stopwords_raw_url).text.split('\n')

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        _X = pd.Series(X)

        _X = _X.apply(lambda text: re.sub(r'http(s?)\S+.', '', text))
        _X = _X.apply(lambda text: re.sub(r'[@#/!.\'‘’\"“”–+-=()%]', '', text))
        _X = _X.apply(lambda text: re.sub(r'\r\n', ' ', text))
        _X = _X.apply(lambda text: text.lower())
        _X = _X.apply(word_tokenize)
        _X = _X.apply(lambda words: ' '.join([word for word in words if word not in self.stopwords]))

        return _X

import base64

@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file) 
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    background-repeat: no-repeat;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

@st.cache(allow_output_mutation=True)
def load_models():
    vectorizers = {'cv':'Count Vectorize', 
                   'tv':'TF-IDF Vectorize'}

    classifiers = {'li':'Linear Regression', 
                   'lo':'Logistic Regression', 
                   'dt':'Decision Tree Classifier', 
                   'rf':'Random Forest Classifier'}

    models = dict() 
    for c in classifiers:
        for v in vectorizers:
            name = classifiers[c] + ' - ' + vectorizers[v]
            models[name] = joblib.load('models/tr_{}_{}.pkl'.format(v, c))

    return models

def process(text, model):
    if model in models:
        return models[model].predict([text])[0] >= 0.5

# MAIN FLOW

# Load
with st.spinner('Loading models... This might take a while.'):
    set_png_as_page_bg('background.png')
    models = load_models()

# Title
st.title('FAKE NEWS DETECTION')

# Input text
input_text = st.text_area('Enter your text', height = 300)

# Model selection
selected_model = st.selectbox('Select a model', [key for key in models])

# Process
if st.button('Submit'):
    with st.spinner('Processing...'):
        result = process(input_text.title(), selected_model)
    if result is None:
        st.warning('Invalid model')
    elif result:
        st.error('Fake news detected!')
    else:
        st.success('This news is real.')
