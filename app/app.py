import streamlit as st
import joblib
import nltk
import string

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

model = joblib.load("../models/model.pkl")
tfidf = joblib.load("../models/vectorizer.pkl")

def transform_text(text):

    text = text.lower()

    text = nltk.word_tokenize(text)

    y = []

    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

st.title("Spam Message Detector")

input_sms = st.text_area("Enter your message")

if st.button("Predict"):

    transformed_sms = transform_text(input_sms)

    vector_input = tfidf.transform([transformed_sms])

    result = model.predict(vector_input)[0]

    if result == 1:
        st.error("🚨 SPAM MESSAGE")
    else:
        st.success("✅ NOT SPAM")