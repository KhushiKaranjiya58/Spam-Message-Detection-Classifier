import streamlit as st
import os
import joblib
import nltk
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# -----------------------------------
# Load Model & Vectorizer
# -----------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = joblib.load(
    os.path.join(BASE_DIR, "models", "model.pkl")
)

tfidf = joblib.load(
    os.path.join(BASE_DIR, "models", "vectorizer.pkl")
)

ps = PorterStemmer()

# -----------------------------------
# Text Preprocessing
# -----------------------------------

def transform_text(text):

    text = text.lower()

    text = nltk.word_tokenize(text)

    y = []

    # Keep only alphanumeric words
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    # Remove stopwords & punctuation
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    # Stemming
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# -----------------------------------
# Explainability Function
# -----------------------------------

def explain_prediction(text):

    transformed = transform_text(text)

    vector = tfidf.transform([transformed])

    feature_names = tfidf.get_feature_names_out()

    indices = np.argsort(vector.toarray()[0])[::-1]

    top_words = []

    for idx in indices[:10]:

        if vector.toarray()[0][idx] > 0:
            top_words.append(feature_names[idx])

    return top_words

# -----------------------------------
# Streamlit UI
# -----------------------------------

st.set_page_config(
    page_title="Spam Detector",
    page_icon="📩",
    layout="centered"
)

st.markdown("""
<style>

.main {
    background-color: #f8f9fa;
}

h1 {
    color: #1f77b4;
    text-align: center;
}

.stButton > button {
    width: 100%;
    border-radius: 10px;
    height: 3em;
    font-size: 18px;
    font-weight: bold;
}

.stTextArea textarea {
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

st.title("📩 Spam Message Detector")

st.caption(
    "Classify SMS messages as Spam or Ham using Machine Learning."
)

with st.sidebar:

    st.markdown("### 📊 Model Performance")

    st.markdown("""
    Accuracy: **97.29%**

    Precision: **99.15%**

    Recall: **81.37%**

    F1 Score: **89.39%**
    """)

    st.divider()

    st.write("Model: Multinomial Naive Bayes")
    st.write("Features: TF-IDF + NLP")

st.sidebar.info("""
Spam Detection Classifier

Model:
• Multinomial Naive Bayes

Features:
• Spam Detection
• Confidence Score
• Explainability Layer
• Prediction History
• Analytics Dashboard
• Download Report
""")

st.markdown("""
### 🤖 AI-Powered Spam Detection System

Detect spam messages using Machine Learning and Natural Language Processing.

""")

# -----------------------------------
# History Storage
# -----------------------------------

if "history" not in st.session_state:
    st.session_state.history = []

# -----------------------------------
# User Input
# -----------------------------------

input_sms = st.text_area(
    "Enter your message"
)

# -----------------------------------
# Prediction
# -----------------------------------

if st.button("Predict"):

    # Empty input check

    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message")
        st.stop()

    transformed_sms = transform_text(input_sms)

    vector_input = tfidf.transform([transformed_sms])

    result = model.predict(vector_input)[0]

    # Confidence Score

    probability = model.predict_proba(vector_input)

    confidence = round(
        max(probability[0]) * 100,
        2
    )

    # Prediction Result

    if result == 1:

        prediction = "SPAM"

        st.error("🚨 SPAM MESSAGE")

    else:

        prediction = "HAM"

        st.success("✅ NOT SPAM")

    # Confidence Display

    st.info(
        f"Confidence Score: {confidence}%"
    )

    st.progress(int(confidence))

    # Save History

    st.session_state.history.append(
        {
            "Message": input_sms,
            "Prediction": prediction,
            "Confidence": f"{confidence}%"
        }
    )

    # Explainability

    important_words = explain_prediction(
        input_sms
    )

    st.subheader("🔍 Words Responsible")

    if len(important_words) > 0:

        for word in important_words:
            st.write(f"• {word}")

    else:

        st.write(
            "No important keywords detected."
        )

    # Download Report

    result_text = f"""
Spam Detection Report

Message:
{input_sms}

Prediction:
{prediction}

Confidence:
{confidence}%
"""

    st.download_button(
        label="📥 Download Result",
        data=result_text,
        file_name="prediction_report.txt",
        mime="text/plain"
    )

# -----------------------------------
# Prediction History
# -----------------------------------

st.divider()

st.subheader("📜 Prediction History")

if len(st.session_state.history) > 0:

    history_df = pd.DataFrame(
        st.session_state.history
    )

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

else:

    st.write(
        "No predictions made yet."
    )

st.divider()

st.subheader("📊 Analytics Dashboard")

if len(st.session_state.history) > 0:

    history_df = pd.DataFrame(st.session_state.history)

    analytics_df = history_df[
        history_df["Prediction"].isin(["SPAM", "HAM"])
    ]

    if len(analytics_df) > 0:

        total_predictions = len(analytics_df)

        spam_count = len(
            analytics_df[
                analytics_df["Prediction"] == "SPAM"
            ]
        )

        ham_count = len(
            analytics_df[
                analytics_df["Prediction"] == "HAM"
            ]
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Predictions",
            total_predictions
        )

        col2.metric(
            "Spam Messages",
            spam_count
        )

        col3.metric(
            "Ham Messages",
            ham_count
        )

        fig, ax = plt.subplots()

        ax.pie(
            [spam_count, ham_count],
            labels=["Spam", "Ham"],
            autopct="%1.1f%%"
        )

        ax.set_title(
            "Spam vs Ham Distribution"
        )

        st.pyplot(fig)

        st.bar_chart(
            analytics_df["Prediction"].value_counts()
        )

    else:

        st.info(
            "No Spam/Ham predictions available yet."
        )

else:

    st.info(
        "No prediction history available."
    )

# -----------------------------------
# Batch Prediction
# -----------------------------------

st.divider()

st.subheader("📂 Batch Message Prediction")

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    if st.button("🔍Run Batch Prediction"):

        batch_df = pd.read_csv(uploaded_file)

        if "message" in batch_df.columns:

            predictions = []

            for msg in batch_df["message"]:

                transformed = transform_text(str(msg))

                vector = tfidf.transform([transformed])

                pred = model.predict(vector)[0]

                prediction = "SPAM" if pred == 1 else "HAM"

                predictions.append(prediction)

            batch_df["Prediction"] = predictions

            st.success("✅ Batch Prediction Complete")

            st.dataframe(
                batch_df,
                use_container_width=True
            )

            st.session_state.history.append({
                "Message": f"📂 {uploaded_file.name}",
                "Prediction": "BATCH",
                "Confidence": f"{len(batch_df)} Messages"
            })
            

            csv = batch_df.to_csv(index=False)

            st.download_button(
                "📥 Download Results",
                csv,
                "batch_predictions.csv",
                "text/csv"
            )

        else:

            st.error(
                "CSV must contain a column named 'message'"
            )