import re
import string
from pathlib import Path

import joblib
import streamlit as st

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "sentiment_model.pkl"
VECTORIZER_PATH = BASE_DIR / "tfidf_vectorizer.pkl"


# Same cleaning steps used in the notebook to create 'clean_text':
# lowercase -> remove punctuation -> remove numbers -> remove extra spaces
def preprocess_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Load the trained model and the fitted TF-IDF vectorizer only once.
# Nothing is trained or fitted here.
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def predict_sentiment(text, model, vectorizer):
    features = vectorizer.transform([preprocess_text(text)])
    label = model.predict(features)[0]
    probabilities = dict(zip(model.classes_, model.predict_proba(features)[0]))
    return label, probabilities


st.set_page_config(page_title="Sentiment Analysis", page_icon="💬", layout="centered")

st.title("💬 Sentiment Analysis")
st.write(
    "Enter a product review or any sentence below. The app uses a machine learning "
    "model trained on customer reviews to classify the text as **Positive**, "
    "**Neutral**, or **Negative**."
)

try:
    model, vectorizer = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model files not found. Make sure `sentiment_model.pkl` and "
        "`tfidf_vectorizer.pkl` are in the same folder as `app.py`."
    )
    st.stop()

review = st.text_area(
    "Your review",
    height=180,
    placeholder="The product quality is excellent and I really enjoyed using it.",
)

if st.button("Analyze Sentiment", type="primary"):
    if not review.strip():
        st.warning("Please enter some text before clicking **Analyze Sentiment**.")
    elif not preprocess_text(review):
        st.warning("Please enter some words. Numbers and punctuation alone can't be analyzed.")
    else:
        label, probabilities = predict_sentiment(review, model, vectorizer)
        confidence = probabilities[label]

        if label == "Positive":
            st.success(f"😊 **Positive** sentiment  (confidence: {confidence:.0%})")
        elif label == "Negative":
            st.error(f"😞 **Negative** sentiment  (confidence: {confidence:.0%})")
        else:
            st.info(f"😐 **Neutral** sentiment  (confidence: {confidence:.0%})")

        st.caption("Probability for each sentiment")
        for sentiment in ["Positive", "Neutral", "Negative"]:
            score = probabilities.get(sentiment, 0.0)
            st.progress(float(score), text=f"{sentiment}: {score:.1%}")

with st.expander("How does this work?"):
    st.markdown(
        f"""
1. **Preprocessing:** the text is lowercased, and punctuation, numbers and extra spaces are removed.
2. **TF-IDF:** the cleaned text is converted into numbers using the TF-IDF vectorizer fitted during training.
3. **Prediction:** the trained **{type(model).__name__}** model predicts the sentiment.

The model was trained on 1,440 Samsung smartphone reviews, where ratings 1–2 = Negative,
3 = Neutral and 4–5 = Positive.
"""
    )
