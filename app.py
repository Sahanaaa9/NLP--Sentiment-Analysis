import re
import string
from pathlib import Path

import joblib
import streamlit as st

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "sentiment_model.pkl"
VECTORIZER_PATH = BASE_DIR / "tfidf_vectorizer.pkl"

EXAMPLES = {
    "😊 Positive example": "The product quality is excellent and I really enjoyed using it. The camera and battery life are amazing!",
    "😞 Negative example": "Worst phone ever. The battery drains fast, it heats up and the camera is terrible. Totally disappointed.",
    "😐 Mixed example": "The phone is okay. Camera is average and battery is fine but the display could be better.",
}

SENTIMENT_STYLE = {
    "Positive": {"emoji": "😊", "color": "#16a34a", "message": "The text expresses a positive opinion."},
    "Neutral": {"emoji": "😐", "color": "#2563eb", "message": "The text is mixed or neutral."},
    "Negative": {"emoji": "😞", "color": "#dc2626", "message": "The text expresses a negative opinion."},
}


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


def set_review(text):
    st.session_state.review = text


def show_result(label, probabilities):
    style = SENTIMENT_STYLE[label]
    confidence = probabilities[label]
    st.markdown(
        f"""
<div style="border-left: 6px solid {style['color']};
            background-color: {style['color']}14;
            border-radius: 10px; padding: 18px 22px; margin: 8px 0 18px 0;">
<div style="font-size: 0.9rem; opacity: 0.75;">Predicted sentiment</div>
<div style="font-size: 2rem; font-weight: 700; color: {style['color']};">{style['emoji']} {label}</div>
<div style="opacity: 0.85;">{style['message']} Confidence: <b>{confidence:.0%}</b></div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("**Confidence by sentiment**")
    for sentiment in ["Positive", "Neutral", "Negative"]:
        score = float(probabilities.get(sentiment, 0.0))
        st.progress(score, text=f"{SENTIMENT_STYLE[sentiment]['emoji']} {sentiment}: {score:.1%}")


st.set_page_config(page_title="Sentiment Analysis", page_icon="💬", layout="centered")

try:
    model, vectorizer = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model files not found. Make sure `sentiment_model.pkl` and "
        "`tfidf_vectorizer.pkl` are in the same folder as `app.py`."
    )
    st.stop()

# ---------- Sidebar ----------
with st.sidebar:
    st.header("About the model")
    st.write(
        "Trained on **1,440 Samsung smartphone reviews**. "
        "Ratings 1–2 = Negative, 3 = Neutral, 4–5 = Positive."
    )

    st.subheader("How it works")
    st.markdown(
        """
1. **Clean** the text: lowercase, remove punctuation, numbers and extra spaces
2. **TF-IDF** converts the text into numbers
3. **Logistic Regression** predicts the sentiment
"""
    )

    st.subheader("Test-set performance")
    col1, col2 = st.columns(2)
    col1.metric("Accuracy", "76.7%")
    col2.metric("F1 Score", "71.5%")
    st.caption(
        "Best of 4 models compared (Logistic Regression, Decision Tree, "
        "Random Forest, KNN), selected by F1 score."
    )
    st.caption(
        "ℹ️ Neutral reviews were rare in the training data, so the model "
        "predicts Neutral less often than Positive or Negative."
    )

# ---------- Main page ----------
st.title("💬 Sentiment Analysis")
st.write(
    "Type a product review or any sentence below and the model will classify it as "
    "**Positive**, **Neutral**, or **Negative**."
)

st.markdown("**Try an example:**")
example_cols = st.columns(len(EXAMPLES))
for col, (label, text) in zip(example_cols, EXAMPLES.items()):
    col.button(label, on_click=set_review, args=(text,))

review = st.text_area(
    "Your review",
    key="review",
    height=160,
    placeholder="The product quality is excellent and I really enjoyed using it.",
)

analyze_col, clear_col, _ = st.columns([2, 1, 4])
analyze = analyze_col.button("🔍 Analyze Sentiment", type="primary")
clear_col.button("Clear", on_click=set_review, args=("",))

if analyze:
    if not review.strip():
        st.warning("Please enter some text before clicking **Analyze Sentiment**.")
    elif not preprocess_text(review):
        st.warning("Please enter some words. Numbers and punctuation alone can't be analyzed.")
    else:
        label, probabilities = predict_sentiment(review, model, vectorizer)
        show_result(label, probabilities)

st.divider()
st.caption("Built with Python, scikit-learn (TF-IDF + Logistic Regression) and Streamlit.")
