import re
import string
from pathlib import Path

import joblib
import streamlit as st

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "sentiment_model.pkl"
VECTORIZER_PATH = BASE_DIR / "tfidf_vectorizer.pkl"

# Readable names for the models trained in the notebook
MODEL_NAMES = {
    "LogisticRegression": "Logistic Regression",
    "DecisionTreeClassifier": "Decision Tree",
    "RandomForestClassifier": "Random Forest",
    "KNeighborsClassifier": "KNN",
}

# Test-set results from the notebook's "Model Comparison" table
# (288 test reviews; precision, recall and F1 are weighted averages)
NOTEBOOK_RESULTS = {
    "Logistic Regression": {"Accuracy": 0.7674, "Precision": 0.7321, "Recall": 0.7674, "F1 Score": 0.7151},
    "Decision Tree": {"Accuracy": 0.6181, "Precision": 0.6107, "Recall": 0.6181, "F1 Score": 0.6139},
    "Random Forest": {"Accuracy": 0.7326, "Precision": 0.6318, "Recall": 0.7326, "F1 Score": 0.6762},
    "KNN": {"Accuracy": 0.6944, "Precision": 0.6403, "Recall": 0.6944, "F1 Score": 0.6610},
}

EXAMPLES = {
    "😊 Positive Example": "The product quality is excellent and I really enjoyed using it. The camera and battery life are amazing!",
    "😞 Negative Example": "Worst phone ever. The battery drains fast, it heats up and the camera is terrible. Totally disappointed.",
    "😐 Neutral / Mixed Example": "The phone is okay. Camera is average and battery is fine but the display could be better.",
}

SENTIMENT_STYLE = {
    "Positive": {"emoji": "😊", "color": "#059669", "soft": "#ecfdf5", "text": "The review expresses a positive opinion."},
    "Neutral": {"emoji": "😐", "color": "#d97706", "soft": "#fffbeb", "text": "The review is mixed or neutral."},
    "Negative": {"emoji": "😞", "color": "#e11d48", "soft": "#fff1f2", "text": "The review expresses a negative opinion."},
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp h1, .stApp h2, .stApp h3, .stApp p, .stApp label, .stApp textarea,
.stApp button, .stApp li, .sa-font { font-family: 'Inter', system-ui, sans-serif; }

.block-container { max-width: 1080px; padding-top: 2.2rem; padding-bottom: 2rem; }
.stDeployButton, [data-testid="stAppDeployButton"], [data-testid="stDecoration"] { display: none; }

@keyframes sa-fade-up { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }
@keyframes sa-grow { from { width: 0; } }

/* Header */
.sa-header { display: flex; justify-content: space-between; align-items: center; gap: 16px;
  flex-wrap: wrap; margin-bottom: 1.4rem; }
.sa-title { font-size: 2.1rem; font-weight: 800; color: #1e1b4b; letter-spacing: -0.02em; line-height: 1.2; }
.sa-subtitle { color: #64748b; font-size: 1.05rem; margin-top: 4px; }
.sa-badge { display: inline-flex; align-items: center; gap: 8px; background: #eef2ff; color: #4338ca;
  border: 1px solid #e0e7ff; padding: 7px 14px; border-radius: 999px; font-size: 0.85rem; font-weight: 600; }
.sa-dot { width: 8px; height: 8px; border-radius: 50%; background: #22c55e; box-shadow: 0 0 0 3px #dcfce7; }

/* Cards */
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .sa-card-marker),
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"] > [data-testid="element-container"] .sa-card-marker) {
  background: #ffffff; border: 1px solid #e8eaf3 !important; border-radius: 18px;
  box-shadow: 0 4px 24px rgba(30, 27, 75, 0.06); padding: 10px 8px; }
.sa-card { background: #ffffff; border: 1px solid #e8eaf3; border-radius: 18px;
  box-shadow: 0 4px 24px rgba(30, 27, 75, 0.06); padding: 22px 24px; height: 100%;
  animation: sa-fade-up 0.35s ease-out; }
.sa-card-title { font-size: 1.25rem; font-weight: 700; color: #1e1b4b; }
.sa-card-subtitle { color: #64748b; font-size: 0.95rem; margin: 2px 0 6px 0; }
.sa-eyebrow { font-size: 0.75rem; font-weight: 700; letter-spacing: 0.12em; color: #6366f1;
  text-transform: uppercase; }
.sa-section { margin: 2.2rem 0 0.9rem 0; }
.sa-section-title { font-size: 1.35rem; font-weight: 700; color: #1e1b4b; }

/* Inputs & buttons */
.stTextArea textarea { font-size: 1rem; line-height: 1.55; }
.stTextArea [data-baseweb="textarea"] { border-radius: 12px; border-color: #e2e5f0; background: #fafbff; }
[data-testid="stColumn"] [data-testid="stElementContainer"]:has(.stButton),
[data-testid="stColumn"] .stButton { width: 100% !important; }
.stButton button { width: 100%; border-radius: 10px; font-weight: 600; border: 1px solid #e2e5f0;
  transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease; }
.stButton button:hover { transform: translateY(-1px); border-color: #a5b4fc;
  box-shadow: 0 4px 12px rgba(79, 70, 229, 0.12); }
.stButton button[kind="primary"], [data-testid="stBaseButton-primary"], [data-testid="baseButton-primary"] {
  background: #4f46e5; border: none; box-shadow: 0 6px 16px rgba(79, 70, 229, 0.28); padding: 0.55rem 1.4rem; }
.stButton button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover,
[data-testid="baseButton-primary"]:hover { background: #4338ca; }

/* Result */
.sa-result { text-align: center; border-radius: 18px; padding: 26px 20px; height: 100%;
  animation: sa-fade-up 0.35s ease-out; border: 1px solid; }
.sa-result-emoji { font-size: 3.2rem; line-height: 1; margin: 14px 0 8px 0; }
.sa-result-label { font-size: 2rem; font-weight: 800; letter-spacing: 0.08em; }
.sa-result-text { color: #475569; margin-top: 6px; font-size: 0.95rem; }
.sa-result-conf { display: inline-block; margin-top: 14px; padding: 5px 14px; border-radius: 999px;
  background: #ffffff; font-weight: 600; font-size: 0.9rem; }

@media (max-width: 640px) { .sa-result { margin-bottom: 14px; } }

/* Probabilities */
.sa-prob-row { display: grid; grid-template-columns: 80px 1fr 56px; align-items: center; gap: 12px; margin: 16px 0; }
.sa-prob-label { font-weight: 600; color: #334155; font-size: 0.95rem; }
.sa-prob-track { background: #eef0f6; border-radius: 999px; height: 12px; overflow: hidden; }
.sa-prob-fill { height: 100%; border-radius: 999px; animation: sa-grow 0.7s ease-out; }
.sa-prob-value { text-align: right; font-weight: 700; color: #1e1b4b; font-variant-numeric: tabular-nums; }

/* Summary & metrics */
.sa-stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 14px; }
.sa-stat { background: #ffffff; border: 1px solid #e8eaf3; border-radius: 14px; padding: 14px 18px;
  animation: sa-fade-up 0.35s ease-out; }
.sa-stat-label { color: #64748b; font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; }
.sa-stat-value { color: #1e1b4b; font-size: 1.35rem; font-weight: 700; margin-top: 4px; }
.sa-metric { background: #ffffff; border: 1px solid #e8eaf3; border-radius: 16px; padding: 18px 20px;
  box-shadow: 0 4px 18px rgba(30, 27, 75, 0.05); }
.sa-metric-value { color: #1e1b4b; font-size: 1.7rem; font-weight: 800; margin-top: 6px; }
.sa-metric-value.sa-small { font-size: 1.15rem; padding-top: 8px; }
.sa-note { color: #64748b; font-size: 0.85rem; margin-top: 12px; }

/* Sidebar */
[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e8eaf3; }
.sa-side-label { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.12em; color: #6366f1;
  text-transform: uppercase; margin: 22px 0 6px 0; }
.sa-side-text { color: #334155; font-size: 0.9rem; line-height: 1.5; }
.sa-side-row { display: flex; justify-content: space-between; font-size: 0.9rem; padding: 5px 0;
  border-bottom: 1px dashed #e8eaf3; color: #334155; }
.sa-side-row b { color: #1e1b4b; }
.sa-pipeline { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; font-size: 0.82rem; }
.sa-nowrap { white-space: nowrap; }
.sa-chip { background: #eef2ff; color: #4338ca; padding: 4px 10px; border-radius: 999px; font-weight: 600; }

/* Footer */
.sa-footer { text-align: center; color: #94a3b8; font-size: 0.85rem; margin-top: 2.6rem;
  padding-top: 1.2rem; border-top: 1px solid #e8eaf3; }
</style>
"""


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
    probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = dict(zip(model.classes_, model.predict_proba(features)[0]))
    return label, probabilities


def html(block):
    # Remove indentation and blank lines so Markdown renders it as plain HTML
    st.markdown("\n".join(line.strip() for line in block.splitlines() if line.strip()),
                unsafe_allow_html=True)


st.set_page_config(page_title="Sentiment Analysis", page_icon="💬", layout="wide")

# ---------- Session state & button callbacks ----------
st.session_state.setdefault("review", "")
st.session_state.setdefault("result", None)
st.session_state.setdefault("notice", None)


def use_example(text):
    st.session_state.review = text
    st.session_state.result = None
    st.session_state.notice = None


def clear_all():
    use_example("")


def run_analysis():
    text = st.session_state.review
    st.session_state.result = None
    st.session_state.notice = None
    if not text.strip():
        st.session_state.notice = "Please enter a review before clicking Analyze Sentiment."
    elif not preprocess_text(text):
        st.session_state.notice = "Please enter some words. Numbers and punctuation alone can't be analyzed."
    else:
        model, vectorizer = load_artifacts()
        label, probabilities = predict_sentiment(text, model, vectorizer)
        st.session_state.result = {"text": text, "label": label, "probabilities": probabilities}


# ---------- Page ----------
st.markdown(CSS, unsafe_allow_html=True)

try:
    model, vectorizer = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model files not found. Make sure `sentiment_model.pkl` and "
        "`tfidf_vectorizer.pkl` are in the same folder as `app.py`."
    )
    st.stop()

model_name = MODEL_NAMES.get(type(model).__name__, type(model).__name__)
metrics = NOTEBOOK_RESULTS.get(model_name)

# ---------- Sidebar ----------
with st.sidebar:
    performance_rows = "".join(
        f'<div class="sa-side-row"><span>{name}</span><b>{value:.1%}</b></div>'
        for name, value in (metrics or {}).items()
    )
    html(f"""
        <div class="sa-font">
        <div class="sa-side-label" style="margin-top:4px">About the project</div>
        <div class="sa-side-text">Classifies customer reviews as Positive, Neutral or Negative using
        classic NLP and machine learning.</div>
        <div class="sa-side-label">Model</div>
        <div class="sa-side-text"><b>{model_name}</b></div>
        <div class="sa-side-label">Dataset</div>
        <div class="sa-side-text">1,440 smartphone customer reviews (title, rating, body).<br>
        Rating 1–2 → Negative · 3 → Neutral · 4–5 → Positive</div>
        <div class="sa-side-label">Pipeline</div>
        <div class="sa-pipeline"><span class="sa-chip">Text preprocessing</span>
        <span class="sa-nowrap">→ <span class="sa-chip">TF-IDF</span></span>
        <span class="sa-nowrap">→ <span class="sa-chip">ML Model</span></span></div>
        {'<div class="sa-side-label">Performance</div>' + performance_rows if metrics else ''}
        </div>
    """)

# ---------- 1. Header ----------
html(f"""
    <div class="sa-header sa-font">
    <div>
    <div class="sa-title">💬 Sentiment Analysis</div>
    <div class="sa-subtitle">Understand customer opinions instantly with machine learning.</div>
    </div>
    <div class="sa-badge"><span class="sa-dot"></span>Model: {model_name}</div>
    </div>
""")

# ---------- 2. Analysis card ----------
with st.container(border=True):
    html("""
        <div class="sa-card-marker sa-font" style="padding: 6px 8px 0 8px">
        <div class="sa-card-title">Analyze a Review</div>
        <div class="sa-card-subtitle">Enter a customer review and let the model identify its sentiment.</div>
        </div>
    """)
    example_cols = st.columns(len(EXAMPLES))
    for col, (label, text) in zip(example_cols, EXAMPLES.items()):
        col.button(label, on_click=use_example, args=(text,))

    st.text_area(
        "Customer review",
        key="review",
        height=170,
        placeholder="The product quality is excellent and I really enjoyed using it.",
        label_visibility="collapsed",
    )

    analyze_col, clear_col, _ = st.columns([2, 1, 3])
    analyze_col.button("✨ Analyze Sentiment", type="primary", on_click=run_analysis)
    clear_col.button("Reset", on_click=clear_all)

    if st.session_state.notice:
        st.warning(st.session_state.notice)

# ---------- 3–5. Result, probabilities, summary ----------
result = st.session_state.result
# Only show a result that belongs to the text currently in the box
if result and result["text"] == st.session_state.review:
    label = result["label"]
    style = SENTIMENT_STYLE.get(label, SENTIMENT_STYLE["Neutral"])
    probabilities = result["probabilities"]

    html('<div class="sa-section sa-font"><div class="sa-eyebrow">Output</div>'
         '<div class="sa-section-title">Analysis Result</div></div>')

    confidence_html = ""
    if probabilities:
        confidence_html = (f'<div class="sa-result-conf" style="color:{style["color"]}">'
                           f'Confidence {probabilities[label]:.0%}</div>')

    result_col, prob_col = st.columns([1, 1.25]) if probabilities else (st.container(), None)
    with result_col:
        html(f"""
            <div class="sa-result sa-font" style="background:{style['soft']}; border-color:{style['color']}33">
            <div class="sa-eyebrow" style="color:{style['color']}">Analysis Result</div>
            <div class="sa-result-emoji">{style['emoji']}</div>
            <div class="sa-result-label" style="color:{style['color']}">{label.upper()}</div>
            <div class="sa-result-text">{style['text']}</div>
            {confidence_html}
            </div>
        """)

    if probabilities:
        with prob_col:
            rows = "".join(
                f"""<div class="sa-prob-row">
                <div class="sa-prob-label">{name}</div>
                <div class="sa-prob-track"><div class="sa-prob-fill"
                style="width:{probabilities.get(name, 0):.1%}; background:{SENTIMENT_STYLE[name]['color']}"></div></div>
                <div class="sa-prob-value">{probabilities.get(name, 0):.0%}</div>
                </div>"""
                for name in ["Positive", "Neutral", "Negative"]
            )
            html(f"""
                <div class="sa-card sa-font">
                <div class="sa-card-title">Class Probabilities</div>
                <div class="sa-card-subtitle">Probability of each sentiment returned by the model.</div>
                {rows}
                </div>
            """)

    text = result["text"]
    html(f"""
        <div class="sa-stats sa-font" style="margin-top:16px">
        <div class="sa-stat"><div class="sa-stat-label">Characters</div>
        <div class="sa-stat-value">{len(text):,}</div></div>
        <div class="sa-stat"><div class="sa-stat-label">Words</div>
        <div class="sa-stat-value">{len(text.split()):,}</div></div>
        <div class="sa-stat"><div class="sa-stat-label">Predicted Sentiment</div>
        <div class="sa-stat-value" style="color:{style['color']}">{label}</div></div>
        </div>
    """)

# ---------- 6. Model information ----------
html('<div class="sa-section sa-font"><div class="sa-eyebrow">Model</div>'
     '<div class="sa-section-title">Model Information</div></div>')

metric_cards = f"""<div class="sa-metric"><div class="sa-stat-label">Model</div>
    <div class="sa-metric-value sa-small">{model_name}</div></div>"""
for name, value in (metrics or {}).items():
    metric_cards += f"""<div class="sa-metric"><div class="sa-stat-label">{name}</div>
        <div class="sa-metric-value">{value:.1%}</div></div>"""

html(f"""
    <div class="sa-font">
    <div class="sa-stats">{metric_cards}</div>
    <div class="sa-note">Evaluated on 288 test reviews (20% stratified split). Precision, recall and F1 are
    weighted averages. Selected as the best of 4 models (Logistic Regression, Decision Tree, Random Forest,
    KNN) by F1 score. Neutral reviews were rare in the training data, so Neutral is predicted less often.</div>
    </div>
""")

# ---------- Footer ----------
html('<div class="sa-footer sa-font">Built with Python • Scikit-learn • TF-IDF • Streamlit</div>')
