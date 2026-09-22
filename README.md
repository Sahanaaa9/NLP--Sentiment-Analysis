# Sentiment Analysis of Customer Reviews

A machine learning project that classifies smartphone customer reviews as **Positive**, **Neutral**, or **Negative**. It is deployed as a **Streamlit** web app.

---

## 1. Project Overview

Online shoppers write thousands of reviews. Reading them all by hand is slow. This project trains a machine learning model that reads a review and predicts its sentiment automatically. The trained model is then served through a simple web app, so anyone can type a review and see the prediction.

## 2. Objective

- Clean and prepare review text for machine learning
- Convert text into numbers using **TF-IDF**
- Train and compare several classification models
- Select the best model using the evaluation metrics
- Deploy the best model as a web application

## 3. Dataset

| Item | Details |
|---|---|
| File | `dataset.xlsx` |
| Rows | 1,440 Samsung smartphone reviews |
| Columns | `title`, `rating` (1–5), `body` (review text) |
| Missing values | None |
| Duplicate rows | None |

**Input text:** `body`

**Target:** `sentiment` is created from `rating`:

| Rating | Sentiment | Count |
|---|---|---|
| 1–2 | Negative | 512 |
| 3 | Neutral | 199 |
| 4–5 | Positive | 729 |

The dataset is **imbalanced**: Neutral has far fewer reviews than the other two classes.

## 4. Text Preprocessing

Each review (`body`) is cleaned into a new column called `clean_text`:

1. Convert to **lowercase**
2. Remove **punctuation**
3. Remove **numbers**
4. Remove **extra spaces**

The app applies **exactly the same steps** (the `preprocess_text` function in `app.py`) to the text you type. The notebook checks that this function reproduces the training `clean_text` column for all 1,440 reviews (result: `True`).

> NLTK stopwords are used in the notebook for EDA only (top-20 words and the word cloud). Stop words for the model are removed by TF-IDF's built-in English stop-word list.

## 5. TF-IDF (Feature Extraction)

**TF-IDF (Term Frequency – Inverse Document Frequency)** turns text into numbers. A word gets a high score when it appears often in a review but is rare across all reviews. This means informative words like *"disappointed"* or *"excellent"* matter more than common words like *"phone"*.

```python
TfidfVectorizer(max_features=5000, stop_words='english')
```

- Keeps the 5,000 most important words
- Resulting matrix: **1,440 reviews × 5,000 features**
- The vectorizer is **fitted once in the notebook** and saved as `tfidf_vectorizer.pkl`. The app only *loads* it and never refits it.

**Train/test split:** 80% training (1,152 reviews) / 20% testing (288 reviews), stratified, `random_state=42`.

## 6. Machine Learning Models Used

1. **Logistic Regression**
2. **Decision Tree**
3. **Random Forest** (100 trees)
4. **K-Nearest Neighbors** (k = 5)

## 7. Model Evaluation

All models were evaluated on the same 288 test reviews. Precision, recall and F1 are **weighted averages**.

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| **Logistic Regression** | **0.7674** | **0.7321** | **0.7674** | **0.7151** |
| Decision Tree | 0.6181 | 0.6107 | 0.6181 | 0.6139 |
| Random Forest | 0.7326 | 0.6318 | 0.7326 | 0.6762 |
| KNN | 0.6944 | 0.6403 | 0.6944 | 0.6610 |

**Logistic Regression — per-class results:**

| Class | Precision | Recall | F1 | Test reviews |
|---|---|---|---|---|
| Negative | 0.77 | 0.80 | 0.79 | 102 |
| Neutral | 0.50 | 0.03 | 0.05 | 40 |
| Positive | 0.77 | 0.95 | 0.85 | 146 |

## 8. Selected (Best) Model

The notebook selects the model with the **highest F1 score**:

```python
best_model = comparison.loc[comparison['F1 Score'].idxmax()]
```

➡️ **Logistic Regression** is selected (F1 = 0.7151). It is also the best model on accuracy, precision and recall. The notebook saves the selected model automatically as `sentiment_model.pkl`, so the deployed model is always the one chosen by the comparison.

**Known limitation:** the model is good at Positive and Negative reviews but rarely predicts **Neutral** (recall 0.03). This is because there are few Neutral reviews and 3-star reviews often mix positive and negative opinions. Neutral or factual sentences may therefore be classified as Positive or Negative.

## 9. Application Workflow

```
User enters text
   → preprocess_text()  (same cleaning as training)
   → tfidf_vectorizer.pkl  (transform only)
   → sentiment_model.pkl   (Logistic Regression → predict)
   → Positive / Neutral / Negative  (+ confidence for each class)
```

The model and vectorizer are loaded once and cached (`@st.cache_resource`). **No training happens in the app.**

## 10. Technologies Used

- **Python**
- **Pandas, NumPy**: data handling (notebook)
- **Matplotlib, Seaborn, WordCloud**: visualisation (notebook)
- **NLTK**: stopwords for EDA (notebook)
- **Scikit-learn**: TF-IDF and ML models
- **Joblib**: saving/loading the model
- **Streamlit**: web application

## 11. Project Structure

```
sentiment-analysis/
├── Project - Sentiment Analysis (1).ipynb   # EDA, preprocessing, training, evaluation, saving
├── dataset.xlsx                             # training data (used by the notebook only)
├── app.py                                   # Streamlit web app
├── sentiment_model.pkl                      # trained Logistic Regression model
├── tfidf_vectorizer.pkl                     # fitted TF-IDF vectorizer
├── requirements.txt                         # packages needed to run the app
├── .streamlit/config.toml                   # app theme colour
├── .gitignore
└── README.md
```

## 12. Install Dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` contains only what the app needs (`streamlit`, `scikit-learn`, `joblib`). `scikit-learn` is pinned to **1.9.0**, the version used to create the `.pkl` files, so the saved model loads correctly.

## 13. Run the Application Locally

```bash
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

To retrain the model, run all cells in the notebook (with `dataset.xlsx` in the same folder). It regenerates both `.pkl` files.

## 14. Deploy on Streamlit Community Cloud

1. Push this project to a **public GitHub repository**. Make sure `app.py`, `sentiment_model.pkl`, `tfidf_vectorizer.pkl` and `requirements.txt` are in the repository root.
2. Go to **https://share.streamlit.io** and sign in with GitHub.
3. Click **Create app** → **Deploy a public app from GitHub**.
4. Choose:
   - **Repository:** `<your-username>/sentiment-analysis`
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Open **Advanced settings** and choose **Python 3.12**.
6. Click **Deploy**. After a few minutes you get a public URL like `https://<your-app-name>.streamlit.app`.
