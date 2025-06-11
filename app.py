import streamlit as st
import joblib
import re
from nltk.stem import WordNetLemmatizer
import nltk
import os

# --- 1. Streamlit App Title and Description (MOVED TO TOP) ---
st.set_page_config(page_title="IMDB Review Sentiment Analyzer", layout="centered")

# --- 2. Download necessary NLTK resources (check) ---

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/punkt')
except LookupError:
    nltk.download('punkt')

# --- 3. Text Preprocessing Function ---
lemmatizer = WordNetLemmatizer()

def preprocess_text_for_inference(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    text = " ".join(lemmatized_words)
    return text

# --- 4. Load trained model and vectorizer ---
@st.cache_resource
def load_model_components():
    model_dir = 'trained_models'
    vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer.joblib')
    model_path = os.path.join(model_dir, 'logistic_regression_model.joblib')

    if not os.path.exists(vectorizer_path) or not os.path.exists(model_path):
        st.error(f"Model or vectorizer files not found. Ensure they are saved in the '{model_dir}' folder.")
        st.stop()

    vectorizer = joblib.load(vectorizer_path)
    model = joblib.load(model_path)
    return vectorizer, model

tfidf_vectorizer, log_reg_model = load_model_components()

st.title("🎬 IMDB Review Sentiment Analyzer")
st.markdown("---")
st.write("Enter a movie review, and the model will predict whether it is positive or negative.")

# --- 5. Text input field ---
user_input = st.text_area("Your review:", height=150, placeholder="E.g., 'This movie was absolutely terrible, a complete waste of time!'")

# --- 6. Prediction button ---
if st.button("Predict Sentiment"):
    if user_input: # If the user entered text
        # 1. Preprocess the text
        processed_text = preprocess_text_for_inference(user_input)

        # 2. Vectorize the text
        vectorized_text = tfidf_vectorizer.transform([processed_text])

        # 3. Predict
        prediction_encoded = log_reg_model.predict(vectorized_text)[0]
        prediction_proba = log_reg_model.predict_proba(vectorized_text)[0] 

        sentiment_map = {0: 'negative', 1: 'positive'}
        predicted_sentiment = sentiment_map[prediction_encoded]

        # 4. Display the result
        st.markdown("### Result:")
        if predicted_sentiment == 'positive':
            st.success(f"**Predicted Sentiment: Positive 👍**")
        else:
            st.error(f"**Predicted Sentiment: Negative 👎**")

        st.write(f"Probability of positive review: `{prediction_proba[1]:.2f}`")
        st.write(f"Probability of negative review: `{prediction_proba[0]:.2f}`")

    else:
        st.warning("Please enter a review for analysis.")

st.markdown("---")
st.markdown("Developed as part of an IMDB sentiment analysis project.")