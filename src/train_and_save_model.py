# Import necessary libraries
import re
import joblib
import os
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Download stopword lists and other necessary NLTK resources
# Check if resources are downloaded, and download if not
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    print("NLTK 'stopwords' corpus successfully downloaded.")

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    print("NLTK 'punkt' tokenizer successfully downloaded.")

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    print("NLTK 'wordnet' corpus successfully downloaded.")

# Initialize NLTK lemmatizer
lemmatizer = WordNetLemmatizer()

# Function for full text preprocessing:
def preprocess_text(text):
    """
    Performs full text preprocessing for sentiment analysis.
    """
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

# Load data
try:
    df = pd.read_csv("../data/IMDB Dataset.csv")
    print("Dataset 'IMDB Dataset.csv' successfully loaded.")
except FileNotFoundError:
    print("Error: 'IMDB Dataset.csv' file not found. Make sure the file is in the same directory.")
    exit()

# Apply the preprocess_text function to the 'review' column to create 'processed_review'
print("Applying full text preprocessing.")
df['processed_review'] = df['review'].apply(preprocess_text)
print("Preprocessing completed.")

# Define features (X) and target variable (y)
X = df['processed_review']
y = df['sentiment']

# Convert target variable 'sentiment' to numerical format (0 for 'negative', 1 for 'positive')
y_encoded = y.map({'negative': 0, 'positive': 1})

# Initialize TfidfVectorizer with best parameters (from GridSearchCV)
tfidf_vectorizer_final = TfidfVectorizer(max_features=30000, ngram_range=(1,2))

print("Training TfidfVectorizer on all preprocessed data and transforming it.")
X_tfidf_final = tfidf_vectorizer_final.fit_transform(X)
print(f"Data dimensionality after TF-IDF: {X_tfidf_final.shape}")

# Initialize Logistic Regression model with best parameters (from GridSearchCV)
log_reg_model_final = LogisticRegression(C=3.0, solver='liblinear', random_state=42)

print("Training Logistic Regression model on all transformed data.")
log_reg_model_final.fit(X_tfidf_final, y_encoded)
print("Model training completed.")

# Create directory for saving models if it doesn't exist
model_dir = '../trained_models'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    print(f"Directory created: {model_dir}")

# Save the trained vectorizer
vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer.joblib')
joblib.dump(tfidf_vectorizer_final, vectorizer_path)
print(f"Vectorizer successfully saved to: {vectorizer_path}")

# Save the trained Logistic Regression model
model_path = os.path.join(model_dir, 'logistic_regression_model.joblib')
joblib.dump(log_reg_model_final, model_path)
print(f"Model successfully saved to: {model_path}")

print("Model and vectorizer are ready for production use.")