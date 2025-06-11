# IMDB Movie Review Sentiment Analysis

## Project Overview
This project focuses on the Sentiment Analysis of movie reviews from the IMDB dataset. The goal of the project is to develop a machine learning model capable of classifying text reviews as "positive" or "negative". The project covers the complete NLP pipeline: from exploratory data analysis (EDA) and text preprocessing to building, training, and evaluating classification models.

Graphs and Classification Reports can be found in the ipynb file.

I deployed an interactive demo of the tonality analysis model. You can test it by following this link: https://sentiment-analysis-of-reviews.streamlit.app/"

## Technologies Used
### Programming Language: Python

### Libraries:

* pandas for data manipulation
* numpy for numerical operations
* matplotlib and seaborn for data visualization
* nltk for natural language processing (lemmatization, stopwords)
* scikit-learn for text vectorization, model building and evaluation (CountVectorizer, TfidfVectorizer, LogisticRegression, MultinomialNB, SVC, Pipeline, GridSearchCV, train_test_split, cross_val_score, classification_report, confusion_matrix, accuracy_score)
* re for regular expressions

## Project Phases

## Phase 1: Data Loading

* Task: Load the IMDB Dataset.csv dataset.
* Results: Data successfully loaded, its structure and initial rows examined.

## Phase 2: Exploratory Data Analysis (EDA)

* Task: Understand the structure and characteristics of the text data, identify patterns, class distribution, and review specifics.
* Key Observations and Conclusions:
  * Class Balance: The dataset is perfectly balanced, containing 25,000 positive and 25,000 negative reviews. This eliminates the need for class balancing.
  * Review Length: Reviews vary significantly in length (from 6 to 2494 words), with an average length of about 229 words. The length distributions for positive and negative reviews are similar, indicating that length itself is not a strong predictor of sentiment. It was decided that TfidfVectorizer with max_features would effectively handle this variability.
  * Word Clouds: Visualization revealed clear sentiment indicators (e.g., "great," "love," "best" for positive; "bad," "worst," "plot," "acting" for negative). Frequent neutral words were also identified and added to the stopword list for more relevant unigram analysis.
  * N-gram Frequency:
    * Uni-grams: Confirmed the presence of sentiment-specific words, as well as common neutral words (e.g., "good" appeared in the top of both categories, highlighting the importance of context).
    * Bi-grams and Tri-grams: Analysis of N-grams with stopwords retained showed that many top combinations (e.g., "of the," "in the") do not carry direct semantic meaning. However, the decision was made not to remove stopwords before N-gram vectorization, as TfidfVectorizer effectively reduces the weight of such common combinations, and removing stopwords could lead to the loss of important sentimental context (e.g., "not good," "very bad").

## Phase 3: Text Preprocessing

* Task: Create a unified function for cleaning and normalizing text data, preparing it for vectorization.
* Key Steps:
  * Convert text to lowercase.
  * Remove HTML tags and URLs.
  * Remove punctuation and numbers.
  * Normalize whitespace.
  * Lemmatization: Reduce words to their base form (e.g., "running" -> "run") to reduce vocabulary size and improve generalization.
  * Note: Stopwords were not removed at this stage to preserve them for forming contextual N-grams, which would be handled by TfidfVectorizer.
* Results: A new column processed_review was created in the DataFrame, containing clean, lemmatized texts.

## Phase 4: Text Vectorization

* Task: Transform preprocessed text data into numerical vectors understandable by machine learning models.
* Key Steps:
  * Split data into training and test sets (X_train, X_test, y_train, y_test) before vectorization to prevent data leakage.
  * Use TfidfVectorizer to create features. This method effectively weighs words/N-grams by their importance in a document relative to the entire collection.
  * Apply ngram_range=(1,2) to include both single words (unigrams) and pairs of words (bigrams) to capture context.
  * Limit max_features to control vocabulary size and computational resources.
* Results: Sparse matrices X_train_tfidf and X_test_tfidf were obtained, containing TF-IDF representations of the reviews.

## Phase 5: Model Building and Training

* Task: Train and evaluate the performance of several classical machine learning models on the vectorized data.
* Models and their Initial Performance:
  * Logistic Regression: Accuracy: 0.9049. Showed very strong and balanced results.
  * Multinomial Naive Bayes: Accuracy: 0.8769. A good baseline model, but performed worse than others.
  * Support Vector Machine (SVC) (Linear Kernel): Accuracy: 0.9086. Showed the best accuracy among the three.
* Model Selection for Further Work:
  * Despite SVC's slightly higher accuracy, Logistic Regression was chosen as the primary model for further optimization. Justification: The difference in accuracy between SVC and Logistic Regression was minor (0.9086 vs. 0.9049), while Logistic Regression is significantly faster in training and prediction, and more interpretable. This makes it a more practical choice for most real-world applications.

## Phase 6: Model Evaluation and Improvement

* Task: Deepen the performance analysis of the selected model (Logistic Regression) and optimize its hyperparameters.
* Key Steps and Results:
  * Analysis of Important Features: Investigated Logistic Regression coefficients to identify the most influential words and N-grams associated with positive (high positive coefficients, e.g., "great," "excellent," "highly recommend") and negative (high negative coefficients, e.g., "bad," "worst," "waste time") sentiment. This confirmed that the model learns meaningful linguistic patterns.
  * Hyperparameter Optimization with GridSearchCV:
    * A pipeline including TfidfVectorizer and LogisticRegression was optimized using GridSearchCV.
    * Best Parameters Found:
      * log_reg__C: 3.0 (slightly less regularization).
      * tfidf__max_features: 30000 (optimal vocabulary size).
      * tfidf__ngram_range: (1, 2) (confirmed the importance of unigrams and bigrams).
    * Optimized Model Accuracy: 0.911 on the test set. This is a small but stable improvement over the baseline Logistic Regression, and a reduction in errors.
  * Cross-Validation:
    * 5-fold stratified cross-validation was performed on the entire dataset using the best hyperparameters found.
    * Average Cross-Validation Accuracy: 0.9094
    * Standard Deviation: 0.0018
    * Conclusion: The cross-validation results confirm the model's high and stable generalization ability. The low standard deviation indicates the reliability of the performance estimate.


## Conclusion

Within this project, we successfully developed and optimized a model for IMDB review sentiment analysis. The Logistic Regression model, vectorized using TF-IDF with unigrams and bigrams, demonstrated high and stable accuracy of approximately 91.1%. This makes it an effective and practical solution for text sentiment classification, considering its speed and interpretability.

The project showcased a complete lifecycle of working with text data: from initial understanding to building and evaluating a high-performing model, including important steps for preprocessing, feature selection, and hyperparameter optimization.

## Next Steps (Potential Improvements)

* More Complex Preprocessing Approaches: Investigate the impact of handling slang, sarcasm, or specific jargon.
* Advanced Vectorization/Embeddings Methods: Explore methods like Word2Vec, GloVe, or even contextual embeddings (e.g., BERT) to compare performance, although this would require deeper exploration of deep learning.
* Using Other Models: Experiment with other machine learning algorithms or even simple deep learning models (RNN/LSTM) to compare their effectiveness and computational costs.
* Prediction Interface: Create a simple web interface or script for interactively testing the model on new reviews.
