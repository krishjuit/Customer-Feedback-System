import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# --- Robust NLTK Resource Downloader ---
def download_nltk_resources():
    resources = {
        'corpora/stopwords': 'stopwords',
        'tokenizers/punkt': 'punkt',
        'corpora/wordnet': 'wordnet',
        'corpora/omw-1.4': 'omw-1.4'
    }
    for path, name in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(name, quiet=True)
            except Exception as e:
                print(f"Warning: Could not download NLTK resource '{name}': {e}")

download_nltk_resources()

# Fallback stopwords in case NLTK downloader fails
FALLBACK_STOPWORDS = set([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", 
    "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", 
    "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", 
    "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", 
    "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", 
    "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", 
    "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", 
    "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", 
    "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", 
    "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", 
    "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", 
    "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
])

# Expansion mapping for common English contractions
CONTRACTION_MAP = {
    "don't": "do not",
    "can't": "cannot",
    "won't": "will not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "doesn't": "does not",
    "shouldn't": "should not",
    "wouldn't": "would not",
    "couldn't": "could not",
    "mustn't": "must not",
    "didn't": "did not",
    "it's": "it is",
    "he's": "he is",
    "she's": "she is",
    "that's": "that is",
    "what's": "what is",
    "where's": "where is",
    "there's": "there is",
    "i'm": "i am",
    "you're": "you are",
    "we're": "we are",
    "they're": "they are",
    "i've": "i have",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have",
    "i'd": "i would",
    "you'd": "you would",
    "we'd": "we would",
    "they'd": "they would",
    "i'll": "i will",
    "you'll": "you will",
    "we'll": "we will",
    "they'll": "they will",
}

# Words indicating negation/contrast that should NOT be removed in sentiment analysis
NEGATION_WORDS = {
    'not', 'no', 'nor', 'neither', 'never', 'none', 'but', 'against', 
    'without', 'hardly', 'scarcely', 'barely', 'few', 'less', 'least'
}

try:
    NLTK_STOPWORDS = set(stopwords.words('english'))
except Exception:
    NLTK_STOPWORDS = FALLBACK_STOPWORDS

# Filter negation words out of the stopword set
STOP_WORDS = NLTK_STOPWORDS - NEGATION_WORDS

try:
    LEMMATIZER = WordNetLemmatizer()
except Exception:
    LEMMATIZER = None

# --- Preprocessing Function ---
def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Expand contractions
    for contraction, expansion in CONTRACTION_MAP.items():
        text = text.replace(contraction, expansion)
    
    # 3. Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # 4. Tokenize
    try:
        tokens = word_tokenize(text)
    except Exception:
        tokens = text.split()
        
    # 5. Remove non-negation stopwords and lemmatize
    cleaned_tokens = []
    for token in tokens:
        if token not in STOP_WORDS and len(token) > 2:
            if LEMMATIZER:
                try:
                    token = LEMMATIZER.lemmatize(token)
                except Exception:
                    pass
            cleaned_tokens.append(token)
            
    return " ".join(cleaned_tokens)

# --- Streamlit Safe Cache Wrapper ---
try:
    import streamlit as st
    cache_data = st.cache_data
except ImportError:
    def cache_data(func):
        return func

@cache_data
def clean_text_list(texts):
    # Convert series or array to list to ensure clean caching behavior
    texts_list = list(texts)
    return [clean_text(t) for t in texts_list]

# --- Sentiment Classifier Pipeline ---
class SentimentModel:
    def __init__(self, model_type='Logistic Regression', C=1.0, alpha=0.5):
        self.model_type = model_type
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1500, sublinear_tf=True, min_df=2)
        
        # Initialize selected model
        if model_type == 'Logistic Regression':
            self.classifier = LogisticRegression(C=C, class_weight='balanced', random_state=42, max_iter=1000)
        elif model_type == 'Linear SVM (SVC)':
            base_svc = LinearSVC(C=C, class_weight='balanced', random_state=42, dual=False, max_iter=2000)
            # Calibrate SVM to output reliable probabilities
            self.classifier = CalibratedClassifierCV(base_svc, cv=3)
        elif model_type == 'Multinomial Naive Bayes':
            self.classifier = MultinomialNB(alpha=alpha)
        else:
            self.classifier = LogisticRegression(C=1.0, class_weight='balanced', random_state=42)
            
        self.is_trained = False
        self.test_accuracy = 0.0
        self.conf_matrix = None
        self.class_report = None
        self.classes = []

    def train(self, texts, labels):
        cleaned_texts = clean_text_list(texts)
        labels = np.array(labels)
        
        # 1. Split into 80% train and 20% test to compute clean, unbiased metrics
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            cleaned_texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # 2. Fit vectorizer on training split
        X_train = self.vectorizer.fit_transform(X_train_raw)
        X_test = self.vectorizer.transform(X_test_raw)
        
        # 3. Fit classifier on training split
        self.classifier.fit(X_train, y_train)
        self.classes = list(self.classifier.classes_)
        
        # 4. Generate evaluation metrics
        test_preds = self.classifier.predict(X_test)
        self.test_accuracy = float(np.mean(test_preds == y_test))
        self.conf_matrix = confusion_matrix(y_test, test_preds, labels=self.classes)
        self.class_report = classification_report(y_test, test_preds, output_dict=True)
        
        # 5. Retrain the model on the full dataset so the prediction engine is as robust as possible
        X_full = self.vectorizer.fit_transform(cleaned_texts)
        self.classifier.fit(X_full, labels)
        self.is_trained = True
        
        return self.test_accuracy

    def predict(self, text):
        if not self.is_trained:
            raise ValueError("Model is not trained yet!")
        cleaned = clean_text(text)
        X = self.vectorizer.transform([cleaned])
        return self.classifier.predict(X)[0]

    def predict_probs(self, text):
        if not self.is_trained:
            raise ValueError("Model is not trained yet!")
        cleaned = clean_text(text)
        X = self.vectorizer.transform([cleaned])
        probs = self.classifier.predict_proba(X)[0]
        classes = self.classifier.classes_
        return dict(zip(classes, probs))

# --- Topic Clustering Pipeline ---
class TopicClustering:
    def __init__(self, n_clusters=4):
        self.n_clusters = n_clusters
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.pca = PCA(n_components=2, random_state=42)
        self.is_fitted = False

    def fit_transform(self, texts):
        cleaned_texts = clean_text_list(texts)
        self.cleaned_texts = cleaned_texts
        
        self.tfidf_matrix = self.vectorizer.fit_transform(cleaned_texts)
        self.cluster_labels = self.kmeans.fit_predict(self.tfidf_matrix)
        
        self.pca_coords = self.pca.fit_transform(self.tfidf_matrix.toarray())
        self.is_fitted = True
        
        return self.cluster_labels, self.pca_coords

    def get_cluster_keywords(self, top_n=5):
        if not self.is_fitted:
            raise ValueError("Clustering is not fitted yet!")
            
        centroids = self.kmeans.cluster_centers_
        feature_names = np.array(self.vectorizer.get_feature_names_out())
        
        cluster_keywords = {}
        for i in range(self.n_clusters):
            sorted_indices = np.argsort(centroids[i])[::-1]
            top_features = feature_names[sorted_indices[:top_n]]
            cluster_keywords[i] = list(top_features)
            
        return cluster_keywords

    def assign_new_text(self, text):
        if not self.is_fitted:
            raise ValueError("Clustering is not fitted yet!")
            
        cleaned = clean_text(text)
        X = self.vectorizer.transform([cleaned])
        
        cluster_label = self.kmeans.predict(X)[0]
        coords = self.pca.transform(X.toarray())[0]
        
        return cluster_label, coords

    def compute_silhouette_scores(self, texts, max_k=8):
        """Compute silhouette score for KMeans clustering for K in range [2, max_k]"""
        cleaned_texts = clean_text_list(texts)
        temp_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
        tfidf_matrix = temp_vectorizer.fit_transform(cleaned_texts)
        
        scores = {}
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(tfidf_matrix)
            score = silhouette_score(tfidf_matrix, labels)
            scores[k] = float(score)
            
        return scores

