# Smart Customer Feedback & Topic Clustering Dashboard

A premium, interactive Natural Language Processing (NLP) dashboard built with **Streamlit** to analyze customer reviews, classify sentiment, group reviews into unsupervised topic clusters, and evaluate machine learning models.

---

## 🚀 Key Features

### 1. Advanced NLP text Preprocessing
*   **Contraction Expansion**: Maps contractions (e.g. *"don't"* $\rightarrow$ *"do not"*, *"can't"* $\rightarrow$ *"cannot"*) before cleaning.
*   **Negation Preservation**: Preserves sentiment-critical negation terms (e.g., *"not"*, *"no"*, *"never"*, *"nor"*) from stopword stripping, ensuring phrases like *"not good"* are classified correctly.
*   **Lemmatization**: Standardizes words to their dictionary forms using NLTK's `WordNetLemmatizer`.

### 2. Multi-Model Sentiment Classification
*   Choose between three popular classifiers from **Scikit-Learn**:
    *   **Logistic Regression** (Balanced class weights)
    *   **Linear Support Vector Classifier (LinearSVC)** (Calibrated for probability estimation)
    *   **Multinomial Naive Bayes**
*   Tune hyperparameters (Regularization strength $C$, Smoothing $\alpha$) directly from the sidebar.

### 3. Model Performance & Diagnostics
*   Evaluates classifiers on an **80/20 train-test split** to report realistic, unbiased out-of-sample accuracy.
*   Generates a Seaborn **Confusion Matrix Heatmap** showing true vs. predicted sentiment frequencies.
*   Renders a detailed **Classification Report** table outlining Precision, Recall, and F1-score for positive, negative, and neutral classes.

### 4. Topic Clustering & Silhouette Optimization
*   Vectorizes feedback using **TF-IDF (unigrams & bigrams)**.
*   Uses **K-Means Clustering** to discover hidden topic groups.
*   Projects the high-dimensional document vectors into a 2D space using **PCA (Principal Component Analysis)** for spatial visualization.
*   **Cluster Count Optimizer**: Runs an on-demand **Silhouette Score Analysis** for $K \in [2, 8]$ and graphs the score. The peak value suggests the mathematically optimal number of topics for the active dataset.

### 5. Live NLP Pipeline Tester
*   Run the entire NLP pipeline interactively.
*   Input custom reviews and instantly view:
    *   Preprocessed tokens side-by-side with original text.
    *   Sentiment predictions and confidence breakdown.
    *   Topic assignment and keywords.
    *   Visual projection of the entry (marked by a star) in 2D semantic space.

---

## 🛠️ Installation & Setup

### Prerequisites
*   Python 3.8 or higher

### Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/krishjuit/Customer-Feedback-System.git
   cd Customer-Feedback-System
   ```

2. **Set up a virtual environment (Recommended):**
   ```bash
   python -m venv .venv
   # Activate on Windows:
   .venv\Scripts\activate
   # Activate on macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

---

## 📁 Project Structure

*   `app.py`: Streamlit frontend dashboard layout, UI blocks, and plotting logic.
*   `model_utils.py`: Text cleaning preprocessing functions, `SentimentModel` pipelines, and `TopicClustering` algorithms.
*   `data_generator.py`: Generates the default synthetic customer review dataset.
*   `feedback_dataset.csv`: The default loaded customer dataset containing 4000+ customer reviews.
*   `requirements.txt`: List of dependencies.
*   `.gitignore`: List of untracked file patterns to ignore.
