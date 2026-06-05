# Resume / CV Description: Customer Feedback System

This document provides ready-to-use bullet points, technical summaries, and interview talking points for your resume.

---

## 📄 Resume Project Section Templates

### Option 1: Bullets-Only Format (Standard)

**AI-Powered Customer Feedback & Topic Clustering System** | *Python, Streamlit, Scikit-Learn, NLTK, PCA, K-Means*
*   **Engineered** an end-to-end NLP analytics dashboard using Streamlit to classify customer sentiment and programmatically categorize 4,000+ customer reviews.
*   **Optimized** text preprocessing pipeline by implementing contraction expansion mapping and negation preservation (excluding terms like "not" and "no" from standard NLTK stopwords), raising sentiment classification robustness.
*   **Designed** and calibrated a multi-model classification engine (Logistic Regression, Linear SVC, Multinomial Naive Bayes), validating performance via an 80/20 train-test split to report out-of-sample accuracy.
*   **Implemented** unsupervised topic clustering using TF-IDF vectorization and K-Means, projecting high-dimensional review vectors onto a 2D PCA semantic space for visual distribution.
*   **Developed** a dynamic cluster optimizer that calculates and plots **Silhouette Scores** ($K \in [2, 8]$) to mathematically determine the optimal number of topics for any uploaded dataset.

---

### Option 2: Short Summary + Bullets (Modern)

**NLP Customer Feedback System (GitHub: [github.com/krishjuit/Customer-Feedback-System](https://github.com/krishjuit/Customer-Feedback-System))**
*Developed an interactive, production-ready machine learning dashboard to parse customer sentiment trends and map unsupervised topic clusters in 2D semantic space.*
*   **Negation-Aware Preprocessing**: Overcame standard stopword-stripping limitations (e.g. "not good" $\rightarrow$ "good") by designing a negation-preserving tokenizer.
*   **Multi-Model Diagnostics**: Integrated Logistic Regression, calibrated SVM, and Naive Bayes classifiers, complete with dynamic hyperparameter sliders, Seaborn confusion matrix heatmaps, and classification reports.
*   **Topic Modeling & PCA**: Vectorized corpus using TF-IDF bigrams and grouped reviews using K-Means clustering, visualized via 2D PCA projections.
*   **Silhouette Optimization**: Added on-demand Silhouette Score curves to automatically guide users to the mathematically optimal cluster count.

---

## 🛠️ Technology Stack Breakdown

*   **Languages**: Python
*   **Machine Learning & NLP**: Scikit-Learn (TF-IDF, Logistic Regression, LinearSVC, MultinomialNB, K-Means, PCA, Silhouette Score), NLTK (Tokenization, Lemmatization, Stopwords)
*   **Data Science**: Pandas, NumPy
*   **Visualization**: Matplotlib, Seaborn
*   **Frontend / UI**: Streamlit
*   **DevOps & Tools**: Git, GitHub, Hugging Face Spaces / Streamlit Sharing

---

## 💬 Interview Talking Points (How to talk about this project)

Be prepared to answer these questions using the actions you performed:

### 1. "What was the biggest technical challenge you faced?"
> **Answer**: *"The biggest challenge was standard stopword removal in NLP. Normally, library stopwords remove words like 'not' or 'no', which breaks sentiment analysis (e.g., 'not clean' becomes 'clean' and gets classified as Positive). I resolved this by custom-filtering the stopword set to retain negation terms and building a contraction expander to turn 'can't' into 'cannot' before character cleaning, ensuring sentiment integrity was preserved."*

### 2. "How did you validate your models?"
> **Answer**: *"Rather than calculating training accuracy (which is biased and looks deceptively high), I implemented a strict 80/20 train-test split. The performance metrics—Confusion Matrix and Classification Report (Precision, Recall, F1)—are computed on the unseen test holdout. Once validated, the model is retrained on the full dataset to optimize its production weight values."*

### 3. "Why did you use K-Means and how did you select the number of clusters?"
> **Answer**: *"We wanted an unsupervised way to discover review topics. I vectorized the reviews using TF-IDF and used K-Means. To choose the number of clusters ($K$), I added a Silhouette Score analyzer. The user runs this analysis on-demand, which calculates the silhouette coefficient for $K=2$ to $K=8$. The peak of this score curve indicates the mathematically optimal cluster configuration for that specific corpus."*
