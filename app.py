import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from model_utils import SentimentModel, TopicClustering, clean_text

# Set Streamlit Page Config
st.set_page_config(
    page_title="Smart Feedback & Topic Clustering Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Injected Custom CSS for Premium Design & Responsiveness (Works in both Dark & Light Themes)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    /* Global Font Settings */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Dashboard Glassmorphic Cards */
    .metric-card {
        background-color: rgba(128, 128, 128, 0.08);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(128, 128, 128, 0.15);
        border-left: 5px solid #4F46E5;
        margin-bottom: 20px;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    }
    .metric-card.positive {
        border-left-color: #10B981;
    }
    .metric-card.neutral {
        border-left-color: #6B7280;
    }
    .metric-card.negative {
        border-left-color: #EF4444;
    }
    .metric-value {
        font-size: 32px;
        font-weight: 700;
        margin: 0;
    }
    .metric-label {
        font-size: 13px;
        color: #888888;
        font-weight: 600;
        margin-top: 6px;
        text-transform: uppercase;
        letter-spacing: 0.7px;
    }
    
    /* Styled badge elements */
    .badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
        text-align: center;
    }
    .badge-positive {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .badge-neutral {
        background-color: rgba(107, 114, 128, 0.15);
        color: #9CA3AF;
        border: 1px solid rgba(107, 114, 128, 0.3);
    }
    .badge-negative {
        background-color: rgba(239, 68, 68, 0.15);
        color: #EF4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    /* Subtitle spacing */
    .dashboard-subtitle {
        color: #888888;
        font-size: 16px;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to get correct paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "feedback_dataset.csv")

# Ensure dataset exists, generate if missing
if not os.path.exists(DATASET_PATH):
    from data_generator import generate_dataset
    generate_dataset()

# Cache data loading
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

# Cache model training based on hyperparameters
@st.cache_resource
def train_sentiment_model(df, model_type, C, alpha):
    model = SentimentModel(model_type=model_type, C=C, alpha=alpha)
    test_accuracy = model.train(df['review_text'], df['sentiment'])
    return model, test_accuracy

# Cache topic clustering pipeline, labels, and coords
@st.cache_resource
def get_topic_clustering(texts, n_clusters):
    pipeline = TopicClustering(n_clusters=n_clusters)
    labels, coords = pipeline.fit_transform(texts)
    return pipeline, labels, coords

# Cache silhouette score calculation to prevent CPU/memory spikes
@st.cache_data
def run_silhouette_analysis(texts, max_k=8):
    pipeline = TopicClustering()
    return pipeline.compute_silhouette_scores(texts, max_k)

# Load default dataset
raw_df = load_data(DATASET_PATH)

# --- SIDEBAR CONTROLS ---
st.sidebar.image("https://img.icons8.com/nolan/96/artificial-intelligence.png", width=80)
st.sidebar.title("Dashboard Controls")
st.sidebar.markdown("---")

# Option 1: File Uploader
uploaded_file = st.sidebar.file_uploader("Upload your own CSV feedback", type=["csv"])
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        if 'review_text' not in df.columns:
            st.sidebar.error("CSV must contain a 'review_text' column!")
            df = raw_df
        else:
            st.sidebar.success("Loaded uploaded dataset successfully!")
            # Add a placeholder if sentiment column is missing
            if 'sentiment' not in df.columns:
                df['sentiment'] = "Neutral"
    except Exception as e:
        st.sidebar.error(f"Error loading file: {e}")
        df = raw_df
else:
    df = raw_df

# Option 2: Classifier Model Selector
selected_model_type = st.sidebar.selectbox(
    "Sentiment Classifier Model",
    options=["Logistic Regression", "Linear SVM (SVC)", "Multinomial Naive Bayes"],
    index=0
)

# Option 3: Model Hyperparameters
with st.sidebar.expander("⚙️ Model Hyperparameters"):
    if selected_model_type in ["Logistic Regression", "Linear SVM (SVC)"]:
        model_c = st.slider("Regularization Strength (C)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
        model_alpha = 0.5
    else:
        model_alpha = st.slider("Smoothing Parameter (Alpha)", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
        model_c = 1.0

# Option 4: Number of Cluster/Topic groups
n_clusters = st.sidebar.slider("Number of Topic Clusters (K)", min_value=2, max_value=8, value=4, step=1)
st.sidebar.markdown("""
<small style='color:#888;'>Adjusting K changes the number of topic groups K-Means finds in your dataset.</small>
""", unsafe_allow_html=True)

# Decide which dataset to train on
train_df = raw_df
if uploaded_file is not None and 'sentiment' in df.columns and df['sentiment'].nunique() > 1:
    train_on_uploaded = st.sidebar.checkbox("Train classifier on uploaded data", value=True)
    if train_on_uploaded:
        train_df = df

# Train sentiment model on selected data
sentiment_model, test_accuracy = train_sentiment_model(train_df, selected_model_type, model_c, model_alpha)

# Fit clustering on the active dataset
clustering_pipeline, cluster_labels, pca_coords = get_topic_clustering(df['review_text'], n_clusters)

# Add cluster labels to our working dataframe
df['cluster_label'] = cluster_labels
df['pca_x'] = pca_coords[:, 0]
df['pca_y'] = pca_coords[:, 1]

# If the uploaded dataset did not have true sentiment labels, let's predict them
if uploaded_file is not None and ('sentiment' not in df.columns or df['sentiment'].nunique() <= 1):
    df['sentiment'] = df['review_text'].apply(lambda x: sentiment_model.predict(x))

# --- MAIN DASHBOARD AREA ---
st.title("📊 Customer Feedback & Topic Clustering Dashboard")
st.markdown("<p class='dashboard-subtitle'>Analyze sentiment trends, categorize reviews, and map text clusters in 2D space with advanced diagnostic accuracy.</p>", unsafe_allow_html=True)

# --- KPI METRIC CARDS ---
total_reviews = len(df)
positive_pct = int(np.mean(df['sentiment'] == 'Positive') * 100) if total_reviews > 0 else 0
neutral_pct = int(np.mean(df['sentiment'] == 'Neutral') * 100) if total_reviews > 0 else 0
negative_pct = int(np.mean(df['sentiment'] == 'Negative') * 100) if total_reviews > 0 else 0

kpi_cols = st.columns(4)

with kpi_cols[0]:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{total_reviews}</div>
        <div class="metric-label">Total Reviews</div>
    </div>
    """, unsafe_allow_html=True)
    
with kpi_cols[1]:
    st.markdown(f"""
    <div class="metric-card positive">
        <div class="metric-value">{positive_pct}%</div>
        <div class="metric-label">Positive Sentiment</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_cols[2]:
    st.markdown(f"""
    <div class="metric-card neutral">
        <div class="metric-value">{neutral_pct}%</div>
        <div class="metric-label">Neutral Sentiment</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_cols[3]:
    st.markdown(f"""
    <div class="metric-card negative">
        <div class="metric-value">{negative_pct}%</div>
        <div class="metric-label">Negative Sentiment</div>
    </div>
    """, unsafe_allow_html=True)

# --- TABS LAYOUT ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Sentiment Analytics", 
    "🎯 Topic Clustering Map", 
    "📊 Model Performance & Diagnostics",
    "✍️ Live Sentiment & Topic Tester"
])

# ==================== TAB 1: SENTIMENT ANALYTICS ====================
with tab1:
    st.subheader("Sentiment Distribution & Feedback Explorer")
    
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.markdown("**Sentiment Proportion**")
        sentiment_counts = df['sentiment'].value_counts()
        
        if len(sentiment_counts) > 0:
            fig, ax = plt.subplots(figsize=(6, 5))
            labels = sentiment_counts.index
            
            # Map colors correctly to labels
            color_map = {'Positive': '#10B981', 'Neutral': '#9CA3AF', 'Negative': '#EF4444'}
            plot_colors = [color_map.get(lbl, '#6B7280') for lbl in labels]
            
            ax.pie(
                sentiment_counts, 
                labels=labels, 
                autopct='%1.1f%%', 
                colors=plot_colors, 
                startangle=90,
                wedgeprops={'edgecolor': 'none', 'linewidth': 1, 'antialiased': True},
                textprops={'fontsize': 11, 'fontweight': 'bold'}
            )
            # Draw transparent center circle for a donut chart style
            centre_circle = plt.Circle((0,0),0.70,fc='white' if st.get_option("theme.base") == "light" else '#0e1117')
            fig.gca().add_artist(centre_circle)
            
            ax.axis('equal')  
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No data available for sentiment proportions.")
        
    with col2:
        st.markdown("**Filter and Explore Reviews**")
        selected_sentiment = st.selectbox(
            "Filter reviews by sentiment:", 
            options=["All", "Positive", "Neutral", "Negative"]
        )
        
        # Filter dataframe
        if selected_sentiment == "All":
            filtered_df = df
        else:
            filtered_df = df[df['sentiment'] == selected_sentiment]
            
        # Display styled dataframe
        disp_cols = [c for c in ['review_text', 'sentiment', 'department'] if c in filtered_df.columns]
        st.dataframe(
            filtered_df[disp_cols],
            column_config={
                "review_text": st.column_config.TextColumn("Customer Review", width="large"),
                "sentiment": st.column_config.TextColumn("Sentiment"),
                "department": st.column_config.TextColumn("Category")
            },
            hide_index=True,
            use_container_width=True
        )

# ==================== TAB 2: TOPIC CLUSTERING ====================
with tab2:
    st.subheader("Unsupervised Topic Clustering (K-Means)")
    st.markdown("""
    Using Scikit-Learn's **K-Means**, reviews are vectorized using TF-IDF and clustered into topic groups. 
    **Principal Component Analysis (PCA)** is then used to reduce the high-dimensional vectors to 2D coordinates ($X, Y$) for plotting.
    """)
    
    col_plot, col_keywords = st.columns([1.2, 1])
    
    with col_plot:
        st.markdown("**PCA Projection of Topic Clusters**")
        
        # Color by options
        color_by = st.radio("Color plot by:", options=["Topic Cluster", "Sentiment"], horizontal=True)
        
        fig, ax = plt.subplots(figsize=(8, 6.5))
        
        # Set plot styles matching base theme
        is_dark = st.get_option("theme.base") == "dark"
        bg_color = '#0e1117' if is_dark else 'white'
        text_color = 'white' if is_dark else 'black'
        
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)
        
        if color_by == "Topic Cluster":
            scatter = sns.scatterplot(
                data=df, 
                x='pca_x', 
                y='pca_y', 
                hue='cluster_label', 
                palette='Set2', 
                s=100, 
                alpha=0.85, 
                ax=ax,
                edgecolor='w' if not is_dark else 'black'
            )
            ax.legend(title="Topic Cluster", facecolor=bg_color, labelcolor=text_color)
        else:
            scatter = sns.scatterplot(
                data=df, 
                x='pca_x', 
                y='pca_y', 
                hue='sentiment', 
                palette={'Positive': '#10B981', 'Neutral': '#9CA3AF', 'Negative': '#EF4444'}, 
                s=100, 
                alpha=0.85, 
                ax=ax,
                edgecolor='w' if not is_dark else 'black'
            )
            ax.legend(title="Sentiment", facecolor=bg_color, labelcolor=text_color)
            
        ax.set_title("2D Review Semantic Space", color=text_color, fontsize=14, fontweight='bold')
        ax.set_xlabel("PCA Component 1", color=text_color)
        ax.set_ylabel("PCA Component 2", color=text_color)
        ax.tick_params(colors=text_color)
        ax.grid(True, linestyle='--', alpha=0.2)
        
        # Remove spines
        for spine in ax.spines.values():
            spine.set_visible(False)
            
        st.pyplot(fig)
        plt.close(fig)
        
    with col_keywords:
        st.markdown("**Auto-Discovered Topic Definitions & Top Keywords**")
        keywords_dict = clustering_pipeline.get_cluster_keywords(top_n=5)
        
        for cluster_id, keywords in keywords_dict.items():
            # Get reviews belonging to this cluster
            cluster_reviews = df[df['cluster_label'] == cluster_id]['review_text'].tolist()
            sample_review = cluster_reviews[0] if cluster_reviews else "No reviews in cluster."
            
            # Format keywords as a nice string
            keyword_tags = " • ".join([f"`{kw}`" for kw in keywords])
            
            st.markdown(f"""
            <div style='background-color: rgba(128, 128, 128, 0.05); padding: 15px; border-radius: 8px; border: 1px solid rgba(128, 128, 128, 0.1); margin-bottom: 12px;'>
                <strong style='font-size: 15px; color:#4F46E5;'>Topic Group #{cluster_id}</strong><br/>
                <span style='font-size: 13px; color:#888;'>Keywords: </span>{keyword_tags}<br/>
                <div style='margin-top: 8px; font-style: italic; font-size: 13px; color: {"#555555" if st.get_option("theme.base") == "light" else "#aaaaaa"};'>
                    "Sample: {sample_review}"
                </div>
            </div>
            """, unsafe_allow_html=True)
            
    # Silhouette Score optimization plot
    st.markdown("---")
    st.subheader("🔍 Optimize Cluster Count (K)")
    st.markdown("""
    Find the mathematically optimal number of topics for your dataset using the **Silhouette Score**. 
    A higher Silhouette Score indicates that clusters are denser and better separated.
    """)
    
    if st.button("📊 Run Silhouette Score Analysis", type="secondary"):
        with st.spinner("Calculating Silhouette Scores for K = 2 to 8 (this may take a few seconds)..."):
            try:
                scores = run_silhouette_analysis(df['review_text'], max_k=8)
                
                # Plot setup
                fig, ax = plt.subplots(figsize=(8, 3.5))
                is_dark = st.get_option("theme.base") == "dark"
                bg_color = '#0e1117' if is_dark else 'white'
                text_color = 'white' if is_dark else 'black'
                fig.patch.set_facecolor(bg_color)
                ax.set_facecolor(bg_color)
                
                k_values = list(scores.keys())
                k_scores = list(scores.values())
                
                ax.plot(k_values, k_scores, marker='o', color='#4F46E5', linewidth=2.5, markersize=8)
                ax.set_title("Silhouette Score vs Number of Topic Clusters", color=text_color, fontsize=12, fontweight='bold')
                ax.set_xlabel("Number of Clusters (K)", color=text_color)
                ax.set_ylabel("Silhouette Score", color=text_color)
                ax.tick_params(colors=text_color)
                ax.grid(True, linestyle='--', alpha=0.1)
                
                # Highlight best K
                best_k = max(scores, key=scores.get)
                ax.axvline(best_k, color='#10B981', linestyle=':', linewidth=2, label=f'Optimal K = {best_k}')
                ax.legend(facecolor=bg_color, labelcolor=text_color)
                
                for spine in ax.spines.values():
                    spine.set_visible(False)
                    
                col_graph, col_info = st.columns([1.5, 1])
                with col_graph:
                    st.pyplot(fig)
                    plt.close(fig)
                with col_info:
                    st.success(f"**Optimal K Found: {best_k}**")
                    st.markdown(f"""
                    The highest Silhouette Score occurs at **K = {best_k}** (Score: `{scores[best_k]:.4f}`). 
                    This suggests that grouping the reviews into **{best_k} topics** will yield the most mathematically cohesive categories.
                    """)
            except Exception as e:
                st.error(f"Error calculating silhouette scores: {e}")

# ==================== TAB 3: MODEL PERFORMANCE & DIAGNOSTICS ====================
with tab3:
    st.subheader("📊 Model Performance & Diagnostics")
    st.markdown("""
    Evaluate the validation metrics of the classifier on unseen data. The model splits the dataset into 
    **80% Training data** and **20% Test data (held-out)** to generate these metrics. 
    This ensures that the accuracy reported is realistic and reflects performance on future feedback.
    """)
    
    metric_cols = st.columns(3)
    with metric_cols[0]:
        st.metric(
            label="Out-of-Sample Test Accuracy", 
            value=f"{test_accuracy:.2%}",
            help="The accuracy of the model on the 20% held-out test set."
        )
    with metric_cols[1]:
        st.metric(
            label="Classifier Model", 
            value=selected_model_type
        )
    with metric_cols[2]:
        st.metric(
            label="Split Size (Train / Test)", 
            value=f"{int(len(train_df)*0.8)} / {int(len(train_df)*0.2)}"
        )
        
    diag_col1, diag_col2 = st.columns([1, 1.2])
    
    with diag_col1:
        st.markdown("**Confusion Matrix Heatmap**")
        if sentiment_model.conf_matrix is not None:
            fig, ax = plt.subplots(figsize=(6, 5))
            is_dark = st.get_option("theme.base") == "dark"
            bg_color = '#0e1117' if is_dark else 'white'
            text_color = 'white' if is_dark else 'black'
            fig.patch.set_facecolor(bg_color)
            ax.set_facecolor(bg_color)
            
            sns.heatmap(
                sentiment_model.conf_matrix,
                annot=True,
                fmt='d',
                cmap='Purples' if is_dark else 'Blues',
                xticklabels=sentiment_model.classes,
                yticklabels=sentiment_model.classes,
                ax=ax,
                cbar=False,
                annot_kws={"size": 12, "weight": "bold"}
            )
            ax.set_title("Confusion Matrix (Holdout Set)", color=text_color, fontsize=12, fontweight='bold')
            ax.set_xlabel("Predicted Sentiment", color=text_color)
            ax.set_ylabel("True Sentiment", color=text_color)
            ax.tick_params(colors=text_color)
            
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No confusion matrix available.")
            
    with diag_col2:
        st.markdown("**Detailed Classification Report**")
        if sentiment_model.class_report is not None:
            # Format report dict
            report_df = pd.DataFrame(sentiment_model.class_report).T
            # Drop aggregate rows
            classes_only_df = report_df.drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore')
            classes_only_df = classes_only_df[['precision', 'recall', 'f1-score', 'support']]
            classes_only_df['support'] = classes_only_df['support'].astype(int)
            
            st.dataframe(
                classes_only_df.style.format({
                    'precision': '{:.2%}',
                    'recall': '{:.2%}',
                    'f1-score': '{:.2%}',
                    'support': '{:,}'
                }),
                use_container_width=True
            )
            
            st.markdown("""
            **Understand the Metrics:**
            *   **Precision (Positive Predictive Value)**: Of all reviews labeled by the model as *Positive/Negative*, how many were correct? High precision means low false positives.
            *   **Recall (Sensitivity)**: Of all actual *Positive/Negative* reviews in the test set, how many did the model find? High recall means low false negatives.
            *   **F1-Score**: The harmonic mean of precision and recall, representing a balanced metric.
            """)
        else:
            st.info("No classification report available.")

# ==================== TAB 4: LIVE FEEDBACK CLASSIFIER ====================
with tab4:
    st.subheader("Test the NLP Pipeline in Real-Time")
    st.markdown("""
    Type in any new customer review below. Click **Run Analytics Pipeline** to process the review:
    1. **Preprocess and Clean**: The pipeline resolves contractions (e.g. *"didn't"* $\rightarrow$ *"did not"*), preserves crucial negation keywords (like *"not"*, *"no"*, *"never"*), removes non-negation stopwords, and applies lemmatization.
    2. **Classify Sentiment**: Predictions and confidence probabilities are computed.
    3. **Map Topic Cluster**: Vectorizes the review and projects it into the 2D topic PCA space.
    """)
    
    # Session state for holding test texts
    if 'test_phrase' not in st.session_state:
        st.session_state.test_phrase = "The delivery was incredibly quick but the item had a scratch on the screen."
        
    input_text = st.text_area(
        "Enter customer review / tweet:", 
        value=st.session_state.test_phrase,
        placeholder="Type here...",
        key="test_area_input"
    )
    
    st.markdown("**Test common negation validation cases:**")
    neg_cols = st.columns(2)
    with neg_cols[0]:
        if st.button("Case 1: 'The customer service was not good.'"):
            st.session_state.test_phrase = "The customer service was not good."
            st.rerun()
    with neg_cols[1]:
        if st.button("Case 2: 'I can't say this is a bad device.'"):
            st.session_state.test_phrase = "I can't say this is a bad device."
            st.rerun()
            
    if st.button("Run Analytics Pipeline", type="primary"):
        cleaned_input = st.session_state.test_area_input
        if cleaned_input.strip() == "":
            st.warning("Please enter some text to analyze.")
        else:
            # 1. Cleaning
            cleaned_text = clean_text(cleaned_input)
            
            # 2. Predict Sentiment & Probs
            sentiment_pred = sentiment_model.predict(cleaned_input)
            sentiment_probs = sentiment_model.predict_probs(cleaned_input)
            
            # 3. Predict Cluster & PCA
            cluster_pred, coords = clustering_pipeline.assign_new_text(cleaned_input)
            keywords_dict = clustering_pipeline.get_cluster_keywords(top_n=5)
            assigned_keywords = ", ".join(keywords_dict[cluster_pred])
            
            # Setup columns for results
            res_col1, res_col2 = st.columns([1, 1.2])
            
            with res_col1:
                st.markdown("### Analysis Results")
                
                # Sentiment Badge
                badge_class = "badge-positive" if sentiment_pred == "Positive" else ("badge-neutral" if sentiment_pred == "Neutral" else "badge-negative")
                st.markdown(f"**Predicted Sentiment:** <span class='badge {badge_class}'>{sentiment_pred}</span>", unsafe_allow_html=True)
                st.markdown("")
                
                # Probs progress bars
                st.markdown("**Model Confidence Scores:**")
                prob_df = pd.DataFrame({
                    'Sentiment': list(sentiment_probs.keys()),
                    'Probability': list(sentiment_probs.values())
                })
                
                for index, row in prob_df.iterrows():
                    st.write(f"{row['Sentiment']}: {int(row['Probability'] * 100)}%")
                    st.progress(row['Probability'])
                
                st.markdown("---")
                st.markdown(f"**Assigned Topic Group:** `#{cluster_pred}`")
                st.markdown(f"**Topic Keywords:** `{assigned_keywords}`")
                
                # Preprocessing validation block
                st.markdown("---")
                st.markdown("**NLP Preprocessing Validation:**")
                st.markdown(f"**Original Text:** *\"{cleaned_input}\"*")
                st.markdown(f"**Cleaned Tokens:** `{cleaned_text}`")
                
            with res_col2:
                st.markdown("### Mapping in Semantic Space")
                st.markdown("The yellow star $\\bigstar$ marks the position of your new review relative to the existing cluster data.")
                
                fig, ax = plt.subplots(figsize=(7, 5.5))
                is_dark = st.get_option("theme.base") == "dark"
                bg_color = '#0e1117' if is_dark else 'white'
                text_color = 'white' if is_dark else 'black'
                
                fig.patch.set_facecolor(bg_color)
                ax.set_facecolor(bg_color)
                
                # Draw existing points
                sns.scatterplot(
                    data=df, 
                    x='pca_x', 
                    y='pca_y', 
                    hue='cluster_label', 
                    palette='Set2', 
                    s=80, 
                    alpha=0.4, 
                    ax=ax,
                    legend=False,
                    edgecolor='none'
                )
                
                # Highlight new point
                ax.scatter(
                    coords[0], 
                    coords[1], 
                    color='#EF4444' if not is_dark else '#F59E0B', 
                    marker='*', 
                    s=300, 
                    edgecolor='black', 
                    linewidth=1.5, 
                    label="New Feedback"
                )
                
                ax.legend(facecolor=bg_color, labelcolor=text_color)
                ax.set_title("New Entry Position in 2D Space", color=text_color, fontsize=12, fontweight='bold')
                ax.set_xlabel("PCA Component 1", color=text_color)
                ax.set_ylabel("PCA Component 2", color=text_color)
                ax.tick_params(colors=text_color)
                ax.grid(True, linestyle='--', alpha=0.1)
                
                for spine in ax.spines.values():
                    spine.set_visible(False)
                    
                st.pyplot(fig)
                plt.close(fig)
