"""
app.py  –  Streamlit UI for the NLP Search Engine
===================================================
Run from the project root:
    streamlit run app.py
 
Input  : يكتب المستخدم query بالإنجليزي
Output : Top-5 نتايج من كل موديل (TF-IDF + Embedding) مع السكور
 
Data   : يتحمل تلقائياً من Kaggle أول ما يشتغل البرنامج
         (محتاج Kaggle username + API key في أول تشغيل)
"""
import os
import sys
import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from kaggle.api.kaggle_api_extended import KaggleApi

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="NLP Search Engine",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Intelligent Search Engine")
st.caption("Project 2 – NLP Course | Faculty of Computing & AI")

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
REVIEWS_PATH = "data/Reviews.csv"
KAGGLE_DATASET = "snap/amazon-fine-food-reviews"

# ─────────────────────────────────────────────
# Kaggle setup from Streamlit secrets
# ─────────────────────────────────────────────
def setup_kaggle():
    """Setup Kaggle API credentials from Streamlit secrets"""
    try:
        kaggle_user = st.secrets["KAGGLE_USERNAME"]
        kaggle_key = st.secrets["KAGGLE_KEY"]
    except Exception:
        st.error("❌ لازم تضيف Kaggle credentials في Streamlit secrets")
        st.stop()

    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)

    with open(os.path.join(kaggle_dir, "kaggle.json"), "w") as f:
        json.dump({"username": kaggle_user, "key": kaggle_key}, f)

    os.chmod(os.path.join(kaggle_dir, "kaggle.json"), 0o600)


def download_dataset():
    """Download dataset using Kaggle API"""
    st.info("📥 Downloading dataset from Kaggle...")

    api = KaggleApi()
    api.authenticate()

    os.makedirs("data", exist_ok=True)

    api.dataset_download_files(
        KAGGLE_DATASET,
        path="data",
        unzip=True
    )

    st.success("✅ Dataset downloaded successfully!")


def ensure_data():
    """Ensure dataset exists"""
    if os.path.exists(REVIEWS_PATH):
        return

    st.warning("⚠️ Dataset not found — downloading automatically...")

    try:
        setup_kaggle()
        download_dataset()
    except Exception as e:
        st.error(f"❌ Failed to download dataset: {e}")
        st.stop()


# ─────────────────────────────────────────────
# Ensure data exists
# ─────────────────────────────────────────────
ensure_data()


# ─────────────────────────────────────────────
# Load models
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Loading models...")
def load_models():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from src.data_loader import load_data, get_documents
    from src.preprocessing import preprocess_documents
    from src.tfidf_search import build_tfidf
    from src.embedding_search import build_embeddings

    df = load_data(REVIEWS_PATH, n_samples=3000)
    documents = get_documents(df)
    cleaned_docs = preprocess_documents(documents)

    vectorizer, tfidf_matrix = build_tfidf(cleaned_docs)
    emb_model, embeddings = build_embeddings(documents)

    return documents, cleaned_docs, vectorizer, tfidf_matrix, emb_model, embeddings


# ─────────────────────────────────────────────
# Load models safely
# ─────────────────────────────────────────────
try:
    documents, cleaned_docs, vectorizer, tfidf_matrix, emb_model, embeddings = load_models()
    models_ready = True
except Exception as e:
    st.error(f"❌ Error loading models: {e}")
    models_ready = False


# ─────────────────────────────────────────────
# Evaluation Helper Functions
# ─────────────────────────────────────────────

QUERY_KEYWORDS = {
    "great coffee and pastries": ["coffee", "pastri", "cafe", "espresso", "latte", "muffin", "donut", "biscuit", "croissant", "baked"],
    "bad service and cold food": ["bad", "cold", "terrible", "awful", "horrible", "disappoint", "poor", "worst", "rude", "slow"],
    "healthy snacks for kids": ["healthy", "kid", "child", "snack", "organic", "natural", "wholesome", "nutritious", "fruit", "veggie"],
}


def _is_relevant(document: str, keywords: list) -> int:
    """Return 1 if the document contains at least one keyword, else 0."""
    doc_lower = document.lower()
    return int(any(kw in doc_lower for kw in keywords))


def _build_relevant_flags(query: str, results: list) -> list:
    """Build a list of 0/1 relevance flags for a list of result dicts."""
    if query in QUERY_KEYWORDS:
        keywords = QUERY_KEYWORDS[query]
    else:
        keywords = [w.lower() for w in query.split() if len(w) > 3]
    
    return [_is_relevant(r["document"], keywords) for r in results]


def precision_at_k(relevant_flags: list, k: int) -> float:
    """Compute Precision@k."""
    if k <= 0:
        return 0.0
    top_k_flags = relevant_flags[:k]
    relevant_count = sum(top_k_flags)
    return round(relevant_count / k, 4)


def evaluate_single_query(query: str, tfidf_results: list, emb_results: list, k: int = 5):
    """Evaluate Precision@k for a single query with both models."""
    tfidf_flags = _build_relevant_flags(query, tfidf_results)
    emb_flags = _build_relevant_flags(query, emb_results)
    
    p_tfidf = precision_at_k(tfidf_flags, k)
    p_emb = precision_at_k(emb_flags, k)
    
    return p_tfidf, p_emb, tfidf_flags[:k], emb_flags[:k]


def create_comparison_table(query: str, tfidf_results: list, emb_results: list, k: int = 5):
    """Create a comparison DataFrame for the UI."""
    p_tfidf, p_emb, tfidf_flags, emb_flags = evaluate_single_query(query, tfidf_results, emb_results, k)
    
    # Determine winner
    if p_tfidf > p_emb:
        winner = "TF-IDF"
    elif p_emb > p_tfidf:
        winner = "Embedding"
    else:
        winner = "Tie"
    
    # Create summary DataFrame
    summary_data = {
        "Metric": ["Precision@" + str(k), "Relevant Docs", "Winner"],
        "TF-IDF": [f"{p_tfidf:.4f}", f"{sum(tfidf_flags)}/{k}", ""],
        "Embedding": [f"{p_emb:.4f}", f"{sum(emb_flags)}/{k}", ""],
        "Winner": ["", "", winner]
    }
    
    summary_df = pd.DataFrame(summary_data)
    
    # Create detailed comparison DataFrame
    detailed_data = []
    for i in range(k):
        detailed_data.append({
            "Rank": i + 1,
            "TF-IDF Score": tfidf_results[i]["score"],
            "TF-IDF Relevant": "✅" if tfidf_flags[i] == 1 else "❌",
            "Embedding Score": emb_results[i]["score"],
            "Embedding Relevant": "✅" if emb_flags[i] == 1 else "❌",
            "TF-IDF Document": tfidf_results[i]["document"][:100] + "...",
            "Embedding Document": emb_results[i]["document"][:100] + "..."
        })
    
    detailed_df = pd.DataFrame(detailed_data)
    
    return summary_df, detailed_df, p_tfidf, p_emb, tfidf_flags, emb_flags


def plot_comparison_chart(query: str, p_tfidf: float, p_emb: float, k: int = 5):
    """Create a comparison bar chart for the UI."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    models = ["TF-IDF", "Embedding"]
    scores = [p_tfidf, p_emb]
    colors = ["#4C72B0", "#DD8452"]
    
    bars = ax.bar(models, scores, color=colors, edgecolor="white", linewidth=1.5, width=0.5)
    
    ax.set_ylabel(f"Precision@{k}", fontsize=12)
    ax.set_title(f"Model Comparison for Query: '{query[:50]}...'", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    
    # Add value labels on bars
    for bar, val in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f"{val:.4f}",
                ha="center", va="bottom", fontsize=12, fontweight="bold")
    
    # Highlight winner
    winner_idx = 0 if p_tfidf >= p_emb else 1
    bars[winner_idx].set_edgecolor("gold")
    bars[winner_idx].set_linewidth(3)
    
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    top_k = st.slider("Top-K results", 1, 10, 5)

    model_choice = st.radio(
        "Search Model",
        ["TF-IDF", "Embedding", "Model Comparison", "All"],
        index=2
    )

    st.markdown("---")
    st.markdown("### Example Queries")

    examples = [
        "great coffee and pastries",
        "bad service and cold food",
        "healthy snacks for kids",
        "delivery issue",
        "sweet cake"
    ]

    for q in examples:
        if st.button(q):
            st.session_state["query"] = q


# ─────────────────────────────────────────────
# Input
# ─────────────────────────────────────────────
query = st.text_input(
    "Enter your query:",
    value=st.session_state.get("query", "")
)

search_btn = st.button("🔎 Search", disabled=not models_ready)


# ─────────────────────────────────────────────
# Search logic
# ─────────────────────────────────────────────
if search_btn and query.strip():

    from src.tfidf_search import search_tfidf
    from src.embedding_search import search_embeddings

    st.subheader(f"Results for: '{query}'")

    def to_df(results, model_name=""):
        return pd.DataFrame([
            {
                "Rank": r["rank"],
                "Score": f"{r['score']:.4f}",
                "Text": r["document"][:120] + "..."
            }
            for r in results
        ])

    # Get search results for both models
    tfidf_results = search_tfidf(query, vectorizer, tfidf_matrix, documents, top_k)
    emb_results = search_embeddings(query, emb_model, embeddings, documents, top_k)

    if model_choice in ["TF-IDF", "All"]:
        st.markdown("### TF-IDF Results")
        st.dataframe(to_df(tfidf_results), use_container_width=True)

    if model_choice in ["Embedding", "All"]:
        st.markdown("### Embedding Results")
        st.dataframe(to_df(emb_results), use_container_width=True)

    if model_choice in ["Model Comparison", "All"]:
        st.markdown("### 📊 Model Comparison")
        
        # Create comparison data
        summary_df, detailed_df, p_tfidf, p_emb, tfidf_flags, emb_flags = create_comparison_table(
            query, tfidf_results, emb_results, top_k
        )
        
        # Display metrics in columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("TF-IDF Precision", f"{p_tfidf:.4f}")
        with col2:
            st.metric("Embedding Precision", f"{p_emb:.4f}")
        with col3:
            if p_tfidf > p_emb:
                winner = "🏆 TF-IDF"
            elif p_emb > p_tfidf:
                winner = "🏆 Embedding"
            else:
                winner = "🤝 Tie"
            st.metric("Winner", winner)
        
        # Display comparison chart
        st.markdown("#### Precision Comparison")
        fig = plot_comparison_chart(query, p_tfidf, p_emb, top_k)
        st.pyplot(fig)
        plt.close(fig)
        
        # Display summary table
        st.markdown("#### Summary")
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        # Display detailed comparison
        st.markdown("#### Detailed Results Comparison")
        st.dataframe(detailed_df, use_container_width=True, hide_index=True)
        
        # Display relevance statistics
        st.markdown("#### Relevance Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**TF-IDF**")
            st.write(f"- Relevant documents in top-{top_k}: {sum(tfidf_flags)}/{top_k}")
            st.write(f"- Precision@{top_k}: {p_tfidf:.4f}")
        with col2:
            st.markdown("**Embedding**")
            st.write(f"- Relevant documents in top-{top_k}: {sum(emb_flags)}/{top_k}")
            st.write(f"- Precision@{top_k}: {p_emb:.4f}")

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.caption("Team: Sama & Team | NLP Project 2")