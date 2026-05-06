"""
tfidf_search.py
TF-IDF + Cosine Similarity  →  Baseline search model.

IMPROVEMENTS (v2)
-----------------
- ngram_range=(1,2)   : captures bigrams like "great coffee", "bad service"
- sublinear_tf=True   : log-normalises term frequency (reduces dominance of common words)
- max_df=0.85         : ignores terms in >85% of docs (too common to be useful)
- min_df=2            : ignores very rare terms (appear in only 1 doc)
- max_features=15000  : keeps vocabulary focused on most useful terms
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.preprocessing import preprocess_text


# ─────────────────────────────────────────────────────────────────────────────
# ORIGINAL (Baseline)
# ─────────────────────────────────────────────────────────────────────────────

def build_tfidf(cleaned_docs: list):
    """
    Build and fit a TF-IDF matrix — original baseline configuration.

    Parameters
    ----------
    cleaned_docs : list of str – preprocessed documents

    Returns
    -------
    vectorizer   : fitted TfidfVectorizer
    tfidf_matrix : sparse matrix of shape (n_docs, n_features)
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(cleaned_docs)
    print(f"[TF-IDF] Matrix shape: {tfidf_matrix.shape}")
    return vectorizer, tfidf_matrix


# ─────────────────────────────────────────────────────────────────────────────
# IMPROVED (v2)
# ─────────────────────────────────────────────────────────────────────────────

def build_tfidf_improved(cleaned_docs: list):
    """
    Build and fit an improved TF-IDF matrix.

    Improvements over baseline:
    - ngram_range=(1,2)  : unigrams + bigrams for richer features
    - sublinear_tf=True  : log-normalised TF dampens high-frequency terms
    - max_df=0.85        : removes near-universal stopwords missed by preprocessing
    - min_df=2           : removes hapax legomena (noise terms)
    - max_features=15000 : focused vocabulary, faster inference

    Parameters
    ----------
    cleaned_docs : list of str – preprocessed documents

    Returns
    -------
    vectorizer   : fitted TfidfVectorizer (improved)
    tfidf_matrix : sparse matrix of shape (n_docs, n_features)
    """
    vectorizer = TfidfVectorizer(
        ngram_range=(1,2),
        sublinear_tf=True,
        max_df=0.90,
        min_df=2,
        max_features=15000,
    )
    tfidf_matrix = vectorizer.fit_transform(cleaned_docs)
    print(f"[TF-IDF Improved] Matrix shape: {tfidf_matrix.shape}")
    return vectorizer, tfidf_matrix


# ─────────────────────────────────────────────────────────────────────────────
# SEARCH  (shared by both original and improved)
# ─────────────────────────────────────────────────────────────────────────────

def search_tfidf(query: str,
                 vectorizer,
                 tfidf_matrix,
                 documents: list,
                 top_k: int = 5) -> list:
    """
    Return the top-k most relevant documents for a query using
    TF-IDF vectors + cosine similarity.

    Works with both the original and improved vectorizer/matrix.

    Parameters
    ----------
    query        : str   – raw user query
    vectorizer   : fitted TfidfVectorizer (original or improved)
    tfidf_matrix : sparse matrix from build_tfidf() or build_tfidf_improved()
    documents    : list of str – original (un-cleaned) documents
    top_k        : int   – how many results to return (default 5)

    Returns
    -------
    list of dict – each dict has keys: 'rank', 'score', 'document'
    """
    cleaned_query = preprocess_text(query)
    if not cleaned_query:
        cleaned_query_str = query.lower()
    else:
        cleaned_query_str = " ".join(cleaned_query)

    query_vector = vectorizer.transform([cleaned_query_str])
    scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

    ranked_indices = scores.argsort()[::-1][:top_k]

    results = []
    for rank, idx in enumerate(ranked_indices, start=1):
        results.append({
            "rank": rank,
            "score": round(float(scores[idx]), 4),
            "document": documents[idx],
        })

    return results
