"""
w2v_cosine_search.py
Word2Vec embeddings + Cosine Similarity  →  Baseline search on Word Embeddings.

IMPROVEMENTS (v2)
-----------------
- vector_size  : 100 → 200  (richer semantic representations)
- window       : 5   → 3    (tighter context = more precise word associations)
- epochs       : 10  → 30   (longer training = better convergence)
- min_count    : 2   → 1    (keep more vocabulary, important for rare food terms)
- sg           : 0 (CBOW) → 1 (Skip-gram, better for rare/specific words)
"""

import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from src.preprocessing import preprocess_text


# ─────────────────────────────────────────────────────────────────────────────
# HELPER  –  average word vectors for a text string
# ─────────────────────────────────────────────────────────────────────────────

def _average_vector(text: str, model: Word2Vec, vector_size: int) -> np.ndarray:
    """
    Return the mean Word2Vec vector for all *known* words in `text`.
    If no word is found in the vocabulary, return a zero vector.
    """
    tokens = text.split() if isinstance(text, str) else text
    known = [t for t in tokens if t in model.wv]

    if not known:
        return np.zeros(vector_size)

    vectors = np.stack([model.wv[t] for t in known])
    return vectors.mean(axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# 1-A.  BUILD  (original baseline)
# ─────────────────────────────────────────────────────────────────────────────

def build_w2v(cleaned_docs: list,
              vector_size: int = 100,
              window: int = 5,
              min_count: int = 2,
              epochs: int = 10):
    """
    Train a Word2Vec model — original baseline configuration.

    Parameters
    ----------
    cleaned_docs : list of str  – preprocessed documents (space-separated tokens)
    vector_size  : int  – embedding dimensionality (default 100)
    window       : int  – context window size (default 5)
    min_count    : int  – minimum word frequency (default 2)
    epochs       : int  – training epochs (default 10)

    Returns
    -------
    w2v_model   : trained gensim Word2Vec model
    doc_vectors : np.ndarray of shape (n_docs, vector_size)
    """
    tokenised = [doc.split() for doc in cleaned_docs]

    print("[Word2Vec] Training model...")
    w2v_model = Word2Vec(
        sentences=tokenised,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        epochs=epochs,
        sg=0,  # CBOW
    )
    vocab_size = len(w2v_model.wv)
    print(f"[Word2Vec] Vocabulary size : {vocab_size:,}")
    print(f"[Word2Vec] Vector size     : {vector_size}")

    doc_vectors = np.stack([
        _average_vector(doc, w2v_model, vector_size)
        for doc in cleaned_docs
    ])
    print(f"[Word2Vec] Document matrix : {doc_vectors.shape}")

    return w2v_model, doc_vectors


# ─────────────────────────────────────────────────────────────────────────────
# 1-B.  BUILD  (improved v2)
# ─────────────────────────────────────────────────────────────────────────────

def build_w2v_improved(cleaned_docs: list):
    """
    Train an improved Word2Vec model.

    Improvements over baseline:
    - vector_size=200  : richer 200-dim representations vs 100
    - window=3         : tighter context window → sharper word associations
    - epochs=30        : 3× more training for better convergence
    - min_count=1      : keeps domain-specific rare food words
    - sg=1             : Skip-gram (better than CBOW for specific/rare words)

    Parameters
    ----------
    cleaned_docs : list of str – preprocessed documents

    Returns
    -------
    w2v_model   : trained gensim Word2Vec model (improved)
    doc_vectors : np.ndarray of shape (n_docs, 200)
    """
    vector_size = 200
    tokenised = [doc.split() for doc in cleaned_docs]

    print("[Word2Vec Improved] Training model (Skip-gram, 200d, 30 epochs)...")
    w2v_model = Word2Vec(
        sentences=tokenised,
        vector_size=vector_size,
        window=3,
        min_count=1,
        workers=4,
        epochs=30,
        sg=1,  # Skip-gram
    )
    vocab_size = len(w2v_model.wv)
    print(f"[Word2Vec Improved] Vocabulary size : {vocab_size:,}")
    print(f"[Word2Vec Improved] Vector size     : {vector_size}")

    doc_vectors = np.stack([
        _average_vector(doc, w2v_model, vector_size)
        for doc in cleaned_docs
    ])
    print(f"[Word2Vec Improved] Document matrix : {doc_vectors.shape}")

    return w2v_model, doc_vectors


# ─────────────────────────────────────────────────────────────────────────────
# 2.  SEARCH  (shared by both original and improved)
# ─────────────────────────────────────────────────────────────────────────────

def search_w2v(query: str,
               w2v_model: Word2Vec,
               doc_vectors: np.ndarray,
               documents: list,
               top_k: int = 5) -> list:
    """
    Return the top-k most relevant documents using Word2Vec + cosine similarity.
    Works with both original and improved models.
    """
    vector_size = w2v_model.vector_size

    cleaned_query = preprocess_text(query)
    if not cleaned_query:
        cleaned_query = query.lower()


    query_vector = _average_vector(cleaned_query, w2v_model, vector_size)

    if not np.any(query_vector):
        print("[Word2Vec] Warning: no query words found in vocabulary.")
        return []

    scores = cosine_similarity(
        query_vector.reshape(1, -1),
        doc_vectors
    ).flatten()

    ranked_indices = scores.argsort()[::-1][:top_k]

    results = []
    for rank, idx in enumerate(ranked_indices, start=1):
        results.append({
            "rank": rank,
            "score": round(float(scores[idx]), 4),
            "document": documents[idx],
        })

    return results


# Alias for backward compatibility
search_w2v_cosine = search_w2v