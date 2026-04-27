"""
w2v_bert_search.py
Word2Vec (feature extraction)  +  BERT Cross-Encoder re-ranking  (advanced model).

Feature Extraction : Word2Vec average vectors
Model              : BERT Cross-Encoder  (sentence-transformers)

How it works  –  2-stage pipeline
----------------------------------
Stage 1  →  Word2Vec + Cosine Similarity  (fast retrieval)
Stage 2  →  BERT Cross-Encoder  (precise re-ranking)
"""

import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import CrossEncoder
from src.preprocessing import preprocess_text
from src.w2v_cosine_search import build_w2v


# ─────────────────────────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _average_vector(text: str, w2v_model, vector_size: int) -> np.ndarray:
    """Return the mean Word2Vec vector for all known words in text."""
    tokens = text.split() if isinstance(text, str) else text
    known = [t for t in tokens if t in w2v_model.wv]
    if not known:
        return np.zeros(vector_size)
    vectors = np.stack([w2v_model.wv[t] for t in known])
    return vectors.mean(axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  BUILD
# ─────────────────────────────────────────────────────────────────────────────

def build_w2v_bert(cleaned_docs: list, documents: list = None):
    """
    Train a Word2Vec model, encode documents, and load the BERT Cross-Encoder.

    Parameters
    ----------
    cleaned_docs : list of str – preprocessed documents (from preprocess_documents)
    documents    : list of str – original documents (unused, kept for API consistency)

    Returns
    -------
    tuple: (cross_encoder, w2v_model, doc_vectors)
        cross_encoder : CrossEncoder model
        w2v_model     : trained gensim Word2Vec model
        doc_vectors   : np.ndarray of shape (n_docs, vector_size)
    """
    # Train Word2Vec and encode documents
    w2v_model, doc_vectors = build_w2v(cleaned_docs)

    # Load Cross-Encoder
    print("[W2V+BERT] Loading Cross-Encoder: cross-encoder/ms-marco-MiniLM-L-6-v2")
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    print("[W2V+BERT] Model loaded.")

    return cross_encoder, w2v_model, doc_vectors


# ─────────────────────────────────────────────────────────────────────────────
# 2.  SEARCH
# ─────────────────────────────────────────────────────────────────────────────

def search_w2v_bert(query: str,
                    w2v_bert_model: tuple,
                    documents: list,
                    top_k: int = 5,
                    candidate_k: int = 50) -> list:
    """
    2-stage search: Word2Vec retrieval → BERT Cross-Encoder re-ranking.

    Parameters
    ----------
    query          : str   – raw user query
    w2v_bert_model : tuple – (cross_encoder, w2v_model, doc_vectors)
                             returned by build_w2v_bert()
    documents      : list of str – original (un-cleaned) documents
    top_k          : int   – final results to return (default 5)
    candidate_k    : int   – how many W2V candidates to re-rank (default 50)

    Returns
    -------
    list of dict  – each dict has keys: 'rank', 'score', 'document'
                    'score' is the BERT Cross-Encoder relevance score.
    """
    cross_encoder, w2v_model, doc_vectors = w2v_bert_model
    vector_size = w2v_model.vector_size

    # ── Stage 1: Word2Vec fast retrieval ─────────────────────────────────────
    cleaned_tokens = preprocess_text(query)
    if not cleaned_tokens:
        cleaned_query = query.lower()
    else:
        cleaned_query = " ".join(cleaned_tokens)

    query_vector = _average_vector(cleaned_query, w2v_model, vector_size)

    if not np.any(query_vector):
        print("[W2V+BERT] Warning: no query words in W2V vocabulary. "
              "Returning empty results.")
        return []

    w2v_scores = cosine_similarity(
        query_vector.reshape(1, -1),
        doc_vectors
    ).flatten()

    n_candidates = min(candidate_k, len(documents))
    candidate_indices = w2v_scores.argsort()[::-1][:n_candidates]
    candidate_docs = [documents[i] for i in candidate_indices]

    # ── Stage 2: BERT Cross-Encoder re-ranking ───────────────────────────────
    pairs = [(query, doc) for doc in candidate_docs]
    bert_scores = cross_encoder.predict(pairs)

    sorted_order = np.argsort(bert_scores)[::-1][:top_k]

    results = []
    for rank, order_idx in enumerate(sorted_order, start=1):
        results.append({
            "rank": rank,
            "score": round(float(bert_scores[order_idx]), 4),
            "document": candidate_docs[order_idx],
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.preprocessing import preprocess_documents

    sample_docs = [
        "The food delivery was very late and cold",
        "I love this product, great quality",
        "Best coffee I ever tasted",
        "Package was delayed for two weeks",
        "Amazing taste and fast shipping",
    ]

    cleaned = preprocess_documents(sample_docs)
    w2v_bert_model = build_w2v_bert(cleaned, sample_docs)

    query = "food delivery problem"
    results = search_w2v_bert(query, w2v_bert_model, sample_docs, top_k=3)

    print(f"\nQuery: '{query}'\n")
    for r in results:
        print(f"  #{r['rank']} | Score: {r['score']:.4f} | {r['document']}")
