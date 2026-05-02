"""
tfidf_bert_search.py
TF-IDF (feature extraction)  +  BERT Cross-Encoder re-ranking  (advanced model).

Feature Extraction : TF-IDF  (same vectorizer as the baseline)
Model              : BERT Cross-Encoder  (sentence-transformers)

How it works  –  2-stage pipeline
----------------------------------
Stage 1  →  TF-IDF + Cosine Similarity  (fast retrieval)
    Use the TF-IDF vectors to quickly retrieve the top-N candidate
    documents from the whole corpus.

Stage 2  →  BERT Cross-Encoder  (precise re-ranking)
    Feed every (query, candidate) pair to a BERT Cross-Encoder.
    Re-rank the candidates by the Cross-Encoder score and return top-k.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import CrossEncoder
from src.preprocessing import preprocess_text


# ─────────────────────────────────────────────────────────────────────────────
# 1.  BUILD
# ─────────────────────────────────────────────────────────────────────────────

def build_tfidf_bert(cleaned_docs: list, documents: list = None):
    """
    Build TF-IDF matrix and load the BERT Cross-Encoder model.

    Parameters
    ----------
    cleaned_docs : list of str – preprocessed documents (from preprocess_documents)
    documents    : list of str – original documents (unused, kept for API consistency)

    Returns
    -------
    tuple: (cross_encoder, vectorizer, tfidf_matrix)
        cross_encoder : CrossEncoder model
        vectorizer    : fitted TfidfVectorizer
        tfidf_matrix  : sparse TF-IDF matrix
    """
    # Build TF-IDF
    vectorizer = TfidfVectorizer(
        ngram_range=(1,2),
        sublinear_tf=True,
        max_df=0.85,
        min_df=1,
        max_features=15000,
    )
    tfidf_matrix = vectorizer.fit_transform(cleaned_docs)
    print(f"[TF-IDF+BERT] TF-IDF matrix shape: {tfidf_matrix.shape}")

    # Load Cross-Encoder
    print("[TF-IDF+BERT] Loading model: cross-encoder/ms-marco-MiniLM-L-6-v2")
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    print("[TF-IDF+BERT] Model loaded.")

    return cross_encoder, vectorizer, tfidf_matrix


# ─────────────────────────────────────────────────────────────────────────────
# 2.  SEARCH
# ─────────────────────────────────────────────────────────────────────────────

def search_tfidf_bert(query: str,
                      tfidf_bert_model: tuple,
                      documents: list,
                      top_k: int = 5,
                      candidate_k: int = 50) -> list:
    """
    2-stage search: TF-IDF retrieval → BERT Cross-Encoder re-ranking.

    Parameters
    ----------
    query            : str   – raw user query
    tfidf_bert_model : tuple – (cross_encoder, vectorizer, tfidf_matrix)
                               returned by build_tfidf_bert()
    documents        : list of str – original (un-cleaned) documents
    top_k            : int   – final results to return (default 5)
    candidate_k      : int   – how many TF-IDF candidates to re-rank (default 50)

    Returns
    -------
    list of dict  – each dict has keys: 'rank', 'score', 'document'
                    'score' is the BERT Cross-Encoder relevance score.
    """
    cross_encoder, vectorizer, tfidf_matrix = tfidf_bert_model

    # ── Stage 1: TF-IDF fast retrieval ──────────────────────────────────────
    cleaned_tokens = preprocess_text(query)
    if not cleaned_tokens:
        cleaned_query_str = query.lower()
    else:
        cleaned_query_str = " ".join(cleaned_tokens)

    query_vector = vectorizer.transform([cleaned_query_str])
    tfidf_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

    n_candidates = min(candidate_k, len(documents))
    candidate_indices = tfidf_scores.argsort()[::-1][:n_candidates]
    candidate_docs = [documents[i] for i in candidate_indices]

    # ── Stage 2: BERT Cross-Encoder re-ranking ──────────────────────────────
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
    tfidf_bert_model = build_tfidf_bert(cleaned, sample_docs)

    query = "food delivery problem"
    results = search_tfidf_bert(query, tfidf_bert_model, sample_docs, top_k=3)

    print(f"\nQuery: '{query}'\n")
    for r in results:
        print(f"  #{r['rank']} | Score: {r['score']:.4f} | {r['document']}")
