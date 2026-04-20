"""
embedding_search.py
Member 4 – Aliaa
Sentence-Transformer embeddings + semantic search engine.
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# 1. BUILD (encode all documents)
# ─────────────────────────────────────────────────────────────────────────────

def build_embeddings(documents: list):
    """
    Encode all documents using a sentence transformer model.

    Parameters
    ----------
    documents : list of str – original (un-cleaned) documents

    Returns
    -------
    model      : SentenceTransformer – loaded model
    embeddings : np.ndarray – shape (n_docs, 384)
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("[Embeddings] Encoding documents...")
    embeddings = model.encode(documents, show_progress_bar=True)
    print(f"[Embeddings] Matrix shape : {embeddings.shape}")
    return model, embeddings


# ─────────────────────────────────────────────────────────────────────────────
# 2. SEARCH (query → top-k)
# ─────────────────────────────────────────────────────────────────────────────

def search_embeddings(query: str,
                    model,
                    embeddings,
                    documents: list,
                    top_k: int = 5) -> list:
    """
    Return the top-k most relevant documents using semantic similarity.

    Parameters
    ----------
    query      : str   – raw user query
    model      : fitted SentenceTransformer
    embeddings : np.ndarray from build_embeddings()
    documents  : list of str – original documents
    top_k      : int   – how many results to return (default 5)

    Returns
    -------
    list of dict – each dict has keys: 'rank', 'score', 'document'
    """
    # Encode the query
    query_embedding = model.encode([query])

    # Cosine similarity between query and all documents
    scores = cosine_similarity(query_embedding, embeddings).flatten()

    # Rank descending
    ranked_indices = scores.argsort()[::-1][:top_k]

    # Build result list
    results = []
    for rank, idx in enumerate(ranked_indices, start=1):
        results.append({
            "rank"    : rank,
            "score"   : round(float(scores[idx]), 4),
            "document": documents[idx],
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample_docs = [
        "The food delivery was very late and cold",
        "I love this product, great quality",
        "Shipment arrived damaged and broken",
        "Best coffee I ever tasted",
        "Package was delayed for two weeks",
        "The courier never showed up",
        "Amazing taste and fast shipping",
    ]

    model, embeddings = build_embeddings(sample_docs)

    query = "food delivery problem"
    results = search_embeddings(query, model, embeddings, sample_docs, top_k=3)

    print(f"\nQuery: '{query}'\n")
    for r in results:
        print(f"  #{r['rank']} | Score: {r['score']:.4f} | {r['document']}")