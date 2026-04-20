"""
tfidf_search.py
Member 3 – Waad
TF-IDF vectorization + cosine-similarity baseline search engine.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.preprocessing import preprocess_text


# ─────────────────────────────────────────────────────────────────────────────
# 1.  BUILD  (fit on corpus)
# ─────────────────────────────────────────────────────────────────────────────

def build_tfidf(documents: list):
    """
    Fit a TF-IDF vectorizer on *cleaned* documents and return the matrix.

    Parameters
    ----------
    documents : list of str
        Already-preprocessed (cleaned) documents.

    Returns
    -------
    vectorizer : TfidfVectorizer  – fitted vectorizer (use to transform queries)
    tfidf_matrix : scipy sparse matrix  – shape (n_docs, n_features)
    """
    vectorizer = TfidfVectorizer(
        max_features=10_000,   # keep the top 10 k terms by document frequency
        ngram_range=(1, 2),    # unigrams + bigrams → captures "delivery problem"
        min_df=2,              # ignore terms that appear in fewer than 2 docs
        sublinear_tf=True,     # apply log(1+tf) → reduces effect of very common terms
    )

    tfidf_matrix = vectorizer.fit_transform(documents)
    print(f"[TF-IDF] Vocabulary size : {len(vectorizer.vocabulary_):,}")
    print(f"[TF-IDF] Matrix shape    : {tfidf_matrix.shape}")
    return vectorizer, tfidf_matrix


# ─────────────────────────────────────────────────────────────────────────────
# 2.  SEARCH  (query → top-k)
# ─────────────────────────────────────────────────────────────────────────────

def search_tfidf(query: str,
                 vectorizer: TfidfVectorizer,
                 tfidf_matrix,
                 documents: list,
                 top_k: int = 5) -> list:
    """
    Return the top-k most relevant *original* documents for a query.

    Steps
    -----
    1. Preprocess the query the same way we preprocessed the corpus.
    2. Transform query with the fitted vectorizer → query vector.
    3. Compute cosine similarity between query vector and every document vector.
    4. Rank documents by similarity (highest first).
    5. Return top-k original (un-cleaned) document strings.

    Parameters
    ----------
    query        : str   – raw user query
    vectorizer   : fitted TfidfVectorizer
    tfidf_matrix : sparse matrix from build_tfidf()
    documents    : list of str – original (un-cleaned) documents
    top_k        : int   – how many results to return (default 5)

    Returns
    -------
    list of dict  – each dict has keys: 'rank', 'score', 'document'
    """
    # Step 1 – clean the query
    cleaned_query = preprocess_text(query)

    if not cleaned_query:
        cleaned_query = str(query).lower()

    # Step 2 – vectorize the query
    query_vector = vectorizer.transform([cleaned_query])

    # Step 3 – cosine similarity against the entire corpus
    scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # Step 4 – rank (argsort descending)
    ranked_indices = scores.argsort()[::-1][:top_k]

    # Step 5 – build result list
    results = []
    for rank, idx in enumerate(ranked_indices, start=1):
        results.append({
            "rank"    : rank,
            "score"   : round(float(scores[idx]), 4),
            "document": documents[idx],
        })

    return results