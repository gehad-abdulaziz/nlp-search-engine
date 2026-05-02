"""
main.py
Runs all 4 search models and the full evaluation.

Models
------
1. TF-IDF        + Cosine Similarity   (Baseline)
2. Word2Vec      + Cosine Similarity   (Baseline on Embeddings)
3. TF-IDF        + BERT Cross-Encoder  (Advanced)
4. Word2Vec      + BERT Cross-Encoder  (Advanced)
"""

from src.data_loader       import load_data, get_documents, save_sample
from src.preprocessing     import preprocess_documents
from src.tfidf_search      import build_tfidf_improved,  search_tfidf
from src.w2v_cosine_search import build_w2v_improved,    search_w2v, search_w2v_cosine
from src.tfidf_bert_search import build_tfidf_bert, search_tfidf_bert
from src.w2v_bert_search   import build_w2v_bert,   search_w2v_bert
from src.evaluation        import evaluate

SEPARATOR = "=" * 70

def print_results(label, results, k=5):
    print(f"\n  [{label}]")
    for r in results[:k]:
        print(f"    #{r['rank']} | score={r['score']:.4f} | {r['document'][:80]}...")

def main():
    # ── 1. Load data ─────────────────────────────────────────────────────────
    print(SEPARATOR)
    print("STEP 1 : Loading data")
    print(SEPARATOR)
    df        = load_data("data/Reviews.csv", n_samples=6000)
    documents = get_documents(df)
    save_sample(df)
    print(f"  Loaded {len(documents)} reviews")

    # ── 2. Preprocessing ─────────────────────────────────────────────────────
    print(f"\n{SEPARATOR}")
    print("STEP 2 : Preprocessing")
    print(SEPARATOR)
    cleaned_docs = preprocess_documents(documents)
    print("  Done — lowercase, remove punctuation/numbers, tokenize, "
          "remove stopwords, lemmatize")

    # ── 3. Build all models ──────────────────────────────────────────────────
    print(f"\n{SEPARATOR}")
    print("STEP 3 : Building models")
    print(SEPARATOR)

    print("\n  [Model 1] TF-IDF + Cosine Similarity  (Baseline)")
    vectorizer, tfidf_matrix = build_tfidf_improved(cleaned_docs)

    print("\n  [Model 2] Word2Vec + Cosine Similarity  (Baseline on Embeddings)")
    w2v_model, w2v_vectors = build_w2v_improved(cleaned_docs)

    print("\n  [Model 3] TF-IDF + BERT Cross-Encoder  (Advanced)")
    # Returns (cross_encoder, vectorizer, tfidf_matrix) as a tuple
    tfidf_bert_model = build_tfidf_bert(cleaned_docs)

    print("\n  [Model 4] Word2Vec + BERT Cross-Encoder  (Advanced)")
    # Returns (cross_encoder, w2v_model, doc_vectors) as a tuple
    w2v_bert_model = build_w2v_bert(cleaned_docs)

    # Unpack for direct use in search functions
    tfidf_ce, _, _ = tfidf_bert_model
    w2v_ce, _, _   = w2v_bert_model

    # ── 4. Demo queries ──────────────────────────────────────────────────────
    print(f"\n{SEPARATOR}")
    print("STEP 4 : Demo search  (top-5 results per model)")
    print(SEPARATOR)

    queries = [
        "great coffee and pastries",
        "bad service and cold food",
        "healthy snacks for kids",
    ]

    TOP_K = 5
    for query in queries:
        print(f"\n{'─'*60}")
        print(f"  QUERY: \"{query}\"")
        print('─'*60)

        print_results("TF-IDF + Cosine (Baseline)",
                      search_tfidf(query, vectorizer, tfidf_matrix, documents, TOP_K))

        print_results("W2V + Cosine (Baseline)",
                      search_w2v(query, w2v_model, w2v_vectors, documents, TOP_K))

        print_results("TF-IDF + BERT (Advanced)",
                      search_tfidf_bert(query, tfidf_bert_model, documents, TOP_K))

        print_results("W2V + BERT (Advanced)",
                      search_w2v_bert(query, w2v_bert_model, documents, TOP_K))

    # ── 5. Evaluation ────────────────────────────────────────────────────────
    print(f"\n{SEPARATOR}")
    print("STEP 5 : Evaluation  (Precision@5)")
    print(SEPARATOR)

    evaluate(
        queries          = queries,
        vectorizer       = vectorizer,
        tfidf_matrix     = tfidf_matrix,
        w2v_model        = w2v_model,
        w2v_vectors      = w2v_vectors,
        tfidf_bert_model = tfidf_bert_model,
        w2v_bert_model   = w2v_bert_model,
        cleaned_docs     = cleaned_docs,
        documents        = documents,
        k                = TOP_K,
    )

    print(f"\n{SEPARATOR}")
    print("  DONE!  Check evaluation_results.png for the charts.")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
