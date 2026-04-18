from src.data_loader import load_data, get_documents, save_sample
from src.preprocessing import preprocess_documents
from src.tfidf_search import build_tfidf, search_tfidf
from src.embedding_search import build_embeddings, search_embeddings
from src.evaluation import evaluate

def main():
    print("Step 1: Loading data...")
    df = load_data("data/Reviews.csv", n_samples=3000)
    documents = get_documents(df)
    save_sample(df)
    print(f"Loaded {len(documents)} reviews")

    print("\nStep 2: Preprocessing...")
    cleaned_docs = preprocess_documents(documents)

    print("\nStep 3: Building TF-IDF model...")
    vectorizer, tfidf_matrix = build_tfidf(cleaned_docs)

    print("\nStep 4: Building embedding model...")
    emb_model, embeddings = build_embeddings(documents)

    print("\nStep 5: Running test queries...")
    queries = [
        "great coffee and pastries",
        "bad service and cold food",
        "healthy snacks for kids"
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        print("  TF-IDF results:")
        tfidf_results = search_tfidf(query, vectorizer, tfidf_matrix, documents, top_k=5)
        for i, result in enumerate(tfidf_results):
            print(f"    {i+1}. {result[:80]}...")

        print("  Embedding results:")
        emb_results = search_embeddings(query, emb_model, embeddings, documents, top_k=5)
        for i, result in enumerate(emb_results):
            print(f"    {i+1}. {result[:80]}...")

    print("\nStep 6: Evaluating...")
    evaluate(queries, vectorizer, tfidf_matrix, emb_model, embeddings, cleaned_docs, documents)
    print("\nDone.")

if __name__ == "__main__":
    main()