def precision_at_k(relevant_flags, k):
    # takes a list of 0s and 1s (relevant or not) and returns precision@k
    pass

def evaluate_model(model_name, results, relevant_flags, k=5):
    # prints precision@k and a summary for one model
    pass

def evaluate(queries, vectorizer, tfidf_matrix, emb_model, embeddings, cleaned_docs, documents):
    # runs evaluation for both models on all queries
    # prints comparison table
    pass