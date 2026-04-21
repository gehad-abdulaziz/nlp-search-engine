"""
evaluation.py
Member 5 – Sama
Precision@k evaluation + comparison tables and charts for both models.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
try:
    from src.tfidf_search import search_tfidf
    from src.embedding_search import search_embeddings
except ModuleNotFoundError:
    from tfidf_search import search_tfidf
    from embedding_search import search_embeddings


# ─────────────────────────────────────────────────────────────────────────────
# RELEVANCE ORACLE
# A simple keyword-based relevance judge.
# For each query we define "relevant keywords" – if a document contains
# at least one of them we treat it as relevant (flag = 1), else 0.
# In a real project you would have human-annotated qrels; here we simulate them.
# ─────────────────────────────────────────────────────────────────────────────

QUERY_KEYWORDS = {
    "great coffee and pastries"  : ["coffee", "pastri", "cafe", "espresso", "latte", "muffin", "donut", "biscuit", "croissant", "baked"],
    "bad service and cold food"  : ["bad", "cold", "terrible", "awful", "horrible", "disappoint", "poor", "worst", "rude", "slow"],
    "healthy snacks for kids"    : ["healthy", "kid", "child", "snack", "organic", "natural", "wholesome", "nutritious", "fruit", "veggie"],
}


def _is_relevant(document: str, keywords: list) -> int:
    """Return 1 if the document contains at least one keyword, else 0."""
    doc_lower = document.lower()
    return int(any(kw in doc_lower for kw in keywords))


def _build_relevant_flags(query: str, results: list) -> list:
    """
    Build a list of 0/1 relevance flags for a list of result dicts.
    Uses QUERY_KEYWORDS if the query is known, otherwise falls back to
    a simple word-overlap heuristic (query words present in document).
    """
    if query in QUERY_KEYWORDS:
        keywords = QUERY_KEYWORDS[query]
    else:
        # Fallback: treat each query word (>3 chars) as a keyword
        keywords = [w.lower() for w in query.split() if len(w) > 3]

    return [_is_relevant(r["document"], keywords) for r in results]


# ─────────────────────────────────────────────────────────────────────────────
# 1.  precision_at_k
# ─────────────────────────────────────────────────────────────────────────────

def precision_at_k(relevant_flags: list, k: int) -> float:
    """
    Compute Precision@k.

    Precision@k = (# relevant documents in top-k) / k

    Parameters
    ----------
    relevant_flags : list of int (0 or 1)
        Relevance flags for the ranked results, in rank order.
    k : int
        Cutoff rank.

    Returns
    -------
    float  – precision value between 0.0 and 1.0
    """
    if k <= 0:
        return 0.0

    top_k_flags = relevant_flags[:k]           # take only the first k flags
    relevant_count = sum(top_k_flags)          # count the 1s
    return round(relevant_count / k, 4)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  evaluate_model
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model_name: str, results: list, relevant_flags: list, k: int = 5):
    """
    Print Precision@k and a per-result summary for one model on one query.

    Parameters
    ----------
    model_name     : str  – label to print (e.g. "TF-IDF" or "Embedding")
    results        : list of dicts with keys 'rank', 'score', 'document'
    relevant_flags : list of int (0 or 1) aligned with results
    k              : int  – cutoff (default 5)

    Returns
    -------
    float  – Precision@k value (also printed to console)
    """
    p_at_k = precision_at_k(relevant_flags, k)

    print(f"\n  -- {model_name} --")
    print(f"  Precision@{k} = {p_at_k:.4f}  "
          f"({sum(relevant_flags[:k])}/{k} relevant)")
    print(f"  {'Rank':<6} {'Score':<8} {'Relevant':<10} Document (first 80 chars)")
    print(f"  {'-'*6} {'-'*8} {'-'*10} {'-'*50}")

    for i, result in enumerate(results[:k]):
        rel_label = "YES" if relevant_flags[i] == 1 else "NO "
        doc_preview = result["document"][:80].replace("\n", " ")
        print(f"  #{result['rank']:<5} {result['score']:<8.4f} {rel_label:<10} {doc_preview}...")

    return p_at_k


# ─────────────────────────────────────────────────────────────────────────────
# 3.  evaluate  (main entry point called from main.py)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(queries: list,
             vectorizer,
             tfidf_matrix,
             emb_model,
             embeddings,
             cleaned_docs: list,
             documents: list,
             k: int = 5):
    """
    Run Precision@k evaluation for both TF-IDF and Embedding models on all
    queries, then print a comparison table and save comparison charts.

    Parameters
    ----------
    queries       : list of str  – raw queries to evaluate
    vectorizer    : fitted TfidfVectorizer
    tfidf_matrix  : sparse matrix from build_tfidf()
    emb_model     : fitted SentenceTransformer
    embeddings    : np.ndarray from build_embeddings()
    cleaned_docs  : list of str – preprocessed documents (for TF-IDF)
    documents     : list of str – original documents
    k             : int  – Precision@k cutoff (default 5)
    """

    tfidf_scores = []
    emb_scores   = []

    print("\n" + "=" * 70)
    print("  EVALUATION  --  Precision@{k}".format(k=k))
    print("=" * 70)

    for query in queries:
        print(f"\n> Query: \"{query}\"")

        # -- TF-IDF ----------------------------------------------------------
        tfidf_results = search_tfidf(
            query, vectorizer, tfidf_matrix, documents, top_k=k
        )
        tfidf_flags = _build_relevant_flags(query, tfidf_results)
        p_tfidf = evaluate_model("TF-IDF", tfidf_results, tfidf_flags, k)

        # -- Embeddings ------------------------------------------------------
        emb_results = search_embeddings(
            query, emb_model, embeddings, documents, top_k=k
        )
        emb_flags = _build_relevant_flags(query, emb_results)
        p_emb = evaluate_model("Embedding", emb_results, emb_flags, k)

        tfidf_scores.append(p_tfidf)
        emb_scores.append(p_emb)

    # -- Comparison Table ----------------------------------------------------
    _print_comparison_table(queries, tfidf_scores, emb_scores, k)

    # -- Charts --------------------------------------------------------------
    _plot_comparison(queries, tfidf_scores, emb_scores, k)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS  – table + chart
# ─────────────────────────────────────────────────────────────────────────────

def _print_comparison_table(queries: list,
                             tfidf_scores: list,
                             emb_scores: list,
                             k: int):
    """Print a side-by-side Precision@k comparison table to the console."""

    col_w = 40
    print("\n" + "=" * 70)
    print(f"  COMPARISON TABLE  --  Precision@{k}")
    print("=" * 70)
    print(f"  {'Query':<{col_w}} {'TF-IDF':>8}  {'Embedding':>10}  {'Winner':>10}")
    print(f"  {'-'*col_w} {'-------':>8}  {'---------':>10}  {'------':>10}")

    for query, p_tf, p_emb in zip(queries, tfidf_scores, emb_scores):
        if p_tf > p_emb:
            winner = "TF-IDF"
        elif p_emb > p_tf:
            winner = "Embedding"
        else:
            winner = "Tie"

        short_q = query[:col_w - 1] if len(query) >= col_w else query
        print(f"  {short_q:<{col_w}} {p_tf:>8.4f}  {p_emb:>10.4f}  {winner:>10}")

    avg_tf  = np.mean(tfidf_scores)
    avg_emb = np.mean(emb_scores)
    print(f"  {'─'*col_w} {'────────':>8}  {'──────────':>10}  {'──────':>10}")
    overall_winner = "TF-IDF" if avg_tf > avg_emb else ("Embedding" if avg_emb > avg_tf else "Tie")
    print(f"  {'AVERAGE':<{col_w}} {avg_tf:>8.4f}  {avg_emb:>10.4f}  {overall_winner:>10}")
    print("=" * 70)


def _plot_comparison(queries: list,
                     tfidf_scores: list,
                     emb_scores: list,
                     k: int):
    """
    Generate and save two charts:
      1. Grouped bar chart  – per-query Precision@k for both models.
      2. Average bar chart  – overall comparison.
    Saved to evaluation_results.png
    """

    short_queries = [q[:30] + "..." if len(q) > 30 else q for q in queries]
    x = np.arange(len(queries))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Search Engine Evaluation  --  Precision@{k}", fontsize=14, fontweight="bold")

    # Chart 1: Per-query grouped bars
    ax1 = axes[0]
    bars1 = ax1.bar(x - width/2, tfidf_scores, width, label="TF-IDF",
                    color="#4C72B0", edgecolor="white", linewidth=0.8)
    bars2 = ax1.bar(x + width/2, emb_scores,   width, label="Embedding",
                    color="#DD8452", edgecolor="white", linewidth=0.8)

    ax1.set_xlabel("Query")
    ax1.set_ylabel(f"Precision@{k}")
    ax1.set_title(f"Per-Query Precision@{k}")
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_queries, rotation=20, ha="right", fontsize=9)
    ax1.set_ylim(0, 1.15)
    ax1.legend()
    ax1.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax1.set_axisbelow(True)

    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 0.02, f"{h:.2f}",
                 ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 0.02, f"{h:.2f}",
                 ha="center", va="bottom", fontsize=8)

    # Chart 2: Average scores
    ax2 = axes[1]
    avg_tf  = float(np.mean(tfidf_scores))
    avg_emb = float(np.mean(emb_scores))
    models  = ["TF-IDF", "Embedding"]
    avgs    = [avg_tf, avg_emb]
    colors  = ["#4C72B0", "#DD8452"]

    bars = ax2.bar(models, avgs, color=colors, edgecolor="white",
                   linewidth=0.8, width=0.4)
    ax2.set_ylabel(f"Average Precision@{k}")
    ax2.set_title(f"Overall Average Precision@{k}")
    ax2.set_ylim(0, 1.15)
    ax2.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax2.set_axisbelow(True)

    for bar, val in zip(bars, avgs):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.02, f"{val:.4f}",
                 ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Highlight winner with gold border
    winner_idx = 0 if avg_tf >= avg_emb else 1
    bars[winner_idx].set_edgecolor("gold")
    bars[winner_idx].set_linewidth(2.5)
    winner_patch = mpatches.Patch(edgecolor="gold", facecolor="none",
                                  linewidth=2.5, label="Winner")
    ax2.legend(handles=[winner_patch])

    plt.tight_layout()
    out_path = "evaluation_results.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Chart saved -> {out_path}")



if __name__ == "__main__":
    import sys
    import os
 
    # Allow running as  python -m src.evaluation  from the project root
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
 
    try:
        from src.data_loader import load_data, get_documents
        from src.preprocessing import preprocess_documents
        from src.tfidf_search import build_tfidf
        from src.embedding_search import build_embeddings
    except ModuleNotFoundError:
        from data_loader import load_data, get_documents
        from preprocessing import preprocess_documents
        from tfidf_search import build_tfidf
        from embedding_search import build_embeddings
 
    print("Step 1: Loading data...")
    df = load_data("data/Reviews.csv", n_samples=3000)
    documents = get_documents(df)
    print(f"Loaded {len(documents)} reviews")
 
    print("\nStep 2: Preprocessing...")
    cleaned_docs = preprocess_documents(documents)
 
    print("\nStep 3: Building TF-IDF model...")
    vectorizer, tfidf_matrix = build_tfidf(cleaned_docs)
 
    print("\nStep 4: Building Embedding model...")
    emb_model, embeddings = build_embeddings(documents)
 
    queries = [
        "great coffee and pastries",
        "bad service and cold food",
        "healthy snacks for kids",
    ]
 
    print("\nStep 5: Running evaluation...")
    evaluate(queries, vectorizer, tfidf_matrix, emb_model, embeddings, cleaned_docs, documents)
 
    print("\nDone! Check evaluation_results.png for the charts.")    