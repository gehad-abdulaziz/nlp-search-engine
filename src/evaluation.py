"""
evaluation.py
Precision@k / Recall@k / F1@k + Confusion Matrix evaluation
for ALL 4 models:
    1. TF-IDF        + Cosine Similarity  (Baseline)
    2. Word2Vec      + Cosine Similarity  (Baseline on Embeddings)
    3. TF-IDF        + BERT Cross-Encoder (Advanced)
    4. Word2Vec      + BERT Cross-Encoder (Advanced)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

try:
    from src.tfidf_search      import search_tfidf
    from src.w2v_cosine_search import search_w2v
    from src.tfidf_bert_search import search_tfidf_bert
    from src.w2v_bert_search   import search_w2v_bert
except ModuleNotFoundError:
    from tfidf_search      import search_tfidf
    from w2v_cosine_search import search_w2v
    from tfidf_bert_search import search_tfidf_bert
    from w2v_bert_search   import search_w2v_bert

# ─────────────────────────────────────────────────────────────────────────────
# RELEVANCE ORACLE
# ─────────────────────────────────────────────────────────────────────────────

QUERY_KEYWORDS = {
    "great coffee and pastries": ["coffee","pastri","cafe","espresso","latte",
                                   "muffin","donut","biscuit","croissant","baked"],
    "bad service and cold food": ["bad","cold","terrible","awful","horrible",
                                   "disappoint","poor","worst","rude","slow"],
    "healthy snacks for kids"  : ["healthy","kid","child","snack","organic",
                                   "natural","wholesome","nutritious","fruit","veggie"],
}

def _is_relevant(document, keywords):
    return int(any(kw in document.lower() for kw in keywords))

def _build_relevant_flags(query, results):
    keywords = QUERY_KEYWORDS.get(query,
               [w.lower() for w in query.split() if len(w) > 3])
    return [_is_relevant(r["document"], keywords) for r in results]

# ─────────────────────────────────────────────────────────────────────────────
# CORE METRICS
# ─────────────────────────────────────────────────────────────────────────────

def precision_at_k(relevant_flags, k):
    """Precision@k = relevant in top-k / k"""
    if k <= 0:
        return 0.0
    return round(sum(relevant_flags[:k]) / k, 4)

def recall_at_k(relevant_flags, total_relevant, k):
    """Recall@k = relevant in top-k / total relevant in corpus (capped at k for IR)"""
    if total_relevant <= 0:
        return 0.0
    # In IR without a full corpus scan, we use min(total_relevant, k) as denominator
    denominator = min(total_relevant, k)
    return round(sum(relevant_flags[:k]) / denominator, 4)

def f1_at_k(p, r):
    """F1@k = harmonic mean of Precision@k and Recall@k"""
    if (p + r) == 0:
        return 0.0
    return round(2 * p * r / (p + r), 4)

def confusion_matrix_at_k(relevant_flags, k):
    """
    Binary confusion matrix for top-k results.
    Each result is either:
        TP: retrieved AND relevant
        FP: retrieved AND not relevant
    For the remaining (corpus - k) documents (not retrieved):
        FN: not retrieved AND relevant  → estimated as total_relevant - TP
        TN: not retrieved AND not relevant → corpus_size - TP - FP - FN
    Since we don't scan the full corpus, we return only TP and FP from top-k.

    Returns
    -------
    dict with keys: TP, FP, retrieved_relevant, retrieved_total
    """
    flags = relevant_flags[:k]
    tp = sum(flags)
    fp = k - tp
    return {"TP": tp, "FP": fp, "k": k}

# ─────────────────────────────────────────────────────────────────────────────
# PER-MODEL EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model_name, results, relevant_flags, k=5):
    """Print and return Precision@k, Recall@k, F1@k for one model."""
    p = precision_at_k(relevant_flags, k)
    total_relevant = sum(relevant_flags)          # relevant found in top-k (IR convention)
    r = recall_at_k(relevant_flags, total_relevant, k)
    f = f1_at_k(p, r)
    cm = confusion_matrix_at_k(relevant_flags, k)

    print(f"\n  -- {model_name} --")
    print(f"  Precision@{k} = {p:.4f}  |  Recall@{k} = {r:.4f}  |  F1@{k} = {f:.4f}")
    print(f"  Confusion Matrix (top-{k}): TP={cm['TP']}  FP={cm['FP']}")
    print(f"  {'Rank':<6} {'Score':<8} {'Relevant':<10} Document (first 80 chars)")
    print(f"  {'-'*6} {'-'*8} {'-'*10} {'-'*50}")
    for i, result in enumerate(results[:k]):
        rel = "YES" if relevant_flags[i] == 1 else "NO "
        doc = result["document"][:80].replace("\n", " ")
        print(f"  #{result['rank']:<5} {result['score']:<8.4f} {rel:<10} {doc}...")

    return p, r, f

# ─────────────────────────────────────────────────────────────────────────────
# MODEL STYLE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

MODEL_COLORS = {
    "TF-IDF + Cosine" : "#4C72B0",
    "W2V + Cosine"    : "#55A868",
    "TF-IDF + BERT"   : "#DD8452",
    "W2V + BERT"      : "#C44E52",
}
MODEL_LABELS = {
    "TF-IDF + Cosine" : "TF-IDF + Cosine\n(Baseline)",
    "W2V + Cosine"    : "W2V + Cosine\n(Baseline)",
    "TF-IDF + BERT"   : "TF-IDF + BERT\n(Advanced)",
    "W2V + BERT"      : "W2V + BERT\n(Advanced)",
}

# ─────────────────────────────────────────────────────────────────────────────
# MAIN EVALUATE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(queries, vectorizer, tfidf_matrix,
             w2v_model, w2v_vectors,
             tfidf_bert_model, w2v_bert_model,
             cleaned_docs, documents, k=5):
    """
    Evaluate all 4 models on the given queries.
    Returns dict of scores with Precision, Recall, F1 per model.
    """
    metrics = {
        m: {"precision": [], "recall": [], "f1": []}
        for m in ["TF-IDF + Cosine", "W2V + Cosine", "TF-IDF + BERT", "W2V + BERT"]
    }

    print("\n" + "="*75)
    print(f"  EVALUATION  --  Precision / Recall / F1  @{k}  (4 Models)")
    print("="*75)

    for query in queries:
        print(f"\n>>> Query: \"{query}\"")

        r1 = search_tfidf(query, vectorizer, tfidf_matrix, documents, top_k=k)
        f1 = _build_relevant_flags(query, r1)
        p, r, f = evaluate_model("TF-IDF + Cosine (Baseline)", r1, f1, k)
        metrics["TF-IDF + Cosine"]["precision"].append(p)
        metrics["TF-IDF + Cosine"]["recall"].append(r)
        metrics["TF-IDF + Cosine"]["f1"].append(f)

        r2 = search_w2v(query, w2v_model, w2v_vectors, documents, top_k=k)
        f2 = _build_relevant_flags(query, r2)
        p, r, f = evaluate_model("W2V + Cosine (Baseline)", r2, f2, k)
        metrics["W2V + Cosine"]["precision"].append(p)
        metrics["W2V + Cosine"]["recall"].append(r)
        metrics["W2V + Cosine"]["f1"].append(f)

        r3 = search_tfidf_bert(query, tfidf_bert_model, documents, top_k=k)
        f3 = _build_relevant_flags(query, r3)
        p, r, f = evaluate_model("TF-IDF + BERT (Advanced)", r3, f3, k)
        metrics["TF-IDF + BERT"]["precision"].append(p)
        metrics["TF-IDF + BERT"]["recall"].append(r)
        metrics["TF-IDF + BERT"]["f1"].append(f)

        r4 = search_w2v_bert(query, w2v_bert_model, documents, top_k=k)
        f4 = _build_relevant_flags(query, r4)
        p, r, f = evaluate_model("W2V + BERT (Advanced)", r4, f4, k)
        metrics["W2V + BERT"]["precision"].append(p)
        metrics["W2V + BERT"]["recall"].append(r)
        metrics["W2V + BERT"]["f1"].append(f)

    _print_comparison_table(queries, metrics, k)
    _plot_comparison(queries, metrics, k)
    _plot_confusion_matrices(queries, metrics, k)
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# PRINT TABLE
# ─────────────────────────────────────────────────────────────────────────────

def _print_comparison_table(queries, metrics, k):
    models = list(metrics.keys())
    col_w  = 28

    for metric_name in ["precision", "recall", "f1"]:
        print("\n" + "="*90)
        print(f"  COMPARISON TABLE  --  {metric_name.upper()}@{k}")
        print("="*90)
        header = f"  {'Query':<{col_w}}"
        for m in models:
            header += f" {m:>18}"
        header += f"  {'Winner':>18}"
        print(header)
        print("  " + "-"*88)

        for i, query in enumerate(queries):
            row_scores = {m: metrics[m][metric_name][i] for m in models}
            winner = max(row_scores, key=row_scores.get)
            short_q = query[:col_w-1] if len(query) >= col_w else query
            row = f"  {short_q:<{col_w}}"
            for m in models:
                row += f" {row_scores[m]:>18.4f}"
            row += f"  {winner:>18}"
            print(row)

        avgs = {m: float(np.mean(metrics[m][metric_name])) for m in models}
        winner = max(avgs, key=avgs.get)
        print("  " + "─"*88)
        row = f"  {'AVERAGE':<{col_w}}"
        for m in models:
            row += f" {avgs[m]:>18.4f}"
        row += f"  {winner:>18}"
        print(row)
        print("="*90)


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1: Precision / Recall / F1 Comparison
# ─────────────────────────────────────────────────────────────────────────────

def _plot_comparison(queries, metrics, k):
    models = list(metrics.keys())
    colors = [MODEL_COLORS[m] for m in models]
    labels = [MODEL_LABELS[m] for m in models]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f"Search Engine Evaluation  –  @{k}  (4 Models)",
                 fontsize=14, fontweight="bold")

    for ax, metric_name in zip(axes, ["precision", "recall", "f1"]):
        avgs = [float(np.mean(metrics[m][metric_name])) for m in models]
        bars = ax.bar(labels, avgs, color=colors,
                      edgecolor="white", linewidth=0.8, width=0.55)
        winner_idx = int(np.argmax(avgs))
        for i, (bar, val) in enumerate(zip(bars, avgs)):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                    f"{val:.4f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")
            if i == winner_idx:
                bar.set_edgecolor("gold")
                bar.set_linewidth(2.5)

        ax.set_ylabel(f"Avg {metric_name.title()}@{k}")
        ax.set_title(f"{metric_name.title()}@{k} – All Models")
        ax.set_ylim(0, 1.25)
        ax.yaxis.grid(True, linestyle="--", alpha=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=7)

    ax.legend(handles=[mpatches.Patch(edgecolor="gold", facecolor="none",
              linewidth=2.5, label="Winner")], fontsize=9)

    plt.tight_layout()
    plt.savefig("evaluation_results.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n  Chart saved → evaluation_results.png")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2: Confusion Matrix (TP / FP per model per query)
# ─────────────────────────────────────────────────────────────────────────────

def _plot_confusion_matrices(queries, metrics, k):
    """
    For each model, show a stacked bar: TP (relevant retrieved) vs FP (not relevant retrieved).
    Since recall is already computed from top-k relevant, TP = round(recall * min(relevant, k)).
    We derive TP from precision: TP = precision * k.
    """
    models = list(metrics.keys())
    n_queries = len(queries)
    short_q = [q[:18] + "..." if len(q) > 18 else q for q in queries]
    x = np.arange(n_queries)
    width = 0.18
    offsets = [-1.5, -0.5, 0.5, 1.5]

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle(f"Confusion Matrix Summary (Top-{k} Retrieved)",
                 fontsize=13, fontweight="bold")

    # ── Subplot 1: TP vs FP per query per model (stacked) ───────────────────
    ax1 = axes[0]
    for (model, color), offset in zip(MODEL_COLORS.items(), offsets):
        tp_vals = [round(metrics[model]["precision"][i] * k) for i in range(n_queries)]
        fp_vals = [k - tp for tp in tp_vals]
        bars_tp = ax1.bar(x + offset * width, tp_vals, width,
                          color=color, label=MODEL_LABELS[model].replace("\n", " "),
                          edgecolor="white", linewidth=0.5)
        ax1.bar(x + offset * width, fp_vals, width,
                bottom=tp_vals, color=color, alpha=0.25,
                edgecolor="white", linewidth=0.5)

    ax1.set_xlabel("Query")
    ax1.set_ylabel(f"Count (out of top-{k})")
    ax1.set_title(f"TP (solid) vs FP (faded) per Query")
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_q, rotation=20, ha="right", fontsize=8)
    ax1.set_ylim(0, k + 1)
    ax1.legend(fontsize=7, ncol=2)
    ax1.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax1.set_axisbelow(True)

    # Add legend for TP/FP
    tp_patch = mpatches.Patch(color="gray", label="TP – Relevant retrieved")
    fp_patch = mpatches.Patch(color="gray", alpha=0.25, label="FP – Not relevant retrieved")
    ax1.legend(handles=[tp_patch, fp_patch], fontsize=8, loc="upper right")

    # ── Subplot 2: Overall TP vs FP (average across queries) ────────────────
    ax2 = axes[1]
    avg_tp = [np.mean([round(metrics[m]["precision"][i] * k) for i in range(n_queries)])
              for m in models]
    avg_fp = [k - tp for tp in avg_tp]
    colors_list = [MODEL_COLORS[m] for m in models]
    short_labels = [MODEL_LABELS[m].replace("\n", " ") for m in models]

    bars_tp2 = ax2.bar(short_labels, avg_tp, color=colors_list,
                       edgecolor="white", linewidth=0.5, width=0.5, label="TP (avg)")
    ax2.bar(short_labels, avg_fp, bottom=avg_tp,
            color=colors_list, alpha=0.25,
            edgecolor="white", linewidth=0.5, width=0.5, label="FP (avg)")

    for bar, tp, fp in zip(bars_tp2, avg_tp, avg_fp):
        ax2.text(bar.get_x() + bar.get_width() / 2, tp / 2,
                 f"TP={tp:.1f}", ha="center", va="center",
                 fontsize=9, fontweight="bold", color="white")
        ax2.text(bar.get_x() + bar.get_width() / 2, tp + fp / 2,
                 f"FP={fp:.1f}", ha="center", va="center",
                 fontsize=9, color="#555")

    ax2.set_ylabel(f"Avg count (out of top-{k})")
    ax2.set_title("Average TP vs FP Across All Queries")
    ax2.set_ylim(0, k + 1)
    ax2.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax2.set_axisbelow(True)
    ax2.tick_params(axis="x", labelsize=7)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig("confusion_matrix_results.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Confusion matrix chart saved → confusion_matrix_results.png")