"""
app.py  –  Streamlit UI for the NLP Search Engine
===================================================
Run from the project root:
    streamlit run app.py

4 Models:
  1. TF-IDF   + Cosine Similarity  (Baseline)
  2. TF-IDF   + BERT Cross-Encoder (Advanced)
  3. Word2Vec + Cosine Similarity  (Baseline)
  4. Word2Vec + BERT Cross-Encoder (Advanced)
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use("Agg")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NLP Search Engine",
    page_icon="🔍",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .model-header {
        background: linear-gradient(135deg, #1e3a5f, #2d6a9f);
        color: white;
        padding: 8px 14px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 14px;
        margin-bottom: 8px;
        text-align: center;
    }
    .model-header.advanced {
        background: linear-gradient(135deg, #4a1060, #8b2fc9);
    }
    .score-bar {
        background: #e8f4f8;
        border-left: 4px solid #2d6a9f;
        padding: 6px 10px;
        margin: 4px 0;
        border-radius: 0 6px 6px 0;
        font-size: 13px;
    }
    .score-bar.advanced {
        background: #f3e8ff;
        border-left-color: #8b2fc9;
    }
    .tag-baseline {
        background: #dbeafe;
        color: #1e40af;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
    }
    .tag-advanced {
        background: #ede9fe;
        color: #6d28d9;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
    }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 12px 16px;
        text-align: center;
        margin: 4px 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔍 Intelligent Search Engine")
st.caption("Project 2 – NLP Course | Faculty of Computing & AI")

# ── Load ALL 4 models (cached) ─────────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Loading all 4 models – please wait...")
def load_all_models():
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from src.data_loader       import load_data, get_documents
    from src.preprocessing     import preprocess_documents
    from src.tfidf_search      import build_tfidf
    from src.w2v_cosine_search import build_w2v
    from src.tfidf_bert_search import build_tfidf_bert
    from src.w2v_bert_search   import build_w2v_bert

    df           = load_data("data/Reviews.csv", n_samples=3000)
    documents    = get_documents(df)
    cleaned_docs = preprocess_documents(documents)

    vectorizer, tfidf_matrix = build_tfidf(cleaned_docs)
    tfidf_bert_model         = build_tfidf_bert(cleaned_docs, documents)
    w2v_model, w2v_vectors   = build_w2v(cleaned_docs)
    w2v_bert_model           = build_w2v_bert(cleaned_docs, documents)

    return (
        documents, cleaned_docs,
        vectorizer, tfidf_matrix,
        tfidf_bert_model,
        w2v_model, w2v_vectors,
        w2v_bert_model,
    )

try:
    (
        documents, cleaned_docs,
        vectorizer, tfidf_matrix,
        tfidf_bert_model,
        w2v_model, w2v_vectors,
        w2v_bert_model,
    ) = load_all_models()
    models_ready = True
except FileNotFoundError:
    st.error(
        "❌ Data file not found.\n\n"
        "Please download **Reviews.csv** from Kaggle and place it in the `data/` folder.\n\n"
        "🔗 https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews"
    )
    models_ready = False

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("Number of Results (Top-K)", min_value=1, max_value=10, value=5)

    st.markdown("---")
    st.markdown("**💡 Example Queries:**")
    example_queries = [
        "great coffee and pastries",
        "bad service and cold food",
        "healthy snacks for kids",
        "food delivery problem",
        "sweet chocolate cake",
        "poor quality product",
    ]
    for q in example_queries:
        if st.button(q, use_container_width=True):
            st.session_state["query_input"] = q

    st.markdown("---")
    st.markdown("""
    **The 4 Models:**
    | # | Feature | Model |
    |---|---------|-------|
    | 1 | TF-IDF | Cosine *(Baseline)* |
    | 2 | TF-IDF | BERT *(Advanced)* |
    | 3 | W2V | Cosine *(Baseline)* |
    | 4 | W2V | BERT *(Advanced)* |
    """)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Search", "📊 Evaluation & Plots", "ℹ️ About"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SEARCH
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    query = st.text_input(
        "Enter your search query:",
        value=st.session_state.get("query_input", ""),
        placeholder="e.g. food delivery problem",
        key="query_input",
    )
    search_clicked = st.button("🔎 Search", type="primary", disabled=not models_ready)

    if search_clicked and query.strip():
        from src.tfidf_search      import search_tfidf
        from src.w2v_cosine_search import search_w2v
        from src.tfidf_bert_search import search_tfidf_bert
        from src.w2v_bert_search   import search_w2v_bert

        st.markdown(f"### Search Results for: `{query}`")

        with st.spinner("Searching across all 4 models..."):
            r1 = search_tfidf(query, vectorizer, tfidf_matrix, documents, top_k=top_k)
            r2 = search_tfidf_bert(query, tfidf_bert_model, documents, top_k=top_k)
            r3 = search_w2v(query, w2v_model, w2v_vectors, documents, top_k=top_k)
            r4 = search_w2v_bert(query, w2v_bert_model, documents, top_k=top_k)

        models_data = [
            ("1️⃣ TF-IDF + Cosine", "Baseline", r1, False),
            ("2️⃣ TF-IDF + BERT",   "Advanced", r2, True),
            ("3️⃣ W2V + Cosine",    "Baseline", r3, False),
            ("4️⃣ W2V + BERT",      "Advanced", r4, True),
        ]

        cols = st.columns(4)
        for col, (name, tag, results, is_adv) in zip(cols, models_data):
            with col:
                tag_class = "tag-advanced" if is_adv else "tag-baseline"
                hdr_class = "model-header advanced" if is_adv else "model-header"
                st.markdown(
                    f'<div class="{hdr_class}">{name}<br>'
                    f'<span class="{tag_class}">{tag}</span></div>',
                    unsafe_allow_html=True,
                )
                for r in results:
                    bar_class = "score-bar advanced" if is_adv else "score-bar"
                    snippet = r["document"][:80].replace("\n", " ")
                    st.markdown(
                        f'<div class="{bar_class}">'
                        f'<b>#{r["rank"]}</b> Score: <b>{r["score"]:.4f}</b><br>'
                        f'<small>{snippet}…</small>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        st.markdown("---")
        st.markdown("### 📊 Score Comparison Across All 4 Models")

        fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=False)
        colors_base = ["#2d6a9f", "#5599cc"]
        colors_adv  = ["#7c3aed", "#a855f7"]

        for ax, (name, tag, results, is_adv) in zip(axes, models_data):
            ranks  = [f"#{r['rank']}" for r in results]
            scores = [r["score"] for r in results]
            color  = colors_adv[0] if is_adv else colors_base[0]
            bars = ax.barh(ranks[::-1], scores[::-1], color=color, alpha=0.85)
            ax.set_title(name.split(" ", 1)[1], fontsize=11, fontweight="bold")
            ax.set_xlabel("Score")
            ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
            ax.spines[["top", "right"]].set_visible(False)
            tag_color = "#7c3aed" if is_adv else "#1e40af"
            ax.text(0.98, 0.02, tag, transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=9, color=tag_color, fontweight="bold")

        plt.suptitle(f'Query: "{query}"', fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("---")
        st.markdown("#### 📖 Full Text of Results")
        for name, tag, results, _ in models_data:
            with st.expander(f"{name} — Top {top_k} Full Results"):
                for r in results:
                    st.markdown(f"**Rank #{r['rank']} | Score: {r['score']:.4f}**")
                    st.write(r["document"])
                    st.markdown("---")

    elif search_clicked and not query.strip():
        st.warning("⚠️ Please enter a query first!")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EVALUATION & PLOTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📊 Evaluation – All Metrics Across All 4 Models")
    st.info("Click the button below to run the full evaluation (Precision, Recall, F1, Confusion Matrix).")

    eval_queries = [
        "food delivery problem",
        "great coffee taste",
        "bad quality product",
        "healthy snacks for kids",
        "poor packaging and shipping",
    ]

    relevance_keywords = {
        "food delivery problem":       ["delivery", "late", "slow", "shipping", "arrived", "damaged", "problem", "issue"],
        "great coffee taste":          ["coffee", "taste", "flavor", "delicious", "great", "aroma", "brew"],
        "bad quality product":         ["bad", "poor", "quality", "broken", "terrible", "awful", "waste", "disappoint"],
        "healthy snacks for kids":     ["healthy", "snack", "kids", "children", "natural", "organic", "nutritious"],
        "poor packaging and shipping": ["packaging", "package", "box", "shipping", "damaged", "broken", "arrived"],
    }

    K_EVAL = 5  # fixed k for evaluation

    def get_relevant_flags(results, query):
        keywords = relevance_keywords.get(query, query.lower().split())
        flags = []
        for r in results:
            doc_lower = r["document"].lower()
            hit = sum(1 for kw in keywords if kw in doc_lower)
            flags.append(1 if hit >= 2 else 0)
        return flags

    def compute_metrics(flags, k=K_EVAL):
        tp = sum(flags[:k])
        fp = k - tp
        # Recall: use TP / min(total_relevant_in_topk, k)
        total_rel = sum(flags)
        denom = min(total_rel, k) if total_rel > 0 else 1
        precision = round(tp / k, 4)
        recall    = round(tp / denom, 4)
        f1        = round(2 * precision * recall / (precision + recall), 4) if (precision + recall) > 0 else 0.0
        return precision, recall, f1, tp, fp

    if st.button("▶️ Run Full Evaluation", type="primary", disabled=not models_ready):
        from src.tfidf_search      import search_tfidf
        from src.w2v_cosine_search import search_w2v
        from src.tfidf_bert_search import search_tfidf_bert
        from src.w2v_bert_search   import search_w2v_bert

        model_names   = ["TF-IDF\n+Cosine\n(Baseline)", "TF-IDF\n+BERT\n(Advanced)",
                         "W2V\n+Cosine\n(Baseline)",    "W2V\n+BERT\n(Advanced)"]
        short_labels  = ["TF-IDF\nCosine", "TF-IDF\nBERT", "W2V\nCosine", "W2V\nBERT"]
        bar_colors    = ["#2d6a9f", "#7c3aed", "#2d9f6a", "#9f2d2d"]

        # Storage
        prec_table = {m: [] for m in model_names}
        rec_table  = {m: [] for m in model_names}
        f1_table   = {m: [] for m in model_names}
        tp_table   = {m: [] for m in model_names}
        fp_table   = {m: [] for m in model_names}

        progress = st.progress(0)
        for i, q in enumerate(eval_queries):
            r1 = search_tfidf(q, vectorizer, tfidf_matrix, documents, top_k=K_EVAL)
            r2 = search_tfidf_bert(q, tfidf_bert_model, documents, top_k=K_EVAL)
            r3 = search_w2v(q, w2v_model, w2v_vectors, documents, top_k=K_EVAL)
            r4 = search_w2v_bert(q, w2v_bert_model, documents, top_k=K_EVAL)

            for m_name, res in zip(model_names, [r1, r2, r3, r4]):
                flags = get_relevant_flags(res, q)
                p, r, f, tp, fp = compute_metrics(flags, K_EVAL)
                prec_table[m_name].append(p)
                rec_table[m_name].append(r)
                f1_table[m_name].append(f)
                tp_table[m_name].append(tp)
                fp_table[m_name].append(fp)

            progress.progress((i + 1) / len(eval_queries))

        # ── Helper averages ───────────────────────────────────────────────────
        avg_prec = {m: round(float(np.mean(prec_table[m])), 4) for m in model_names}
        avg_rec  = {m: round(float(np.mean(rec_table[m])),  4) for m in model_names}
        avg_f1   = {m: round(float(np.mean(f1_table[m])),   4) for m in model_names}
        avg_tp   = {m: round(float(np.mean(tp_table[m])),   2) for m in model_names}
        avg_fp   = {m: round(float(np.mean(fp_table[m])),   2) for m in model_names}

        # ── Summary metric cards ──────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 🏅 Average Metrics Summary (all queries)")
        col_h = st.columns(len(model_names))
        for col, m, color in zip(col_h, model_names, bar_colors):
            with col:
                label = m.replace("\n", " ")
                st.markdown(
                    f'<div class="metric-card" style="border-top: 4px solid {color};">'
                    f'<b style="color:{color}">{label}</b><br>'
                    f'<span style="font-size:13px">Precision: <b>{avg_prec[m]:.2f}</b></span><br>'
                    f'<span style="font-size:13px">Recall: <b>{avg_rec[m]:.2f}</b></span><br>'
                    f'<span style="font-size:13px">F1: <b>{avg_f1[m]:.2f}</b></span><br>'
                    f'<span style="font-size:12px; color:#555">TP≈{avg_tp[m]} | FP≈{avg_fp[m]}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # ── Table: Precision ──────────────────────────────────────────────────
        st.markdown("---")
        st.markdown(f"#### 📋 Precision@{K_EVAL} per Model per Query")
        df_prec = pd.DataFrame(prec_table, index=eval_queries)
        df_prec.index.name = "Query"
        df_prec.loc["Average"] = df_prec.mean().round(4)
        st.dataframe(df_prec.style.highlight_max(axis=1, color="#d1fae5"), use_container_width=True)

        # ── Table: Recall ─────────────────────────────────────────────────────
        st.markdown(f"#### 📋 Recall@{K_EVAL} per Model per Query")
        df_rec = pd.DataFrame(rec_table, index=eval_queries)
        df_rec.index.name = "Query"
        df_rec.loc["Average"] = df_rec.mean().round(4)
        st.dataframe(df_rec.style.highlight_max(axis=1, color="#dbeafe"), use_container_width=True)

        # ── Table: F1 ─────────────────────────────────────────────────────────
        st.markdown(f"#### 📋 F1@{K_EVAL} per Model per Query")
        df_f1 = pd.DataFrame(f1_table, index=eval_queries)
        df_f1.index.name = "Query"
        df_f1.loc["Average"] = df_f1.mean().round(4)
        st.dataframe(df_f1.style.highlight_max(axis=1, color="#ede9fe"), use_container_width=True)

        # ── Classification Report Table ───────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 📄 Classification Report (averaged across all queries)")
        report_rows = []
        for m in model_names:
            p  = avg_prec[m]
            r  = avg_rec[m]
            f  = avg_f1[m]
            tp = avg_tp[m]
            fp = avg_fp[m]
            fn = round(K_EVAL - tp, 2)   # estimated: relevant not retrieved
            support = K_EVAL
            report_rows.append({
                "Model": m.replace("\n", " "),
                "Precision": p,
                "Recall": r,
                "F1-Score": f,
                "TP (avg)": tp,
                "FP (avg)": fp,
                "FN (est.)": fn,
                "Support (k)": support,
            })
        df_report = pd.DataFrame(report_rows).set_index("Model")

        def color_f1(val):
            if isinstance(val, float):
                green = int(val * 200)
                return f"background-color: rgba(0,{green},100,0.15)"
            return ""

        st.dataframe(
            df_report.style
                .format("{:.4f}", subset=["Precision","Recall","F1-Score"])
                .format("{:.1f}",  subset=["TP (avg)","FP (avg)","FN (est.)"])
                .applymap(color_f1, subset=["F1-Score"])
                .highlight_max(axis=0, color="#d1fae5", subset=["Precision","Recall","F1-Score"]),
            use_container_width=True,
        )

        # ── Plot 1: Grouped Bar – Precision / Recall / F1 ─────────────────────
        st.markdown("---")
        st.markdown("#### 📊 Plot 1: Precision / Recall / F1 Comparison")
        fig1, axes1 = plt.subplots(1, 3, figsize=(17, 5))
        fig1.suptitle(f"Search Evaluation – @{K_EVAL}  (4 Models)", fontsize=13, fontweight="bold")

        for ax, metric_vals, metric_name, highlight_color in zip(
            axes1,
            [avg_prec, avg_rec, avg_f1],
            ["Precision", "Recall", "F1-Score"],
            ["#2d6a9f", "#2d9f6a", "#9f2d2d"],
        ):
            vals  = [metric_vals[m] for m in model_names]
            slabs = [m.replace("\n", " ") for m in model_names]
            bars  = ax.bar(short_labels, vals, color=bar_colors, alpha=0.85, width=0.55)
            winner_idx = int(np.argmax(vals))
            for idx, (bar, val) in enumerate(zip(bars, vals)):
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                        f"{val:.3f}", ha="center", va="bottom",
                        fontsize=9, fontweight="bold")
                if idx == winner_idx:
                    bar.set_edgecolor("gold"); bar.set_linewidth(2.5)
            ax.set_title(f"Avg {metric_name}@{K_EVAL}", fontweight="bold")
            ax.set_ylim(0, 1.25)
            ax.yaxis.grid(True, linestyle="--", alpha=0.5)
            ax.set_axisbelow(True)
            ax.tick_params(axis="x", labelsize=8)
            ax.spines[["top", "right"]].set_visible(False)

        plt.tight_layout()
        st.pyplot(fig1)
        plt.close()

        # ── Plot 2: Confusion Matrix Heatmaps (one per model) ─────────────────
        st.markdown("---")
        st.markdown(f"#### 🟥 Plot 2: Confusion Matrix per Model (averaged across {len(eval_queries)} queries)")
        st.caption("TP = Relevant Retrieved  |  FP = Not-Relevant Retrieved  |  FN = Relevant NOT Retrieved  |  TN = Not-Relevant NOT Retrieved (estimated as corpus - k)")

        fig2, axes2 = plt.subplots(1, 4, figsize=(16, 4))
        fig2.suptitle(f"Confusion Matrices – Top-{K_EVAL} Results (Avg across queries)", fontsize=12, fontweight="bold")

        CORPUS_SIZE = 3000
        for ax, m, color in zip(axes2, model_names, bar_colors):
            tp = avg_tp[m]
            fp = avg_fp[m]
            fn = round(K_EVAL - tp, 2)               # relevant not in top-k
            tn = round(CORPUS_SIZE - tp - fp - fn, 2) # large

            # Normalize for display
            cm_vals = np.array([[tp, fp], [fn, min(tn, K_EVAL)]])  # cap TN for display
            im = ax.imshow(cm_vals, cmap="Blues", vmin=0, vmax=K_EVAL)

            ax.set_xticks([0, 1]); ax.set_xticklabels(["Predicted\nRelevant", "Predicted\nNot-Relevant"], fontsize=8)
            ax.set_yticks([0, 1]); ax.set_yticklabels(["Actually\nRelevant", "Actually\nNot-Relevant"], fontsize=8)

            for i_row in range(2):
                for j_col in range(2):
                    val = cm_vals[i_row, j_col]
                    label = ["TP","FP","FN","TN"][i_row * 2 + j_col]
                    ax.text(j_col, i_row, f"{label}\n{val:.1f}",
                            ha="center", va="center",
                            fontsize=12, fontweight="bold",
                            color="white" if val > K_EVAL * 0.5 else "#333")

            ax.set_title(m.replace("\n", " "), fontsize=9, fontweight="bold", color=color)

        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

        # ── Plot 3: Stacked TP/FP bar ─────────────────────────────────────────
        st.markdown("---")
        st.markdown(f"#### 📊 Plot 3: TP vs FP per Query per Model")
        x    = np.arange(len(eval_queries))
        width = 0.18
        offsets = [-1.5, -0.5, 0.5, 1.5]
        short_q = [q[:18] + "…" if len(q) > 18 else q for q in eval_queries]

        fig3, ax3 = plt.subplots(figsize=(14, 5))
        for m, color, offset in zip(model_names, bar_colors, offsets):
            tp_vals = tp_table[m]
            fp_vals = fp_table[m]
            ax3.bar(x + offset * width, tp_vals, width, color=color, label=m.replace("\n", " "))
            ax3.bar(x + offset * width, fp_vals, width, bottom=tp_vals, color=color, alpha=0.25)

        ax3.set_xticks(x); ax3.set_xticklabels(short_q, rotation=15, ha="right", fontsize=9)
        ax3.set_ylabel(f"Count out of top-{K_EVAL}")
        ax3.set_title(f"TP (solid) vs FP (faded) per Query  –  Top-{K_EVAL}", fontweight="bold")
        ax3.set_ylim(0, K_EVAL + 1)
        ax3.legend(fontsize=8, ncol=2)
        ax3.yaxis.grid(True, linestyle="--", alpha=0.5); ax3.set_axisbelow(True)
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

        # ── Plot 4: Heatmap Precision ─────────────────────────────────────────
        st.markdown("---")
        st.markdown(f"#### 🔥 Plot 4: Heatmap – Precision@{K_EVAL} per Query")
        fig4, ax4 = plt.subplots(figsize=(11, 4))
        data_matrix = np.array([prec_table[m] for m in model_names])  # (4, n_queries)
        im4 = ax4.imshow(data_matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
        ax4.set_xticks(range(len(eval_queries))); ax4.set_xticklabels(short_q, rotation=20, ha="right", fontsize=9)
        ax4.set_yticks(range(len(model_names))); ax4.set_yticklabels([m.replace("\n", " ") for m in model_names], fontsize=9)
        for i in range(len(model_names)):
            for j in range(len(eval_queries)):
                ax4.text(j, i, f"{data_matrix[i, j]:.2f}", ha="center", va="center",
                         fontsize=10, fontweight="bold",
                         color="white" if data_matrix[i, j] > 0.5 else "black")
        plt.colorbar(im4, ax=ax4, label=f"Precision@{K_EVAL}")
        ax4.set_title(f"Heatmap – Precision@{K_EVAL} per Query per Model", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()

        # ── Plot 5: Line chart ────────────────────────────────────────────────
        st.markdown("---")
        st.markdown(f"#### 📈 Plot 5: F1@{K_EVAL} per Query Across All Models")
        fig5, ax5 = plt.subplots(figsize=(11, 4))
        line_colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12"]
        for j, q in enumerate(eval_queries):
            vals = [f1_table[m][j] for m in model_names]
            ax5.plot(short_labels, vals, marker="o", linewidth=2,
                     color=line_colors[j], label=q[:28], alpha=0.85)
        ax5.set_ylabel(f"F1@{K_EVAL}"); ax5.set_ylim(-0.05, 1.15)
        ax5.set_title(f"F1@{K_EVAL} per Query Across All 4 Models", fontweight="bold")
        ax5.legend(fontsize=8, loc="upper left"); ax5.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig5)
        plt.close()

        # ── Plot 6: Radar ─────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 🕸️ Plot 6: Radar Chart – F1 per Query")
        N      = len(eval_queries)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        fig6, ax6 = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
        for m, color in zip(model_names, bar_colors):
            vals = f1_table[m] + [f1_table[m][0]]
            ax6.plot(angles, vals, "o-", linewidth=2, color=color, label=m.replace("\n", " "))
            ax6.fill(angles, vals, alpha=0.1, color=color)
        ax6.set_xticks(angles[:-1])
        ax6.set_xticklabels([q[:20] for q in eval_queries], fontsize=8)
        ax6.set_ylim(0, 1)
        ax6.set_title(f"Radar – F1@{K_EVAL} per Model", fontsize=12, fontweight="bold", pad=20)
        ax6.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
        st.pyplot(fig6)
        plt.close()

        # ── Final Summary ─────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 🏆 Summary")
        best_m  = max(avg_f1, key=avg_f1.get)
        worst_m = min(avg_f1, key=avg_f1.get)
        st.success(f"✅ **Best Model (F1):** {best_m.replace(chr(10),' ')}  |  F1@{K_EVAL} = {avg_f1[best_m]:.4f}")
        st.warning(f"⚠️ **Weakest Model (F1):** {worst_m.replace(chr(10),' ')}  |  F1@{K_EVAL} = {avg_f1[worst_m]:.4f}")

        baseline_f1 = np.mean([avg_f1[model_names[0]], avg_f1[model_names[2]]])
        advanced_f1 = np.mean([avg_f1[model_names[1]], avg_f1[model_names[3]]])
        if advanced_f1 > baseline_f1:
            pct = ((advanced_f1 - baseline_f1) / baseline_f1 * 100) if baseline_f1 > 0 else 0
            st.info(f"🚀 Advanced models outperform Baseline by **{pct:.1f}%** avg F1.")
        else:
            st.info("📊 Baseline models achieved comparable results to Advanced models on this dataset.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ABOUT
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("""
    ### 📚 About the Project

    **Dataset:** Amazon Fine Food Reviews – first 3,000 reviews

    ---

    ### ⚙️ Preprocessing Steps
    - Lowercase
    - Remove punctuation & numbers
    - Tokenization
    - Remove stopwords
    - Lemmatization

    ---

    ### 🤖 The 4 Models

    | # | Feature Extraction | Model | Type |
    |---|--------------------|-------|------|
    | 1 | **TF-IDF** | Cosine Similarity | Baseline |
    | 2 | **TF-IDF** | BERT Cross-Encoder | Advanced |
    | 3 | **Word2Vec** (avg vectors) | Cosine Similarity | Baseline |
    | 4 | **Word2Vec** (avg vectors) | BERT Cross-Encoder | Advanced |

    ---

    ### 📊 Evaluation Metrics
    - **Precision@K** – fraction of top-K results that are relevant
    - **Recall@K** – fraction of relevant docs successfully retrieved
    - **F1@K** – harmonic mean of Precision and Recall
    - **Confusion Matrix** – TP / FP / FN breakdown per model

    ---

    ### 👩‍💻 Team
    | Member | Role |
    |--------|------|
    | Gehad | Project Manager · Data Loader · tfidf_bert_search.py · w2v_bert_search.py |
    | Alaa | Text Preprocessing |
    | Waad | TF-IDF Baseline Search |
    | Aliaa | Word2Vec Cosine Similarity Search |
    | Sama | Evaluation & Comparison |
    | Aya | Report & Demo Notebook |

    **Capital University – Faculty of Computing & AI | NLP Course 2025-2026**
    """)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("👩‍💻 NLP Project 2 | Capital University | Spring 2025-2026")