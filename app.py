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
</style>
""", unsafe_allow_html=True)

st.title("🔍 Intelligent Search Engine")
st.caption("Project 2 – NLP Course | Faculty of Computing & AI")

# ── Load ALL 4 models (cached) ─────────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ جارٍ تحميل الـ 4 موديلات – دقيقة واحدة...")
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

    # 1. TF-IDF + Cosine (Baseline)
    vectorizer, tfidf_matrix = build_tfidf(cleaned_docs)

    # 2. TF-IDF + BERT (Advanced)
    # Returns (cross_encoder, vectorizer, tfidf_matrix) as a tuple
    tfidf_bert_model = build_tfidf_bert(cleaned_docs, documents)

    # 3. Word2Vec + Cosine (Baseline)
    w2v_model, w2v_vectors = build_w2v(cleaned_docs)

    # 4. Word2Vec + BERT (Advanced)
    # Returns (cross_encoder, w2v_model, doc_vectors) as a tuple
    w2v_bert_model = build_w2v_bert(cleaned_docs, documents)

    return (
        documents, cleaned_docs,
        vectorizer, tfidf_matrix,
        tfidf_bert_model,
        w2v_model, w2v_vectors,
        w2v_bert_model,
    )

# ── Try loading ────────────────────────────────────────────────────────────────
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
        "❌ ملف البيانات مش موجود.\n\n"
        "حملي **Reviews.csv** من Kaggle وحطيه في مجلد `data/`\n\n"
        "🔗 https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews"
    )
    models_ready = False

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ الإعدادات")
    top_k = st.slider("عدد النتايج (Top-K)", min_value=1, max_value=10, value=5)

    st.markdown("---")
    st.markdown("**💡 أمثلة جاهزة:**")
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
    **الـ 4 Models:**
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
        "اكتبي الـ Query بتاعك هنا:",
        value=st.session_state.get("query_input", ""),
        placeholder="مثلاً: food delivery problem",
        key="query_input",
    )
    search_clicked = st.button("🔎 ابحثي", type="primary", disabled=not models_ready)

    if search_clicked and query.strip():
        from src.tfidf_search      import search_tfidf
        from src.w2v_cosine_search import search_w2v
        from src.tfidf_bert_search import search_tfidf_bert
        from src.w2v_bert_search   import search_w2v_bert

        st.markdown(f"### نتايج البحث عن: `{query}`")

        # ── Run all 4 models ───────────────────────────────────────────────────
        with st.spinner("جارٍ البحث في الـ 4 موديلات..."):
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

        # ── 4 columns layout ───────────────────────────────────────────────────
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

        # ── Score comparison bar chart ─────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📊 مقارنة الـ Scores للـ 4 موديلات")

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
                    ha="right", va="bottom", fontsize=9, color=tag_color,
                    fontweight="bold")

        plt.suptitle(f'Query: "{query}"', fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # ── Full document expanders ────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 📖 النص الكامل للنتايج")
        for name, tag, results, _ in models_data:
            with st.expander(f"{name} — Top {top_k} نتايج كاملة"):
                for r in results:
                    st.markdown(f"**Rank #{r['rank']} | Score: {r['score']:.4f}**")
                    st.write(r["document"])
                    st.markdown("---")

    elif search_clicked and not query.strip():
        st.warning("⚠️ اكتبي query الأول!")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EVALUATION & PLOTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📊 Evaluation – Precision@K مقارنة بين الـ 4 موديلات")
    st.info("اضغطي الزر عشان تشغلي الـ evaluation على queries تجريبية.")

    eval_queries = [
        "food delivery problem",
        "great coffee taste",
        "bad quality product",
        "healthy snacks for kids",
        "poor packaging and shipping",
    ]

    # Manual relevance keywords per query
    relevance_keywords = {
        "food delivery problem":      ["delivery", "late", "slow", "shipping", "arrived", "damaged", "problem", "issue"],
        "great coffee taste":         ["coffee", "taste", "flavor", "delicious", "great", "aroma", "brew"],
        "bad quality product":        ["bad", "poor", "quality", "broken", "terrible", "awful", "waste", "disappoint"],
        "healthy snacks for kids":    ["healthy", "snack", "kids", "children", "natural", "organic", "nutritious"],
        "poor packaging and shipping":["packaging", "package", "box", "shipping", "damaged", "broken", "arrived"],
    }

    def is_relevant(doc, query):
        keywords = relevance_keywords.get(query, query.lower().split())
        doc_lower = doc.lower()
        return sum(1 for kw in keywords if kw in doc_lower) >= 2

    if st.button("▶️ Run Full Evaluation", type="primary", disabled=not models_ready):
        from src.tfidf_search      import search_tfidf
        from src.w2v_cosine_search import search_w2v
        from src.tfidf_bert_search import search_tfidf_bert
        from src.w2v_bert_search   import search_w2v_bert

        model_names = [
            "TF-IDF\n+Cosine\n(Baseline)",
            "TF-IDF\n+BERT\n(Advanced)",
            "W2V\n+Cosine\n(Baseline)",
            "W2V\n+BERT\n(Advanced)",
        ]

        results_table = {m: [] for m in model_names}

        progress = st.progress(0)
        for i, q in enumerate(eval_queries):
            r1 = search_tfidf(q, vectorizer, tfidf_matrix, documents, top_k=5)
            r2 = search_tfidf_bert(q, tfidf_bert_model, documents, top_k=5)
            r3 = search_w2v(q, w2v_model, w2v_vectors, documents, top_k=5)
            r4 = search_w2v_bert(q, w2v_bert_model, documents, top_k=5)

            for model_name, results in zip(model_names, [r1, r2, r3, r4]):
                relevant_count = sum(1 for r in results if is_relevant(r["document"], q))
                precision = relevant_count / 5
                results_table[model_name].append(round(precision, 2))

            progress.progress((i + 1) / len(eval_queries))

        # ── Table ──────────────────────────────────────────────────────────────
        df_eval = pd.DataFrame(results_table, index=eval_queries)
        df_eval.index.name = "Query"
        df_eval.loc["**Average**"] = df_eval.mean().round(2)

        st.markdown("#### 📋 Precision@5 لكل موديل لكل query")
        st.dataframe(df_eval.style.highlight_max(axis=1, color="#d1fae5"), use_container_width=True)

        avg_scores = df_eval.drop("**Average**").mean()

        # ── Plot 1: Average Precision Bar Chart ───────────────────────────────
        st.markdown("#### 📊 Plot 1: متوسط الـ Precision@5")
        fig1, ax1 = plt.subplots(figsize=(9, 4))
        bar_colors = ["#2d6a9f", "#7c3aed", "#2d9f6a", "#9f2d2d"]
        bars = ax1.bar(model_names, avg_scores.values, color=bar_colors, alpha=0.85, width=0.5)
        ax1.bar_label(bars, fmt="%.2f", padding=4, fontsize=11, fontweight="bold")
        ax1.set_ylabel("Avg Precision@5")
        ax1.set_title("مقارنة متوسط الـ Precision@5 بين الـ 4 موديلات", fontsize=13, fontweight="bold")
        ax1.set_ylim(0, 1.1)
        ax1.spines[["top", "right"]].set_visible(False)
        ax1.axhline(avg_scores.mean(), color="gray", linestyle="--", linewidth=1, label=f"Overall avg: {avg_scores.mean():.2f}")
        ax1.legend()
        st.pyplot(fig1)
        plt.close()

        # ── Plot 2: Per-query heatmap ─────────────────────────────────────────
        st.markdown("#### 🔥 Plot 2: Heatmap – Precision@5 لكل query")
        fig2, ax2 = plt.subplots(figsize=(11, 4))
        data_matrix = df_eval.drop("**Average**").values
        im = ax2.imshow(data_matrix.T, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
        ax2.set_xticks(range(len(eval_queries)))
        ax2.set_xticklabels([q[:25] for q in eval_queries], rotation=20, ha="right", fontsize=9)
        ax2.set_yticks(range(len(model_names)))
        ax2.set_yticklabels(model_names, fontsize=9)
        for i in range(len(model_names)):
            for j in range(len(eval_queries)):
                ax2.text(j, i, f"{data_matrix[j, i]:.2f}", ha="center", va="center",
                         fontsize=10, fontweight="bold",
                         color="white" if data_matrix[j, i] > 0.5 else "black")
        plt.colorbar(im, ax=ax2, label="Precision@5")
        ax2.set_title("Heatmap – Precision@5 per Query per Model", fontsize=12, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

        # ── Plot 3: Line chart per query ──────────────────────────────────────
        st.markdown("#### 📈 Plot 3: الـ Precision لكل query عبر الـ 4 موديلات")
        fig3, ax3 = plt.subplots(figsize=(11, 4))
        short_names = ["TF-IDF\nCosine", "TF-IDF\nBERT", "W2V\nCosine", "W2V\nBERT"]
        line_colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12"]
        for j, q in enumerate(eval_queries):
            vals = [results_table[m][j] for m in model_names]
            ax3.plot(short_names, vals, marker="o", linewidth=2,
                     color=line_colors[j], label=q[:30], alpha=0.8)
        ax3.set_ylabel("Precision@5")
        ax3.set_title("Precision@5 لكل query عبر الـ 4 موديلات", fontsize=12, fontweight="bold")
        ax3.legend(fontsize=8, loc="upper left")
        ax3.set_ylim(-0.05, 1.1)
        ax3.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

        # ── Plot 4: Radar chart ───────────────────────────────────────────────
        st.markdown("#### 🕸️ Plot 4: Radar Chart – مقارنة شاملة")
        import numpy as np
        categories = [q[:20] for q in eval_queries]
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        fig4, ax4 = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
        for model_name, color in zip(model_names, bar_colors):
            vals = results_table[model_name] + [results_table[model_name][0]]
            ax4.plot(angles, vals, "o-", linewidth=2, color=color,
                     label=model_name.replace("\n", " "))
            ax4.fill(angles, vals, alpha=0.1, color=color)

        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(categories, fontsize=8)
        ax4.set_ylim(0, 1)
        ax4.set_title("Radar Chart – Precision@5 لكل موديل", fontsize=12, fontweight="bold", pad=20)
        ax4.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
        st.pyplot(fig4)
        plt.close()

        # ── Summary ───────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 🏆 الخلاصة")
        best_model = avg_scores.idxmax()
        worst_model = avg_scores.idxmin()
        st.success(f"✅ **أفضل موديل:** {best_model.replace(chr(10), ' ')} بـ Precision@5 = {avg_scores.max():.2f}")
        st.warning(f"⚠️ **أضعف موديل:** {worst_model.replace(chr(10), ' ')} بـ Precision@5 = {avg_scores.min():.2f}")

        baseline_avg = (avg_scores.iloc[0] + avg_scores.iloc[2]) / 2
        advanced_avg = (avg_scores.iloc[1] + avg_scores.iloc[3]) / 2
        if advanced_avg > baseline_avg:
            improvement = ((advanced_avg - baseline_avg) / baseline_avg * 100) if baseline_avg > 0 else 0
            st.info(f"🚀 الـ Advanced models أحسن من الـ Baseline بنسبة **{improvement:.1f}%** في المتوسط")
        else:
            st.info("📊 الـ Baseline models حققت نتايج قريبة من الـ Advanced على الـ dataset ده")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ABOUT
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("""
    ### 📚 عن المشروع

    **Dataset:** Amazon Fine Food Reviews – أول 3000 review

    ---

    ### ⚙️ Preprocessing
    - Lowercase
    - Remove punctuation & numbers
    - Tokenization
    - Remove stopwords
    - Lemmatization

    ---

    ### 🤖 الـ 4 Models

    | # | Feature Extraction | Model | النوع |
    |---|--------------------|-------|-------|
    | 1 | **TF-IDF** | Cosine Similarity | Baseline |
    | 2 | **TF-IDF** | BERT Cross-Encoder | Advanced |
    | 3 | **Word2Vec** (avg vectors) | Cosine Similarity | Baseline |
    | 4 | **Word2Vec** (avg vectors) | BERT Cross-Encoder | Advanced |

    ---

    ### 📊 Evaluation
    - **Precision@K**: عدد النتايج الـ relevant / K
    - مقارنة بين الـ 4 models على 5 queries مختلفة

    ---

    ### 👩‍💻 الفريق
    Gehad · Alaa · Waad · Aliaa · Sama · Aya

    **Capital University – Faculty of Computing & AI | NLP Course 2025-2026**
    """)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("👩‍💻 NLP Project 2 | Capital University | Spring 2025-2026")