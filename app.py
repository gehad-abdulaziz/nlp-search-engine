"""
app.py  –  Streamlit UI for the NLP Search Engine
===================================================
Run from the project root:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use("Agg")

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS  (defined first so they're available everywhere)
# ═══════════════════════════════════════════════════════════════════════════════

def _search_tfidf_bert_improved(query, tfidf_bert_model, vec_imp, tfidf_imp, documents, top_k=5, candidate_k=100):
    """Use improved TF-IDF for candidate retrieval + same BERT cross-encoder re-ranking."""
    from sklearn.metrics.pairwise import cosine_similarity
    from src.preprocessing import preprocess_text

    cross_encoder, _, _ = tfidf_bert_model
    cleaned_tokens    = preprocess_text(query)
    cleaned_query_str = " ".join(cleaned_tokens) if cleaned_tokens else query.lower()

    query_vector = vec_imp.transform([cleaned_query_str])
    tfidf_scores = cosine_similarity(query_vector, tfidf_imp).flatten()

    n_candidates      = min(candidate_k, len(documents))
    candidate_indices = tfidf_scores.argsort()[::-1][:n_candidates]
    candidate_docs    = [documents[i] for i in candidate_indices]

    pairs       = [(query, doc) for doc in candidate_docs]
    bert_scores = cross_encoder.predict(pairs)
    sorted_order = np.argsort(bert_scores)[::-1][:top_k]

    return [{"rank": rank, "score": round(float(bert_scores[o]), 4), "document": candidate_docs[o]}
            for rank, o in enumerate(sorted_order, 1)]


def _search_w2v_bert_improved(query, w2v_bert_model, w2v_imp, w2v_vec_imp, documents, top_k=5, candidate_k=100):
    """Use improved W2V for candidate retrieval + same BERT cross-encoder re-ranking."""
    from sklearn.metrics.pairwise import cosine_similarity
    from src.preprocessing import preprocess_text

    cross_encoder, _, _ = w2v_bert_model
    vector_size = w2v_imp.vector_size

    cleaned_tokens = preprocess_text(query)
    cleaned_query  = " ".join(cleaned_tokens) if cleaned_tokens else query.lower()

    tokens = cleaned_query.split()
    known  = [t for t in tokens if t in w2v_imp.wv]
    if not known:
        return []
    query_vector = np.stack([w2v_imp.wv[t] for t in known]).mean(axis=0)

    w2v_scores        = cosine_similarity(query_vector.reshape(1, -1), w2v_vec_imp).flatten()
    n_candidates      = min(candidate_k, len(documents))
    candidate_indices = w2v_scores.argsort()[::-1][:n_candidates]
    candidate_docs    = [documents[i] for i in candidate_indices]

    pairs        = [(query, doc) for doc in candidate_docs]
    bert_scores  = cross_encoder.predict(pairs)
    sorted_order = np.argsort(bert_scores)[::-1][:top_k]

    return [{"rank": rank, "score": round(float(bert_scores[o]), 4), "document": candidate_docs[o]}
            for rank, o in enumerate(sorted_order, 1)]


def get_bonus_flags(results, query, keyword_map):
    keywords = keyword_map.get(query, query.lower().split())
    return [1 if sum(1 for kw in keywords if kw in r["document"].lower()) >= 2 else 0
            for r in results]


def compute_metrics(flags, k=5):
    tp        = sum(flags[:k])
    fp        = k - tp
    total_rel = sum(flags)
    denom     = min(total_rel, k) if total_rel > 0 else 1
    precision = round(tp / k, 4)
    recall    = round(tp / denom, 4)
    f1        = round(2 * precision * recall / (precision + recall), 4) if (precision + recall) > 0 else 0.0
    return precision, recall, f1, tp, fp


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="NLP Search Engine", page_icon="🔍", layout="wide")

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base Cards ── */
.score-bar, .score-bar.advanced,
.bonus-before, .bonus-after {
    padding: 10px 14px;
    margin: 6px 0;
    border-radius: 10px;
    font-size: 13px;
    color: #f1f5f9 !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}
.score-bar          { background: #1e3a5f; border-left: 4px solid #60a5fa; }
.score-bar.advanced { background: #3b1f6e; border-left: 4px solid #a78bfa; }
.bonus-before       { background: #7c2d12; border-left: 4px solid #fb923c; }
.bonus-after        { background: #14532d; border-left: 4px solid #4ade80; }

.score-bar b, .score-bar.advanced b,
.bonus-before b, .bonus-after b         { color: #ffffff !important; font-weight: 700; }
.score-bar small, .score-bar.advanced small,
.bonus-before small, .bonus-after small { color: #cbd5e1 !important; line-height: 1.6; }

/* ── Model Headers ── */
.model-header {
    background: linear-gradient(135deg, #0f2044, #1e4080);
    color: #ffffff !important;
    padding: 10px 14px;
    border-radius: 10px;
    font-weight: 700;
    font-size: 14px;
    margin-bottom: 10px;
    text-align: center;
    border: 1px solid #2d6a9f;
    box-shadow: 0 4px 12px rgba(0,0,0,0.4);
}
.model-header.advanced {
    background: linear-gradient(135deg, #2d0a4e, #5b1a8a);
    border: 1px solid #7c3aed;
    box-shadow: 0 4px 12px rgba(124,58,237,0.4);
}

/* ── Tags ── */
.tag-baseline {
    background: rgba(96,165,250,0.2);
    color: #93c5fd !important;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
    border: 1px solid #3b82f6;
}
.tag-advanced {
    background: rgba(167,139,250,0.2);
    color: #c4b5fd !important;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
    border: 1px solid #7c3aed;
}

/* ── Metric Cards ── */
.metric-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 14px 16px;
    text-align: center;
    margin: 4px 0;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    color: #f1f5f9 !important;
}
.metric-card b     { color: #ffffff !important; font-size: 15px; }
.metric-card small { color: #94a3b8 !important; }

/* ── Improvement Badges ── */
.improvement-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 700;
    margin: 2px;
}
.badge-up   { background: #14532d; color: #4ade80 !important; border: 1px solid #16a34a; }
.badge-down { background: #7f1d1d; color: #f87171 !important; border: 1px solid #dc2626; }
.badge-same { background: #1e293b; color: #94a3b8 !important; border: 1px solid #475569; }

/* ── Dataframe Fix ── */
.stDataFrame td, .stDataFrame th {
    color: #f1f5f9 !important;
    background-color: #1e293b !important;
}
.stDataFrame tr:hover td {
    background-color: #273449 !important;
}
.stDataFrame td {
    color: #f1f5f9 !important; 
}

[data-testid="stTable"] td div {
    color: inherit !important;
}
</style>
""", unsafe_allow_html=True)

st.title("🔍 Intelligent Search Engine")
st.caption("Project 2 – NLP Course | Faculty of Computing & AI")

# ── Load models ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Loading all 4 models – please wait...")
def load_all_models():
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from src.data_loader       import load_data, get_documents
    from src.preprocessing     import preprocess_documents
    from src.tfidf_search      import build_tfidf_improved
    from src.w2v_cosine_search import build_w2v_improved
    from src.tfidf_bert_search import build_tfidf_bert
    from src.w2v_bert_search   import build_w2v_bert

    df           = load_data("data/Reviews.csv", n_samples=3000)
    documents    = get_documents(df)
    cleaned_docs = preprocess_documents(documents)

    vectorizer, tfidf_matrix = build_tfidf_improved(cleaned_docs)
    tfidf_bert_model         = build_tfidf_bert(cleaned_docs, documents)
    w2v_model, w2v_vectors   = build_w2v_improved(cleaned_docs)
    w2v_bert_model           = build_w2v_bert(cleaned_docs, documents)

    return (documents, cleaned_docs,
            vectorizer, tfidf_matrix,
            tfidf_bert_model,
            w2v_model, w2v_vectors,
            w2v_bert_model)

try:
    (documents, cleaned_docs,
     vectorizer, tfidf_matrix,
     tfidf_bert_model,
     w2v_model, w2v_vectors,
     w2v_bert_model) = load_all_models()
    models_ready = True
except FileNotFoundError:
    st.error("❌ Data file not found. Place Reviews.csv in the data/ folder.")
    models_ready = False

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("Number of Results (Top-K)", 1, 10, 5)
    st.markdown("---")
    st.markdown("**💡 Example Queries:**")
    example_queries = [
        "great coffee and pastries", "bad service and cold food",
        "healthy snacks for kids",   "food delivery problem",
        "sweet chocolate cake",      "poor quality product",
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
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Search", "📊 Evaluation & Plots",
    "🚀 Bonus: Before vs After", "ℹ️ About"
])


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

        st.markdown(f"### Results for: `{query}`")

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
                    unsafe_allow_html=True)
                for r in results:
                    bar_class = "score-bar advanced" if is_adv else "score-bar"
                    snippet = r["document"][:80].replace("\n", " ")
                    st.markdown(
                        f'<div class="{bar_class}">'
                        f'<b>#{r["rank"]}</b> Score: <b>{r["score"]:.4f}</b><br>'
                        f'<small>{snippet}…</small></div>',
                        unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 📊 Score Comparison")
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        clrs = ["#2d6a9f","#7c3aed","#2d9f6a","#9f2d2d"]
        for ax, (name, tag, results, is_adv), color in zip(axes, models_data, clrs):
            ranks  = [f"#{r['rank']}" for r in results]
            scores = [r["score"] for r in results]
            bars = ax.barh(ranks[::-1], scores[::-1], color=color, alpha=0.85)
            ax.set_title(name.split(" ",1)[1], fontsize=11, fontweight="bold")
            ax.set_xlabel("Score")
            ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
            ax.spines[["top","right"]].set_visible(False)
        plt.suptitle(f'Query: "{query}"', fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout(); st.pyplot(fig); plt.close()

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
# TAB 2 — EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📊 Evaluation – All Metrics Across All 4 Models")
    st.info("Click the button below to run the full evaluation.")

    eval_queries = [
        "food delivery problem", "great coffee taste",
        "bad quality product",   "healthy snacks for kids",
        "poor packaging and shipping",
    ]
    relevance_keywords = {
        "food delivery problem":       ["delivery","late","slow","shipping","arrived","damaged","problem","issue"],
        "great coffee taste":          ["coffee","taste","flavor","delicious","great","aroma","brew"],
        "bad quality product":         ["bad","poor","quality","broken","terrible","awful","waste","disappoint"],
        "healthy snacks for kids":     ["healthy","snack","kids","children","natural","organic","nutritious"],
        "poor packaging and shipping": ["packaging","package","box","shipping","damaged","broken","arrived"],
    }
    K_EVAL = 5

    if st.button("▶️ Run Full Evaluation", type="primary", disabled=not models_ready):
        from src.tfidf_search      import search_tfidf
        from src.w2v_cosine_search import search_w2v
        from src.tfidf_bert_search import search_tfidf_bert
        from src.w2v_bert_search   import search_w2v_bert

        model_names  = ["TF-IDF\n+Cosine\n(Baseline)","TF-IDF\n+BERT\n(Advanced)",
                        "W2V\n+Cosine\n(Baseline)",   "W2V\n+BERT\n(Advanced)"]
        short_labels = ["TF-IDF\nCosine","TF-IDF\nBERT","W2V\nCosine","W2V\nBERT"]
        bar_colors   = ["#2d6a9f","#7c3aed","#2d9f6a","#9f2d2d"]

        prec_table={m:[] for m in model_names}; rec_table={m:[] for m in model_names}
        f1_table  ={m:[] for m in model_names}; tp_table ={m:[] for m in model_names}
        fp_table  ={m:[] for m in model_names}

        progress = st.progress(0)
        for i, q in enumerate(eval_queries):
            r1=search_tfidf(q, vectorizer, tfidf_matrix, documents, top_k=K_EVAL)
            r2=search_tfidf_bert(q, tfidf_bert_model, documents, top_k=K_EVAL)
            r3=search_w2v(q, w2v_model, w2v_vectors, documents, top_k=K_EVAL)
            r4=search_w2v_bert(q, w2v_bert_model, documents, top_k=K_EVAL)
            for m_name, res in zip(model_names, [r1,r2,r3,r4]):
                flags = get_bonus_flags(res, q, relevance_keywords)
                p,r,f,tp,fp = compute_metrics(flags, K_EVAL)
                prec_table[m_name].append(p); rec_table[m_name].append(r)
                f1_table[m_name].append(f);   tp_table[m_name].append(tp)
                fp_table[m_name].append(fp)
            progress.progress((i+1)/len(eval_queries))

        avg_prec={m:round(float(np.mean(prec_table[m])),4) for m in model_names}
        avg_rec ={m:round(float(np.mean(rec_table[m])), 4) for m in model_names}
        avg_f1  ={m:round(float(np.mean(f1_table[m])),  4) for m in model_names}
        avg_tp  ={m:round(float(np.mean(tp_table[m])),  2) for m in model_names}
        avg_fp  ={m:round(float(np.mean(fp_table[m])),  2) for m in model_names}

        st.markdown("---"); st.markdown("#### 🏅 Average Metrics Summary")
        col_h = st.columns(4)
        for col,m,color in zip(col_h, model_names, bar_colors):
            with col:
                label = m.replace("\n"," ")
                st.markdown(
                    f'<div class="metric-card" style="border-top:4px solid {color};">'
                    f'<b style="color:{color}">{label}</b><br>'
                    f'Precision: <b>{avg_prec[m]:.2f}</b><br>'
                    f'Recall: <b>{avg_rec[m]:.2f}</b><br>'
                    f'F1: <b>{avg_f1[m]:.2f}</b><br>'
                    f'<small>TP≈{avg_tp[m]} | FP≈{avg_fp[m]}</small></div>',
                    unsafe_allow_html=True)

        st.markdown("---")
        for title, table, hl in [
            (f"📋 Precision@{K_EVAL}", prec_table, "#064e3b"),  # Dark green
            (f"📋 Recall@{K_EVAL}",    rec_table,  "#1e3a8a"),  # Dark blue
            (f"📋 F1@{K_EVAL}",        f1_table,   "#4c1d95"),  # Dark purple
        ]:
            st.markdown(f"#### {title} per Model per Query")
            df_ = pd.DataFrame(table, index=eval_queries)
            df_.index.name = "Query"
            df_.loc["Average"] = df_.mean().round(4)
            st.dataframe(df_.style.highlight_max(axis=1, color=hl), use_container_width=True)

        st.markdown("---"); st.markdown("#### 📄 Classification Report")
        rows=[]
        for m in model_names:
            rows.append({"Model":m.replace("\n"," "),
                         "Precision":avg_prec[m],"Recall":avg_rec[m],"F1-Score":avg_f1[m],
                         "TP (avg)":avg_tp[m],"FP (avg)":avg_fp[m],
                         "FN (est.)":round(K_EVAL-avg_tp[m],2),"Support (k)":K_EVAL})
        df_report = pd.DataFrame(rows).set_index("Model")

        def color_f1(val):
            if isinstance(val, float):
                green = int(val*200)
                return f"background-color:rgba(0,{green},100,0.15)"
            return ""

        # Using map for newer pandas, applymap for older. We'll use applymap which is more backward compatible or map if it's new.
        # Actually highlight_max on F1-Score is safer and cleaner, or just use applymap
        if hasattr(df_report.style, "map"):
            styled_report = df_report.style.map(color_f1, subset=["F1-Score"])
        else:
            styled_report = df_report.style.applymap(color_f1, subset=["F1-Score"])
            
        st.dataframe(styled_report, use_container_width=True)
        # ── Plot 1 ────────────────────────────────────────────────────────────
        st.markdown("---"); st.markdown("#### 📊 Plot 1: Precision / Recall / F1")
        fig1, axes1 = plt.subplots(1,3, figsize=(17,5))
        fig1.suptitle(f"Search Evaluation @{K_EVAL}", fontsize=13, fontweight="bold")
        for ax, metric_vals, metric_name in zip(axes1,
            [avg_prec,avg_rec,avg_f1], ["Precision","Recall","F1-Score"]):
            vals  = [metric_vals[m] for m in model_names]
            bars  = ax.bar(short_labels, vals, color=bar_colors, alpha=0.85, width=0.55)
            wi    = int(np.argmax(vals))
            for idx,(bar,val) in enumerate(zip(bars,vals)):
                ax.text(bar.get_x()+bar.get_width()/2, val+0.02, f"{val:.3f}",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")
                if idx==wi: bar.set_edgecolor("gold"); bar.set_linewidth(2.5)
            ax.set_title(f"Avg {metric_name}@{K_EVAL}", fontweight="bold")
            ax.set_ylim(0,1.25); ax.yaxis.grid(True,linestyle="--",alpha=0.5)
            ax.set_axisbelow(True); ax.tick_params(axis="x",labelsize=8)
            ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout(); st.pyplot(fig1); plt.close()

        # ── Plot 2: Confusion Matrices ────────────────────────────────────────
        st.markdown("---"); st.markdown(f"#### 🟥 Plot 2: Confusion Matrices")
        fig2, axes2 = plt.subplots(1,4, figsize=(16,4))
        for ax,m,color in zip(axes2, model_names, bar_colors):
            tp=avg_tp[m]; fp=avg_fp[m]; fn=round(K_EVAL-tp,2)
            cm_vals = np.array([[tp,fp],[fn,min(3000-tp-fp-fn,K_EVAL)]])
            ax.imshow(cm_vals, cmap="Blues", vmin=0, vmax=K_EVAL)
            ax.set_xticks([0,1]); ax.set_xticklabels(["Pred\nRelev","Pred\nNot-R"],fontsize=8)
            ax.set_yticks([0,1]); ax.set_yticklabels(["Act\nRelev","Act\nNot-R"],fontsize=8)
            for i_r in range(2):
                for j_c in range(2):
                    val=cm_vals[i_r,j_c]; lbl=["TP","FP","FN","TN"][i_r*2+j_c]
                    ax.text(j_c,i_r,f"{lbl}\n{val:.1f}",ha="center",va="center",
                            fontsize=12,fontweight="bold",
                            color="white" if val>K_EVAL*0.5 else "#333")
            ax.set_title(m.replace("\n"," "),fontsize=9,fontweight="bold",color=color)
        plt.tight_layout(); st.pyplot(fig2); plt.close()

        # ── Plot 3: TP vs FP ──────────────────────────────────────────────────
        st.markdown("---"); st.markdown(f"#### 📊 Plot 3: TP vs FP per Query")
        x=np.arange(len(eval_queries)); width=0.18; offsets=[-1.5,-0.5,0.5,1.5]
        short_q=[q[:18]+"…" if len(q)>18 else q for q in eval_queries]
        fig3,ax3=plt.subplots(figsize=(14,5))
        for m,color,offset in zip(model_names,bar_colors,offsets):
            ax3.bar(x+offset*width, tp_table[m], width, color=color, label=m.replace("\n"," "))
            ax3.bar(x+offset*width, fp_table[m], width, bottom=tp_table[m], color=color, alpha=0.25)
        ax3.set_xticks(x); ax3.set_xticklabels(short_q,rotation=15,ha="right",fontsize=9)
        ax3.set_ylabel(f"Count / top-{K_EVAL}")
        ax3.set_title("TP (solid) vs FP (faded)",fontweight="bold")
        ax3.set_ylim(0,K_EVAL+1); ax3.legend(fontsize=8,ncol=2)
        ax3.yaxis.grid(True,linestyle="--",alpha=0.5); ax3.set_axisbelow(True)
        plt.tight_layout(); st.pyplot(fig3); plt.close()

        # ── Plot 4: Heatmap ───────────────────────────────────────────────────
        st.markdown("---"); st.markdown(f"#### 🔥 Plot 4: Precision Heatmap")
        fig4,ax4=plt.subplots(figsize=(11,4))
        data_m=np.array([prec_table[m] for m in model_names])
        im4=ax4.imshow(data_m,aspect="auto",cmap="YlOrRd",vmin=0,vmax=1)
        ax4.set_xticks(range(len(eval_queries))); ax4.set_xticklabels(short_q,rotation=20,ha="right",fontsize=9)
        ax4.set_yticks(range(len(model_names))); ax4.set_yticklabels([m.replace("\n"," ") for m in model_names],fontsize=9)
        for i in range(len(model_names)):
            for j in range(len(eval_queries)):
                ax4.text(j,i,f"{data_m[i,j]:.2f}",ha="center",va="center",fontsize=10,fontweight="bold",
                         color="white" if data_m[i,j]>0.5 else "black")
        plt.colorbar(im4,ax=ax4,label=f"Precision@{K_EVAL}")
        ax4.set_title(f"Precision@{K_EVAL} Heatmap",fontweight="bold")
        plt.tight_layout(); st.pyplot(fig4); plt.close()

        # ── Plot 5: Line ──────────────────────────────────────────────────────
        st.markdown("---"); st.markdown(f"#### 📈 Plot 5: F1 per Query")
        fig5,ax5=plt.subplots(figsize=(11,4))
        lc=["#e74c3c","#3498db","#2ecc71","#9b59b6","#f39c12"]
        for j,q in enumerate(eval_queries):
            ax5.plot(short_labels,[f1_table[m][j] for m in model_names],
                     marker="o",linewidth=2,color=lc[j],label=q[:28],alpha=0.85)
        ax5.set_ylabel(f"F1@{K_EVAL}"); ax5.set_ylim(-0.05,1.15)
        ax5.set_title("F1 per Query Across Models",fontweight="bold")
        ax5.legend(fontsize=8,loc="upper left"); ax5.spines[["top","right"]].set_visible(False)
        plt.tight_layout(); st.pyplot(fig5); plt.close()

        # ── Plot 6: Radar ─────────────────────────────────────────────────────
        st.markdown("---"); st.markdown("#### 🕸️ Plot 6: Radar – F1")
        N=len(eval_queries); angles=[n/float(N)*2*np.pi for n in range(N)]; angles+=angles[:1]
        fig6,ax6=plt.subplots(figsize=(7,7),subplot_kw=dict(polar=True))
        for m,color in zip(model_names,bar_colors):
            vals=f1_table[m]+[f1_table[m][0]]
            ax6.plot(angles,vals,"o-",linewidth=2,color=color,label=m.replace("\n"," "))
            ax6.fill(angles,vals,alpha=0.1,color=color)
        ax6.set_xticks(angles[:-1]); ax6.set_xticklabels([q[:20] for q in eval_queries],fontsize=8)
        ax6.set_ylim(0,1); ax6.set_title("Radar – F1",fontsize=12,fontweight="bold",pad=20)
        ax6.legend(loc="upper right",bbox_to_anchor=(1.35,1.1),fontsize=9)
        st.pyplot(fig6); plt.close()

        st.markdown("---"); st.markdown("#### 🏆 Summary")
        best_m=max(avg_f1,key=avg_f1.get); worst_m=min(avg_f1,key=avg_f1.get)
        st.success(f"✅ Best: {best_m.replace(chr(10),' ')} | F1={avg_f1[best_m]:.4f}")
        st.warning(f"⚠️ Weakest: {worst_m.replace(chr(10),' ')} | F1={avg_f1[worst_m]:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — BONUS: BEFORE vs AFTER
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## 🚀 Bonus: Model Improvement — Before vs After")
    st.markdown("Evaluate **improved versions** of all 4 models vs the originals on the same queries.")

    with st.expander("📋 What was improved in each model?", expanded=True):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("""
**1️⃣ TF-IDF + Cosine**
| Param | Before | After |
|-------|--------|-------|
| ngram_range | (1,1) | **(1,2)** |
| sublinear_tf | False | **True** |
| max_df | 1.0 | **0.90** |
| min_df | 1 | **1** |
| max_features | None | **15,000** |

**3️⃣ W2V + Cosine**
| Param | Before | After |
|-------|--------|-------|
| vector_size | 100 | **200** |
| window | 5 | **3** |
| epochs | 10 | **30** |
| min_count | 2 | **1** |
| algorithm | CBOW | **Skip-gram** |
""")
        with col_b:
            st.markdown("""
**2️⃣ TF-IDF + BERT**
| Param | Before | After |
|-------|--------|-------|
| TF-IDF stage | baseline | **improved** |
| candidate_k | 50 | **100** |

**4️⃣ W2V + BERT**
| Param | Before | After |
|-------|--------|-------|
| W2V stage | baseline | **improved** |
| candidate_k | 50 | **100** |
""")

    st.markdown("---")

    bonus_queries = [
        "great coffee and pastries", "bad service and cold food",
        "healthy snacks for kids",   "food delivery problem",
        "poor quality product",
    ]
    bonus_keywords = {
        "great coffee and pastries": ["coffee","pastry","pastries","great","delicious","taste","flavor","cake","bakery"],
        "bad service and cold food": ["bad","cold","service","poor","terrible","awful","disappointing","worst"],
        "healthy snacks for kids":   ["healthy","snack","kids","children","natural","organic","nutritious"],
        "food delivery problem":     ["delivery","late","slow","shipping","arrived","damaged","problem","issue"],
        "poor quality product":      ["bad","poor","quality","broken","terrible","awful","waste","disappoint"],
    }
    K_BONUS = 5

    # ── Per-query search viewer ───────────────────────────────────────────────
    st.markdown("### 🔎 Try a Query — See Before vs After Results")
    bonus_query = st.selectbox("Choose a query to inspect:", bonus_queries, key="bonus_q")

    if st.button("🔍 Show Before vs After Results", key="btn_bonus_search", disabled=not models_ready):
        from src.tfidf_search      import search_tfidf, build_tfidf_improved
        from src.w2v_cosine_search import search_w2v, build_w2v_improved
        from src.tfidf_bert_search import search_tfidf_bert
        from src.w2v_bert_search   import search_w2v_bert

        with st.spinner("Building improved models and searching..."):
            vec_imp, tfidf_imp   = build_tfidf_improved(cleaned_docs)
            w2v_imp, w2v_vec_imp = build_w2v_improved(cleaned_docs)

            # BEFORE results
            rb1 = search_tfidf(bonus_query, vectorizer, tfidf_matrix, documents, top_k=K_BONUS)
            rb2 = search_tfidf_bert(bonus_query, tfidf_bert_model, documents, top_k=K_BONUS)
            rb3 = search_w2v(bonus_query, w2v_model, w2v_vectors, documents, top_k=K_BONUS)
            rb4 = search_w2v_bert(bonus_query, w2v_bert_model, documents, top_k=K_BONUS)

            # AFTER results
            ra1 = search_tfidf(bonus_query, vec_imp, tfidf_imp, documents, top_k=K_BONUS)
            ra2 = _search_tfidf_bert_improved(bonus_query, tfidf_bert_model, vec_imp, tfidf_imp, documents, K_BONUS)
            ra3 = search_w2v(bonus_query, w2v_imp, w2v_vec_imp, documents, top_k=K_BONUS)
            ra4 = _search_w2v_bert_improved(bonus_query, w2v_bert_model, w2v_imp, w2v_vec_imp, documents, K_BONUS)

        st.markdown(f"#### Results for: `{bonus_query}`")

        model_pairs = [
            ("TF-IDF + Cosine", rb1, ra1),
            ("TF-IDF + BERT",   rb2, ra2),
            ("W2V + Cosine",    rb3, ra3),
            ("W2V + BERT",      rb4, ra4),
        ]

        for model_name, before_res, after_res in model_pairs:
            st.markdown(f"**{model_name}**")
            c_before, c_after = st.columns(2)

            with c_before:
                st.markdown("🟠 **Before**")
                for r in before_res:
                    snippet = r["document"][:70].replace("\n", " ")
                    st.markdown(
                        f'<div class="bonus-before"><b>#{r["rank"]}</b> '
                        f'Score: <b>{r["score"]:.4f}</b><br>'
                        f'<small>{snippet}…</small></div>',
                        unsafe_allow_html=True)

            with c_after:
                st.markdown("🟢 **After**")
                for r in after_res:
                    snippet = r["document"][:70].replace("\n", " ")
                    st.markdown(
                        f'<div class="bonus-after"><b>#{r["rank"]}</b> '
                        f'Score: <b>{r["score"]:.4f}</b><br>'
                        f'<small>{snippet}…</small></div>',
                        unsafe_allow_html=True)

            # Relevance flags
            bf = get_bonus_flags(before_res, bonus_query, bonus_keywords)
            af = get_bonus_flags(after_res,  bonus_query, bonus_keywords)
            pb,rb_,fb,_,_ = compute_metrics(bf, K_BONUS)
            pa,ra_,fa,_,_ = compute_metrics(af, K_BONUS)
            delta_f1 = fa - fb
            badge = "badge-up" if delta_f1>0 else ("badge-down" if delta_f1<0 else "badge-same")
            sign  = "▲" if delta_f1>0 else ("▼" if delta_f1<0 else "—")
            st.markdown(
                f'Before F1={fb:.3f} → After F1={fa:.3f} '
                f'<span class="improvement-badge {badge}">{sign} {abs(delta_f1)*100:.1f}%</span>',
                unsafe_allow_html=True)
            st.markdown("---")

    # ── Full evaluation button ────────────────────────────────────────────────
    st.markdown("### 📊 Full Before vs After Evaluation (all queries)")
    if st.button("▶️ Run Full Bonus Evaluation", type="primary", disabled=not models_ready):
        from src.tfidf_search      import search_tfidf, build_tfidf_improved
        from src.w2v_cosine_search import search_w2v, build_w2v_improved
        from src.tfidf_bert_search import search_tfidf_bert
        from src.w2v_bert_search   import search_w2v_bert

        with st.spinner("Building improved models... (~1 min)"):
            vec_imp, tfidf_imp   = build_tfidf_improved(cleaned_docs)
            w2v_imp, w2v_vec_imp = build_w2v_improved(cleaned_docs)
        st.success("✅ Improved models ready!")

        model_labels_before = ["TF-IDF+Cosine\n(Before)","TF-IDF+BERT\n(Before)",
                               "W2V+Cosine\n(Before)",   "W2V+BERT\n(Before)"]
        model_labels_after  = ["TF-IDF+Cosine\n(After)", "TF-IDF+BERT\n(After)",
                               "W2V+Cosine\n(After)",    "W2V+BERT\n(After)"]
        short_names = ["TF-IDF\nCosine","TF-IDF\nBERT","W2V\nCosine","W2V\nBERT"]
        bar_colors  = ["#2d6a9f","#7c3aed","#2d9f6a","#9f2d2d"]

        before_f1={m:[] for m in model_labels_before}
        after_f1 ={m:[] for m in model_labels_after}
        before_prec={m:[] for m in model_labels_before}
        after_prec ={m:[] for m in model_labels_after}
        before_rec={m:[] for m in model_labels_before}
        after_rec ={m:[] for m in model_labels_after}

        progress2 = st.progress(0)
        for i, q in enumerate(bonus_queries):
            rb1=search_tfidf(q, vectorizer, tfidf_matrix, documents, top_k=K_BONUS)
            rb2=search_tfidf_bert(q, tfidf_bert_model, documents, top_k=K_BONUS)
            rb3=search_w2v(q, w2v_model, w2v_vectors, documents, top_k=K_BONUS)
            rb4=search_w2v_bert(q, w2v_bert_model, documents, top_k=K_BONUS)

            ra1=search_tfidf(q, vec_imp, tfidf_imp, documents, top_k=K_BONUS)
            ra2=_search_tfidf_bert_improved(q, tfidf_bert_model, vec_imp, tfidf_imp, documents, K_BONUS)
            ra3=search_w2v(q, w2v_imp, w2v_vec_imp, documents, top_k=K_BONUS)
            ra4=_search_w2v_bert_improved(q, w2v_bert_model, w2v_imp, w2v_vec_imp, documents, K_BONUS)

            for mb,ma,rb,ra in zip(model_labels_before, model_labels_after,
                                   [rb1,rb2,rb3,rb4], [ra1,ra2,ra3,ra4]):
                fb_f = get_bonus_flags(rb, q, bonus_keywords)
                fa_f = get_bonus_flags(ra, q, bonus_keywords)
                pb,rb_,fb,_,_ = compute_metrics(fb_f, K_BONUS)
                pa,ra_,fa,_,_ = compute_metrics(fa_f, K_BONUS)
                before_prec[mb].append(pb); before_rec[mb].append(rb_); before_f1[mb].append(fb)
                after_prec[ma].append(pa);  after_rec[ma].append(ra_);  after_f1[ma].append(fa)

            progress2.progress((i+1)/len(bonus_queries))

        avg_bf1={m:round(float(np.mean(v)),4) for m,v in before_f1.items()}
        avg_af1={m:round(float(np.mean(v)),4) for m,v in after_f1.items()}
        avg_bp ={m:round(float(np.mean(v)),4) for m,v in before_prec.items()}
        avg_ap ={m:round(float(np.mean(v)),4) for m,v in after_prec.items()}
        avg_br ={m:round(float(np.mean(v)),4) for m,v in before_rec.items()}
        avg_ar ={m:round(float(np.mean(v)),4) for m,v in after_rec.items()}

        # ── Summary cards ─────────────────────────────────────────────────────
        st.markdown("### 📊 Average F1 — Before vs After")
        cols4 = st.columns(4)
        for col,mb,ma,color,sn in zip(cols4, model_labels_before, model_labels_after, bar_colors, short_names):
            with col:
                bf_=avg_bf1[mb]; af_=avg_af1[ma]; delta=af_-bf_
                sign="▲" if delta>0 else ("▼" if delta<0 else "—")
                badge="badge-up" if delta>0 else ("badge-down" if delta<0 else "badge-same")
                st.markdown(
                    f'<div class="metric-card" style="border-top:4px solid {color};">'
                    f'<b style="color:{color}">{sn.replace(chr(10)," ")}</b><br>'
                    f'Before F1: <b>{bf_:.4f}</b><br>After F1: <b>{af_:.4f}</b><br>'
                    f'<span class="improvement-badge {badge}">{sign} {abs(delta)*100:.1f}%</span>'
                    f'</div>', unsafe_allow_html=True)

        # ── Comparison table ───────────────────────────────────────────────────
        st.markdown("---"); st.markdown("### 📋 Detailed Comparison Table")
        rows=[]
        for mb,ma,sn in zip(model_labels_before,model_labels_after,short_names):
            rows.append({"Model":sn.replace("\n"," "),
                "P (Before)":avg_bp[mb],"P (After)":avg_ap[ma],"P Δ":round(avg_ap[ma]-avg_bp[mb],4),
                "R (Before)":avg_br[mb],"R (After)":avg_ar[ma],"R Δ":round(avg_ar[ma]-avg_br[mb],4),
                "F1 (Before)":avg_bf1[mb],"F1 (After)":avg_af1[ma],"F1 Δ":round(avg_af1[ma]-avg_bf1[mb],4)})
        df_cmp=pd.DataFrame(rows).set_index("Model")

        def color_delta(val):
            if isinstance(val,float):
                if val>0:  return "background-color:#064e3b;color:#ffffff !important"
                if val<0:  return "background-color:#7f1d1d;color:#ffffff !important"
            return ""

        st.dataframe(
            df_cmp.style.format("{:.4f}").map(color_delta, subset=["P Δ","R Δ","F1 Δ"]),
            use_container_width=True)

        # ── Plot: Grouped bar ──────────────────────────────────────────────────
        st.markdown("---"); st.markdown("### 📊 Before vs After F1 — Bar Chart")
        fig_ba,ax_ba=plt.subplots(figsize=(12,5))
        x_ba=np.arange(len(short_names)); w_ba=0.35
        b_vals=[avg_bf1[mb] for mb in model_labels_before]
        a_vals=[avg_af1[ma] for ma in model_labels_after]
        bars_b=ax_ba.bar(x_ba-w_ba/2, b_vals, w_ba, label="Before", color="#94a3b8", alpha=0.9)
        bars_a=ax_ba.bar(x_ba+w_ba/2, a_vals, w_ba, label="After",  color=bar_colors, alpha=0.9)
        for bar,val in zip(bars_b,b_vals):
            ax_ba.text(bar.get_x()+bar.get_width()/2, val+0.01, f"{val:.3f}",
                       ha="center",va="bottom",fontsize=9,color="#475569",fontweight="bold")
        for bar,val,col in zip(bars_a,a_vals,bar_colors):
            ax_ba.text(bar.get_x()+bar.get_width()/2, val+0.01, f"{val:.3f}",
                       ha="center",va="bottom",fontsize=9,color=col,fontweight="bold")
        ax_ba.set_xticks(x_ba)
        ax_ba.set_xticklabels([sn.replace("\n"," ") for sn in short_names],fontsize=10)
        ax_ba.set_ylabel(f"Avg F1@{K_BONUS}"); ax_ba.set_ylim(0,1.2)
        ax_ba.set_title("F1 Before vs After",fontsize=13,fontweight="bold")
        ax_ba.legend(fontsize=10); ax_ba.yaxis.grid(True,linestyle="--",alpha=0.5)
        ax_ba.set_axisbelow(True); ax_ba.spines[["top","right"]].set_visible(False)
        plt.tight_layout(); st.pyplot(fig_ba); plt.close()

        # ── Plot: Per-query line chart ─────────────────────────────────────────
        st.markdown("---"); st.markdown("### 📈 F1 per Query — Before (dashed) vs After (solid)")
        fig_l,axes_l=plt.subplots(2,2,figsize=(14,8)); axes_l=axes_l.flatten()
        short_q_b=[q[:22]+"…" if len(q)>22 else q for q in bonus_queries]
        for idx,(mb,ma,sn,color) in enumerate(zip(model_labels_before,model_labels_after,short_names,bar_colors)):
            ax=axes_l[idx]
            bv=before_f1[mb]; av=after_f1[ma]
            ax.plot(range(len(bonus_queries)),bv,"o--",color="#94a3b8",linewidth=2,label="Before",markersize=7)
            ax.plot(range(len(bonus_queries)),av,"o-", color=color,    linewidth=2.5,label="After", markersize=7)
            for xi,(bval,aval) in enumerate(zip(bv,av)):
                fc="#dcfce7" if aval>=bval else "#fee2e2"
                ax.fill_between([xi-0.1,xi+0.1],[bval,bval],[aval,aval],color=fc,alpha=0.5)
            ax.set_xticks(range(len(bonus_queries)))
            ax.set_xticklabels(short_q_b,rotation=15,ha="right",fontsize=8)
            ax.set_ylim(-0.05,1.15); ax.set_ylabel("F1")
            ax.set_title(sn.replace("\n"," "),fontweight="bold",color=color)
            ax.legend(fontsize=9); ax.spines[["top","right"]].set_visible(False)
            ax.yaxis.grid(True,linestyle="--",alpha=0.4)
        plt.suptitle(f"F1@{K_BONUS} Before vs After — Per Query",fontsize=13,fontweight="bold")
        plt.tight_layout(); st.pyplot(fig_l); plt.close()

        # ── Conclusion ─────────────────────────────────────────────────────────
        st.markdown("---"); st.markdown("### 🏁 Conclusion")
        improved_count = sum(1 for mb,ma in zip(model_labels_before,model_labels_after)
                             if avg_af1[ma]>avg_bf1[mb])
        avg_delta = np.mean([avg_af1[ma]-avg_bf1[mb]
                             for mb,ma in zip(model_labels_before,model_labels_after)])
        if avg_delta > 0:
            st.success(f"✅ {improved_count}/4 models improved. Avg F1 gain: +{avg_delta*100:.1f}%")
        else:
            st.info("📊 Mixed results — expected on small evaluation set.")
        st.markdown("""
**Key Takeaways:**
- **TF-IDF**: Bigrams + sublinear TF better capture multi-word food concepts
- **W2V**: Skip-gram with 200d learns richer domain-specific semantics
- **BERT models**: Benefit from better candidate retrieval in Stage 1
- **Lesson**: Even small hyperparameter changes have measurable IR impact
""")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ABOUT
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("""
### 📚 About the Project
**Dataset:** Amazon Fine Food Reviews – first 3,000 reviews

---
### ⚙️ Preprocessing
Lowercase → Remove punctuation/numbers → Tokenize → Remove stopwords → Lemmatize

---
### 🤖 The 4 Models
| # | Feature | Model | Type |
|---|---------|-------|------|
| 1 | TF-IDF | Cosine Similarity | Baseline |
| 2 | TF-IDF | BERT Cross-Encoder | Advanced |
| 3 | Word2Vec (avg) | Cosine Similarity | Baseline |
| 4 | Word2Vec (avg) | BERT Cross-Encoder | Advanced |

---
### 🚀 Bonus Improvements
| Model | What Changed |
|-------|-------------|
| TF-IDF Cosine | bigrams, sublinear TF, max_df=0.85 |
| TF-IDF BERT | improved TF-IDF retriever, candidate_k=100 |
| W2V Cosine | 200d Skip-gram, window=3, 30 epochs |
| W2V BERT | improved W2V retriever, candidate_k=100 |

---
### 👩‍💻 Team
| Member | Role |
|--------|------|
| Gehad | Project Manager · Data Loader · tfidf_bert_search · w2v_bert_search |
| Alaa | Text Preprocessing |
| Waad | TF-IDF Baseline Search |
| Aliaa | Word2Vec Cosine Similarity |
| Sama | Evaluation & Comparison |
| Aya | Report & Demo Notebook |

**Capital University – Faculty of Computing & AI | NLP Course 2025-2026**
""")

st.markdown("---")
st.caption("👩‍💻 NLP Project 2 | Capital University | Spring 2025-2026")
