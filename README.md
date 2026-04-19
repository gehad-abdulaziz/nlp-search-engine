# Food Reviews Semantic Search Engine

A smart search engine that finds relevant food reviews based on meaning, not just exact keywords.

## Example
Query: "spicy food with bad service"
Returns reviews like "terrible experience, way too hot" even if the exact words are different.

## Dataset
Amazon Fine Food Reviews - 3000 reviews
Source: https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews

## Project Structure
```
nlp-search-engine/
├── data/
│   └── Reviews.csv          (download from Kaggle - not on GitHub)
├── src/
│   ├── data_loader.py        (Gehad)
│   ├── preprocessing.py      (Alaa)
│   ├── tfidf_search.py       (Waad)
│   ├── embedding_search.py   (Aliaa)
│   └── evaluation.py         (Sama)
├── notebooks/
│   └── demo.ipynb            (Aya)
├── main.py
├── requirements.txt
└── README.md
```

## Team
| Member | Role |
|--------|------|
| Gehad | Project Manager + Data Loader + GitHub Setup |
| Alaa | Text Preprocessing |
| Waad | TF-IDF Baseline Search |
| Aliaa | Embedding Advanced Search |
| Sama | Evaluation and Comparison |
| Aya | Report + Demo Notebook |

## How to Run
```bash
git clone https://github.com/gehad-abdulaziz/nlp-search-engine
cd nlp-search-engine
python -m pip install -r requirements.txt
python main.py
```

## Models
- Baseline: TF-IDF + Cosine Similarity
- Advanced: Sentence Transformers (BERT embeddings)

## Evaluation
- Metric: Precision@5
- Comparison between TF-IDF and Embedding models
