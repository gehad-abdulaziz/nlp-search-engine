import pandas as pd

def load_data(filepath="data/Reviews.csv", n_samples=3000):
    df = pd.read_csv(filepath, nrows=n_samples)
    df = df[["Text", "Summary", "Score"]].dropna()
    df["full_review"] = df["Summary"] + ". " + df["Text"]
    return df

def get_documents(df):
    return df["full_review"].tolist()

def save_sample(df, out_path="data/sample_500.csv"):
    df.head(500).to_csv(out_path, index=False)

if __name__ == "__main__":
    df = load_data()
    print(f"Loaded {len(df)} reviews")
    save_sample(df)
    print("Sample saved to data/sample_500.csv")