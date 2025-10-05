import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

def preprocess_data(csv_path: str):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    for col in ["description", "tags", "required_skills", "feedback", "name"]:
        if col in df.columns:
            df[col] = df[col].fillna("")

    # Synthetic features (for supervised training)
    np.random.seed(42)
    if "tags" in df.columns:
        df["donor_interest"] = df["tags"].apply(lambda x: x.split(",")[0] if isinstance(x, str) and x.strip() != "" else "general")
    else:
        df["donor_interest"] = "general"
    df["volunteer_skills"] = df["required_skills"] if "required_skills" in df.columns else ""
    df["availability"] = np.random.choice(["weekdays", "weekends", "anytime"], size=len(df))
    # Synthetic labels for training (replace with real interactions when available)
    df["donated"] = np.random.choice([0, 1], size=len(df), p=[0.6, 0.4])
    df["matched"] = np.random.choice([0, 1], size=len(df), p=[0.5, 0.5])

    df["combined_text"] = df.get("description", "") + " " + df.get("tags", "")

    tfidf = TfidfVectorizer(stop_words="english", max_features=1500)
    tfidf_matrix = tfidf.fit_transform(df["combined_text"].astype(str))

    joblib.dump(df, "models/processed_df.joblib")
    joblib.dump(tfidf, "models/tfidf_vectorizer.joblib")
    joblib.dump(tfidf_matrix, "models/tfidf_matrix.joblib")

    df.to_csv("data/npo_dataset_processed.csv", index=False)
    print(f"[INFO] Processed {len(df)} records. TF-IDF vocab size: {len(tfidf.vocabulary_)}")
    return df, tfidf, tfidf_matrix
