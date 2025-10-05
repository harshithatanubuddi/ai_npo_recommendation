# src/donor_recommender_ml.py

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer


def train_donor_model(df: pd.DataFrame):
    df = df.copy()
    
    # Safe concatenation of donor_interest and tags
    if "tags" in df.columns:
        df["text_features"] = df["donor_interest"].astype(str) + " " + df["tags"].astype(str)
    else:
        df["text_features"] = df["donor_interest"].astype(str)

    y = df["donated"].astype(int)

    vectorizer = TfidfVectorizer(stop_words='english', max_features=1500)
    X = vectorizer.fit_transform(df["text_features"].astype(str))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("Donor Model Performance:")
    print(classification_report(y_test, preds))

    # Save models
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/donor_recommender.joblib")
    joblib.dump(vectorizer, "models/donor_vectorizer.joblib")

    return model, vectorizer


def recommend_for_donor(user_interest: str, df: pd.DataFrame, top_k=5):
    model_path = "models/donor_recommender.joblib"
    vectorizer_path = "models/donor_vectorizer.joblib"

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    # Prepare candidate texts safely
    if "tags" in df.columns:
        candidate_texts = df["donor_interest"].astype(str) + " " + df["tags"].astype(str)
    else:
        candidate_texts = df["donor_interest"].astype(str)

    X_candidates = vectorizer.transform(candidate_texts)
    X_query = vectorizer.transform([user_interest])

    # Compute match probabilities
    probs = model.predict_proba(X_candidates)[:, 1]

    df = df.copy()
    df["match_score"] = probs

    return df.sort_values("match_score", ascending=False).head(top_k)[["name", "tags" if "tags" in df.columns else "donor_interest", "match_score"]]
