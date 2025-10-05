import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

def train_volunteer_model(df: pd.DataFrame):
    df = df.copy()
    df["skill_overlap"] = df.apply(
        lambda r: len(set(str(r.get("volunteer_skills", "")).split()) & set(str(r.get("required_skills", "")).split())), axis=1)
    df["availability_match"] = df["availability"].apply(lambda x: 1 if x == "anytime" else 0)
    # Synthetic distance (replace with real geo distances when available)
    df["distance_km"] = np.random.randint(1, 100, len(df))

    X = df[["skill_overlap", "availability_match", "distance_km"]]
    y = df["matched"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("Volunteer Model Performance:")
    print(classification_report(y_test, preds))

    joblib.dump(model, "models/volunteer_recommender.joblib")
    return model

def recommend_for_volunteer(skills: str, availability: str, df: pd.DataFrame, top_k=5):
    model = joblib.load("models/volunteer_recommender.joblib")
    df = df.copy()
    df["skill_overlap"] = df.get("required_skills","").apply(lambda x: len(set(str(x).split()) & set(skills.split())))
    df["availability_match"] = 1 if availability == "anytime" else 0
    df["distance_km"] = np.random.randint(1, 100, len(df))
    features = df[["skill_overlap","availability_match","distance_km"]]
    probs = model.predict_proba(features)[:, 1]
    df["match_score"] = probs
    return df.sort_values("match_score", ascending=False).head(top_k)[["name","required_skills","match_score"]]
