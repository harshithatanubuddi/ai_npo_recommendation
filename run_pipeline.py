from src.data_preprocessing import preprocess_data
from src.donor_recommender_ml import train_donor_model
from src.volunteer_recommender_ml import train_volunteer_model
from src.sentiment_analysis import analyze_sentiment
import os

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    csv_path = "data/npo_dataset_hybrid.csv"
    print("[INFO] Preprocessing data...")
    df, tfidf, tfidf_matrix = preprocess_data(csv_path)

    print("[INFO] Analyzing sentiment...")
    df = analyze_sentiment(df)
    df.to_csv("data/npo_dataset_processed.csv", index=False)

    print("[INFO] Training donor model...")
    train_donor_model(df)

    print("[INFO] Training volunteer model...")
    train_volunteer_model(df)

    print("[DONE] Pipeline complete. Models saved to /models/")
