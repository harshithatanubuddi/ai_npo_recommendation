# streamlit_app.py

import sys
import os
import streamlit as st
import pandas as pd

# -------------------------------
# Fix module imports
# -------------------------------
# Add parent folder to path so 'src' can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.donor_recommender_ml import recommend_for_donor, train_donor_model
from src.volunteer_recommender_ml import recommend_for_volunteer, train_volunteer_model
from src.sentiment_analysis import analyze_sentiment
from src.data_preprocessing import preprocess_data

# -------------------------------
# Streamlit page setup
# -------------------------------
st.set_page_config(page_title="AI NPO Recommender", layout="wide")
st.title("AI NPO Recommender")

# -------------------------------
# Ensure models exist
# -------------------------------
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

donor_model_path = "models/donor_recommender.joblib"
donor_vectorizer_path = "models/donor_vectorizer.joblib"
volunteer_model_path = "models/volunteer_recommender.joblib"

# Check if donor or volunteer models exist
if not (os.path.exists(donor_model_path) and os.path.exists(donor_vectorizer_path) and os.path.exists(volunteer_model_path)):
    st.info("Training models... This may take a minute.")
    # Load raw CSV
    csv_path = "data/npo_dataset_hybrid.csv"
    df, _, _ = preprocess_data(csv_path)
    # Train models
    train_donor_model(df)
    train_volunteer_model(df)
    # Analyze sentiment
    df = analyze_sentiment(df)
    # Save processed CSV
    df.to_csv("data/npo_dataset_processed.csv", index=False)

# -------------------------------
# Load dataset
# -------------------------------
processed_csv_path = "data/npo_dataset_processed.csv"
if os.path.exists(processed_csv_path):
    df = pd.read_csv(processed_csv_path)
else:
    df = pd.read_csv("data/npo_dataset_hybrid.csv")

# -------------------------------
# Streamlit UI
# -------------------------------
menu = st.sidebar.selectbox("Mode", ["Donor", "Volunteer", "NPO Overview"])

if menu == "Donor":
    interests = st.text_input("Enter your interests (comma separated):", "education, healthcare")
    if st.button("Recommend NPOs"):
        results = recommend_for_donor(interests, df)
        st.write("Top recommended NPOs:")
        st.dataframe(results)

elif menu == "Volunteer":
    skills = st.text_input("Enter your skills (comma separated):", "teaching, first_aid")
    availability = st.selectbox("Availability", ["weekdays", "weekends", "anytime"])
    if st.button("Find Opportunities"):
        results = recommend_for_volunteer(skills, availability, df)
        st.write("Top volunteering opportunities:")
        st.dataframe(results)

else:
    st.write("NPO Feedback Sentiment Summary")
    df2 = analyze_sentiment(df)
    st.write(df2["sentiment"].value_counts())
