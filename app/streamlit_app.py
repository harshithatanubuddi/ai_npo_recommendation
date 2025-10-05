import os
import streamlit as st
import pandas as pd
from src.donor_recommender_ml import recommend_for_donor
from src.volunteer_recommender_ml import recommend_for_volunteer

st.set_page_config(page_title="AI NPO Recommender", layout="wide")
st.title("AI NPO Recommender")

menu = st.sidebar.selectbox("Mode", ["Donor", "Volunteer", "NPO Overview"])

df = pd.read_csv("data/npo_dataset_processed.csv") if os.path.exists("data/npo_dataset_processed.csv") else pd.read_csv("data/npo_dataset_hybrid.csv")

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
    from src.sentiment_analysis import analyze_sentiment
    df2 = analyze_sentiment(df)
    st.write(df2["sentiment"].value_counts())
