import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon', quiet=True)

def analyze_sentiment(df: pd.DataFrame):
    df = df.copy()
    if "feedback" not in df.columns:
        df["feedback"] = ""
    sid = SentimentIntensityAnalyzer()
    scores = df["feedback"].fillna("").astype(str).apply(lambda t: sid.polarity_scores(t)["compound"] if t.strip() else None)
    def label(x):
        if x is None: return "no_feedback"
        if x >= 0.05: return "positive"
        if x <= -0.05: return "negative"
        return "neutral"
    df["sentiment"] = scores.apply(label)
    return df
