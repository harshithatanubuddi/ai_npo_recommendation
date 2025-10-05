# AI NPO Recommender

AI NPO Recommender is an AI-powered recommendation and sentiment analysis platform for
non-profit organizations (NPOs). This repository contains preprocessing, supervised ML
recommendation models for donors and volunteers, and a Streamlit app to demo the system.

## What’s included

- `data/` — contains `npo_dataset_hybrid.csv` (provided by the user) and processed outputs.
- `src/` — source modules: preprocessing, donor/volunteer models, sentiment, geo utils.
- `models/` — saved model artifacts (will be populated after running the pipeline).
- `app/streamlit_app.py` — Streamlit web app (entrypoint).
- `run_pipeline.py` — runs preprocessing, trains models, and saves artifacts.
- `requirements.txt` — Python dependencies.

## Quick start

1. Create a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # Linux / macOS
.venv\Scripts\activate    # Windows (PowerShell)
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the pipeline (preprocess data & train models):
```bash
python run_pipeline.py
```

4. Start the Streamlit app:
```bash
streamlit run app/streamlit_app.py
```

## Notes
- The dataset included here is synthetic and meant for prototyping. Replace with real data for production.
- For better recommendations, consider using semantic embeddings (`sentence-transformers`) and collecting interaction logs for collaborative filtering.

---
Created for the user. Title set to **AI NPO Recommender**.
