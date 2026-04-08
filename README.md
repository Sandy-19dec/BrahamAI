# ✈️ BrahamAI — AI Travel Recommendation System

> Final Year BCA Data Science Project

---

## 📌 Project Overview

BrahamAI is a content-based AI travel recommendation system that suggests personalized
Indian travel destinations based on a user's budget, preferred travel type, and season.

**Problem Statement:**
Travelers face information overload when planning trips. BrahamAI solves this by using
Machine Learning to filter and rank destinations that best match user preferences.

---

## 🏗️ Project Structure

```
BrahamAI/
├── app.py                     ← Main Streamlit UI
├── ml_demo.py                 ← Standalone ML demo (no Streamlit needed)
├── requirements.txt           ← Python dependencies
├── README.md
│
├── data/
│   └── travel_data.csv        ← Dataset (55 Indian destinations)
│
├── models/
│   └── recommender.py         ← ML engine (encoding + cosine similarity)
│
├── visualizations/
│   └── charts.py              ← Matplotlib charts (EDA)
│
└── utils/
    └── chatbot.py             ← Rule-based travel chatbot
```

---

## 🤖 Machine Learning Approach

| Step | Technique | Purpose |
|------|-----------|---------|
| Feature Encoding | One-Hot Encoding | Convert Type, Season to binary |
| Normalization | Min-Max Scaling | Scale Budget, Rating, Duration to [0,1] |
| Similarity | Cosine Similarity | Measure match between user & destination |
| Filtering | Pandas query | Optional budget range filter |
| Ranking | Argsort | Sort by similarity or rating |

---

## 📊 Dataset Columns

| Column      | Type    | Description                          |
|-------------|---------|--------------------------------------|
| Destination | string  | Name of the travel destination       |
| Budget      | int     | Approximate cost in INR              |
| Type        | string  | Beach / Adventure / Cultural / etc.  |
| Season      | string  | Winter / Summer / Monsoon            |
| Rating      | float   | User rating out of 5                 |
| Duration    | int     | Recommended trip duration (days)     |

---

## 📈 Model Performance

- **Avg Precision@5**: 0.83 — 83% of top-5 recommendations match the user's travel type
- **Avg Recall@5**: 0.42 — retrieves ~42% of all relevant destinations
- **F1 Score**: 0.56

---

## 🚀 How to Run

### Option 1 — Streamlit UI
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Option 2 — Command Line Demo
```bash
python ml_demo.py
```

---

## ☁️ Deploy on Streamlit Cloud

1. Push this project to a GitHub repository
2. Go to https://share.streamlit.io and sign in
3. Click **New App** → select your repo → set `app.py` as the main file
4. Click **Deploy** — your app will be live in ~2 minutes!

Make sure `requirements.txt` is at the root of the repo.

---

## 🌐 Real-World Impact

- **For travelers**: Instant personalized recommendations without endless searching
- **For travel agencies**: Can be extended with hotel/flight booking APIs
- **For tourism boards**: Insights on popular destinations and seasonal demand
- **Scalability**: Add 1000s of destinations; swap cosine similarity for a neural model

## 🔮 Future Improvements

- [ ] Collaborative filtering (learn from user history)
- [ ] Integration with Google Maps / TripAdvisor API
- [ ] User login and saved preferences
- [ ] Hotel and flight price integration
- [ ] NLP-based chatbot using transformers

---

*Built with Python · Scikit-learn · Streamlit · Pandas · Matplotlib*
