"""
BrahamAI — Standalone ML Demo Script
Run this without Streamlit to test the recommendation engine directly.
Usage: python ml_demo.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
from models.recommender import recommend, evaluate_model, load_data, build_feature_matrix, compute_similarity

# ─────────────────────────────────────────
# Pretty print helper
# ─────────────────────────────────────────
def separator(title=""):
    print("\n" + "═" * 60)
    if title:
        print(f"  {title}")
        print("═" * 60)

# ─────────────────────────────────────────
# 1. Load & Preview Dataset
# ─────────────────────────────────────────
separator("📂 DATASET PREVIEW")
df = load_data("data/travel_data.csv")
print(f"\nTotal Destinations: {len(df)}")
print(f"Columns          : {list(df.columns)}")
print(f"\nFirst 8 rows:\n")
print(df.head(8).to_string(index=False))
print(f"\nDataset Statistics:\n")
print(df.describe().round(2))

# ─────────────────────────────────────────
# 2. Feature Matrix (One-Hot + Normalization)
# ─────────────────────────────────────────
separator("⚙️  FEATURE MATRIX (first 5 rows)")
feature_matrix, scaler = build_feature_matrix(df)
print(f"\nFeature matrix shape: {feature_matrix.shape}")
print(f"Columns: {list(feature_matrix.columns)}\n")
print(feature_matrix.head(5).round(3).to_string())

# ─────────────────────────────────────────
# 3. Cosine Similarity Sample
# ─────────────────────────────────────────
separator("🔢 COSINE SIMILARITY (top-left 5×5)")
sim = compute_similarity(feature_matrix)
import numpy as np
sim_df = pd.DataFrame(sim[:5, :5],
                      index=df["Destination"][:5],
                      columns=df["Destination"][:5])
print(f"\n{sim_df.round(3).to_string()}")

# ─────────────────────────────────────────
# 4. Recommendations — Test Cases
# ─────────────────────────────────────────
separator("🗺️  RECOMMENDATIONS — Test Case 1")
print("\n  User: Budget=₹15,000 | Type=Beach | Season=Winter")
result = recommend(15000, "Beach", "Winter", top_n=5, data_path="data/travel_data.csv")
print(result.to_string(index=False))

separator("🗺️  RECOMMENDATIONS — Test Case 2")
print("\n  User: Budget=₹25,000 | Type=Adventure | Season=Summer")
result2 = recommend(25000, "Adventure", "Summer", top_n=5, data_path="data/travel_data.csv")
print(result2.to_string(index=False))

separator("🗺️  RECOMMENDATIONS — With Budget Filter")
print("\n  User: Budget=₹20,000 | Type=Cultural | Season=Winter | Max=₹18,000")
result3 = recommend(20000, "Cultural", "Winter", top_n=5,
                    max_budget=18000, data_path="data/travel_data.csv")
print(result3.to_string(index=False))

separator("🗺️  RECOMMENDATIONS — Sorted by Rating")
print("\n  User: Budget=₹15,000 | Type=Nature | Season=Monsoon | Sort=Rating")
result4 = recommend(15000, "Nature", "Monsoon", top_n=5,
                    sort_by_rating=True, data_path="data/travel_data.csv")
print(result4.to_string(index=False))

# ─────────────────────────────────────────
# 5. Model Evaluation
# ─────────────────────────────────────────
separator("📈 MODEL EVALUATION METRICS")
metrics = evaluate_model("data/travel_data.csv")
print()
for k, v in metrics.items():
    print(f"  {k:<30}: {v}")

print("\n  Interpretation:")
print("  • Precision@5: Of the 5 recommendations, how many share the user's travel type")
print("  • Recall@5   : Of all relevant destinations in the DB, how many did we retrieve")
print("  • F1 Score   : Harmonic mean of Precision and Recall")

# ─────────────────────────────────────────
# 6. Chatbot Demo
# ─────────────────────────────────────────
separator("🤖 CHATBOT DEMO")
from utils.chatbot import chatbot_response
test_messages = [
    "Hello",
    "Suggest a beach trip",
    "What are good adventure destinations?",
    "I have a low budget",
    "Best places in winter",
]
for msg in test_messages:
    print(f"\n  User : {msg}")
    print(f"  Bot  : {chatbot_response(msg)}")

separator("✅ DEMO COMPLETE")
print("\n  To launch the full Streamlit UI, run:")
print("  $ streamlit run app.py\n")
