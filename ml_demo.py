"""
BrahamAI — Standalone ML Demo
Run: python ml_demo.py

No Streamlit required. Demonstrates the ML pipeline end-to-end:
  1. Load data
  2. Build feature matrix (OHE + MinMax scaling)
  3. Build user profile vector
  4. Compute cosine similarity
  5. Return ranked recommendations
  6. Print evaluation metrics (Precision@5, Recall@5, F1)
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from models.recommender import (
    load_data,
    build_feature_matrix,
    build_user_vector,
    compute_similarity,
    recommend,
    evaluate_model,
)

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "travel_data.csv")

DIVIDER = "─" * 65


def print_header(title):
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


def demo_data_overview():
    """Print a summary of the dataset."""
    print_header("📊 Dataset Overview")
    df = load_data(DATA_PATH)
    print(f"  Total Destinations : {len(df.drop_duplicates('Destination'))}")
    print(f"  Travel Types       : {', '.join(df['Type'].unique())}")
    print(f"  Seasons            : {', '.join(df['Season'].unique())}")
    print(f"  States Covered     : {df['State'].nunique()}")
    print(f"  Budget Range       : ₹{int(df['Budget'].min()):,} — ₹{int(df['Budget'].max()):,}")
    print(f"  Rating Range       : {df['Rating'].min()} — {df['Rating'].max()}")
    print(f"\n  Type Distribution:")
    for t, c in df["Type"].value_counts().items():
        bar = "█" * int(c / 2)
        print(f"    {t:<12} {bar} ({c})")


def demo_feature_matrix():
    """Show the feature matrix structure."""
    print_header("🧮 Feature Matrix (first 5 rows)")
    df = load_data(DATA_PATH)
    fm, scaler = build_feature_matrix(df)
    print(f"  Shape: {fm.shape}  (rows=destinations, cols=features)")
    print(f"  Columns: {list(fm.columns)}\n")
    print(fm.head(5).round(3).to_string(index=False))


def demo_recommendations(budget, travel_type, season, top_n=5):
    """Show top recommendations for given preferences."""
    print_header(f"🤖 Recommendations — {travel_type} | {season} | ₹{budget:,}")
    results = recommend(
        budget=budget,
        travel_type=travel_type,
        season=season,
        top_n=top_n,
        data_path=DATA_PATH,
    )
    if results.empty:
        print("  ⚠️  No destinations found. Try adjusting filters.")
        return

    for rank, (_, row) in enumerate(results.iterrows(), 1):
        score_pct   = int(row.get("Score", row["Similarity"]) * 100)
        sim_pct     = int(row["Similarity"] * 100)
        stars       = "★" * int(row["Rating"])
        print(
            f"  #{rank}  {row['Destination']:<22} "
            f"₹{int(row['Budget']):>7,}  "
            f"{stars} {row['Rating']}  "
            f"AI Score: {score_pct}%  (Sim: {sim_pct}%)"
        )


def demo_similarity_matrix():
    """Print 5×5 cosine similarity sample."""
    print_header("📐 Cosine Similarity Sample (5×5)")
    df = load_data(DATA_PATH)
    fm, _ = build_feature_matrix(df)
    sim = compute_similarity(fm)
    dest_names = df["Destination"].drop_duplicates().values[:5]
    sim_df = pd.DataFrame(sim[:5, :5], index=dest_names, columns=dest_names).round(3)
    print(sim_df.to_string())


def demo_evaluation():
    """Print model evaluation metrics."""
    print_header("📈 Model Evaluation")
    metrics = evaluate_model(DATA_PATH)
    print(f"  Avg Precision@5   : {metrics['Avg Precision@5']}")
    print(f"  Avg Recall@5      : {metrics['Avg Recall@5']}")
    print(f"  F1 Score          : {metrics['F1 Score']}")
    print(f"  Total Destinations: {metrics['Total Destinations']}")
    if "Per Type Precision" in metrics:
        print(f"\n  Per-Type Precision@5:")
        for t, p in metrics["Per Type Precision"].items():
            bar = "█" * int(p * 20)
            print(f"    {t:<12} {bar:<20} {p:.2f}")


if __name__ == "__main__":
    print("\n" + "═" * 65)
    print("  ✈️  BrahamAI — AI Travel Recommendation System")
    print("  Standalone ML Demo · BCA Final Year Project")
    print("═" * 65)

    demo_data_overview()
    demo_feature_matrix()

    # --- Example 1: Beach trip in winter, ₹15,000 budget ---
    demo_recommendations(budget=15000, travel_type="Beach", season="Winter", top_n=5)

    # --- Example 2: Adventure in summer, ₹25,000 budget ---
    demo_recommendations(budget=25000, travel_type="Adventure", season="Summer", top_n=5)

    # --- Example 3: Nature in monsoon, ₹12,000 budget ---
    demo_recommendations(budget=12000, travel_type="Nature", season="Monsoon", top_n=5)

    demo_similarity_matrix()
    demo_evaluation()

    print(f"\n{DIVIDER}")
    print("  ✅ Demo complete. Run 'streamlit run app.py' for the full UI.")
    print(DIVIDER + "\n")
