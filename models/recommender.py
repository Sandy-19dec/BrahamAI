"""
BrahamAI — Recommendation Engine
Content-Based Filtering using One-Hot Encoding + Cosine Similarity
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity


# ─────────────────────────────────────────
# 1. Load & Prepare Data
# ─────────────────────────────────────────

def load_data(path="data/travel_data.csv"):
    """Load the travel dataset."""
    df = pd.read_csv(path)
    return df


def build_feature_matrix(df):
    """
    Build the feature matrix for content-based filtering.

    Steps:
    1. One-Hot Encode categorical columns (Type, Season)
    2. Normalize numeric columns (Budget, Rating, Duration)
    3. Concatenate into a single feature matrix
    """

    # --- One-Hot Encoding for categorical features ---
    type_dummies   = pd.get_dummies(df["Type"],   prefix="type")
    season_dummies = pd.get_dummies(df["Season"], prefix="season")

    # --- Normalize numeric features to [0, 1] ---
    scaler = MinMaxScaler()
    numeric = pd.DataFrame(
        scaler.fit_transform(df[["Budget", "Rating", "Duration"]]),
        columns=["Budget_norm", "Rating_norm", "Duration_norm"]
    )

    # --- Combine all features ---
    feature_matrix = pd.concat(
        [type_dummies, season_dummies, numeric],
        axis=1
    )

    return feature_matrix, scaler


def compute_similarity(feature_matrix):
    """Compute cosine similarity between all destination pairs."""
    similarity = cosine_similarity(feature_matrix)
    return similarity


# ─────────────────────────────────────────
# 2. Build User Profile Vector
# ─────────────────────────────────────────

def build_user_vector(budget, travel_type, season, feature_matrix, scaler, df):
    """
    Encode user preferences the same way as the dataset,
    then return a 1-row vector aligned to feature_matrix columns.
    """

    # All possible categories (from training data)
    all_types   = [c for c in feature_matrix.columns if c.startswith("type_")]
    all_seasons = [c for c in feature_matrix.columns if c.startswith("season_")]

    user_dict = {}

    # One-hot type
    for col in all_types:
        user_dict[col] = 1 if col == f"type_{travel_type}" else 0

    # One-hot season
    for col in all_seasons:
        user_dict[col] = 1 if col == f"season_{season}" else 0

    # Normalize numeric features using the fitted scaler
    # We need to pass all 3 features; use median rating/duration as neutral
    median_rating   = df["Rating"].median()
    median_duration = df["Duration"].median()

    normalized = scaler.transform([[budget, median_rating, median_duration]])[0]
    user_dict["Budget_norm"]   = normalized[0]
    user_dict["Rating_norm"]   = normalized[1]
    user_dict["Duration_norm"] = normalized[2]

    user_vector = pd.DataFrame([user_dict])[feature_matrix.columns]
    return user_vector


# ─────────────────────────────────────────
# 3. Recommendation Function
# ─────────────────────────────────────────

def recommend(budget, travel_type, season,
              top_n=5, min_budget=None, max_budget=None, sort_by_rating=False,
              data_path="data/travel_data.csv"):
    """
    Return top-N travel recommendations based on user preferences.

    Parameters:
        budget       : int   — user's travel budget (INR)
        travel_type  : str   — e.g. 'Beach', 'Adventure', 'Cultural'
        season       : str   — e.g. 'Winter', 'Summer', 'Monsoon'
        top_n        : int   — number of recommendations
        min_budget   : int   — optional budget filter (lower bound)
        max_budget   : int   — optional budget filter (upper bound)
        sort_by_rating: bool — if True, sort results by rating instead of similarity
        data_path    : str   — path to CSV

    Returns:
        DataFrame with recommended destinations and similarity scores
    """

    df = load_data(data_path)

    # Optional budget filter
    filtered_df = df.copy()
    if min_budget is not None:
        filtered_df = filtered_df[filtered_df["Budget"] >= min_budget]
    if max_budget is not None:
        filtered_df = filtered_df[filtered_df["Budget"] <= max_budget]

    if filtered_df.empty:
        return pd.DataFrame(columns=["Destination","Budget","Type","Season","Rating","Duration","Similarity"])

    filtered_df = filtered_df.reset_index(drop=True)

    # Build feature matrix on filtered data
    feature_matrix, scaler = build_feature_matrix(filtered_df)

    # Build user vector
    user_vector = build_user_vector(budget, travel_type, season, feature_matrix, scaler, filtered_df)

    # Cosine similarity between user and all destinations
    scores = cosine_similarity(user_vector, feature_matrix)[0]

    filtered_df["Similarity"] = np.round(scores, 4)

    if sort_by_rating:
        result = filtered_df.sort_values("Rating", ascending=False).head(top_n)
    else:
        result = filtered_df.sort_values("Similarity", ascending=False).head(top_n)

    return result[["Destination", "Budget", "Type", "Season", "Rating", "Duration", "Similarity"]]


# ─────────────────────────────────────────
# 4. Evaluation Helpers
# ─────────────────────────────────────────

def evaluate_model(data_path="data/travel_data.csv"):
    """
    Simulate precision/recall evaluation.

    Strategy:
    - For each destination, treat its own Type+Season as the 'query'
    - A recommendation is 'relevant' if it shares the same Type
    - Compute precision@5 across all destinations
    """

    df = load_data(data_path)
    feature_matrix, scaler = build_feature_matrix(df)
    sim_matrix = compute_similarity(feature_matrix)

    precisions = []
    recalls    = []

    for i, row in df.iterrows():
        true_type = row["Type"]
        # Get top-5 most similar (excluding itself)
        sim_scores = list(enumerate(sim_matrix[i]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        top5 = [idx for idx, _ in sim_scores[1:6]]   # skip index 0 (itself)

        # Relevant = same Type
        relevant_in_dataset = df[df["Type"] == true_type].index.tolist()
        relevant_in_dataset = [x for x in relevant_in_dataset if x != i]

        hits = sum(1 for idx in top5 if df.loc[idx, "Type"] == true_type)
        precision = hits / 5
        recall    = hits / len(relevant_in_dataset) if relevant_in_dataset else 0

        precisions.append(precision)
        recalls.append(recall)

    avg_precision = np.mean(precisions)
    avg_recall    = np.mean(recalls)
    f1            = (2 * avg_precision * avg_recall / (avg_precision + avg_recall + 1e-9))

    return {
        "Avg Precision@5": round(avg_precision, 4),
        "Avg Recall@5"   : round(avg_recall, 4),
        "F1 Score"       : round(f1, 4),
        "Total Destinations": len(df)
    }
