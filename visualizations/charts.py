"""
BrahamAI — Data Visualizations
Generates charts for EDA and model insights using Matplotlib & Seaborn
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os

# Use non-interactive backend (safe for Streamlit & scripts)
import matplotlib
matplotlib.use("Agg")

OUTPUT_DIR = "visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PALETTE = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
           "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9"]


def load_data(path="data/travel_data.csv"):
    return pd.read_csv(path)


# ─────────────────────────────────────────
# Chart 1: Travel Type Distribution (Pie + Bar)
# ─────────────────────────────────────────

def plot_type_distribution(df=None, save=True):
    if df is None:
        df = load_data()

    type_counts = df["Type"].value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#0F1117")

    # Pie chart
    axes[0].pie(
        type_counts.values,
        labels=type_counts.index,
        autopct="%1.1f%%",
        colors=PALETTE[:len(type_counts)],
        startangle=140,
        textprops={"color": "white", "fontsize": 11}
    )
    axes[0].set_title("Travel Type Distribution", color="white", fontsize=14, pad=15)
    axes[0].set_facecolor("#0F1117")

    # Bar chart
    bars = axes[1].bar(type_counts.index, type_counts.values,
                       color=PALETTE[:len(type_counts)], edgecolor="white", linewidth=0.5)
    axes[1].set_title("Destinations by Travel Type", color="white", fontsize=14)
    axes[1].set_xlabel("Travel Type", color="white")
    axes[1].set_ylabel("Number of Destinations", color="white")
    axes[1].tick_params(colors="white")
    axes[1].set_facecolor("#1E1E2E")
    for spine in axes[1].spines.values():
        spine.set_edgecolor("#444")
    for bar, val in zip(bars, type_counts.values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                     str(val), ha="center", va="bottom", color="white", fontsize=10)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/type_distribution.png"
    if save:
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0F1117")
    return fig


# ─────────────────────────────────────────
# Chart 2: Budget vs Rating (Scatter)
# ─────────────────────────────────────────

def plot_budget_vs_rating(df=None, save=True):
    if df is None:
        df = load_data()

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#0F1117")
    ax.set_facecolor("#1E1E2E")

    # Assign color per type
    types = df["Type"].unique()
    color_map = {t: PALETTE[i] for i, t in enumerate(types)}
    colors = df["Type"].map(color_map)

    scatter = ax.scatter(
        df["Budget"], df["Rating"],
        c=colors, s=df["Duration"] * 20,
        alpha=0.85, edgecolors="white", linewidths=0.4
    )

    # Labels for standout destinations
    for _, row in df[df["Rating"] >= 4.8].iterrows():
        ax.annotate(row["Destination"],
                    (row["Budget"], row["Rating"]),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=8, color="white", alpha=0.9)

    # Legend
    legend_patches = [mpatches.Patch(color=color_map[t], label=t) for t in types]
    ax.legend(handles=legend_patches, facecolor="#1E1E2E", labelcolor="white",
              edgecolor="#555", fontsize=9, title="Travel Type",
              title_fontsize=9)

    ax.set_title("Budget vs Rating  (bubble size = Duration)", color="white", fontsize=14)
    ax.set_xlabel("Budget (₹)", color="white")
    ax.set_ylabel("Rating (out of 5)", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/budget_vs_rating.png"
    if save:
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0F1117")
    return fig


# ─────────────────────────────────────────
# Chart 3: Top 10 Destinations by Rating
# ─────────────────────────────────────────

def plot_top_destinations(df=None, save=True):
    if df is None:
        df = load_data()

    top10 = df.sort_values("Rating", ascending=False).drop_duplicates("Destination").head(10)

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#0F1117")
    ax.set_facecolor("#1E1E2E")

    bars = ax.barh(top10["Destination"], top10["Rating"],
                   color=PALETTE[:10], edgecolor="white", linewidth=0.4)
    ax.set_xlim(4.0, 5.05)
    ax.set_title("Top 10 Destinations by Rating", color="white", fontsize=14)
    ax.set_xlabel("Rating (out of 5)", color="white")
    ax.tick_params(colors="white")
    ax.invert_yaxis()
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    for bar, val in zip(bars, top10["Rating"]):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val}", va="center", color="white", fontsize=9)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/top_destinations.png"
    if save:
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0F1117")
    return fig


# ─────────────────────────────────────────
# Chart 4: Season-wise Destination Count
# ─────────────────────────────────────────

def plot_season_distribution(df=None, save=True):
    if df is None:
        df = load_data()

    season_counts = df["Season"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#0F1117")
    ax.set_facecolor("#1E1E2E")

    wedges, texts, autotexts = ax.pie(
        season_counts.values,
        labels=season_counts.index,
        autopct="%1.1f%%",
        colors=["#45B7D1", "#FF6B6B", "#96CEB4"],
        startangle=90,
        textprops={"color": "white", "fontsize": 12},
        wedgeprops={"edgecolor": "#0F1117", "linewidth": 2}
    )
    ax.set_title("Destinations by Season", color="white", fontsize=14)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/season_distribution.png"
    if save:
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0F1117")
    return fig


# ─────────────────────────────────────────
# Generate all charts at once
# ─────────────────────────────────────────

def generate_all_charts():
    df = load_data()
    plot_type_distribution(df)
    plot_budget_vs_rating(df)
    plot_top_destinations(df)
    plot_season_distribution(df)
    print("✅  All charts saved to /visualizations/")


if __name__ == "__main__":
    generate_all_charts()
