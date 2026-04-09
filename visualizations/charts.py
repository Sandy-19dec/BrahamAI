"""
BrahamAI — Reusable Chart Functions
Dark-themed charts using Matplotlib and Plotly
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# ─────────────────────────────────────────
# Palette & Theme Constants
# ─────────────────────────────────────────

PALETTE = [
    "#667eea", "#764ba2", "#f0932b", "#34d399",
    "#f472b6", "#38bdf8", "#fb923c", "#a78bfa",
    "#fbbf24", "#6ee7b7"
]

TYPE_COLORS = {
    "Beach":     "#38bdf8",
    "Adventure": "#fb923c",
    "Cultural":  "#a78bfa",
    "Nature":    "#34d399",
    "Wildlife":  "#fbbf24",
    "City":      "#94a3b8",
}

DARK_BG = "#0d0d18"
DARK_AX = "#12121f"


def dark_fig(w=10, h=5):
    """Return a dark-themed matplotlib (fig, ax) pair."""
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_AX)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2a3a")
    ax.tick_params(colors="#888", labelsize=9)
    ax.xaxis.label.set_color("#888")
    ax.yaxis.label.set_color("#888")
    ax.title.set_color("white")
    return fig, ax


# ─────────────────────────────────────────
# Matplotlib Charts
# ─────────────────────────────────────────

def plot_type_distribution(df):
    """Pie chart — destination count by travel type."""
    type_counts = df["Type"].value_counts()
    fig, ax = dark_fig(6, 5)
    colors = list(TYPE_COLORS.values())
    wedges, texts, autotexts = ax.pie(
        type_counts.values,
        labels=type_counts.index,
        autopct="%1.1f%%",
        colors=colors,
        startangle=140,
        textprops={"color": "white", "fontsize": 10},
        wedgeprops={"edgecolor": DARK_BG, "linewidth": 2},
    )
    for at in autotexts:
        at.set_color("white")
        at.set_fontsize(9)
    ax.set_title("Destinations by Travel Type", color="white", pad=15)
    return fig


def plot_season_distribution(df):
    """Bar chart — destination count by season."""
    season_counts = df["Season"].value_counts()
    fig, ax = dark_fig(6, 5)
    bars = ax.bar(
        season_counts.index,
        season_counts.values,
        color=["#38bdf8", "#fb923c", "#6ee7b7"],
        edgecolor=DARK_BG,
        linewidth=2,
        width=0.5,
    )
    ax.set_title("Destinations by Season", color="white")
    for bar, val in zip(bars, season_counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            str(val), ha="center", color="white",
            fontsize=11, fontweight="600",
        )
    ax.set_ylim(0, season_counts.max() + 10)
    return fig


def plot_budget_vs_rating(df):
    """Bubble scatter — Budget vs Rating; bubble size = Duration."""
    fig, ax = dark_fig(12, 5)
    for ttype, group in df.groupby("Type"):
        ax.scatter(
            group["Budget"], group["Rating"],
            c=TYPE_COLORS.get(ttype, "#667eea"),
            s=group["Duration"] * 25,
            alpha=0.8, edgecolors="white", linewidths=0.3,
            label=ttype,
        )
    for _, row in df[df["Rating"] >= 4.85].iterrows():
        ax.annotate(
            row["Destination"], (row["Budget"], row["Rating"]),
            textcoords="offset points", xytext=(8, 4),
            fontsize=8, color="white", alpha=0.8,
        )
    ax.legend(
        facecolor=DARK_AX, labelcolor="white",
        edgecolor="#2a2a3a", fontsize=9,
        title="Travel Type", title_fontsize=9,
    )
    ax.set_xlabel("Budget (₹)", color="#888")
    ax.set_ylabel("Rating", color="#888")
    ax.set_title("Budget vs Rating — All Destinations", color="white")
    return fig


def plot_top_rated(df, n=15):
    """Horizontal bar — top N destinations by rating."""
    top_n = df.sort_values("Rating", ascending=False).drop_duplicates("Destination").head(n)
    fig, ax = dark_fig(7, 6)
    colors_bar = [PALETTE[i % len(PALETTE)] for i in range(len(top_n))]
    bars = ax.barh(top_n["Destination"], top_n["Rating"], color=colors_bar, edgecolor=DARK_BG, linewidth=1.5)
    ax.set_xlim(4.0, 5.1)
    ax.invert_yaxis()
    ax.set_title(f"Top {n} Rated Destinations", color="white")
    for bar, val in zip(bars, top_n["Rating"]):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2, f"{val}", va="center", color="white", fontsize=8.5)
    return fig


def plot_avg_budget_by_type(df):
    """Horizontal bar — average budget per travel type."""
    avg_budget = df.groupby("Type")["Budget"].mean().sort_values(ascending=True)
    fig, ax = dark_fig(7, 6)
    bar_colors = [TYPE_COLORS.get(t, "#667eea") for t in avg_budget.index]
    bars = ax.barh(avg_budget.index, avg_budget.values / 1000, color=bar_colors, edgecolor=DARK_BG, linewidth=1.5)
    ax.set_title("Average Budget by Type (₹ thousands)", color="white")
    for bar, val in zip(bars, avg_budget.values):
        ax.text(val / 1000 + 0.2, bar.get_y() + bar.get_height() / 2, f"₹{val/1000:.0f}K", va="center", color="white", fontsize=8.5)
    return fig


def plot_per_type_precision(per_type_precision: dict):
    """Bar chart — Precision@5 per travel type."""
    types  = list(per_type_precision.keys())
    values = list(per_type_precision.values())
    colors = [TYPE_COLORS.get(t, "#667eea") for t in types]
    fig, ax = dark_fig(8, 4)
    bars = ax.bar(types, values, color=colors, edgecolor=DARK_BG, linewidth=1.5, width=0.55)
    ax.set_ylim(0, 1.18)
    ax.axhline(y=0.8, color="#ffffff", alpha=0.2, linestyle="--", linewidth=1, label="0.80 target")
    ax.set_title("Precision@5 by Travel Type", color="white")
    ax.set_ylabel("Precision", color="#888")
    ax.legend(facecolor=DARK_AX, labelcolor="white", edgecolor="#2a2a3a", fontsize=9)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.025,
            f"{val:.2f}", ha="center", color="white", fontsize=10, fontweight="600",
        )
    return fig


def plot_duration_boxplot(df):
    """Box plot of trip duration by travel type."""
    fig, ax = dark_fig(10, 5)
    types = df["Type"].unique()
    data  = [df[df["Type"] == t]["Duration"].values for t in types]
    bp = ax.boxplot(data, labels=types, patch_artist=True, notch=False)
    colors = [TYPE_COLORS.get(t, "#667eea") for t in types]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for element in ["whiskers", "fliers", "caps"]:
        for item in bp[element]:
            item.set(color="#888", linewidth=1.2)
    for median in bp["medians"]:
        median.set(color="white", linewidth=2)
    ax.set_title("Trip Duration Distribution by Travel Type (days)", color="white")
    ax.set_ylabel("Days", color="#888")
    return fig


# ─────────────────────────────────────────
# Plotly Interactive Map
# ─────────────────────────────────────────

def plot_india_map(df, color_col="Type", title="🗺️ Indian Travel Destinations"):
    """
    Interactive Plotly map of destinations.
    Requires 'Lat' and 'Lon' columns in df.
    """
    if "Lat" not in df.columns or "Lon" not in df.columns:
        return None

    plot_df = df.drop_duplicates("Destination").copy()
    plot_df["BudgetStr"] = plot_df["Budget"].apply(lambda x: f"₹{int(x):,}")

    fig = px.scatter_geo(
        plot_df,
        lat="Lat",
        lon="Lon",
        color=color_col,
        size="Rating",
        hover_name="Destination",
        hover_data={
            "Type":       True,
            "Season":     True,
            "BudgetStr":  True,
            "Rating":     True,
            "State":      True,
            "Lat":        False,
            "Lon":        False,
        },
        color_discrete_map=TYPE_COLORS,
        size_max=20,
        title=title,
        template="plotly_dark",
        labels={"BudgetStr": "Budget"},
    )
    fig.update_geos(
        scope="asia",
        center={"lat": 23, "lon": 82},
        projection_scale=4.5,
        showland=True,    landcolor="#1a1a2e",
        showocean=True,   oceancolor="#0a0a0f",
        showlakes=True,   lakecolor="#0a0a0f",
        showcountries=True, countrycolor="#3a3a5a",
        showsubunits=True,  subunitcolor="#2a2a4a",
        bgcolor="#0a0a0f",
    )
    fig.update_layout(
        paper_bgcolor="#0a0a0f",
        plot_bgcolor="#0a0a0f",
        font_color="white",
        title_font_size=16,
        legend=dict(
            bgcolor="rgba(255,255,255,0.05)",
            bordercolor="rgba(255,255,255,0.15)",
            borderwidth=1,
            font=dict(color="white"),
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        height=580,
    )
    return fig
