"""
BrahamAI — Dashboard Page
Full analytics, model evaluation, chatbot, dataset explorer, map

Improvements over v1:
  - @st.cache_data on all data/model loading
  - Dynamic stats (no hardcoded counts)
  - New "🌍 Destination Map" tab with interactive Plotly geo map
  - Per-type Precision@5 bar chart in Model Evaluation
  - Weighted Score shown in Get Recommendations
  - Uses visualizations/charts module for reusable chart functions
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import streamlit as st
import pandas as pd
import utils.database as db
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from models.recommender import (
    recommend, evaluate_model, load_data,
    build_feature_matrix, compute_similarity
)
from utils.chatbot import chatbot_response
from visualizations.charts import (
    dark_fig, TYPE_COLORS, PALETTE,
    plot_type_distribution, plot_season_distribution,
    plot_budget_vs_rating, plot_top_rated, plot_avg_budget_by_type,
    plot_per_type_precision, plot_duration_boxplot, plot_india_map
)

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "travel_data.csv")


# ── Cached Helpers ────────────────────────────────────────────────────────────
@st.cache_data
def cached_load_data(path):
    return load_data(path)

import requests
@st.cache_data
def get_wiki_image(title):
    """Fetch an image thumbnail from Wikipedia API."""
    url = f"https://en.wikipedia.org/w/api.php?action=query&titles={title}&prop=pageimages&format=json&pithumbsize=600"
    headers = {"User-Agent": "BrahamAITravelBot/1.0 (BrahamAI Project)"}
    try:
        r = requests.get(url, headers=headers).json()
        pages = r.get("query", {}).get("pages", {})
        for p_id in pages:
            if "thumbnail" in pages[p_id]:
                return pages[p_id]["thumbnail"]["source"]
    except Exception:
        pass
    return None

@st.cache_data
def cached_evaluate(path):
    return evaluate_model(path)

@st.cache_data
def cached_feature_matrix(path):
    df = load_data(path)
    fm, scaler = build_feature_matrix(df)
    return df, fm, scaler


# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BrahamAI Dashboard",
    page_icon="📊",
    layout="wide",
)

# ── ADMIN AUTHENTICATION CHECK ────────────────────────────────────────────────
if "is_admin" not in st.session_state:
    st.session_state["is_admin"] = False

if not st.session_state["is_admin"]:
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .main { background: #0a0a0f; }
    header { background: transparent !important; }
    [data-testid="stSidebar"] { display: none; }
    .admin-card {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 3rem;
        text-align: center;
        box-shadow: 0 20px 40px rgba(0,0,0,0.5);
        margin-top: 5rem;
    }
    .admin-icon { font-size: 3rem; margin-bottom: 1rem; }
    .admin-title { color: white; font-size: 1.8rem; font-weight: 800; margin-bottom: 0.5rem; }
    .admin-sub { color: #888; font-size: 0.95rem; margin-bottom: 2.5rem; }
    div[data-testid="stForm"] { border: none !important; background: transparent !important; padding:0; }
    div[data-testid="stButton"] button {
        background: #667eea !important; color: white !important; font-weight: 600 !important;
        border-radius: 8px !important; border: none !important; padding: 0.6rem !important; width: 100%;
        margin-top: 1rem;
    }
    div[data-testid="stButton"] button:hover { opacity: 0.8 !important; }
    </style>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 1.2, 1])
    with c2:
        st.markdown('<div class="admin-card">', unsafe_allow_html=True)
        st.markdown('<div class="admin-icon">🛡️</div>', unsafe_allow_html=True)
        st.markdown('<div class="admin-title">Admin Restricted Area</div>', unsafe_allow_html=True)
        st.markdown('<div class="admin-sub">Please verify your identity to access BrahamAI Analytics and metrics.</div>', unsafe_allow_html=True)
        
        with st.form("admin_login_form"):
            a_user = st.text_input("Username", placeholder="Authorized Personnel Only")
            a_pass = st.text_input("Password", type="password", placeholder="••••••••")
            if st.form_submit_button("Authenticate"):
                user_info = db.verify_user(a_user, a_pass)
                if user_info and user_info.get("is_admin"):
                    st.session_state["logged_in"] = True
                    st.session_state["username"] = user_info["username"]
                    st.session_state["is_admin"] = True
                    st.rerun()
                else:
                    st.error("Authentication Failed. Invalid credentials or insufficient permissions.")
        st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif; }
.main { background: #0a0a0f; }
[data-testid="stSidebar"] { background: #0d0d18 !important; border-right: 1px solid rgba(255,255,255,0.07); }
[data-testid="stSidebar"] * { color: rgba(255,255,255,0.8) !important; }
header { background: transparent !important; }

.dash-header {
    background: linear-gradient(135deg, rgba(102,126,234,0.15), rgba(118,75,162,0.1));
    border: 1px solid rgba(102,126,234,0.2); border-radius: 20px;
    padding: 2rem 2.5rem; margin-bottom: 1.5rem;
    display: flex; align-items: center; gap: 1.5rem;
}
.dash-icon { font-size: 3rem; }
.dash-title { font-size: 1.8rem; font-weight: 800; color: white; margin: 0; }
.dash-sub { color: rgba(255,255,255,0.4); font-size: 0.9rem; margin-top: 0.3rem; }

.kpi-card {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px; padding: 1.5rem; text-align: center; height: 100%;
}
.kpi-icon { font-size: 1.5rem; margin-bottom: 0.5rem; }
.kpi-value { font-size: 1.9rem; font-weight: 700; color: white; }
.kpi-label { font-size: 0.75rem; color: rgba(255,255,255,0.35); margin-top: 0.3rem; }
.kpi-delta { font-size: 0.78rem; color: #34d399; margin-top: 0.3rem; }

.section-head { font-size: 1.1rem; font-weight: 700; color: white;
    border-left: 3px solid #667eea; padding-left: 0.8rem; margin: 1.5rem 0 1rem; }

.chart-card {
    background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px; padding: 1.5rem;
}

.chat-user {
    background: rgba(102,126,234,0.12); border: 1px solid rgba(102,126,234,0.2);
    border-radius: 12px 12px 4px 12px; padding: 0.8rem 1rem;
    margin: 0.5rem 0; color: white; font-size: 0.9rem; text-align: right;
}
.chat-bot {
    background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px 12px 12px 4px; padding: 0.8rem 1rem;
    margin: 0.5rem 0; color: rgba(255,255,255,0.85); font-size: 0.9rem;
}
.chat-label-u { color: rgba(102,126,234,0.8); font-size: 0.72rem; text-align: right; margin-bottom: 0.2rem; }
.chat-label-b { color: rgba(52,211,153,0.8); font-size: 0.72rem; margin-bottom: 0.2rem; }

.dest-row {
    background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px; padding: 1rem 1.2rem; margin-bottom: 0.6rem;
    display: flex; align-items: center; gap: 1rem;
}
.dest-row-name { color: white; font-weight: 600; font-size: 0.95rem; flex: 1; }
.dest-row-info { color: rgba(255,255,255,0.4); font-size: 0.8rem; }

div[data-testid="stButton"] button {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important; border: none !important; border-radius: 10px !important;
    font-weight: 600 !important;
}
.stSelectbox > div > div { background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important; border-radius: 10px !important; }
.stDataFrame { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ✈️ BrahamAI")
    st.markdown("---")
    page = st.radio("Navigate", [
        "📊 Overview",
        "📈 Data Insights",
        "🤖 Model Evaluation",
        "🌍 Destination Map",
        "💬 Travel Chatbot",
        "📂 Dataset Explorer",
        "🗺️ Get Recommendations",
    ])
    st.markdown("---")
    st.markdown('<div style="color:rgba(255,255,255,0.25);font-size:0.75rem;">BCA Data Science Project<br>Final Year 2024-25</div>', unsafe_allow_html=True)

df = cached_load_data(DATA_PATH)
type_info = {
    "Beach":     ("🏖️", "#38bdf8"),
    "Adventure": ("🏔️", "#fb923c"),
    "Cultural":  ("🏛️", "#a78bfa"),
    "Nature":    ("🌿", "#34d399"),
    "Wildlife":  ("🐯", "#fbbf24"),
    "City":      ("🏙️", "#94a3b8"),
}

# ═══════════════════════════════════
# PAGE: OVERVIEW
# ═══════════════════════════════════
if page == "📊 Overview":
    st.markdown("""
    <div class="dash-header">
        <div class="dash-icon">📊</div>
        <div>
            <div class="dash-title">BrahamAI Dashboard</div>
            <div class="dash-sub">Real-time analytics and insights for your travel recommendation engine</div>
        </div>
    </div>""", unsafe_allow_html=True)

    total_dest = df["Destination"].nunique()
    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        ("🗺️", total_dest,                       "Destinations",  f"↑ {total_dest} total"),
        ("🏷️", df["Type"].nunique(),              "Travel Types",  "6 categories"),
        ("📍", df["State"].nunique(),             "States",        "Across India"),
        ("⭐", df["Rating"].max(),                "Best Rating",   "Out of 5.0"),
        ("💰", f"₹{int(df['Budget'].mean()):,}", "Avg Budget",    "Per trip"),
    ]
    for col, (icon, val, label, delta) in zip([c1, c2, c3, c4, c5], kpis):
        with col:
            st.markdown(f"""<div class="kpi-card">
                <div class="kpi-icon">{icon}</div>
                <div class="kpi-value">{val}</div>
                <div class="kpi-label">{label}</div>
                <div class="kpi-delta">{delta}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-head">Destination Breakdown by Type</div>', unsafe_allow_html=True)

    type_counts = df["Type"].value_counts()
    cols = st.columns(len(type_counts))
    for col, (ttype, count) in zip(cols, type_counts.items()):
        icon, color = type_info.get(ttype, ("📍", "#667eea"))
        pct = int(count / len(df) * 100)
        with col:
            st.markdown(f"""<div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.06);
                border-radius:14px;padding:1.2rem;text-align:center;">
                <div style="font-size:1.8rem;">{icon}</div>
                <div style="font-size:1.4rem;font-weight:700;color:{color};margin:0.3rem 0;">{count}</div>
                <div style="color:white;font-size:0.85rem;font-weight:600;">{ttype}</div>
                <div style="color:rgba(255,255,255,0.3);font-size:0.75rem;">{pct}% of total</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-head">Top 5 Destinations by Rating</div>', unsafe_allow_html=True)
    top5 = df.sort_values("Rating", ascending=False).drop_duplicates("Destination").head(5)
    for i, (_, row) in enumerate(top5.iterrows(), 1):
        emoji = type_info.get(row["Type"], ("📍", ""))[0]
        stars = "⭐" * int(row["Rating"])
        st.markdown(f"""<div class="dest-row">
            <div style="font-size:1.5rem;min-width:40px;text-align:center;">#{i}</div>
            <div style="font-size:1.4rem;">{emoji}</div>
            <div class="dest-row-name">{row['Destination']} <span style="color:rgba(255,255,255,0.3);font-weight:400;font-size:0.82rem;">— {row.get('State','')}</span></div>
            <div class="dest-row-info">{row['Type']} · {row['Season']} · {int(row['Duration'])} days</div>
            <div style="color:#fbbf24;font-size:0.9rem;">{stars} {row['Rating']}</div>
            <div style="color:#34d399;font-size:0.9rem;font-weight:600;">₹{int(row['Budget']):,}</div>
        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════
# PAGE: DATA INSIGHTS
# ═══════════════════════════════════
elif page == "📈 Data Insights":
    st.markdown("""<div class="dash-header">
        <div class="dash-icon">📈</div>
        <div><div class="dash-title">Data Insights</div>
        <div class="dash-sub">Exploratory Data Analysis of all Indian travel destinations</div></div>
    </div>""", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-head">Travel Type Distribution</div>', unsafe_allow_html=True)
        st.pyplot(plot_type_distribution(df), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-head">Season Distribution</div>', unsafe_allow_html=True)
        st.pyplot(plot_season_distribution(df), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="chart-card" style="margin:1rem 0;">', unsafe_allow_html=True)
    st.markdown('<div class="section-head">Budget vs Rating (bubble size = trip duration)</div>', unsafe_allow_html=True)
    st.pyplot(plot_budget_vs_rating(df), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-head">Top 15 Destinations by Rating</div>', unsafe_allow_html=True)
        st.pyplot(plot_top_rated(df, n=15), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_d:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-head">Average Budget by Travel Type</div>', unsafe_allow_html=True)
        st.pyplot(plot_avg_budget_by_type(df), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="chart-card" style="margin:1rem 0;">', unsafe_allow_html=True)
    st.markdown('<div class="section-head">Trip Duration Distribution by Travel Type</div>', unsafe_allow_html=True)
    st.pyplot(plot_duration_boxplot(df), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════
# PAGE: MODEL EVALUATION
# ═══════════════════════════════════
elif page == "🤖 Model Evaluation":
    st.markdown("""<div class="dash-header">
        <div class="dash-icon">🤖</div>
        <div><div class="dash-title">Model Evaluation</div>
        <div class="dash-sub">Content-Based Filtering performance metrics and analysis</div></div>
    </div>""", unsafe_allow_html=True)

    with st.spinner("Running model evaluation..."):
        metrics = cached_evaluate(DATA_PATH)

    m1, m2, m3, m4 = st.columns(4)
    metric_items = [
        ("Precision@5",  metrics["Avg Precision@5"],    "Of top-5, correct type matches",      "#667eea"),
        ("Recall@5",     metrics["Avg Recall@5"],        "All relevant destinations retrieved",  "#34d399"),
        ("F1 Score",     metrics["F1 Score"],            "Harmonic mean of Precision & Recall", "#f0932b"),
        ("Dataset Size", metrics["Total Destinations"],  "Destinations in model",               "#a78bfa"),
    ]
    for col, (label, val, desc, color) in zip([m1, m2, m3, m4], metric_items):
        with col:
            st.markdown(f"""<div class="kpi-card">
                <div class="kpi-value" style="color:{color};">{val}</div>
                <div class="kpi-label">{label}</div>
                <div style="color:rgba(255,255,255,0.25);font-size:0.72rem;margin-top:0.5rem;">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Per-type precision chart
    st.markdown('<div class="section-head">📊 Precision@5 by Travel Type</div>', unsafe_allow_html=True)
    if "Per Type Precision" in metrics and metrics["Per Type Precision"]:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.pyplot(plot_per_type_precision(metrics["Per Type Precision"]), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.markdown('<div class="section-head">📖 How the Model Works — Step by Step</div>', unsafe_allow_html=True)
        steps = [
            ("1", "One-Hot Encoding",    "Categorical columns (Type, Season) → binary columns. Beach becomes type_Beach=1, all others=0.",    "#667eea"),
            ("2", "Min-Max Normalization", "Numeric columns (Budget, Rating, Duration) are scaled to [0,1] range so no feature dominates.",    "#34d399"),
            ("3", "User Profile Vector", "Your preferences are encoded into the exact same feature vector format as the destinations.",         "#f0932b"),
            ("4", "Cosine Similarity",   "We compute the angle between your vector and every destination. Score 1.0 = perfect match.",         "#a78bfa"),
            ("5", "Weighted Ranking",    "Final score = 70% cosine similarity + 30% normalised rating, so top-rated places rank higher.",      "#f472b6"),
        ]
        for num, title, desc, color in steps:
            st.markdown(f"""<div style="display:flex;gap:1rem;margin-bottom:1rem;padding:1rem;
                background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.06);border-radius:12px;">
                <div style="min-width:32px;height:32px;border-radius:50%;background:{color};
                    display:flex;align-items:center;justify-content:center;
                    font-weight:700;color:white;font-size:0.85rem;">{num}</div>
                <div>
                    <div style="color:white;font-weight:600;margin-bottom:0.3rem;">{title}</div>
                    <div style="color:rgba(255,255,255,0.4);font-size:0.83rem;line-height:1.5;">{desc}</div>
                </div>
            </div>""", unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="section-head">📐 Feature Matrix Sample</div>', unsafe_allow_html=True)
        _, fm, _ = cached_feature_matrix(DATA_PATH)
        sample = fm.head(5).round(3)
        st.dataframe(sample, use_container_width=True)

        st.markdown('<div class="section-head" style="margin-top:1.5rem;">🔢 Cosine Similarity (5×5 sample)</div>', unsafe_allow_html=True)
        sim = compute_similarity(fm)
        sim_df = pd.DataFrame(
            sim[:5, :5],
            index=df["Destination"][:5].values,
            columns=df["Destination"][:5].values,
        ).round(3)
        st.dataframe(sim_df, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-head">📊 Evaluation Metrics Explained</div>', unsafe_allow_html=True)
    ecol1, ecol2, ecol3 = st.columns(3)
    explanations = [
        ("Precision@5",  f"{metrics['Avg Precision@5']}",  "Of the top-5 recommended destinations, this fraction shares the correct travel type. High precision = fewer irrelevant suggestions.",    "#667eea"),
        ("Recall@5",     f"{metrics['Avg Recall@5']}",     "Of all relevant destinations in the database, about this fraction appears in top-5. Lower recall is expected — we only return 5.",       "#34d399"),
        ("F1 Score",     f"{metrics['F1 Score']}",         "Harmonic mean of precision and recall. Balances both metrics — a solid content-based score for a 115-destination dataset.",              "#f0932b"),
    ]
    for col, (title, val, desc, color) in zip([ecol1, ecol2, ecol3], explanations):
        with col:
            st.markdown(f"""<div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.07);
                border-radius:14px;padding:1.5rem;">
                <div style="font-size:2rem;font-weight:800;color:{color};margin-bottom:0.5rem;">{val}</div>
                <div style="color:white;font-weight:600;margin-bottom:0.5rem;">{title}</div>
                <div style="color:rgba(255,255,255,0.4);font-size:0.83rem;line-height:1.6;">{desc}</div>
            </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════
# PAGE: DESTINATION MAP
# ═══════════════════════════════════
elif page == "🌍 Destination Map":
    st.markdown("""<div class="dash-header">
        <div class="dash-icon">🌍</div>
        <div><div class="dash-title">Destination Map</div>
        <div class="dash-sub">Interactive geo map of all Indian travel destinations — colour by type, size by rating</div></div>
    </div>""", unsafe_allow_html=True)

    # Filter controls
    mf1, mf2, mf3 = st.columns(3)
    with mf1:
        map_types = st.multiselect("Filter by Travel Type", options=sorted(df["Type"].unique()), default=list(df["Type"].unique()))
    with mf2:
        map_seasons = st.multiselect("Filter by Season", options=sorted(df["Season"].unique()), default=list(df["Season"].unique()))
    with mf3:
        map_rating = st.slider("Minimum Rating", 4.0, 5.0, 4.0, 0.1)

    map_df = df[
        df["Type"].isin(map_types) &
        df["Season"].isin(map_seasons) &
        (df["Rating"] >= map_rating)
    ]

    st.markdown(f'<div style="color:rgba(255,255,255,0.4);font-size:0.85rem;margin:0.5rem 0;">Showing <b style="color:white;">{map_df["Destination"].nunique()}</b> of {df["Destination"].nunique()} destinations</div>', unsafe_allow_html=True)

    if map_df.empty:
        st.warning("No destinations match these filters. Try widening your selection.")
    elif "Lat" not in map_df.columns:
        st.error("Map requires Lat/Lon columns in the dataset.")
    else:
        fig_map = plot_india_map(map_df, color_col="Type", title="India Travel Destinations")
        if fig_map:
            st.plotly_chart(fig_map, use_container_width=True)

    # Legend
    st.markdown("---")
    st.markdown('<div class="section-head">Map Legend</div>', unsafe_allow_html=True)
    leg_cols = st.columns(len(TYPE_COLORS))
    for col, (ttype, color) in zip(leg_cols, TYPE_COLORS.items()):
        icon = type_info.get(ttype, ("📍", ""))[0]
        count = len(df[df["Type"] == ttype].drop_duplicates("Destination"))
        with col:
            st.markdown(f"""<div style="text-align:center;padding:0.8rem;background:rgba(255,255,255,0.02);
                border:1px solid {color}33;border-radius:10px;">
                <div style="font-size:1.4rem;">{icon}</div>
                <div style="color:{color};font-weight:600;font-size:0.85rem;margin-top:0.3rem;">{ttype}</div>
                <div style="color:rgba(255,255,255,0.3);font-size:0.75rem;">{count} places</div>
            </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════
# PAGE: CHATBOT
# ═══════════════════════════════════
elif page == "💬 Travel Chatbot":
    st.markdown("""<div class="dash-header">
        <div class="dash-icon">💬</div>
        <div><div class="dash-title">BrahamAI Travel Chatbot</div>
        <div class="dash-sub">Ask me about destinations, seasons, budgets, and travel tips — I understand multi-topic queries!</div></div>
    </div>""", unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            ("bot", "👋 Hi! I'm BrahamAI. Ask me about beach trips, adventure spots, budget travel, or seasonal recommendations! Try the quick buttons below.")
        ]

    chat_container = st.container()
    with chat_container:
        for role, msg in st.session_state.chat_history:
            if role == "bot":
                st.markdown(f'<div class="chat-label-b">🤖 BrahamAI</div><div class="chat-bot">{msg}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-label-u">You 👤</div><div class="chat-user">{msg}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**⚡ Quick Questions**")
    qcols = st.columns(4)
    quick = ["🏖️ Beach destinations", "🏔️ Adventure trips", "💰 Budget under ₹10K", "❄️ Best in winter"]
    for col, prompt in zip(qcols, quick):
        with col:
            if st.button(prompt, key=f"q_{prompt}"):
                response = chatbot_response(prompt)
                st.session_state.chat_history.append(("user", prompt))
                st.session_state.chat_history.append(("bot", response))
                st.rerun()

    qcols2 = st.columns(4)
    quick2 = ["🌿 Nature getaway", "🦁 Wildlife safari", "🌧️ Monsoon travel", "🏛️ Cultural heritage"]
    for col, prompt in zip(qcols2, quick2):
        with col:
            if st.button(prompt, key=f"q2_{prompt}"):
                response = chatbot_response(prompt)
                st.session_state.chat_history.append(("user", prompt))
                st.session_state.chat_history.append(("bot", response))
                st.rerun()

    qcols3 = st.columns(4)
    quick3 = ["🏙️ City breaks", "🏔️ Adventure beach combo", "☀️ Summer hills", "🎒 Budget monsoon trip"]
    for col, prompt in zip(qcols3, quick3):
        with col:
            if st.button(prompt, key=f"q3_{prompt}"):
                response = chatbot_response(prompt)
                st.session_state.chat_history.append(("user", prompt))
                st.session_state.chat_history.append(("bot", response))
                st.rerun()

    st.markdown("---")
    with st.form("chat_form", clear_on_submit=True):
        col_inp, col_btn = st.columns([5, 1])
        with col_inp:
            user_msg = st.text_input("", placeholder="e.g. 'beach adventure under ₹15000 in winter'...", label_visibility="collapsed")
        with col_btn:
            send = st.form_submit_button("Send ➤")
        if send and user_msg.strip():
            bot_reply = chatbot_response(user_msg)
            st.session_state.chat_history.append(("user", user_msg))
            st.session_state.chat_history.append(("bot", bot_reply))
            st.rerun()

    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# ═══════════════════════════════════
# PAGE: DATASET EXPLORER
# ═══════════════════════════════════
elif page == "📂 Dataset Explorer":
    st.markdown("""<div class="dash-header">
        <div class="dash-icon">📂</div>
        <div><div class="dash-title">Dataset Explorer</div>
        <div class="dash-sub">Browse, filter and download all Indian travel destinations</div></div>
    </div>""", unsafe_allow_html=True)

    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    with col_f1:
        type_f = st.multiselect("Travel Type", df["Type"].unique(), default=list(df["Type"].unique()))
    with col_f2:
        season_f = st.multiselect("Season", df["Season"].unique(), default=list(df["Season"].unique()))
    with col_f3:
        budget_f = st.slider("Max Budget (₹)", int(df["Budget"].min()), int(df["Budget"].max()), int(df["Budget"].max()), 1000)
    with col_f4:
        rating_f = st.slider("Min Rating", float(df["Rating"].min()), float(df["Rating"].max()), float(df["Rating"].min()), 0.1)

    filtered = df[
        df["Type"].isin(type_f) &
        df["Season"].isin(season_f) &
        (df["Budget"] <= budget_f) &
        (df["Rating"] >= rating_f)
    ].copy()

    st.markdown(f'<div style="color:rgba(255,255,255,0.4);font-size:0.85rem;margin:0.5rem 0;">Showing <b style="color:white;">{len(filtered)}</b> of {len(df)} rows</div>', unsafe_allow_html=True)

    display = filtered.copy()
    display["Budget"]   = display["Budget"].apply(lambda x: f"₹{int(x):,}")
    display["Rating"]   = display["Rating"].apply(lambda x: f"⭐ {x}")
    display["Duration"] = display["Duration"].apply(lambda x: f"{int(x)} days")

    display_cols = ["Destination", "State", "Type", "Season", "Budget", "Rating", "Duration", "Description"]
    display_cols = [c for c in display_cols if c in display.columns]
    st.dataframe(display[display_cols], use_container_width=True, hide_index=True, height=450)

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        csv = filtered.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Filtered Data", csv, "brahamai_filtered.csv", "text/csv", use_container_width=True)
    with col_dl2:
        full_csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Full Dataset", full_csv, "travel_data_full.csv", "text/csv", use_container_width=True)

# ═══════════════════════════════════
# PAGE: GET RECOMMENDATIONS
# ═══════════════════════════════════
elif page == "🗺️ Get Recommendations":
    st.markdown("""<div class="dash-header">
        <div class="dash-icon">🗺️</div>
        <div><div class="dash-title">Get AI Recommendations</div>
        <div class="dash-sub">Personalised destination matching — weighted by cosine similarity + rating</div></div>
    </div>""", unsafe_allow_html=True)

    with st.form("rec_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            budget = st.slider("💰 Your Budget (₹)", int(df["Budget"].min()), int(df["Budget"].max()), 15000, 1000)
        with c2:
            travel_type = st.selectbox("🏷️ Travel Type", ["Beach", "Adventure", "Cultural", "Nature", "Wildlife", "City"])
        with c3:
            season = st.selectbox("🗓️ Season", ["Winter", "Summer", "Monsoon"])

        c4, c5, c6 = st.columns(3)
        with c4:
            top_n = st.slider("Number of results", 3, 10, 5)
        with c5:
            max_bud = st.slider("Max Budget filter (₹)", int(df["Budget"].min()), int(df["Budget"].max()), int(df["Budget"].max()), 1000)
        with c6:
            sort_by = st.selectbox("Sort by", ["AI Score (recommended)", "Rating", "Similarity only"])

        submitted = st.form_submit_button("🚀 Find Destinations", use_container_width=True)

    if submitted:
        with st.spinner("🤖 Calculating AI recommendations..."):
            sort_rating = (sort_by == "Rating")
            results = recommend(
                budget=budget, travel_type=travel_type, season=season,
                top_n=top_n, max_budget=max_bud,
                sort_by_rating=sort_rating,
                data_path=DATA_PATH
            )

        type_emojis      = {"Beach":"🏖️","Adventure":"🏔️","Cultural":"🏛️","Nature":"🌿","Wildlife":"🐯","City":"🏙️"}
        type_colors_map  = TYPE_COLORS

        st.markdown(f"### ✅ {len(results)} destinations found for **{travel_type}** in **{season}** with budget ₹{budget:,}")

        if results.empty:
            st.warning("No destinations found. Try adjusting filters.")
        else:
            for rank, (_, row) in enumerate(results.iterrows(), 1):
                score_pct = int(row.get("Score", row["Similarity"]) * 100)
                sim_pct   = int(row["Similarity"] * 100)
                emoji     = type_emojis.get(row["Type"], "📍")
                color     = type_colors_map.get(row["Type"], "#667eea")
                desc      = row.get("Description", "")
                desc_text = str(desc) if desc and desc == desc else ""
                stars     = "⭐" * int(row["Rating"])
                state     = row.get("State", "India")

                img_url = get_wiki_image(f"{row['Destination']}, {state}") or get_wiki_image(row['Destination'])
                if img_url:
                    media_html = f'<img src="{img_url}" style="width:80px;height:80px;border-radius:16px;object-fit:cover;flex-shrink:0;box-shadow:0 4px 15px rgba(0,0,0,0.3);" />'
                else:
                    media_html = f'<div style="font-size:3rem;width:80px;height:80px;display:flex;align-items:center;justify-content:center;background:rgba(255,255,255,0.02);border-radius:16px;flex-shrink:0;">{emoji}</div>'

                st.markdown(f"""
                <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.07);
                    border-radius:16px;padding:1.5rem;margin-bottom:1rem;
                    display:flex;align-items:center;gap:1.5rem;">
                    <div style="font-size:2.2rem;font-weight:800;min-width:50px;text-align:center;
                        background:linear-gradient(135deg,{color},#764ba2);
                        -webkit-background-clip:text;-webkit-text-fill-color:transparent;">#{rank}</div>
                    {media_html}
                    <div style="flex:1;">
                        <div style="color:white;font-size:1.1rem;font-weight:700;">{row['Destination']} <span style="color:rgba(255,255,255,0.35);font-weight:400;font-size:0.8rem;">— {state}</span></div>
                        <div style="color:rgba(255,255,255,0.35);font-size:0.8rem;margin:0.2rem 0 0.6rem;">
                            {desc_text[:100]}{'...' if len(desc_text)>100 else ''}
                        </div>
                        <div style="color:rgba(255,255,255,0.4);font-size:0.82rem;">
                            💰 ₹{int(row['Budget']):,} &nbsp;·&nbsp; {stars} {row['Rating']}
                            &nbsp;·&nbsp; 🗓️ {int(row['Duration'])} days
                            &nbsp;·&nbsp; 📍 {row.get('Season','')}
                        </div>
                    </div>
                    <div style="text-align:center;padding:1rem 1.5rem;background:rgba(255,255,255,0.03);
                        border-radius:12px;border:1px solid rgba(255,255,255,0.06);">
                        <div style="font-size:1.6rem;font-weight:800;color:{color};">{score_pct}%</div>
                        <div style="color:rgba(255,255,255,0.3);font-size:0.72rem;">AI Score</div>
                        <div style="color:rgba(255,255,255,0.2);font-size:0.68rem;margin-top:0.2rem;">({sim_pct}% sim)</div>
                    </div>
                </div>""", unsafe_allow_html=True)

            # Map of results
            if "Lat" in results.columns:
                st.markdown("---")
                st.markdown('<div class="section-head">📍 Recommended Destinations on Map</div>', unsafe_allow_html=True)
                fig_map = plot_india_map(results, title=f"Top {len(results)} {travel_type} picks")
                if fig_map:
                    st.plotly_chart(fig_map, use_container_width=True)

            csv = results.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Results", csv, "my_recommendations.csv", "text/csv")
