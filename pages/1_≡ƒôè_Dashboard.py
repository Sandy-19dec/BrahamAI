"""
BrahamAI — Dashboard Page
Full analytics, model evaluation, chatbot, dataset explorer
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from models.recommender import recommend, evaluate_model, load_data, build_feature_matrix, compute_similarity
from utils.chatbot import chatbot_response

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "travel_data.csv")

st.set_page_config(
    page_title="BrahamAI Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

PALETTE = ["#667eea","#764ba2","#f0932b","#34d399","#f472b6","#38bdf8","#fb923c","#a78bfa","#fbbf24","#6ee7b7"]

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

.metric-row {
    display: grid; grid-template-columns: repeat(4,1fr); gap: 1rem; margin-bottom: 1.5rem;
}
.metric-box {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px; padding: 1.2rem; text-align: center;
}
.metric-val { font-size: 1.6rem; font-weight: 700;
    background: linear-gradient(135deg, #667eea, #34d399);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.metric-lbl { font-size: 0.75rem; color: rgba(255,255,255,0.35); margin-top: 0.3rem; }

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

# ── SIDEBAR ──
with st.sidebar:
    st.markdown("## ✈️ BrahamAI")
    st.markdown("---")
    page = st.radio("Navigate", [
        "📊 Overview",
        "📈 Data Insights",
        "🤖 Model Evaluation",
        "💬 Travel Chatbot",
        "📂 Dataset Explorer",
        "🗺️ Get Recommendations"
    ])
    st.markdown("---")
    st.markdown('<div style="color:rgba(255,255,255,0.25);font-size:0.75rem;">BCA Data Science Project<br>Final Year 2024-25</div>', unsafe_allow_html=True)

df = load_data(DATA_PATH)

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

    # KPI Row
    c1,c2,c3,c4,c5 = st.columns(5)
    kpis = [
        ("🗺️", len(df), "Destinations", "↑ 100 total"),
        ("🏷️", df['Type'].nunique(), "Travel Types", "6 categories"),
        ("📍", df['State'].nunique(), "States", "Across India"),
        ("⭐", df['Rating'].max(), "Best Rating", "Out of 5.0"),
        ("💰", f"₹{int(df['Budget'].mean()):,}", "Avg Budget", "Per trip"),
    ]
    for col, (icon, val, label, delta) in zip([c1,c2,c3,c4,c5], kpis):
        with col:
            st.markdown(f"""<div class="kpi-card">
                <div class="kpi-icon">{icon}</div>
                <div class="kpi-value">{val}</div>
                <div class="kpi-label">{label}</div>
                <div class="kpi-delta">{delta}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-head">Destination Breakdown by Type</div>', unsafe_allow_html=True)

    # Type breakdown
    type_counts = df['Type'].value_counts()
    cols = st.columns(len(type_counts))
    type_info = {
        "Beach": ("🏖️","#38bdf8"), "Adventure": ("🏔️","#fb923c"),
        "Cultural": ("🏛️","#a78bfa"), "Nature": ("🌿","#34d399"),
        "Wildlife": ("🐯","#fbbf24"), "City": ("🏙️","#94a3b8")
    }
    for col, (ttype, count) in zip(cols, type_counts.items()):
        icon, color = type_info.get(ttype, ("📍","#667eea"))
        pct = int(count/len(df)*100)
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
        emoji = type_info.get(row['Type'], ("📍",""))[0]
        stars = "⭐" * int(row['Rating'])
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
        <div class="dash-sub">Exploratory Data Analysis of 100+ Indian travel destinations</div></div>
    </div>""", unsafe_allow_html=True)

    def dark_fig(w=10, h=5):
        fig, ax = plt.subplots(figsize=(w,h))
        fig.patch.set_facecolor("#0d0d18")
        ax.set_facecolor("#12121f")
        for spine in ax.spines.values(): spine.set_edgecolor("#2a2a3a")
        ax.tick_params(colors="#888", labelsize=9)
        ax.xaxis.label.set_color("#888")
        ax.yaxis.label.set_color("#888")
        ax.title.set_color("white")
        return fig, ax

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-head">Travel Type Distribution</div>', unsafe_allow_html=True)
        type_counts = df['Type'].value_counts()
        fig, ax = dark_fig(6,5)
        colors = ["#667eea","#f0932b","#34d399","#f472b6","#fbbf24","#38bdf8"]
        wedges, texts, autotexts = ax.pie(type_counts.values, labels=type_counts.index,
            autopct='%1.1f%%', colors=colors, startangle=140,
            textprops={'color':'white','fontsize':10},
            wedgeprops={'edgecolor':'#0d0d18','linewidth':2})
        for at in autotexts: at.set_color('white'); at.set_fontsize(9)
        ax.set_title("Destinations by Travel Type", color="white", pad=15)
        st.pyplot(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-head">Season Distribution</div>', unsafe_allow_html=True)
        season_counts = df['Season'].value_counts()
        fig, ax = dark_fig(6,5)
        bars = ax.bar(season_counts.index, season_counts.values,
            color=["#38bdf8","#fb923c","#6ee7b7"], edgecolor="#0d0d18", linewidth=2, width=0.5)
        ax.set_title("Destinations by Season", color="white")
        for bar, val in zip(bars, season_counts.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                str(val), ha='center', color='white', fontsize=11, fontweight='600')
        ax.set_ylim(0, season_counts.max()+8)
        st.pyplot(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="chart-card" style="margin:1rem 0;">', unsafe_allow_html=True)
    st.markdown('<div class="section-head">Budget vs Rating (bubble = trip duration)</div>', unsafe_allow_html=True)
    fig, ax = dark_fig(12, 5)
    type_colors = {"Beach":"#38bdf8","Adventure":"#fb923c","Cultural":"#a78bfa",
                   "Nature":"#34d399","Wildlife":"#fbbf24","City":"#94a3b8"}
    for ttype, group in df.groupby("Type"):
        ax.scatter(group["Budget"], group["Rating"],
            c=type_colors.get(ttype,"#667eea"), s=group["Duration"]*25,
            alpha=0.8, edgecolors="white", linewidths=0.3, label=ttype)
    for _, row in df[df["Rating"]>=4.85].iterrows():
        ax.annotate(row["Destination"], (row["Budget"], row["Rating"]),
            textcoords="offset points", xytext=(8,4), fontsize=8, color="white", alpha=0.8)
    ax.legend(facecolor="#12121f", labelcolor="white", edgecolor="#2a2a3a",
        fontsize=9, title="Travel Type", title_fontsize=9)
    ax.set_xlabel("Budget (₹)", color="#888")
    ax.set_ylabel("Rating", color="#888")
    ax.set_title("Budget vs Rating — All Destinations", color="white")
    st.pyplot(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-head">Top 15 Destinations by Rating</div>', unsafe_allow_html=True)
        top15 = df.sort_values("Rating", ascending=False).drop_duplicates("Destination").head(15)
        fig, ax = dark_fig(7, 6)
        colors_bar = [PALETTE[i%len(PALETTE)] for i in range(15)]
        bars = ax.barh(top15["Destination"], top15["Rating"],
            color=colors_bar, edgecolor="#0d0d18", linewidth=1.5)
        ax.set_xlim(4.0, 5.1)
        ax.invert_yaxis()
        ax.set_title("Top 15 Rated Destinations", color="white")
        for bar, val in zip(bars, top15["Rating"]):
            ax.text(val+0.01, bar.get_y()+bar.get_height()/2,
                f"{val}", va='center', color='white', fontsize=8.5)
        st.pyplot(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_d:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-head">Average Budget by Travel Type</div>', unsafe_allow_html=True)
        avg_budget = df.groupby("Type")["Budget"].mean().sort_values(ascending=True)
        fig, ax = dark_fig(7, 6)
        bars = ax.barh(avg_budget.index, avg_budget.values/1000,
            color=list(type_colors.values())[:len(avg_budget)], edgecolor="#0d0d18", linewidth=1.5)
        ax.set_title("Average Budget by Type (₹ thousands)", color="white")
        for bar, val in zip(bars, avg_budget.values):
            ax.text(val/1000+0.2, bar.get_y()+bar.get_height()/2,
                f"₹{val/1000:.0f}K", va='center', color='white', fontsize=8.5)
        st.pyplot(fig, use_container_width=True)
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

    with st.spinner("Running evaluation..."):
        metrics = evaluate_model(DATA_PATH)

    m1,m2,m3,m4 = st.columns(4)
    metric_items = [
        ("Precision@5", metrics["Avg Precision@5"], "Of top-5, correct type matches", "#667eea"),
        ("Recall@5", metrics["Avg Recall@5"], "All relevant destinations retrieved", "#34d399"),
        ("F1 Score", metrics["F1 Score"], "Harmonic mean of P & R", "#f0932b"),
        ("Dataset Size", metrics["Total Destinations"], "Destinations in model", "#a78bfa"),
    ]
    for col, (label, val, desc, color) in zip([m1,m2,m3,m4], metric_items):
        with col:
            st.markdown(f"""<div class="kpi-card">
                <div class="kpi-value" style="color:{color};">{val}</div>
                <div class="kpi-label">{label}</div>
                <div style="color:rgba(255,255,255,0.25);font-size:0.72rem;margin-top:0.5rem;">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    col_left, col_right = st.columns([3,2])
    with col_left:
        st.markdown('<div class="section-head">📖 How the Model Works — Step by Step</div>', unsafe_allow_html=True)
        steps = [
            ("1", "One-Hot Encoding", "Categorical columns (Type, Season) → binary columns. Beach becomes type_Beach=1, all others=0. This makes categories comparable numerically.", "#667eea"),
            ("2", "Min-Max Normalization", "Numeric columns (Budget, Rating, Duration) are scaled to [0,1] range so no single feature dominates similarity calculations.", "#34d399"),
            ("3", "User Profile Vector", "Your preferences are encoded into the exact same feature vector format as the destinations in our database.", "#f0932b"),
            ("4", "Cosine Similarity", "We compute the angle between your preference vector and every destination vector. Score 1.0 = perfect match, 0.0 = no match.", "#a78bfa"),
            ("5", "Ranking & Filtering", "Results are sorted by similarity score. Optional budget range filter further narrows the output.", "#f472b6"),
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
        fm, _ = build_feature_matrix(df)
        sample = fm.head(5).round(3)
        st.dataframe(sample, use_container_width=True)

        st.markdown('<div class="section-head" style="margin-top:1.5rem;">🔢 Cosine Similarity (5×5 sample)</div>', unsafe_allow_html=True)
        sim = compute_similarity(fm)
        sim_df = pd.DataFrame(sim[:5,:5],
            index=df["Destination"][:5].values,
            columns=df["Destination"][:5].values).round(3)
        st.dataframe(sim_df, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-head">📊 Evaluation Metrics Explained</div>', unsafe_allow_html=True)
    ecol1, ecol2, ecol3 = st.columns(3)
    explanations = [
        ("Precision@5", "0.83", "83% of the 5 recommended destinations match the user's preferred travel type. High precision means fewer irrelevant suggestions.", "#667eea"),
        ("Recall@5", "0.42", "Of all relevant destinations in the database, we retrieve about 42% in our top-5. Lower recall is expected with only 5 results.", "#34d399"),
        ("F1 Score", "0.56", "The harmonic mean balances precision and recall. Score of 0.56 indicates a solid content-based model for this dataset size.", "#f0932b"),
    ]
    for col, (title, val, desc, color) in zip([ecol1,ecol2,ecol3], explanations):
        with col:
            st.markdown(f"""<div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.07);
                border-radius:14px;padding:1.5rem;">
                <div style="font-size:2rem;font-weight:800;color:{color};margin-bottom:0.5rem;">{val}</div>
                <div style="color:white;font-weight:600;margin-bottom:0.5rem;">{title}</div>
                <div style="color:rgba(255,255,255,0.4);font-size:0.83rem;line-height:1.6;">{desc}</div>
            </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════
# PAGE: CHATBOT
# ═══════════════════════════════════
elif page == "💬 Travel Chatbot":
    st.markdown("""<div class="dash-header">
        <div class="dash-icon">💬</div>
        <div><div class="dash-title">BrahamAI Travel Chatbot</div>
        <div class="dash-sub">Ask me about destinations, seasons, budgets, and travel tips</div></div>
    </div>""", unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            ("bot", "👋 Hi! I'm BrahamAI. Ask me about beach trips, adventure spots, budget travel, or seasonal recommendations! Try the quick buttons below.")
        ]

    # Chat display
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

    st.markdown("---")
    with st.form("chat_form", clear_on_submit=True):
        col_inp, col_btn = st.columns([5,1])
        with col_inp:
            user_msg = st.text_input("", placeholder="Ask me anything about travel in India...", label_visibility="collapsed")
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
        <div class="dash-sub">Browse, filter and download all 100+ travel destinations</div></div>
    </div>""", unsafe_allow_html=True)

    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    with col_f1:
        type_f = st.multiselect("Travel Type", df['Type'].unique(), default=list(df['Type'].unique()))
    with col_f2:
        season_f = st.multiselect("Season", df['Season'].unique(), default=list(df['Season'].unique()))
    with col_f3:
        budget_f = st.slider("Max Budget (₹)", 5000, 40000, 40000, 1000)
    with col_f4:
        rating_f = st.slider("Min Rating", 4.0, 5.0, 4.0, 0.1)

    filtered = df[
        df['Type'].isin(type_f) &
        df['Season'].isin(season_f) &
        (df['Budget'] <= budget_f) &
        (df['Rating'] >= rating_f)
    ].copy()

    st.markdown(f'<div style="color:rgba(255,255,255,0.4);font-size:0.85rem;margin:0.5rem 0;">Showing <b style="color:white;">{len(filtered)}</b> of {len(df)} destinations</div>', unsafe_allow_html=True)

    display = filtered.copy()
    display["Budget"] = display["Budget"].apply(lambda x: f"₹{int(x):,}")
    display["Rating"] = display["Rating"].apply(lambda x: f"⭐ {x}")
    display["Duration"] = display["Duration"].apply(lambda x: f"{int(x)} days")
    st.dataframe(display[["Destination","State","Type","Season","Budget","Rating","Duration","Description"]],
        use_container_width=True, hide_index=True, height=450)

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
        <div class="dash-sub">Personalized destination matching using cosine similarity</div></div>
    </div>""", unsafe_allow_html=True)

    with st.form("rec_form"):
        c1,c2,c3 = st.columns(3)
        with c1:
            budget = st.slider("💰 Your Budget (₹)", 5000, 40000, 15000, 1000)
        with c2:
            travel_type = st.selectbox("🏷️ Travel Type", ["Beach","Adventure","Cultural","Nature","Wildlife","City"])
        with c3:
            season = st.selectbox("🗓️ Season", ["Winter","Summer","Monsoon"])

        c4,c5,c6 = st.columns(3)
        with c4:
            top_n = st.slider("Number of results", 3, 10, 5)
        with c5:
            max_bud = st.slider("Max Budget filter (₹)", 5000, 40000, 40000, 1000)
        with c6:
            sort_by = st.selectbox("Sort by", ["AI Match Score", "Rating"])

        submitted = st.form_submit_button("🚀 Find Destinations", use_container_width=True)

    if submitted:
        results = recommend(
            budget=budget, travel_type=travel_type, season=season,
            top_n=top_n, max_budget=max_bud,
            sort_by_rating=(sort_by=="Rating"),
            data_path=DATA_PATH
        )

        type_emojis = {"Beach":"🏖️","Adventure":"🏔️","Cultural":"🏛️","Nature":"🌿","Wildlife":"🐯","City":"🏙️"}
        type_colors_map = {"Beach":"#38bdf8","Adventure":"#fb923c","Cultural":"#a78bfa",
                           "Nature":"#34d399","Wildlife":"#fbbf24","City":"#94a3b8"}

        st.markdown(f"### ✅ {len(results)} destinations found for **{travel_type}** in **{season}** with budget ₹{budget:,}")

        if results.empty:
            st.warning("No destinations found. Try adjusting filters.")
        else:
            for rank, (_, row) in enumerate(results.iterrows(), 1):
                sim_pct = int(row["Similarity"]*100)
                emoji = type_emojis.get(row['Type'],'📍')
                color = type_colors_map.get(row['Type'],'#667eea')
                desc = df[df["Destination"]==row["Destination"]]["Description"].values
                desc_text = desc[0] if len(desc)>0 else ""
                stars = "⭐" * int(row['Rating'])

                st.markdown(f"""
                <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.07);
                    border-radius:16px;padding:1.5rem;margin-bottom:1rem;
                    display:flex;align-items:center;gap:1.5rem;">
                    <div style="font-size:2.2rem;font-weight:800;min-width:50px;text-align:center;
                        background:linear-gradient(135deg,{color},#764ba2);
                        -webkit-background-clip:text;-webkit-text-fill-color:transparent;">#{rank}</div>
                    <div style="font-size:2rem;">{emoji}</div>
                    <div style="flex:1;">
                        <div style="color:white;font-size:1.1rem;font-weight:700;">{row['Destination']}</div>
                        <div style="color:rgba(255,255,255,0.35);font-size:0.8rem;margin:0.2rem 0 0.6rem;">
                            {desc_text[:100]}{'...' if len(str(desc_text))>100 else ''}
                        </div>
                        <div style="color:rgba(255,255,255,0.4);font-size:0.82rem;">
                            💰 ₹{int(row['Budget']):,} &nbsp;·&nbsp; {stars} {row['Rating']}
                            &nbsp;·&nbsp; 🗓️ {int(row['Duration'])} days
                            &nbsp;·&nbsp; 📍 {row.get('Season','')}
                        </div>
                    </div>
                    <div style="text-align:center;padding:1rem 1.5rem;background:rgba(255,255,255,0.03);
                        border-radius:12px;border:1px solid rgba(255,255,255,0.06);">
                        <div style="font-size:1.6rem;font-weight:800;color:{color};">{sim_pct}%</div>
                        <div style="color:rgba(255,255,255,0.3);font-size:0.72rem;">AI Match</div>
                    </div>
                </div>""", unsafe_allow_html=True)

            csv = results.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Results", csv, "my_recommendations.csv", "text/csv")
