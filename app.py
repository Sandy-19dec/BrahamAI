"""
╔══════════════════════════════════════════════════════╗
║   BrahamAI — AI Travel Recommendation System         ║
║   Built for Final-Year BCA Data Science Project      ║
╚══════════════════════════════════════════════════════╝
Run: streamlit run app.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd

from models.recommender  import recommend, evaluate_model, load_data
from visualizations.charts import (plot_type_distribution, plot_budget_vs_rating,
                                    plot_top_destinations, plot_season_distribution)
from utils.chatbot import chatbot_response

# ─────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────
st.set_page_config(
    page_title="BrahamAI — Travel Recommender",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0F1117; }
    .block-container { padding-top: 1.5rem; }

    /* Hero banner */
    .hero-banner {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 40%, #0f3460 100%);
        border: 1px solid #e94560;
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .hero-title {
        font-size: 2.8rem; font-weight: 800;
        background: linear-gradient(90deg, #e94560, #45B7D1);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .hero-sub { color: #aaa; font-size: 1.05rem; margin-top: 0.3rem; }

    /* Metric cards */
    .metric-card {
        background: #1e1e2e; border: 1px solid #333;
        border-radius: 12px; padding: 1rem 1.2rem;
        text-align: center;
    }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #45B7D1; }
    .metric-label { font-size: 0.8rem; color: #888; margin-top: 0.2rem; }

    /* Rec card */
    .rec-card {
        background: linear-gradient(135deg, #1e1e2e, #16213e);
        border: 1px solid #e94560; border-radius: 12px;
        padding: 1rem 1.2rem; margin-bottom: 0.8rem;
    }
    .rec-title { font-size: 1.1rem; font-weight: 700; color: #45B7D1; }
    .rec-badge {
        display: inline-block; background: #e94560;
        color: white; font-size: 0.7rem; font-weight: 600;
        padding: 2px 8px; border-radius: 20px; margin-right: 6px;
    }
    .rec-detail { color: #bbb; font-size: 0.88rem; margin-top: 0.4rem; }

    /* Chat bubble */
    .chat-user {
        background: #1a1a40; border-left: 3px solid #45B7D1;
        border-radius: 8px; padding: 0.6rem 1rem; margin: 0.4rem 0;
        color: #ddd;
    }
    .chat-bot {
        background: #1a2e1a; border-left: 3px solid #4ECDC4;
        border-radius: 8px; padding: 0.6rem 1rem; margin: 0.4rem 0;
        color: #ddd;
    }

    /* Section headers */
    .section-header {
        font-size: 1.3rem; font-weight: 700;
        color: #e94560; border-bottom: 1px solid #333;
        padding-bottom: 0.4rem; margin: 1.2rem 0 0.8rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #1a1a2e !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# Hero Banner
# ─────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <p class="hero-title">✈️ BrahamAI</p>
    <p class="hero-sub">AI-Powered Travel Recommendation System &nbsp;|&nbsp; Final Year BCA Data Science Project</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# Sidebar — User Preferences
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎯 Your Travel Preferences")
    st.markdown("---")

    budget = st.slider(
        "💰 Budget (₹)",
        min_value=5000, max_value=40000,
        value=15000, step=1000,
        help="Your total travel budget in Indian Rupees"
    )

    travel_type = st.selectbox(
        "🏷️ Travel Type",
        ["Beach", "Adventure", "Cultural", "Nature", "Wildlife", "City"]
    )

    season = st.selectbox(
        "🗓️ Preferred Season",
        ["Winter", "Summer", "Monsoon"]
    )

    st.markdown("---")
    st.markdown("### 🔧 Advanced Filters")

    enable_budget_filter = st.checkbox("Filter by budget range")
    min_budget, max_budget = None, None
    if enable_budget_filter:
        min_budget, max_budget = st.slider(
            "Budget Range (₹)",
            5000, 40000, (8000, 25000), step=1000
        )

    sort_by_rating = st.checkbox("Sort results by Rating (not similarity)")
    top_n = st.slider("Number of recommendations", 3, 10, 5)

    st.markdown("---")
    recommend_btn = st.button("🚀 Get Recommendations", use_container_width=True, type="primary")

# ─────────────────────────────────────────
# Tab Layout
# ─────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺️ Recommendations",
    "📊 Data Insights",
    "📈 Model Evaluation",
    "🤖 Travel Chatbot",
    "📂 Dataset"
])

# ══════════════════════════════════════════
# TAB 1 — Recommendations
# ══════════════════════════════════════════
with tab1:
    # Quick stats row
    df_all = load_data("data/travel_data.csv")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{len(df_all)}</div>
            <div class="metric-label">Destinations</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{df_all['Type'].nunique()}</div>
            <div class="metric-label">Travel Types</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{df_all['Rating'].max()}</div>
            <div class="metric-label">Top Rating</div></div>""", unsafe_allow_html=True)
    with c4:
        avg_b = int(df_all['Budget'].mean())
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">₹{avg_b:,}</div>
            <div class="metric-label">Avg Budget</div></div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>🎯 Recommended Destinations</div>", unsafe_allow_html=True)

    if recommend_btn or True:  # auto-run on page load
        results = recommend(
            budget=budget,
            travel_type=travel_type,
            season=season,
            top_n=top_n,
            min_budget=min_budget,
            max_budget=max_budget,
            sort_by_rating=sort_by_rating,
            data_path="data/travel_data.csv"
        )

        if results.empty:
            st.warning("⚠️ No destinations found for this filter. Try widening your budget range.")
        else:
            # Cards view
            for rank, (_, row) in enumerate(results.iterrows(), 1):
                stars = "⭐" * int(row["Rating"]) + ("½" if row["Rating"] % 1 >= 0.5 else "")
                sim_pct = int(row["Similarity"] * 100)
                st.markdown(f"""
                <div class="rec-card">
                    <div class="rec-title">#{rank} &nbsp; {row['Destination']}</div>
                    <div style="margin: 0.3rem 0;">
                        <span class="rec-badge">{row['Type']}</span>
                        <span class="rec-badge" style="background:#0f3460;">{row['Season']}</span>
                        <span class="rec-badge" style="background:#1a5276;">Match {sim_pct}%</span>
                    </div>
                    <div class="rec-detail">
                        💰 Budget: ₹{int(row['Budget']):,} &nbsp;|&nbsp;
                        {stars} {row['Rating']}/5 &nbsp;|&nbsp;
                        🗓️ {int(row['Duration'])} days
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("**📋 Results Table**")
            display_df = results.copy()
            display_df["Budget"] = display_df["Budget"].apply(lambda x: f"₹{int(x):,}")
            display_df["Similarity"] = display_df["Similarity"].apply(lambda x: f"{x:.2%}")
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            csv = results.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Results", csv, "brahamai_recommendations.csv", "text/csv")

# ══════════════════════════════════════════
# TAB 2 — Data Insights
# ══════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-header'>📊 Exploratory Data Analysis</div>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Travel Type Distribution**")
        fig1 = plot_type_distribution(df_all, save=False)
        st.pyplot(fig1)
    with col_b:
        st.markdown("**Season-wise Distribution**")
        fig4 = plot_season_distribution(df_all, save=False)
        st.pyplot(fig4)

    st.markdown("**Budget vs Rating (bubble size = trip duration)**")
    fig2 = plot_budget_vs_rating(df_all, save=False)
    st.pyplot(fig2)

    st.markdown("**Top 10 Destinations by Rating**")
    fig3 = plot_top_destinations(df_all, save=False)
    st.pyplot(fig3)

# ══════════════════════════════════════════
# TAB 3 — Model Evaluation
# ══════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-header'>📈 Model Performance</div>", unsafe_allow_html=True)

    with st.spinner("Running evaluation..."):
        metrics = evaluate_model("data/travel_data.csv")

    m1, m2, m3, m4 = st.columns(4)
    metric_items = list(metrics.items())
    for col, (label, val) in zip([m1, m2, m3, m4], metric_items):
        with col:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div></div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
### 📖 How the Model Works

**Step 1 — One-Hot Encoding**
Categorical columns (`Type`, `Season`) are converted to binary columns. For example, `Type=Beach` becomes `type_Beach=1` with all other type columns = 0.

**Step 2 — Min-Max Normalization**
Numeric columns (`Budget`, `Rating`, `Duration`) are scaled to [0, 1] so they don't dominate similarity calculations.

**Step 3 — User Profile Vector**
When a user enters their preferences, we build the same kind of feature vector — matching the exact encoding used for destinations.

**Step 4 — Cosine Similarity**
We compute the angle between the user's vector and every destination vector. A score of 1.0 = perfect match; 0.0 = no similarity.

**Evaluation Strategy**
- For each destination, we treat its own Type + Season as the query
- A recommendation is **relevant** if it shares the same Travel Type
- We compute **Precision@5** (how many of the top-5 are relevant) and **Recall@5** (what fraction of all relevant items were retrieved)
    """)

# ══════════════════════════════════════════
# TAB 4 — Chatbot
# ══════════════════════════════════════════
with tab4:
    st.markdown("<div class='section-header'>🤖 BrahamAI Travel Chatbot</div>", unsafe_allow_html=True)
    st.caption("Ask me anything about travel destinations, seasons, or budget tips!")

    # Init chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            ("bot", "👋 Hi! I'm BrahamAI. Ask me about beach trips, budget travel, adventure spots, or seasonal recommendations!")
        ]

    # Display history
    for role, msg in st.session_state.chat_history:
        css_class = "chat-bot" if role == "bot" else "chat-user"
        icon = "🤖" if role == "bot" else "🧑"
        st.markdown(f'<div class="{css_class}">{icon} {msg}</div>', unsafe_allow_html=True)

    # Quick suggestion buttons
    st.markdown("**Quick questions:**")
    quick_cols = st.columns(4)
    quick_prompts = ["Suggest a beach trip", "Best adventure places", "Budget under ₹10,000", "Winter destinations"]
    for col, prompt in zip(quick_cols, quick_prompts):
        with col:
            if st.button(prompt, key=f"qp_{prompt}"):
                response = chatbot_response(prompt)
                st.session_state.chat_history.append(("user", prompt))
                st.session_state.chat_history.append(("bot", response))
                st.rerun()

    # Text input
    with st.form("chat_form", clear_on_submit=True):
        user_msg = st.text_input("Type your message...", placeholder="e.g. Suggest adventure trips in summer")
        submitted = st.form_submit_button("Send 📨")
        if submitted and user_msg.strip():
            bot_reply = chatbot_response(user_msg)
            st.session_state.chat_history.append(("user", user_msg))
            st.session_state.chat_history.append(("bot", bot_reply))
            st.rerun()

    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# ══════════════════════════════════════════
# TAB 5 — Dataset
# ══════════════════════════════════════════
with tab5:
    st.markdown("<div class='section-header'>📂 Travel Dataset</div>", unsafe_allow_html=True)

    col_filter1, col_filter2 = st.columns(2)
    with col_filter1:
        type_filter = st.multiselect("Filter by Type", options=df_all["Type"].unique(), default=list(df_all["Type"].unique()))
    with col_filter2:
        season_filter = st.multiselect("Filter by Season", options=df_all["Season"].unique(), default=list(df_all["Season"].unique()))

    filtered = df_all[df_all["Type"].isin(type_filter) & df_all["Season"].isin(season_filter)]
    st.write(f"Showing **{len(filtered)}** of {len(df_all)} destinations")

    # Format budget display
    display = filtered.copy()
    display["Budget"] = display["Budget"].apply(lambda x: f"₹{x:,}")
    st.dataframe(display, use_container_width=True, hide_index=True)

    full_csv = df_all.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Full Dataset", full_csv, "travel_data.csv", "text/csv")

# ─────────────────────────────────────────
# Footer
# ─────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center style='color:#555; font-size:0.8rem;'>"
    "BrahamAI &nbsp;|&nbsp; Final Year BCA Data Science Project &nbsp;|&nbsp; "
    "Built with Python · Scikit-learn · Streamlit · Matplotlib"
    "</center>",
    unsafe_allow_html=True
)
