"""
BrahamAI — Home Page (Main Landing)
Run: streamlit run app.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd

from models.recommender import recommend, load_data
from utils.chatbot import chatbot_response

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "travel_data.csv")

st.set_page_config(
    page_title="BrahamAI — AI Travel Planner",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif; }
.main { background: #0a0a0f; }
.block-container { padding: 0 !important; max-width: 100% !important; }
[data-testid="stSidebar"] { display: none; }
header { display: none !important; }
footer { display: none !important; }

/* ── NAV ── */
.nav-bar {
    background: rgba(10,10,15,0.95);
    backdrop-filter: blur(20px);
    border-bottom: 1px solid rgba(255,255,255,0.08);
    padding: 1rem 3rem;
    display: flex; align-items: center; justify-content: space-between;
    position: sticky; top: 0; z-index: 100;
}
.nav-logo { font-size: 1.4rem; font-weight: 800;
    background: linear-gradient(135deg, #667eea, #764ba2);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.nav-links { display: flex; gap: 2rem; }
.nav-link { color: rgba(255,255,255,0.6); font-size: 0.9rem;
    text-decoration: none; transition: color 0.2s; cursor: pointer; }
.nav-link:hover { color: white; }

/* ── HERO ── */
.hero {
    min-height: 92vh;
    background: radial-gradient(ellipse at 20% 50%, rgba(102,126,234,0.15) 0%, transparent 60%),
                radial-gradient(ellipse at 80% 20%, rgba(118,75,162,0.12) 0%, transparent 60%),
                radial-gradient(ellipse at 60% 80%, rgba(240,147,43,0.08) 0%, transparent 50%),
                linear-gradient(180deg, #0a0a0f 0%, #0d0d18 100%);
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; padding: 4rem 2rem; text-align: center;
}
.hero-badge {
    display: inline-flex; align-items: center; gap: 0.5rem;
    background: rgba(102,126,234,0.15); border: 1px solid rgba(102,126,234,0.3);
    border-radius: 50px; padding: 0.4rem 1rem; font-size: 0.8rem;
    color: #a78bfa; margin-bottom: 1.5rem; letter-spacing: 0.05em;
}
.hero-title {
    font-size: clamp(2.5rem, 6vw, 5rem); font-weight: 800; line-height: 1.1;
    color: white; margin: 0 0 1rem;
}
.hero-title span {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 40%, #f0932b 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero-sub {
    font-size: 1.15rem; color: rgba(255,255,255,0.5); max-width: 600px;
    line-height: 1.7; margin: 0 auto 2.5rem;
}
.hero-stats {
    display: flex; gap: 3rem; justify-content: center;
    margin: 2rem 0; flex-wrap: wrap;
}
.hero-stat { text-align: center; }
.hero-stat-num { font-size: 2rem; font-weight: 700; color: white; }
.hero-stat-label { font-size: 0.8rem; color: rgba(255,255,255,0.4); margin-top: 0.2rem; }

/* ── SEARCH CARD ── */
.search-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 24px; padding: 2.5rem;
    max-width: 800px; width: 100%; margin: 0 auto;
    backdrop-filter: blur(20px);
}
.search-title { color: white; font-size: 1.1rem; font-weight: 600; margin-bottom: 1.5rem; }

/* ── DESTINATION CARDS ── */
.section { padding: 4rem 3rem; background: #0d0d18; }
.section-label { color: #667eea; font-size: 0.8rem; font-weight: 600;
    letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.5rem; }
.section-title { color: white; font-size: 2rem; font-weight: 700; margin-bottom: 0.5rem; }
.section-sub { color: rgba(255,255,255,0.4); font-size: 0.95rem; margin-bottom: 2.5rem; }

.dest-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 1.2rem; }

.dest-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px; padding: 1.5rem;
    transition: all 0.3s ease; cursor: pointer; position: relative; overflow: hidden;
}
.dest-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #667eea, #764ba2);
    opacity: 0; transition: opacity 0.3s;
}
.dest-card:hover { background: rgba(102,126,234,0.08);
    border-color: rgba(102,126,234,0.3); transform: translateY(-2px); }
.dest-card:hover::before { opacity: 1; }
.dest-emoji { font-size: 2rem; margin-bottom: 0.8rem; }
.dest-name { color: white; font-size: 1.05rem; font-weight: 600; margin-bottom: 0.3rem; }
.dest-state { color: rgba(255,255,255,0.35); font-size: 0.78rem; margin-bottom: 0.8rem; }
.dest-desc { color: rgba(255,255,255,0.5); font-size: 0.82rem; line-height: 1.5; margin-bottom: 1rem; }
.dest-footer { display: flex; align-items: center; justify-content: space-between; }
.dest-budget { color: #34d399; font-size: 0.82rem; font-weight: 500; }
.dest-rating { color: #fbbf24; font-size: 0.82rem; }
.dest-tags { display: flex; gap: 0.4rem; flex-wrap: wrap; margin-bottom: 0.8rem; }
.dest-tag {
    font-size: 0.68rem; padding: 0.2rem 0.6rem; border-radius: 20px;
    font-weight: 500;
}
.tag-beach { background: rgba(56,189,248,0.15); color: #38bdf8; border: 1px solid rgba(56,189,248,0.2); }
.tag-adventure { background: rgba(251,146,60,0.15); color: #fb923c; border: 1px solid rgba(251,146,60,0.2); }
.tag-cultural { background: rgba(167,139,250,0.15); color: #a78bfa; border: 1px solid rgba(167,139,250,0.2); }
.tag-nature { background: rgba(52,211,153,0.15); color: #34d399; border: 1px solid rgba(52,211,153,0.2); }
.tag-wildlife { background: rgba(251,191,36,0.15); color: #fbbf24; border: 1px solid rgba(251,191,36,0.2); }
.tag-city { background: rgba(148,163,184,0.15); color: #94a3b8; border: 1px solid rgba(148,163,184,0.2); }
.tag-winter { background: rgba(96,165,250,0.15); color: #60a5fa; }
.tag-summer { background: rgba(253,186,116,0.15); color: #fdba74; }
.tag-monsoon { background: rgba(110,231,183,0.15); color: #6ee7b7; }

/* ── REC RESULTS ── */
.rec-result {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px; padding: 1.5rem; margin-bottom: 1rem;
    display: flex; align-items: center; gap: 1.5rem;
    transition: all 0.2s;
}
.rec-result:hover { background: rgba(102,126,234,0.06); border-color: rgba(102,126,234,0.2); }
.rec-rank {
    font-size: 2rem; font-weight: 800; min-width: 50px; text-align: center;
    background: linear-gradient(135deg, #667eea, #764ba2);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.rec-info { flex: 1; }
.rec-name { color: white; font-size: 1.1rem; font-weight: 600; margin-bottom: 0.3rem; }
.rec-meta { color: rgba(255,255,255,0.4); font-size: 0.82rem; margin-bottom: 0.5rem; }
.rec-right { text-align: right; }
.match-score {
    font-size: 1.4rem; font-weight: 700;
    background: linear-gradient(135deg, #34d399, #06b6d4);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.match-label { color: rgba(255,255,255,0.3); font-size: 0.72rem; }

/* ── HOW IT WORKS ── */
.how-section { padding: 4rem 3rem; background: #0a0a0f; }
.steps-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1.5rem; margin-top: 2rem; }
.step-card {
    background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px; padding: 1.8rem; text-align: center;
}
.step-num {
    width: 48px; height: 48px; border-radius: 50%;
    background: linear-gradient(135deg, #667eea, #764ba2);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem; font-weight: 700; color: white;
    margin: 0 auto 1rem;
}
.step-title { color: white; font-weight: 600; margin-bottom: 0.5rem; }
.step-desc { color: rgba(255,255,255,0.4); font-size: 0.85rem; line-height: 1.6; }

/* ── FOOTER ── */
.footer {
    background: #070710; border-top: 1px solid rgba(255,255,255,0.06);
    padding: 2rem 3rem; text-align: center;
    color: rgba(255,255,255,0.25); font-size: 0.82rem;
}

/* Streamlit overrides */
.stSelectbox > div > div { background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important; color: white !important; border-radius: 12px !important; }
.stSlider { padding: 0.5rem 0; }
div[data-testid="stButton"] button {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important; border: none !important; border-radius: 12px !important;
    padding: 0.7rem 2rem !important; font-weight: 600 !important; font-size: 1rem !important;
    width: 100% !important; transition: opacity 0.2s !important;
}
div[data-testid="stButton"] button:hover { opacity: 0.9 !important; }
</style>
""", unsafe_allow_html=True)

# ── NAV ──
st.markdown("""
<div class="nav-bar">
    <div class="nav-logo">✈ BrahamAI</div>
    <div class="nav-links">
        <span class="nav-link">Home</span>
        <span class="nav-link">Explore</span>
        <span class="nav-link">About</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Load data
df = load_data(DATA_PATH)

# ── HERO ──
st.markdown("""
<div class="hero">
    <div class="hero-badge">✨ AI-Powered Travel Intelligence</div>
    <h1 class="hero-title">Discover Your Perfect<br><span>Indian Adventure</span></h1>
    <p class="hero-sub">BrahamAI uses machine learning to match you with destinations that fit your budget, style, and season — instantly.</p>
</div>
""", unsafe_allow_html=True)

# ── SEARCH FORM ──
st.markdown('<div style="background:#0a0a0f; padding: 0 3rem 4rem;">', unsafe_allow_html=True)
st.markdown('<div class="search-card" style="max-width:900px;margin:0 auto;">', unsafe_allow_html=True)
st.markdown('<div class="search-title">🎯 Find Your Perfect Destination</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    budget = st.slider("💰 Budget (₹)", 5000, 40000, 15000, 1000)
with col2:
    travel_type = st.selectbox("🏷️ Travel Style", ["Beach", "Adventure", "Cultural", "Nature", "Wildlife", "City"])
with col3:
    season = st.selectbox("🗓️ Season", ["Winter", "Summer", "Monsoon"])

search_btn = st.button("🚀 Find My Destinations", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ── STATS ──
col_s1, col_s2, col_s3, col_s4 = st.columns(4)
stats = [
    ("🗺️", str(len(df)), "Destinations"),
    ("🏷️", str(df['Type'].nunique()), "Travel Types"),
    ("⭐", str(df['Rating'].max()), "Top Rating"),
    ("📍", str(df['State'].nunique()), "States Covered"),
]
for col, (icon, num, label) in zip([col_s1, col_s2, col_s3, col_s4], stats):
    with col:
        st.markdown(f"""
        <div style="text-align:center; padding:1.5rem; background:rgba(255,255,255,0.02);
            border:1px solid rgba(255,255,255,0.06); border-radius:16px; margin:0.5rem 0.2rem;">
            <div style="font-size:1.8rem;">{icon}</div>
            <div style="font-size:1.8rem; font-weight:700; color:white; margin:0.3rem 0;">{num}</div>
            <div style="font-size:0.78rem; color:rgba(255,255,255,0.35);">{label}</div>
        </div>""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ── RESULTS ──
if search_btn:
    results = recommend(budget, travel_type, season, top_n=6, data_path=DATA_PATH)
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown(f'<div class="section-label">AI Recommendations</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">Top picks for {travel_type} in {season}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-sub">Based on your ₹{budget:,} budget • Sorted by AI match score</div>', unsafe_allow_html=True)

    type_emoji = {"Beach":"🏖️","Adventure":"🏔️","Cultural":"🏛️","Nature":"🌿","Wildlife":"🐯","City":"🏙️"}
    emoji = type_emoji.get(travel_type, "📍")

    if results.empty:
        st.warning("No destinations found. Try adjusting your filters.")
    else:
        for rank, (_, row) in enumerate(results.iterrows(), 1):
            sim_pct = int(row["Similarity"] * 100)
            desc = df[df["Destination"] == row["Destination"]]["Description"].values
            desc_text = desc[0] if len(desc) > 0 else ""
            tag_class = f"tag-{row['Type'].lower()}"
            season_class = f"tag-{row['Season'].lower()}"
            stars = "⭐" * int(row["Rating"])

            st.markdown(f"""
            <div class="rec-result">
                <div class="rec-rank">#{rank}</div>
                <div style="font-size:2rem;">{emoji}</div>
                <div class="rec-info">
                    <div class="rec-name">{row['Destination']}</div>
                    <div style="display:flex;gap:0.4rem;margin-bottom:0.5rem;">
                        <span class="dest-tag {tag_class}">{row['Type']}</span>
                        <span class="dest-tag {season_class}">{row['Season']}</span>
                    </div>
                    <div class="rec-meta">{desc_text}</div>
                    <div style="color:rgba(255,255,255,0.4);font-size:0.8rem;margin-top:0.4rem;">
                        💰 ₹{int(row['Budget']):,} &nbsp;|&nbsp; {stars} {row['Rating']} &nbsp;|&nbsp; 🗓️ {int(row['Duration'])} days
                    </div>
                </div>
                <div class="rec-right">
                    <div class="match-score">{sim_pct}%</div>
                    <div class="match-label">AI Match</div>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ── FEATURED DESTINATIONS ──
type_emojis = {"Beach":"🏖️","Adventure":"🏔️","Cultural":"🏛️","Nature":"🌿","Wildlife":"🐯","City":"🏙️"}
featured = df.sort_values("Rating", ascending=False).drop_duplicates("Destination").head(12)

st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown('<div class="section-label">Top Rated</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Trending Destinations</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Highest rated destinations loved by travelers across India</div>', unsafe_allow_html=True)

cols = st.columns(4)
for i, (_, row) in enumerate(featured.iterrows()):
    tag_cls = f"tag-{row['Type'].lower()}"
    season_cls = f"tag-{row['Season'].lower()}"
    emoji = type_emojis.get(row['Type'], "📍")
    desc = row.get('Description', '')
    with cols[i % 4]:
        st.markdown(f"""
        <div class="dest-card">
            <div class="dest-emoji">{emoji}</div>
            <div class="dest-name">{row['Destination']}</div>
            <div class="dest-state">📍 {row.get('State','India')}</div>
            <div class="dest-tags">
                <span class="dest-tag {tag_cls}">{row['Type']}</span>
                <span class="dest-tag {season_cls}">{row['Season']}</span>
            </div>
            <div class="dest-desc">{desc[:80]}{'...' if len(str(desc))>80 else ''}</div>
            <div class="dest-footer">
                <span class="dest-budget">₹{int(row['Budget']):,}</span>
                <span class="dest-rating">⭐ {row['Rating']}</span>
            </div>
        </div>""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ── HOW IT WORKS ──
st.markdown("""
<div class="how-section">
    <div class="section-label">The Science</div>
    <div class="section-title">How BrahamAI Works</div>
    <div class="section-sub">4 simple steps from your preference to perfect destination</div>
    <div class="steps-grid">
        <div class="step-card">
            <div class="step-num">1</div>
            <div class="step-title">You set preferences</div>
            <div class="step-desc">Tell us your budget, travel style, and preferred season</div>
        </div>
        <div class="step-card">
            <div class="step-num">2</div>
            <div class="step-title">AI encodes features</div>
            <div class="step-desc">One-Hot Encoding converts your choices into a numeric vector</div>
        </div>
        <div class="step-card">
            <div class="step-num">3</div>
            <div class="step-title">Cosine similarity</div>
            <div class="step-desc">Your vector is matched against 100+ destinations in our database</div>
        </div>
        <div class="step-card">
            <div class="step-num">4</div>
            <div class="step-title">Ranked results</div>
            <div class="step-desc">Top matches are returned with a % match score just for you</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── FOOTER ──
st.markdown("""
<div class="footer">
    ✈️ BrahamAI &nbsp;|&nbsp; Final Year BCA Data Science Project &nbsp;|&nbsp;
    Built with Python · Scikit-learn · Streamlit · Pandas
</div>
""", unsafe_allow_html=True)
