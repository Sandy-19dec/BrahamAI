import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("  BrahamAI — Project Validation")
print("=" * 60)

errors = []

# ── Syntax check ─────────────────────────────────────────────
files_to_check = [
    "models/recommender.py",
    "utils/chatbot.py",
    "visualizations/charts.py",
    "ml_demo.py",
]
for f in files_to_check:
    try:
        with open(f, encoding="utf-8") as fh:
            compile(fh.read(), f, "exec")
        print(f"  [OK]  Syntax: {f}")
    except SyntaxError as e:
        errors.append(f"  [ERR] Syntax {f}: {e}")

# ── Dataset ───────────────────────────────────────────────────
try:
    from models.recommender import load_data, recommend, evaluate_model
    df = load_data("data/travel_data.csv")
    n_dest   = df["Destination"].nunique()
    n_states = df["State"].nunique()
    has_latlon = "Lat" in df.columns and "Lon" in df.columns
    print(f"  [OK]  Dataset: {n_dest} destinations, {n_states} states, Lat/Lon={'yes' if has_latlon else 'MISSING'}")
    print(f"        Types:   {sorted(df['Type'].unique())}")
    print(f"        Seasons: {sorted(df['Season'].unique())}")
    city_count = len(df[df['Type'] == 'City'].drop_duplicates('Destination'))
    monsoon_count = len(df[df['Season'] == 'Monsoon'].drop_duplicates('Destination'))
    print(f"        City destinations: {city_count} | Monsoon destinations: {monsoon_count}")
except Exception as e:
    errors.append(f"  [ERR] Dataset: {e}")

# ── recommend() ───────────────────────────────────────────────
try:
    results = recommend(15000, "Beach", "Winter", top_n=3, data_path="data/travel_data.csv")
    has_score = "Score" in results.columns
    has_lat   = "Lat" in results.columns
    print(f"  [OK]  recommend(): {len(results)} results | Score col={has_score} | Lat/Lon={has_lat}")
    print(f"        Top pick: {results.iloc[0]['Destination']} (Score={results.iloc[0]['Score']:.3f})")
except Exception as e:
    errors.append(f"  [ERR] recommend(): {e}")

# ── evaluate_model() ─────────────────────────────────────────
try:
    metrics = evaluate_model("data/travel_data.csv")
    has_per_type = "Per Type Precision" in metrics
    print(f"  [OK]  evaluate_model(): P@5={metrics['Avg Precision@5']}  R@5={metrics['Avg Recall@5']}  F1={metrics['F1 Score']}")
    print(f"        Per-type precision: {has_per_type}")
    if has_per_type:
        for t, v in metrics["Per Type Precision"].items():
            print(f"          {t:<12}: {v:.3f}")
except Exception as e:
    errors.append(f"  [ERR] evaluate_model(): {e}")

# ── chatbot ───────────────────────────────────────────────────
try:
    from utils.chatbot import chatbot_response
    tests = [
        ("beach adventure under 15000", "multi-keyword + budget"),
        ("monsoon nature trip",          "multi-keyword"),
        ("hi",                           "greeting"),
        ("12000 budget",                 "budget-only"),
        ("city nightlife",               "city keyword"),
    ]
    for inp, label in tests:
        resp = chatbot_response(inp)
        safe = resp.encode("ascii", "replace").decode("ascii")
        print(f"  [OK]  Chatbot ({label}): \"{safe[:70]}\"")
except Exception as e:
    errors.append(f"  [ERR] Chatbot: {e}")

# ── charts module ─────────────────────────────────────────────
try:
    from visualizations.charts import plot_india_map, plot_per_type_precision
    print("  [OK]  visualizations/charts.py imported successfully")
except Exception as e:
    errors.append(f"  [ERR] charts: {e}")

# ── plotly check ─────────────────────────────────────────────
try:
    import plotly
    print(f"  [OK]  Plotly {plotly.__version__} available")
except ImportError:
    errors.append("  [ERR] Plotly not installed — run: pip install plotly")

# ── Summary ───────────────────────────────────────────────────
import sys
sys.stdout.reconfigure(encoding="utf-8")
print("\n" + "=" * 60)
if errors:
    print(f"  FAILED — {len(errors)} error(s):")
    for e in errors:
        print(e)
else:
    print("  ALL CHECKS PASSED [OK]")
print("=" * 60)
