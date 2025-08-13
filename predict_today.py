# predict_today.py
"""
Daily automated script to:
1. Load or train models
2. Predict today's unplayed games
3. Save predictions to the database
"""

import sqlite3
import pandas as pd
import numpy as np
import joblib
from datetime import date
from mlb_feature_engineering import load_data, train_models

# --- Step 1: Load or Train Models ---
try:
    clf = joblib.load("model_winner.pkl")
    reg_margin = joblib.load("model_margin.pkl")
    reg_total = joblib.load("model_total.pkl")
    print("[INFO] Models loaded from disk.")
except Exception:
    print("[INFO] Training models from scratch...")
    X, y = load_data("mlb_predictions.db")
    clf, reg_margin, reg_total = train_models(X, y)
    joblib.dump(clf, "model_winner.pkl")
    joblib.dump(reg_margin, "model_margin.pkl")
    joblib.dump(reg_total, "model_total.pkl")
    print("[INFO] Models trained and saved.")

# --- Step 2: Connect to DB ---
conn = sqlite3.connect("mlb_predictions.db")
cursor = conn.cursor()

# --- Step 3: Get Today's Games ---
today_str = date.today().isoformat()
games_today = pd.read_sql_query(
    """
    SELECT game_id, date, home_team, away_team
    FROM games
    WHERE date = ? AND home_score IS NULL AND away_score IS NULL
    """,
    conn, params=[today_str]
)

if games_today.empty:
    print(f"[WARN] No unplayed games found for {today_str}.")
    conn.close()
    exit()

# --- Step 4: Load Team Stats ---
team_stats = pd.read_sql("SELECT * FROM team_stats", conn)
stats_map = team_stats.set_index("name").to_dict("index")

# --- Step 5: Generate Predictions ---
predictions = []

def safe_div(n, d): return n / d if d else 0.0

for _, row in games_today.iterrows():
    g_id = row.game_id
    home = row.home_team
    away = row.away_team

    stats_home = stats_map.get(home)
    stats_away = stats_map.get(away)
    if not stats_home or not stats_away:
        print(f"[WARN] Skipping {home} vs {away} (missing stats)")
        continue

    features = [
        safe_div(stats_home['wins'], stats_home['wins'] + stats_home['losses']),
        safe_div(stats_away['wins'], stats_away['wins'] + stats_away['losses']),
        safe_div(stats_home['runs_scored'], stats_home['wins'] + stats_home['losses']),
        safe_div(stats_away['runs_scored'], stats_away['wins'] + stats_away['losses']),
        safe_div(stats_home['runs_allowed'], stats_home['wins'] + stats_home['losses']),
        safe_div(stats_away['runs_allowed'], stats_away['wins'] + stats_away['losses']),
        1.0  # Home field advantage
    ]

    win = clf.predict([features])[0]
    margin = float(reg_margin.predict([features])[0])
    total = float(reg_total.predict([features])[0])
    predicted = home if win == 1 else away

    predictions.append((g_id, today_str, home, away, predicted, margin, total))

# --- Step 6: Store Predictions ---
if predictions:
    cursor.executemany(
        """
        INSERT OR REPLACE INTO predictions
        (game_id, date, home_team, away_team, predicted_winner, predicted_margin, predicted_total)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        predictions
    )
    conn.commit()
    print(f"[INFO] Inserted {len(predictions)} predictions for {today_str}.")
else:
    print(f"[WARN] No usable games to predict for {today_str}.")

conn.close()
