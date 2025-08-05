"""
Daily update and prediction script for MLB games.
This script should be run daily (e.g., as a scheduled job) to:
1. Fetch yesterday's game results and update the database.
2. Update team stats (wins, losses, runs, etc.) in the database.
3. Retrain prediction models with the latest data.
4. Predict outcomes for today's games and store these predictions.
5. Save the trained models for use in the Streamlit app.
"""
# predict_today.py
# ----------------
# Loads today’s scheduled games, builds features, applies model, saves predictions

import sqlite3
import pandas as pd
import numpy as np
from datetime import date
import joblib
from mlb_feature_engineering import load_data, train_models

DB_PATH = "mlb_predictions.db"
TODAY = date.today().isoformat()

# Connect to database
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

# Step 1: Load & train models (or load existing)
try:
    clf = joblib.load("model_winner.pkl")
    reg_margin = joblib.load("model_margin.pkl")
    reg_total = joblib.load("model_total.pkl")
    print("[INFO] Models loaded from disk.")
except:
    print("[INFO] Training new models...")
    X, y = load_data(DB_PATH)
    clf, reg_margin, reg_total = train_models(X, y)
    joblib.dump(clf, "model_winner.pkl")
    joblib.dump(reg_margin, "model_margin.pkl")
    joblib.dump(reg_total, "model_total.pkl")
    print("[INFO] Models trained and saved.")

# Step 2: Get today’s unplayed games
games_today = pd.read_sql("""
    SELECT game_id, home_team, away_team
    FROM games
    WHERE date = ? AND home_score IS NULL AND away_score IS NULL
""", conn, params=[TODAY])

if games_today.empty:
    print(f"[WARN] No unplayed games found for {TODAY}.")
    conn.close()
    exit()

# Step 3: Load team stats
stats_df = pd.read_sql("SELECT * FROM team_stats", conn)
stats_map = stats_df.set_index("name").to_dict("index")

def safe_div(n, d): return n / d if d > 0 else 0.0

predictions = []

for _, game in games_today.iterrows():
    gid = game.game_id
    home = game.home_team
    away = game.away_team

    s_home = stats_map.get(home)
    s_away = stats_map.get(away)

    if s_home is None or s_away is None:
        print(f"[WARN] Skipping {away} vs {home} (missing stats)")
        continue

    wins_home = s_home['wins']
    wins_away = s_away['wins']
    losses_home = s_home['losses']
    losses_away = s_away['losses']

    games_home = wins_home + losses_home
    games_away = wins_away + losses_away

    features = [
        safe_div(wins_home, games_home),
        safe_div(wins_away, games_away),
        safe_div(s_home['runs_scored'], games_home),
        safe_div(s_away['runs_scored'], games_away),
        safe_div(s_home['runs_allowed'], games_home),
        safe_div(s_away['runs_allowed'], games_away),
        1.0  # home field advantage
    ]

    win_pred = clf.predict([features])[0]
    margin_pred = float(reg_margin.predict([features])[0])
    total_pred = float(reg_total.predict([features])[0])
    predicted_winner = home if win_pred == 1 else away

    predictions.append((gid, TODAY, away, home, predicted_winner, margin_pred, total_pred))

# Step 4: Store predictions
if predictions:
    c.executemany("""
        INSERT INTO predictions (game_id, date, away_team, home_team, predicted_winner, predicted_margin, predicted_total)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, predictions)
    conn.commit()
    print(f"[INFO] Saved {len(predictions)} predictions for {TODAY}.")
else:
    print(f"[WARN] No predictions generated.")

conn.close()
