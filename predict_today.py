"""
Daily update and prediction script for MLB games.
This script should be run daily (e.g., as a scheduled job) to:
1. Fetch yesterday's game results and update the database.
2. Update team stats (wins, losses, runs, etc.) in the database.
3. Retrain prediction models with the latest data.
4. Predict outcomes for today's games and store these predictions.
5. Save the trained models for use in the Streamlit app.
"""
# predict_today.py regenerated
# -----------------------------
# Loads models, predicts outcomes for today's games, saves to SQLite

import sqlite3
import pandas as pd
import numpy as np
import joblib
from mlb_feature_engineering import load_data, train_models
from datetime import date

# 1. Load or train models
try:
    clf = joblib.load("model_winner.pkl")
    reg_margin = joblib.load("model_margin.pkl")
    reg_total = joblib.load("model_total.pkl")
    print("[INFO] Models loaded from disk.")
except Exception:
    X, y = load_data()
    clf, reg_margin, reg_total = train_models(X, y)
    joblib.dump(clf, "model_winner.pkl")
    joblib.dump(reg_margin, "model_margin.pkl")
    joblib.dump(reg_total, "model_total.pkl")
    print("[INFO] Models trained and saved.")

# 2. Connect to database
conn = sqlite3.connect("mlb_predictions.db")
c = conn.cursor()

# 3. Fetch today's unplayed games
today = date.today().isoformat()
query = """
    SELECT game_id, home_team, away_team
    FROM games
    WHERE date = ? AND home_score IS NULL AND away_score IS NULL
"""
today_games = pd.read_sql_query(query, conn, params=[today])

if today_games.empty:
    print(f"[WARN] No unplayed games found for {today}.")
    exit()

# 4. Fetch team stats
team_stats = pd.read_sql_query("SELECT * FROM team_stats", conn)
stats_map = team_stats.groupby("team_name").last().to_dict(orient="index")

# 5. Predict each game
predictions = []
for _, row in today_games.iterrows():
    g_id = row["game_id"]
    home = row["home_team"]
    away = row["away_team"]
    stats_home = stats_map.get(home)
    stats_away = stats_map.get(away)
    if not stats_home or not stats_away:
        print(f"[WARN] Skipping {home} vs {away} (missing stats)")
        continue

    features = [
        stats_home['wins'] / (stats_home['wins'] + stats_home['losses']),
        stats_away['wins'] / (stats_away['wins'] + stats_away['losses']),
        stats_home['runs_scored'] / (stats_home['wins'] + stats_home['losses']),
        stats_away['runs_scored'] / (stats_away['wins'] + stats_away['losses']),
        stats_home['runs_allowed'] / (stats_home['wins'] + stats_home['losses']),
        stats_away['runs_allowed'] / (stats_away['wins'] + stats_away['losses']),
        1.0  # home field advantage
    ]

    win = clf.predict([features])[0]
    margin = float(reg_margin.predict([features])[0])
    total = float(reg_total.predict([features])[0])
    predicted = home if win == 1 else away

    predictions.append((g_id, today, away, home, predicted, margin, total))

# 6. Insert into predictions table
if predictions:
    c.executemany("""
        INSERT OR REPLACE INTO predictions
        (game_id, date, away_team, home_team, predicted_winner, predicted_margin, predicted_total)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, predictions)
    conn.commit()
    print(f"[INFO] Inserted {len(predictions)} predictions for {today}.")
else:
    print(f"[WARN] No usable games to predict for {today}.")

conn.close()
