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
import datetime
from mlb_feature_engineering import load_data, train_models

# Step 1: Load trained models
X, y = load_data()
clf, reg_margin, reg_total = train_models(X, y)

# Step 2: Get today's date
today = datetime.date.today().isoformat()

# Step 3: Connect to DB and get today's unplayed games
conn = sqlite3.connect("mlb_predictions.db")
c = conn.cursor()

games_today = pd.read_sql_query("""
    SELECT * FROM games
    WHERE game_date = ? AND home_score IS NULL AND away_score IS NULL
""", conn, params=(today,))

if games_today.empty:
    print(f"[INFO] No games to predict for {today}.")
    conn.close()
    exit()

# Step 4: Load team stats for today’s games
team_stats = pd.read_sql_query("SELECT * FROM team_stats", conn)

features = []
meta = []

for _, game in games_today.iterrows():
    game_id = game["game_id"]
    home_team = game["home_team"]
    away_team = game["away_team"]

    home = team_stats[(team_stats["game_id"] == game_id) & (team_stats["is_home"] == 1)]
    away = team_stats[(team_stats["game_id"] == game_id) & (team_stats["is_home"] == 0)]

    if home.empty or away.empty:
        print(f"[WARN] Skipping {home_team} vs {away_team} (missing stats)")
        continue

    # Drop identifiers and non-numeric
    h = home.drop(columns=["game_id", "team_id", "team_name", "is_home"])
    a = away.drop(columns=["game_id", "team_id", "team_name", "is_home"])

    row = pd.concat([h.reset_index(drop=True), a.reset_index(drop=True)], axis=1)
    features.append(row)
    meta.append({
        "game_id": game_id,
        "game_date": today,
        "home_team": home_team,
        "away_team": away_team
    })

if not features:
    print(f"[WARN] No usable games to predict for {today}.")
    conn.close()
    exit()

# Step 5: Build feature DataFrame
X_pred = pd.concat(features, axis=0).reset_index(drop=True)
X_pred = X_pred.select_dtypes(include=[np.number])  # keep numeric only

# Step 6: Predict
preds = pd.DataFrame(meta)
preds["win_prob"]   = clf.predict_proba(X_pred)[:, 1]
preds["pred_margin"] = reg_margin.predict(X_pred)
preds["pred_total"]  = reg_total.predict(X_pred)

# Step 7: Save predictions
preds.to_sql("predictions_today", conn, if_exists="replace", index=False)
print(f"[INFO] Saved {len(preds)} predictions for {today}.")

conn.close()
