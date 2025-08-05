# predict_today.py
# -------------------------------------
# Predict outcomes for today's MLB games using trained ML models

import sqlite3
import pandas as pd
import numpy as np
from datetime import date
from sklearn.impute import SimpleImputer
from mlb_feature_engineering import load_and_train

# 1. Connect to the database
db_path = "mlb_predictions.db"
conn = sqlite3.connect(db_path)

# 2. Check that the required table exists
check_table = pd.read_sql_query("""
    SELECT name FROM sqlite_master
    WHERE type='table' AND name='game_schedule';
""", conn)

if check_table.empty:
    raise RuntimeError("Table 'game_schedule' not found. Run db_setup.py to generate schedule.")

# 3. Load today's games that have not yet occurred
today = date.today().isoformat()
games_today = pd.read_sql_query(f'''
    SELECT gr.game_pk, gr.home_team, gr.away_team
    FROM game_schedule gr
    LEFT JOIN game_results r ON gr.game_pk = r.game_pk
    WHERE gr.game_date = '{today}' AND r.game_pk IS NULL
''', conn)

if games_today.empty:
    print(f"[INFO] No games to predict for today ({today})")
    exit(0)

# 4. Train models on historical games
clf, reg_margin, reg_total = load_and_train(db_path)

# 5. Load latest team stats to generate features
team_stats = pd.read_sql_query("SELECT * FROM team_stats_by_date", conn)
latest_stats = team_stats.groupby("team_name").last().reset_index()

# 6. Build features for each game
rows = []
for _, row in games_today.iterrows():
    home = latest_stats[latest_stats["team_name"] == row.home_team]
    away = latest_stats[latest_stats["team_name"] == row.away_team]

    if home.empty or away.empty:
        print(f"[WARN] Skipping {row.home_team} vs {row.away_team} - stats missing")
        continue

    # Select numeric columns and prefix
    home_feats = home[['AB','R','H','HR','RBI','BB','K','ERA','WHIP']].add_prefix("home_")
    away_feats = away[['AB','R','H','HR','RBI','BB','K','ERA','WHIP']].add_prefix("away_")
    feats = pd.concat([home_feats.reset_index(drop=True), away_feats.reset_index(drop=True)], axis=1)
    feats["game_pk"] = row.game_pk
    rows.append(feats)

# 7. Create feature DataFrame
if not rows:
    print("[WARN] Could not build features for today's games (missing team stats).")
    exit(0)

X = pd.concat(rows).set_index("game_pk")

# 8. Drop all-NaN columns and impute missing
all_nan_cols = X.columns[X.isna().all()].tolist()
if all_nan_cols:
    print(f"[INFO] Dropping all-NaN columns: {all_nan_cols}")
    X = X.drop(columns=all_nan_cols)

imputer = SimpleImputer(strategy="mean")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

# 9. Predict using trained models
preds = pd.DataFrame(index=X.index)
preds["win_prob"]    = clf.predict_proba(X)[:, 1]
preds["margin_pred"] = reg_margin.predict(X)
preds["total_pred"]  = reg_total.predict(X)

# 10. Join game metadata
preds = preds.join(games_today.set_index("game_pk"))

# 11. Store predictions in table
preds["game_date"] = today
preds.to_sql("predictions", conn, if_exists="append", index=False)

print(f"[INFO] Saved {len(preds)} predictions for {today}.")
