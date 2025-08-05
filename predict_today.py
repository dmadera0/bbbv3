"""
Daily update and prediction script for MLB games.
This script should be run daily (e.g., as a scheduled job) to:
1. Fetch yesterday's game results and update the database.
2. Update team stats (wins, losses, runs, etc.) in the database.
3. Retrain prediction models with the latest data.
4. Predict outcomes for today's games and store these predictions.
5. Save the trained models for use in the Streamlit app.
"""
import sqlite3
import requests
import datetime
import joblib
from mlb_feature_engineering import train_models

# Connect to the database
conn = sqlite3.connect("mlb_predictions.db")
c = conn.cursor()

# Determine yesterday's date (to update yesterday's games)
today = datetime.date.today()
yesterday = today - datetime.timedelta(days=1)
yesterday_str = yesterday.strftime("%Y-%m-%d")

# Fetch yesterday's game results from MLB Stats API
date_param = yesterday.strftime("%m/%d/%Y")
schedule_url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date_param}&gameTypes=R"
try:
    resp = requests.get(schedule_url)
    schedule_data = resp.json()
except Exception:
    schedule_data = {}

# Update games table with yesterday's final scores
for date_block in schedule_data.get("dates", []):
    for game in date_block.get("games", []):
        game_id = game.get("gamePk")
        status = game.get("status", {}).get("detailedState", "")
        # Only update if the game has concluded
        if status == "Final" or status == "Completed Early":
            home_score = game.get("teams", {}).get("home", {}).get("score")
            away_score = game.get("teams", {}).get("away", {}).get("score")
            c.execute("""
                UPDATE games
                SET home_score = ?, away_score = ?, status = ?
                WHERE game_id = ?
            """, (home_score, away_score, status, game_id))

# Recalculate and update team_stats for all teams based on updated games
c.execute("SELECT id FROM teams")
team_ids = [row[0] for row in c.fetchall()]
for team_id in team_ids:
    # Wins: count games where team won
    c.execute("""
        SELECT COUNT(*) FROM games
        WHERE (home_team_id = ? AND home_score > away_score)
           OR (away_team_id = ? AND away_score > home_score)
    """, (team_id, team_id))
    wins = c.fetchone()[0]
    # Losses: count games where team lost
    c.execute("""
        SELECT COUNT(*) FROM games
        WHERE (home_team_id = ? AND home_score < away_score)
           OR (away_team_id = ? AND away_score < home_score)
    """, (team_id, team_id))
    losses = c.fetchone()[0]
    # Runs scored by the team
    c.execute("SELECT SUM(home_score) FROM games WHERE home_team_id = ?", (team_id,))
    home_runs = c.fetchone()[0] or 0
    c.execute("SELECT SUM(away_score) FROM games WHERE away_team_id = ?", (team_id,))
    away_runs = c.fetchone()[0] or 0
    runs_scored = home_runs + away_runs
    # Runs allowed by the team
    c.execute("SELECT SUM(away_score) FROM games WHERE home_team_id = ?", (team_id,))
    runs_allowed_home = c.fetchone()[0] or 0
    c.execute("SELECT SUM(home_score) FROM games WHERE away_team_id = ?", (team_id,))
    runs_allowed_away = c.fetchone()[0] or 0
    runs_allowed = runs_allowed_home + runs_allowed_away
    # Fetch latest batting average and ERA from API
    batting_avg = 0.0
    era = 0.0
    stats_batting_url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/stats?season=2025&group=hitting&stats=season"
    stats_pitching_url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/stats?season=2025&group=pitching&stats=season"
    try:
        r_bat = requests.get(stats_batting_url)
        r_pitch = requests.get(stats_pitching_url)
        stats_batting = r_bat.json().get("stats", [])
        stats_pitching = r_pitch.json().get("stats", [])
        if stats_batting:
            avg_str = stats_batting[0]["splits"][0]["stat"].get("avg")
            if avg_str is not None:
                batting_avg = float(avg_str)
        if stats_pitching:
            era_str = stats_pitching[0]["splits"][0]["stat"].get("era")
            if era_str is not None:
                era = float(era_str)
    except Exception:
        # On failure, retain existing stats from the database
        c.execute("SELECT batting_avg, era FROM team_stats WHERE team_id = ?", (team_id,))
        row = c.fetchone()
        if row:
            batting_avg, era = row[0], row[1]
    # Update team_stats table for this team
    c.execute("""
        UPDATE team_stats
        SET wins = ?, losses = ?, runs_scored = ?, runs_allowed = ?, batting_avg = ?, era = ?
        WHERE team_id = ?
    """, (wins, losses, runs_scored, runs_allowed, batting_avg, era, team_id))

# Save updates to games and team_stats
conn.commit()

# Retrain models with the latest data (2025 season up to yesterday)
clf, margin_model, total_model = train_models("mlb_predictions.db")

# Predict outcomes for today's games
today_str = today.strftime("%Y-%m-%d")
c.execute("SELECT game_id, home_team_id, away_team_id FROM games WHERE date = ? AND home_score IS NULL", (today_str,))
today_games = c.fetchall()
# Ensure predictions table exists
c.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        game_id INTEGER PRIMARY KEY,
        date TEXT,
        predicted_winner_id INTEGER,
        predicted_margin REAL,
        predicted_total REAL
    )
""")
# Remove any prior predictions for today (to avoid duplicates if re-run)
c.execute("DELETE FROM predictions WHERE date = ?", (today_str,))
predictions_to_insert = []
for game_id, home_id, away_id in today_games:
    # Get current stats for home and away teams
    c.execute("SELECT wins, losses, runs_scored, runs_allowed FROM team_stats WHERE team_id = ?", (home_id,))
    wins_home, losses_home, runs_home, allowed_home = c.fetchone()
    c.execute("SELECT wins, losses, runs_scored, runs_allowed FROM team_stats WHERE team_id = ?", (away_id,))
    wins_away, losses_away, runs_away, allowed_away = c.fetchone()
    games_home = wins_home + losses_home
    games_away = wins_away + losses_away
    home_win_pct = wins_home / games_home if games_home > 0 else 0.0
    away_win_pct = wins_away / games_away if games_away > 0 else 0.0
    home_runs_per_game = runs_home / games_home if games_home > 0 else 0.0
    away_runs_per_game = runs_away / games_away if games_away > 0 else 0.0
    home_runs_allowed_per_game = allowed_home / games_home if games_home > 0 else 0.0
    away_runs_allowed_per_game = allowed_away / games_away if games_away > 0 else 0.0
    features = [
        home_win_pct,
        away_win_pct,
        home_runs_per_game,
        away_runs_per_game,
        home_runs_allowed_per_game,
        away_runs_allowed_per_game,
        1.0  # home_field indicator
    ]
    # Generate predictions using the trained models
    win_pred = clf.predict([features])[0]        # 1 if home wins, 0 if away wins
    margin_pred = margin_model.predict([features])[0]
    total_pred = total_model.predict([features])[0]
    # Determine predicted winning team
    predicted_winner_id = home_id if win_pred == 1 else away_id
    predictions_to_insert.append((game_id, today_str, predicted_winner_id, float(margin_pred), float(total_pred)))

# Insert today's predictions into the database
if predictions_to_insert:
    c.executemany("""
        INSERT OR REPLACE INTO predictions (game_id, date, predicted_winner_id, predicted_margin, predicted_total)
        VALUES (?, ?, ?, ?, ?)
    """, predictions_to_insert)
    conn.commit()

# Save trained models to disk for the Streamlit app
joblib.dump(clf, "model_winner.pkl")
joblib.dump(margin_model, "model_margin.pkl")
joblib.dump(total_model, "model_total.pkl")

# Close the database connection
conn.close()
