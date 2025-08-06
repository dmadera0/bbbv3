# db_setup.py
# ------------------------------
# Builds SQLite database with:
# - Full MLB 2025 schedule
# - Placeholder team stats
# ------------------------------

import requests
import pandas as pd
import sqlite3
import datetime
import numpy as np

# Connect to SQLite database
conn = sqlite3.connect("mlb_predictions.db")
c = conn.cursor()

# 1. Create Tables
def create_tables():
    c.execute("DROP TABLE IF EXISTS games")
    c.execute("DROP TABLE IF EXISTS team_stats")
    c.execute("DROP TABLE IF EXISTS predictions")

    c.execute("""
        CREATE TABLE games (
            game_id INTEGER PRIMARY KEY,
            date TEXT,
            away_team TEXT,
            home_team TEXT,
            away_score INTEGER,
            home_score INTEGER,
            status TEXT
        )
    """)
    c.execute("""
        CREATE TABLE team_stats (
            name TEXT PRIMARY KEY,
            wins INTEGER,
            losses INTEGER,
            runs_scored INTEGER,
            runs_allowed INTEGER
        )
    """)
    c.execute("""
        CREATE TABLE predictions (
            game_id INTEGER,
            predicted_winner TEXT,
            predicted_margin REAL,
            predicted_total REAL,
            date TEXT
        )
    """)
    conn.commit()
    print("[INFO] Tables created.")

# 2. Ingest 2025 Game Schedule
def fetch_schedule():
    print("[INFO] Fetching full schedule...")
    start_date = datetime.date(2025, 3, 28)
    end_date = datetime.date(2025, 10, 1)
    current = start_date
    all_games = []

    while current <= end_date:
        url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={current.isoformat()}"
        try:
            resp = requests.get(url, timeout=10)
            data = resp.json()
            for date_info in data.get("dates", []):
                for game in date_info.get("games", []):
                    all_games.append({
                        "game_id": game["gamePk"],
                        "date": game["gameDate"][:10],
                        "away_team": game["teams"]["away"]["team"]["name"],
                        "home_team": game["teams"]["home"]["team"]["name"],
                        "away_score": game["teams"]["away"].get("score"),
                        "home_score": game["teams"]["home"].get("score"),
                        "status": game.get("status", {}).get("detailedState", "")
                    })
        except Exception as e:
            print(f"[WARN] Failed to fetch {current}: {e}")
        current += datetime.timedelta(days=1)

    df = pd.DataFrame(all_games)
    if not df.empty:
        df.drop_duplicates(subset="game_id", inplace=True)
        df.to_sql("games", conn, if_exists="append", index=False)
        print(f"[INFO] Inserted {len(df)} scheduled games.")
    else:
        print("[WARN] No games fetched.")

# 3. Simulate Team Stats (Placeholder until real data source is integrated)
def simulate_team_stats():
    print("[INFO] Simulating team stats... (placeholder)")
    teams = pd.read_sql("SELECT DISTINCT home_team FROM games", conn)
    stats = []
    for team in teams["home_team"]:
        stats.append({
            "name": team,
            "wins": int(60 + 40 * np.random.rand()),  # 60–100 wins
            "losses": int(60 + 40 * np.random.rand()),
            "runs_scored": int(500 + 300 * np.random.rand()),
            "runs_allowed": int(500 + 300 * np.random.rand())
        })
    df_stats = pd.DataFrame(stats)
    df_stats.to_sql("team_stats", conn, if_exists="replace", index=False)
    print(f"[INFO] Simulated stats for {len(df_stats)} teams.")

# --------------------------
# MAIN SCRIPT
# --------------------------
if __name__ == "__main__":
    create_tables()
    fetch_schedule()
    simulate_team_stats()
    print("[DONE] Database setup complete.")
