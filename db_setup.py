# db_setup.py
# ----------------------------
# Sets up SQLite database with MLB 2025 games and team-level stats

import requests
import sqlite3
import pandas as pd
from datetime import date, timedelta

DB_PATH = "mlb_predictions.db"
SEASON_YEAR = 2025

# ----------------------------
# Connect to SQLite
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

# ----------------------------
# Create tables
def create_tables():
    c.execute("DROP TABLE IF EXISTS games")
    c.execute("DROP TABLE IF EXISTS team_stats")
    c.execute("DROP TABLE IF EXISTS predictions")

    c.execute("""
        CREATE TABLE games (
            game_id INTEGER PRIMARY KEY,
            date TEXT,
            home_team TEXT,
            away_team TEXT,
            home_score INTEGER,
            away_score INTEGER
        )
    """)

    c.execute("""
        CREATE TABLE team_stats (
            team_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            wins INTEGER,
            losses INTEGER,
            runs_scored INTEGER,
            runs_allowed INTEGER
        )
    """)

    c.execute("""
        CREATE TABLE predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id INTEGER,
            date TEXT,
            away_team TEXT,
            home_team TEXT,
            predicted_winner TEXT,
            predicted_margin REAL,
            predicted_total REAL
        )
    """)

    conn.commit()
    print("[INFO] Tables created.")

# ----------------------------
# Fetch full 2025 schedule from MLB API
def fetch_schedule():
    print("[INFO] Fetching full schedule...")
    schedule_url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&season={SEASON_YEAR}&gameTypes=R"
    response = requests.get(schedule_url)
    data = response.json()

    rows = []
    for date_info in data.get("dates", []):
        for game in date_info.get("games", []):
            game_id = game["gamePk"]
            game_date = game["gameDate"][:10]
            home = game["teams"]["home"]["team"]["name"]
            away = game["teams"]["away"]["team"]["name"]

            home_score = game["teams"]["home"].get("score")
            away_score = game["teams"]["away"].get("score")

            rows.append((game_id, game_date, home, away, home_score, away_score))

    df = pd.DataFrame(rows, columns=["game_id", "date", "home_team", "away_team", "home_score", "away_score"])
    df.to_sql("games", conn, if_exists="append", index=False)
    print(f"[INFO] Inserted {len(df)} games into the database.")

# ----------------------------
# Dummy team stats for now (you can replace with API later)
def generate_team_stats():
    print("[INFO] Generating dummy team stats...")
    teams = pd.read_sql("SELECT DISTINCT home_team AS name FROM games UNION SELECT DISTINCT away_team AS name FROM games", conn)
    stats = []
    for name in teams["name"]:
        wins = int(40 + 20 * (hash(name) % 3))         # 40–80 wins
        losses = 162 - wins
        runs_scored = int(wins * 4.5)
        runs_allowed = int(losses * 4.3)
        stats.append((name, wins, losses, runs_scored, runs_allowed))

    df_stats = pd.DataFrame(stats, columns=["name", "wins", "losses", "runs_scored", "runs_allowed"])
    df_stats.to_sql("team_stats", conn, if_exists="append", index=False)
    print(f"[INFO] Inserted team stats for {len(df_stats)} teams.")

# ----------------------------
# Main execution
if __name__ == "__main__":
    create_tables()
    fetch_schedule()
    generate_team_stats()
    conn.close()
    print("[INFO] Database setup complete.")
