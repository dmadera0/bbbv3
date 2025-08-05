# db_setup.py
# ---------------------------
# Initializes and populates mlb_predictions.db
# - Creates tables for games, stats, and predictions
# - Pulls schedule and results from MLB API
# - Pulls team-level boxscore stats

import sqlite3
import requests
import datetime
import time

# ---------------------------
# Configuration
# ---------------------------
DB_PATH = "mlb_predictions.db"
SEASON = 2025
START_DATE = datetime.date(SEASON, 3, 28)
TODAY = datetime.date.today()

# ---------------------------
# Connect to DB and create tables
# ---------------------------
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS games (
    game_id INTEGER PRIMARY KEY,
    game_date TEXT,
    home_team TEXT,
    away_team TEXT,
    home_score INTEGER,
    away_score INTEGER
)
""")

c.execute("""
CREATE TABLE IF NOT EXISTS team_stats (
    game_id INTEGER,
    team_name TEXT,
    is_home INTEGER,
    R INTEGER,
    H INTEGER,
    HR INTEGER,
    BB INTEGER,
    SO INTEGER,
    ERA REAL,
    WHIP REAL,
    FOREIGN KEY (game_id) REFERENCES games(game_id)
)
""")

c.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    game_id INTEGER,
    prediction_date TEXT,
    predicted_winner TEXT,
    win_prob REAL,
    predicted_margin REAL,
    predicted_total REAL,
    FOREIGN KEY (game_id) REFERENCES games(game_id)
)
""")

conn.commit()

# ---------------------------
# Fetch schedule + results
# ---------------------------
def fetch_schedule(start_date, end_date):
    all_games = []
    date = start_date
    while date <= end_date:
        url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date}"
        r = requests.get(url)
        data = r.json()
        for date_info in data.get("dates", []):
            for game in date_info.get("games", []):
                gid = game["gamePk"]
                game_date = game["gameDate"][:10]
                home = game["teams"]["home"]["team"]["name"]
                away = game["teams"]["away"]["team"]["name"]
                home_score = game["teams"]["home"].get("score")
                away_score = game["teams"]["away"].get("score")
                all_games.append((gid, game_date, home, away, home_score, away_score))
        date += datetime.timedelta(days=1)
        time.sleep(0.25)
    return all_games

# ---------------------------
# Fetch boxscore stats per game
# ---------------------------
def fetch_boxscore(game_id):
    url = f"https://statsapi.mlb.com/api/v1/game/{game_id}/boxscore"
    r = requests.get(url)
    if not r.ok:
        return []
    data = r.json()
    result = []
    for side in ["home", "away"]:
        team = data["teams"][side]
        name = team["team"]["name"]
        stats = team["teamStats"]
        batting = stats.get("batting", {})
        pitching = stats.get("pitching", {})
        result.append((
            game_id,
            name,
            1 if side == "home" else 0,
            batting.get("runs"),
            batting.get("hits"),
            batting.get("homeRuns"),
            batting.get("baseOnBalls"),
            batting.get("strikeOuts"),
            pitching.get("era"),
            pitching.get("whip")
        ))
    return result

# ---------------------------
# Ingest data
# ---------------------------
print("[INFO] Fetching full schedule...")
games = fetch_schedule(START_DATE, TODAY)
print(f"[INFO] Found {len(games)} total games")

c.executemany("""
    INSERT OR IGNORE INTO games (game_id, game_date, home_team, away_team, home_score, away_score)
    VALUES (?, ?, ?, ?, ?, ?)
""", games)
conn.commit()

print("[INFO] Ingesting boxscores...")
boxscore_rows = []
for game in games:
    if game[4] is not None and game[5] is not None:
        boxscore_rows.extend(fetch_boxscore(game[0]))
        time.sleep(0.1)

c.executemany("""
    INSERT OR IGNORE INTO team_stats (game_id, team_name, is_home, R, H, HR, BB, SO, ERA, WHIP)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
""", boxscore_rows)
conn.commit()
print("[INFO] Data ingestion complete.")
