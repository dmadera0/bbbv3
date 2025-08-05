# db_setup.py
# ----------------------------------
# This script:
# 1. Creates SQLite tables for team stats and historical games (with game_pk)
# 2. Scrapes team batting & pitching stats from Yahoo Sports
# 3. Uses MLB Stats API to pull historical game results with gamePk
# 4. Backfills per-game team boxscore stats via MLB boxscore API
# 5. Stores all data in mlb_predictions.db

import warnings
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

import requests
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3
import datetime

# Connect to the database (or create it)
conn = sqlite3.connect("mlb_predictions.db")
cursor = conn.cursor()

# ---------------------
# 1. TABLE DEFINITIONS
# ---------------------
def create_tables():
    """
    Drops and recreates tables:
      - team_batting_stats
      - team_pitching_stats
      - game_results (including game_pk)
      - team_stats_by_date
    """
    # Drop existing tables for clean slate
    cursor.execute("DROP TABLE IF EXISTS team_batting_stats")
    cursor.execute("DROP TABLE IF EXISTS team_pitching_stats")
    cursor.execute("DROP TABLE IF EXISTS game_results")
    cursor.execute("DROP TABLE IF EXISTS team_stats_by_date")

    # Create batting stats table
    cursor.execute('''
        CREATE TABLE team_batting_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            team_name TEXT,
            date TEXT,
            AVG REAL,
            OBP REAL,
            SLG REAL,
            OPS REAL,
            AB INTEGER,
            R INTEGER,
            H INTEGER,
            "2B" INTEGER,
            "3B" INTEGER,
            HR INTEGER,
            RBI INTEGER,
            BB INTEGER,
            K INTEGER,
            SO INTEGER,
            SB INTEGER,
            CS INTEGER,
            AVG_RANK INTEGER,
            OBP_RANK INTEGER,
            SLG_RANK INTEGER,
            OPS_RANK INTEGER
        )
    ''')
    # Create pitching stats table
    cursor.execute('''
        CREATE TABLE team_pitching_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            team_name TEXT,
            date TEXT,
            ERA REAL,
            H INTEGER,
            BB INTEGER,
            K INTEGER,
            SV INTEGER,
            WHIP REAL
        )
    ''')
    # Create game results table with game_pk
    cursor.execute('''
        CREATE TABLE game_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_pk INTEGER,
            game_date TEXT,
            home_team TEXT,
            away_team TEXT,
            home_score INTEGER,
            away_score INTEGER
        )
    ''')
    # Create per-game, per-team boxscore stats table
    cursor.execute('''
        CREATE TABLE team_stats_by_date (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_pk INTEGER,
            team_name TEXT,
            date TEXT,
            AB INTEGER,
            R INTEGER,
            H INTEGER,
            HR INTEGER,
            RBI INTEGER,
            BB INTEGER,
            K INTEGER,
            ERA REAL,
            WHIP REAL
        )
    ''')
    conn.commit()
    print("[INFO] Tables created: batting, pitching, game_results (with game_pk), team_stats_by_date")

# ---------------------
# 2. SCRAPE TEAM BATTING STATS
# ---------------------
def scrape_team_batting_stats():
    """
    Scrapes Yahoo team batting stats and inserts into team_batting_stats.
    """
    url = "https://sports.yahoo.com/mlb/stats/team/"
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table")
    headers = [th.text.strip() for th in table.find_all("th")]
    rows = table.find("tbody").find_all("tr")
    today = datetime.date.today().isoformat()

    expected = {
        'Team':'team_name','AVG':'AVG','OBP':'OBP','SLG':'SLG','OPS':'OPS',
        'AB':'AB','R':'R','H':'H','2B':'"2B"','3B':'"3B"',
        'HR':'HR','RBI':'RBI','BB':'BB','K':'K','SO':'SO',
        'SB':'SB','CS':'CS','AVG Rank':'AVG_RANK','OBP Rank':'OBP_RANK',
        'SLG Rank':'SLG_RANK','OPS Rank':'OPS_RANK'
    }
    for row in rows:
        cells = row.find_all("td")
        if not cells:
            continue
        data = {'date': today}
        for idx, header in enumerate(headers):
            if header not in expected:
                continue
            col = expected[header]
            text = cells[idx].text.strip().replace(',', '').replace('%', '')
            if header == 'Team':
                a = cells[idx].find('a')
                data[col] = a.text.strip() if a else text
            else:
                try:
                    data[col] = float(text)
                except ValueError:
                    data[col] = None
        cols = ", ".join(data.keys())
        placeholders = ", ".join(["?" for _ in data])
        cursor.execute(
            f"INSERT INTO team_batting_stats ({cols}) VALUES ({placeholders})",
            list(data.values())
        )
    conn.commit()
    print(f"[INFO] Team batting stats scraped for {today}")

# ---------------------
# 3. SCRAPE TEAM PITCHING STATS
# ---------------------
def scrape_team_pitching_stats():
    """
    Scrapes Yahoo team pitching stats and inserts into team_pitching_stats.
    """
    url = "https://sports.yahoo.com/mlb/stats/team/?selectedTable=1"
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table")
    headers = [th.text.strip() for th in table.find_all("th")]
    rows = table.find("tbody").find_all("tr")
    today = datetime.date.today().isoformat()

    expected = {'Team':'team_name','ERA':'ERA','H':'H','BB':'BB','K':'K','SV':'SV','WHIP':'WHIP'}
    for row in rows:
        cells = row.find_all("td")
        if not cells:
            continue
        data = {'date': today}
        for idx, header in enumerate(headers):
            if header not in expected:
                continue
            col = expected[header]
            text = cells[idx].text.strip().replace(',', '').replace('%', '')
            if header == 'Team':
                a = cells[idx].find('a')
                data[col] = a.text.strip() if a else text
            else:
                try:
                    data[col] = float(text)
                except ValueError:
                    data[col] = None
        cols = ", ".join(data.keys())
        placeholders = ", ".join(["?" for _ in data])
        cursor.execute(
            f"INSERT INTO team_pitching_stats ({cols}) VALUES ({placeholders})",
            list(data.values())
        )
    conn.commit()
    print(f"[INFO] Team pitching stats scraped for {today}")

# ---------------------
# 4. SCRAPE HISTORICAL GAME RESULTS WITH gamePk
# ---------------------
def scrape_game_results_mlb_api():
    """
    Uses MLB Stats API to pull schedule from Opening Day through today.
    Inserts game_pk, date, teams, and scores into game_results.
    """
    season_start = "2025-03-28"
    current = datetime.date.fromisoformat(season_start)
    today = datetime.date.today()
    while current <= today:
        date_str = current.isoformat()
        url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date_str}"
        data = requests.get(url).json()
        days = data.get('dates', [])
        count = 0
        for day in days:
            for game in day.get('games', []):
                pk = game.get('gamePk')
                away = game['teams']['away']
                home = game['teams']['home']
                away_score = away.get('score')
                home_score = home.get('score')
                if away_score is None or home_score is None:
                    continue
                cursor.execute(
                    "INSERT INTO game_results (game_pk, game_date, home_team, away_team, home_score, away_score) VALUES (?, ?, ?, ?, ?, ?)",
                    (pk, date_str, home['team']['name'], away['team']['name'], home_score, away_score)
                )
                count += 1
        conn.commit()
        print(f"[INFO] MLB API: {count} games for {date_str} stored")
        current += datetime.timedelta(days=1)

# ---------------------
# 5. BACKFILL PER-GAME TEAM BOX SCORE STATS
# ---------------------
def scrape_team_stats_for_games():
    """
    For each game in game_results, fetch the boxscore via MLB API
    and insert per-team stats into team_stats_by_date.
    Includes debug statements to verify data.
    """
    # Load the list of games to backfill
    games = pd.read_sql_query("SELECT game_pk, game_date FROM game_results", conn)
    print(f"[DEBUG] Found {len(games)} games to backfill team_stats_by_date")
    inserted = 0
    for _, row in games.iterrows():
        pk = row['game_pk']
        game_date = row['game_date']
        url = f"https://statsapi.mlb.com/api/v1/game/{pk}/boxscore"
        data = requests.get(url).json()
        # Ensure data contains expected keys
        if 'teams' not in data:
            print(f"[WARN] No boxscore data for gamePk {pk}")
            continue
        for side in ['home', 'away']:
            team_name = data['teams'][side]['team']['name']
            batting = data['teams'][side].get('teamStats', {}).get('batting', {})
            pitching = data['teams'][side].get('teamStats', {}).get('pitching', {})
            # Extract stats with default None
            AB = batting.get('atBats')
            R  = batting.get('runs')
            H  = batting.get('hits')
            HR = batting.get('homeRuns')
            RBI= batting.get('rbi')
            BB = batting.get('baseOnBalls')
            K  = batting.get('strikeOuts')
            ERA= pitching.get('era')
            WHIP = pitching.get('hitsPerInning')
            cursor.execute(
                '''INSERT INTO team_stats_by_date (game_pk, team_name, date, AB, R, H, HR, RBI, BB, K, ERA, WHIP)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (
                    pk,
                    team_name,
                    game_date,
                    AB,
                    R,
                    H,
                    HR,
                    RBI,
                    BB,
                    K,
                    ERA,
                    WHIP
                )
            )
            inserted += 1
    conn.commit()
    print(f"[INFO] team_stats_by_date backfilled for all games ({inserted} rows inserted)")
# ---------------------
# 6. INGEST FULL SEASON SCHEDULE
# ---------------------
def ingest_game_schedule():
    """
    Pulls the full MLB schedule for the 2025 season and stores in game_schedule.
    Needed for predicting future games.
    """
    print("[INFO] Ingesting full season game schedule...")

    url = "https://statsapi.mlb.com/api/v1/schedule"
    season_start = "2025-03-20"
    season_end = "2025-11-01"

    params = {
        "sportId": 1,
        "startDate": season_start,
        "endDate": season_end,
    }

    response = requests.get(url, params=params)
    data = response.json()

    games = []
    for date_info in data.get("dates", []):
        game_date = date_info["date"]
        for game in date_info["games"]:
            game_pk = game["gamePk"]
            home = game["teams"]["home"]["team"]["name"]
            away = game["teams"]["away"]["team"]["name"]
            games.append((game_pk, game_date, home, away))

    # Create table if not exists
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS game_schedule (
            game_pk INTEGER PRIMARY KEY,
            game_date TEXT,
            home_team TEXT,
            away_team TEXT
        )
    ''')

    cursor.executemany("""
        INSERT OR IGNORE INTO game_schedule (game_pk, game_date, home_team, away_team)
        VALUES (?, ?, ?, ?)
    """, games)

    conn.commit()
    print(f"[INFO] Stored {len(games)} games in game_schedule.")

# ---------------------
# MAIN EXECUTION
# ---------------------
if __name__ == "__main__":
    create_tables()
    scrape_team_batting_stats()
    scrape_team_pitching_stats()
    scrape_game_results_mlb_api()
    scrape_team_stats_for_games()
    ingest_game_schedule()  # ✅ Add this
    print("[INFO] Data ingestion complete.")
