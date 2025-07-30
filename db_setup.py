# MLB Game Outcome Prediction Model and Application - PART 1: Setup & Scraper Base
# ===============================================================================" +
"# Step-by-step walkthrough: In this part, we will:
" +
"# 1. Silence urllib3 SSL warnings
" +
"# 2. Scrape team batting stats (already working)
" +
"# 3. Scrape team pitching stats
" +
"# 4. Add a simple pandas viewer to inspect stored stats

# ----------------------------
# STEP 1: Import Dependencies
# ----------------------------
import warnings  # to silence SSL warning
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

import requests
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3
import datetime
import time

# ----------------------------
# STEP 2: Setup SQLite Database
# ----------------------------
conn = sqlite3.connect("mlb_predictions.db")
cursor = conn.cursor()

# Drop old team_pitching_stats if testing (optional for dev use)
cursor.execute("DROP TABLE IF EXISTS team_pitching_stats")

# Create wide-format tables for team stats
def create_tables():
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS team_batting_stats (
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

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS team_pitching_stats (
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

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS game_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_date TEXT,
            home_team TEXT,
            away_team TEXT,
            home_score INTEGER,
            away_score INTEGER
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_date TEXT,
            home_team TEXT,
            away_team TEXT,
            predicted_winner TEXT,
            win_confidence REAL,
            predicted_margin REAL,
            predicted_total_runs REAL,
            actual_home_score INTEGER,
            actual_away_score INTEGER
        )
    ''')

    conn.commit()

create_tables()

# ---------------------------------------------
# STEP 3a: Scraper for Yahoo Team Batting Stats
# ---------------------------------------------
def scrape_team_batting_stats():
    url = "https://sports.yahoo.com/mlb/stats/team/"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    table = soup.find("table")
    headers = [th.text.strip() for th in table.find_all("th")]
    rows = table.find("tbody").find_all("tr")
    today = datetime.date.today().isoformat()

    for row in rows:
        cells = row.find_all("td")
        team_data = {
            'team_name': cells[1].text.strip(),
            'date': today
        }

        for i in range(2, len(cells)):
            stat_name = headers[i]
            clean_name = stat_name.replace(" ", "_").replace("%", "").replace("-", "_")
            if clean_name in ["2B", "3B"]:
                clean_name = f'"{clean_name}"'

            try:
                team_data[clean_name] = float(cells[i].text.strip().replace("%", "").replace(",", ""))
            except:
                team_data[clean_name] = None

        expected_columns = [
            'AVG', 'OBP', 'SLG', 'OPS', 'AB', 'R', 'H', '"2B"', '"3B"', 'HR',
            'RBI', 'BB', 'K', 'SO', 'SB', 'CS',
            'AVG_RANK', 'OBP_RANK', 'SLG_RANK', 'OPS_RANK'
        ]
        for col in expected_columns:
            if col not in team_data:
                team_data[col] = None

        columns = ", ".join(team_data.keys())
        placeholders = ", ".join(["?" for _ in team_data])
        values = list(team_data.values())

        cursor.execute(f"""
            INSERT INTO team_batting_stats ({columns})
            VALUES ({placeholders})
        """, values)

    conn.commit()
    print(f"[INFO] Team batting stats (wide format) scraped and stored for {today}")

# ---------------------------------------------
# STEP 3b: Scraper for Yahoo Team Pitching Stats
# ---------------------------------------------
def scrape_team_pitching_stats():
    url = "https://sports.yahoo.com/mlb/stats/team/?selectedTable=1"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    table = soup.find("table")
    headers = [th.text.strip() for th in table.find_all("th")]
    rows = table.find("tbody").find_all("tr")
    today = datetime.date.today().isoformat()

    for row in rows:
        cells = row.find_all("td")
        team_data = {
            'team_name': cells[1].text.strip(),
            'date': today
        }

        for i in range(2, len(cells)):
            stat_name = headers[i].replace(" ", "_").replace("%", "").replace("-", "_")
            try:
                team_data[stat_name] = float(cells[i].text.strip().replace("%", "").replace(",", ""))
            except:
                team_data[stat_name] = None

        expected_columns = ["ERA", "H", "BB", "K", "SV", "WHIP"]
        for col in expected_columns:
            if col not in team_data:
                team_data[col] = None

        columns = ", ".join(team_data.keys())
        placeholders = ", ".join(["?" for _ in team_data])
        values = list(team_data.values())

        cursor.execute(f"""
            INSERT INTO team_pitching_stats ({columns})
            VALUES ({placeholders})
        """, values)

    conn.commit()
    print(f"[INFO] Team pitching stats (wide format) scraped and stored for {today}")

# --------------------------------------------------
# STEP 4: Run Scrapers and Preview in Pandas
# --------------------------------------------------
scrape_team_batting_stats()
scrape_team_pitching_stats()

# Load and display a preview using pandas
print("\n[INFO] Batting Stats Sample:")
batting_df = pd.read_sql_query("SELECT * FROM team_batting_stats ORDER BY date DESC LIMIT 5", conn)
print(batting_df)

print("\n[INFO] Pitching Stats Sample:")
pitching_df = pd.read_sql_query("SELECT * FROM team_pitching_stats ORDER BY date DESC LIMIT 5", conn)
print(pitching_df)

# Note: This output is for developer inspection.
# The next step is to automate daily runs and integrate predictions and front-end.
