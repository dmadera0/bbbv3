# MLB Game Outcome Prediction Model - Data Scraper and Viewer
# -------------------------------------------------------------
# This script:
# 1. Scrapes team batting and pitching stats from Yahoo Sports
# 2. Stores them in a local SQLite database in wide format
# 3. Displays a preview using pandas for developer inspection

import warnings
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

import requests
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3
import datetime

# Connect to SQLite database (or create if it doesn't exist)
conn = sqlite3.connect("mlb_predictions.db")
cursor = conn.cursor()

# ---------------------
# Table Definitions
# ---------------------
# Drop existing tables for a clean state (development only)
cursor.execute("DROP TABLE IF EXISTS team_batting_stats")
cursor.execute("DROP TABLE IF EXISTS team_pitching_stats")

# Create wide-format table for team batting stats
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

# Create wide-format table for team pitching stats
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
conn.commit()

# ---------------------
# Scrape Team Batting Stats
# ---------------------
def scrape_team_batting_stats():
    """
    Scrapes team batting stats from Yahoo and inserts into team_batting_stats table.
    """
    url = "https://sports.yahoo.com/mlb/stats/team/"
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, "html.parser")

    # Read table headers
    table = soup.find("table")
    raw_headers = [th.text.strip() for th in table.find_all("th")]
    # Normalize headers: remove empty or irrelevant
    headers = []
    for h in raw_headers:
        if h and h != "#":
            headers.append(h)
    # Expected batting statistics to extract
    expected = [
        'Team', 'AVG', 'OBP', 'SLG', 'OPS', 'AB', 'R', 'H', '2B', '3B',
        'HR', 'RBI', 'BB', 'K', 'SO', 'SB', 'CS', 'AVG Rank', 'OBP Rank',
        'SLG Rank', 'OPS Rank'
    ]
    # Map column names to DB-safe names
    colmap = {h: h.replace(" ", "_").replace("%", "").replace("-", "_") for h in expected}
    colmap['2B'] = '"2B"'
    colmap['3B'] = '"3B"'
    colmap['AVG Rank'] = 'AVG_RANK'
    colmap['OBP Rank'] = 'OBP_RANK'
    colmap['SLG Rank'] = 'SLG_RANK'
    colmap['OPS Rank'] = 'OPS_RANK'

    rows = table.find("tbody").find_all("tr")
    today = datetime.date.today().isoformat()

    for row in rows:
        cells = row.find_all("td")
        if not cells:
            continue
        # Build team_data dict
        team_data = {'date': today}
        # Extract team name cell
        # The first header in 'expected' is 'Team'
        idx_team = headers.index('Team')
        team_cell = cells[idx_team]
        # Team name usually in <a> tag
        name_tag = team_cell.find('a')
        team_name = name_tag.text.strip() if name_tag else team_cell.text.strip()
        team_data['team_name'] = team_name

        # Extract each expected stat
        for stat in expected[1:]:  # skip 'Team'
            if stat not in headers:
                team_data[colmap[stat]] = None
                continue
            idx = headers.index(stat)
            cell_text = cells[idx].text.strip().replace(',', '').replace('%', '')
            try:
                team_data[colmap[stat]] = float(cell_text)
            except ValueError:
                team_data[colmap[stat]] = None

        # Insert into database
        cols = ", ".join(team_data.keys())
        placeholders = ", ".join(["?" for _ in team_data])
        vals = list(team_data.values())
        cursor.execute(f"INSERT INTO team_batting_stats ({cols}) VALUES ({placeholders})", vals)

    conn.commit()
    print(f"[INFO] Team batting stats scraped and stored for {today}")

# ---------------------
# Scrape Team Pitching Stats
# ---------------------
def scrape_team_pitching_stats():
    """
    Scrapes team pitching stats from Yahoo and inserts into team_pitching_stats table.
    """
    url = "https://sports.yahoo.com/mlb/stats/team/?selectedTable=1"
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, "html.parser")

    table = soup.find("table")
    raw_headers = [th.text.strip() for th in table.find_all("th")]
    headers = [h for h in raw_headers if h and h != "#"]
    expected = ['Team', 'ERA', 'H', 'BB', 'K', 'SV', 'WHIP']

    rows = table.find("tbody").find_all("tr")
    today = datetime.date.today().isoformat()

    for row in rows:
        cells = row.find_all("td")
        if not cells:
            continue
        team_data = {'date': today}
        # Team name
        idx_team = headers.index('Team')
        name_tag = cells[idx_team].find('a')
        team_data['team_name'] = name_tag.text.strip() if name_tag else cells[idx_team].text.strip()

        # Pitching stats
        for stat in expected[1:]:
            if stat not in headers:
                team_data[stat] = None
                continue
            idx = headers.index(stat)
            raw = cells[idx].text.strip().replace(',', '').replace('%', '')
            try:
                team_data[stat] = float(raw)
            except ValueError:
                team_data[stat] = None

        # Insert into database
        cols = ", ".join(team_data.keys())
        placeholders = ", ".join(["?" for _ in team_data])
        vals = list(team_data.values())
        cursor.execute(f"INSERT INTO team_pitching_stats ({cols}) VALUES ({placeholders})", vals)

    conn.commit()
    print(f"[INFO] Team pitching stats scraped and stored for {today}")

# ---------------------
# Main Execution
# ---------------------
if __name__ == "__main__":
    scrape_team_batting_stats()
    scrape_team_pitching_stats()

    # Preview data
    print("\n[INFO] Batting Stats Sample:")
    df_b = pd.read_sql_query("SELECT * FROM team_batting_stats ORDER BY date DESC LIMIT 5", conn)
    print(df_b)

    print("\n[INFO] Pitching Stats Sample:")
    df_p = pd.read_sql_query("SELECT * FROM team_pitching_stats ORDER BY date DESC LIMIT 5", conn)
    print(df_p)
