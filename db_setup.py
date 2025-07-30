# MLB Game Outcome Prediction Model and Application - PART 1: Setup & Scraper Base
# ===============================================================================
# Step-by-step walkthrough: In this first part, we'll set up the base project structure,
# install necessary libraries, and begin building the scraping functions to extract data
# from Yahoo Sports.

# ----------------------------
# STEP 1: Import Dependencies
# ----------------------------
import requests
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3
import datetime
import time

# ----------------------------
# STEP 2: Setup SQLite Database
# ----------------------------
# We'll use a local SQLite database for simplicity. You can scale this later to PostgreSQL/MySQL.
conn = sqlite3.connect("mlb_predictions.db")
cursor = conn.cursor()

# Create tables if they don't exist already
def create_tables():
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS team_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            team_name TEXT,
            date TEXT,
            stat_type TEXT,
            stat_name TEXT,
            stat_value REAL
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
# STEP 3: Define Scraper for Yahoo Team Batting
# ---------------------------------------------
def scrape_team_batting_stats():
    url = "https://sports.yahoo.com/mlb/stats/team/"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    # Find the stats table
    table = soup.find("table")
    headers = [th.text.strip() for th in table.find_all("th")]

    # Iterate through each row of team data
    rows = table.find("tbody").find_all("tr")
    today = datetime.date.today().isoformat()

    for row in rows:
        cells = row.find_all("td")
        team_name = cells[1].text.strip()  # Usually the 2nd column has team name

        for i in range(2, len(cells)):
            stat_name = headers[i]
            stat_value = cells[i].text.strip().replace("%", "")
            try:
                stat_value = float(stat_value)
            except:
                stat_value = None

            cursor.execute("""
                INSERT INTO team_stats (team_name, date, stat_type, stat_name, stat_value)
                VALUES (?, ?, ?, ?, ?)
            """, (team_name, today, "batting", stat_name, stat_value))

    conn.commit()
    print(f"[INFO] Team batting stats scraped and stored for {today}")

# -------------------------------
# STEP 4: Run Scraper and Confirm
# -------------------------------
scrape_team_batting_stats()

# You can run this script daily to collect updated stats.
# Next steps will involve:
# - Scraping team pitching and individual stats
# - Scraping game results by date
# - Feature engineering and model training
# - Prediction generation and Streamlit front-end integration
