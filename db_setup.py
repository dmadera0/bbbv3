# MLB Game Outcome Prediction Model - Data Scraper, Feature Engineering & Model Training
# ---------------------------------------------------------------------------------
# This script performs the following tasks:
# 1. Defines and creates SQLite tables for team stats and game results
# 2. Scrapes team batting and pitching statistics from Yahoo Sports
# 3. Pulls historical game results from the official MLB Stats API
# 4. Performs feature engineering by merging stats with game outcomes
# 5. Trains machine learning models to predict game outcomes, margin, and total runs

import warnings
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

import requests
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3
import datetime

# ---------------------
# 1. DATABASE SETUP
# ---------------------
def create_tables(conn):
    """
    Drops existing tables and recreates:
      - team_batting_stats: stores daily batting metrics per team
      - team_pitching_stats: stores daily pitching metrics per team
      - game_results: stores historical game scores
    """
    cursor = conn.cursor()
    # Remove old tables if they exist (development reset)
    cursor.execute("DROP TABLE IF EXISTS team_batting_stats")
    cursor.execute("DROP TABLE IF EXISTS team_pitching_stats")
    cursor.execute("DROP TABLE IF EXISTS game_results")

    # Create batting stats table in wide format (one row per team per date)
    cursor.execute('''
        CREATE TABLE team_batting_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            team_name TEXT,          -- Team identifier
            date TEXT,               -- Statistics date (ISO format)
            AVG REAL,                -- Batting average
            OBP REAL,                -- On-base percentage
            SLG REAL,                -- Slugging percentage
            OPS REAL,                -- On-base plus slugging
            AB INTEGER,              -- At bats
            R INTEGER,               -- Runs scored
            H INTEGER,               -- Hits
            "2B" INTEGER,           -- Doubles
            "3B" INTEGER,           -- Triples
            HR INTEGER,              -- Home runs
            RBI INTEGER,             -- Runs batted in
            BB INTEGER,              -- Walks
            K INTEGER,               -- Strikeouts
            SO INTEGER,              -- Strikeouts (alternate stat)
            SB INTEGER,              -- Stolen bases
            CS INTEGER,              -- Caught stealing
            AVG_RANK INTEGER,        -- League ranking for AVG
            OBP_RANK INTEGER,        -- League ranking for OBP
            SLG_RANK INTEGER,        -- League ranking for SLG
            OPS_RANK INTEGER         -- League ranking for OPS
        )
    ''')

    # Create pitching stats table in wide format
    cursor.execute('''
        CREATE TABLE team_pitching_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            team_name TEXT,          -- Team identifier
            date TEXT,               -- Statistics date (ISO format)
            ERA REAL,                -- Earned run average
            H INTEGER,               -- Hits allowed
            BB INTEGER,              -- Walks issued
            K INTEGER,               -- Strikeouts
            SV INTEGER,              -- Saves
            WHIP REAL                -- Walks + hits per inning pitched
        )
    ''')

    # Create game results table for historical outcomes
    cursor.execute('''
        CREATE TABLE game_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_date TEXT,          -- Date of game (ISO format)
            home_team TEXT,          -- Home team name
            away_team TEXT,          -- Away team name
            home_score INTEGER,      -- Home team runs
            away_score INTEGER       -- Away team runs
        )
    ''')

    conn.commit()

# ---------------------
# 2. SCRAPE TEAM BATTING STATS
# ---------------------
def scrape_team_batting_stats(conn):
    """
    Fetches the team batting statistics page from Yahoo,
    parses the table, and inserts daily metrics into the
    team_batting_stats SQLite table.
    """
    url = "https://sports.yahoo.com/mlb/stats/team/"
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, "html.parser")

    # Extract table headers and rows
    table = soup.find("table")
    headers = [th.text.strip() for th in table.find_all("th")]
    rows = table.select("tbody tr")
    today = datetime.date.today().isoformat()

    # Mapping of Yahoo header labels to DB column names
    expected = {
        'Team': 'team_name', 'AVG': 'AVG', 'OBP': 'OBP', 'SLG': 'SLG', 'OPS': 'OPS',
        'AB': 'AB', 'R': 'R', 'H': 'H', '2B': '"2B"', '3B': '"3B"',
        'HR': 'HR', 'RBI': 'RBI', 'BB': 'BB', 'K': 'K', 'SO': 'SO',
        'SB': 'SB', 'CS': 'CS', 'AVG Rank': 'AVG_RANK', 'OBP Rank': 'OBP_RANK',
        'SLG Rank': 'SLG_RANK', 'OPS Rank': 'OPS_RANK'
    }

    cursor = conn.cursor()
    for row in rows:
        cells = row.find_all("td")
        if not cells:
            continue

        # Initialize a dict for the row, always include the date
        team_data = {'date': today}
        for idx, header in enumerate(headers):
            if header not in expected:
                continue
            db_col = expected[header]
            text = cells[idx].text.strip().replace(',', '').replace('%', '')
            if header == 'Team':
                # Team name is a link inside the cell
                link = cells[idx].find('a')
                team_data[db_col] = link.text.strip() if link else text
            else:
                # Convert numeric fields to float
                try:
                    team_data[db_col] = float(text)
                except ValueError:
                    team_data[db_col] = None

        # Build and execute the INSERT statement dynamically
        cols = ", ".join(team_data.keys())
        placeholders = ", ".join(["?" for _ in team_data])
        vals = list(team_data.values())
        cursor.execute(
            f"INSERT INTO team_batting_stats ({cols}) VALUES ({placeholders})", vals
        )
    conn.commit()
    print(f"[INFO] Team batting stats scraped and stored for {today}")

# ---------------------
# 3. SCRAPE TEAM PITCHING STATS
# ---------------------
def scrape_team_pitching_stats(conn):
    """
    Fetches the team pitching statistics page from Yahoo,
    parses the table, and inserts daily metrics into the
    team_pitching_stats SQLite table.
    """
    url = "https://sports.yahoo.com/mlb/stats/team/?selectedTable=1"
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, "html.parser")

    table = soup.find("table")
    headers = [th.text.strip() for th in table.find_all("th")]
    rows = table.select("tbody tr")
    today = datetime.date.today().isoformat()

    # Mapping of Yahoo header labels to DB column names
    expected = {
        'Team': 'team_name', 'ERA': 'ERA', 'H': 'H', 'BB': 'BB',
        'K': 'K', 'SV': 'SV', 'WHIP': 'WHIP'
    }

    cursor = conn.cursor()
    for row in rows:
        cells = row.find_all("td")
        if not cells:
            continue

        team_data = {'date': today}
        for idx, header in enumerate(headers):
            if header not in expected:
                continue
            db_col = expected[header]
            text = cells[idx].text.strip().replace(',', '').replace('%', '')
            if header == 'Team':
                link = cells[idx].find('a')
                team_data[db_col] = link.text.strip() if link else text
            else:
                try:
                    team_data[db_col] = float(text)
                except ValueError:
                    team_data[db_col] = None

        cols = ", ".join(team_data.keys())
        placeholders = ", ".join(["?" for _ in team_data])
        vals = list(team_data.values())
        cursor.execute(
            f"INSERT INTO team_pitching_stats ({cols}) VALUES ({placeholders})", vals
        )
    conn.commit()
    print(f"[INFO] Team pitching stats scraped and stored for {today}")

# ---------------------
# 4. SCRAPE HISTORICAL GAME RESULTS VIA MLB STATS API
# ---------------------
def scrape_game_results_mlb_api(conn, start_date, end_date):
    """
    Calls the official MLB Stats API's schedule endpoint for each date
    between start_date and end_date. Parses JSON to extract completed
    game scores and inserts into the game_results table.
    """
    cursor = conn.cursor()
    current = datetime.date.fromisoformat(start_date)
    end = datetime.date.fromisoformat(end_date)

    while current <= end:
        date_str = current.isoformat()
        api_url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date_str}"
        resp = requests.get(api_url)
        data = resp.json()
        games_list = data.get('dates', [])

        if not games_list:
            print(f"[WARN] No MLB API data for {date_str}")
            current += datetime.timedelta(days=1)
            continue

        count = 0
        # Iterate through each game object in the JSON
        for day in games_list:
            for game in day.get('games', []):
                away = game['teams']['away']
                home = game['teams']['home']
                away_team = away['team']['name']
                home_team = home['team']['name']
                away_score = away.get('score')
                home_score = home.get('score')
                # Only store if scores are present (game completed)
                if away_score is None or home_score is None:
                    continue
                cursor.execute(
                    "INSERT INTO game_results (game_date, home_team, away_team, home_score, away_score) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (date_str, home_team, away_team, home_score, away_score)
                )
                count += 1
        conn.commit()
        print(f"[INFO] MLB API: {count} games for {date_str} scraped and stored")
        current += datetime.timedelta(days=1)

# ---------------------
# 5. FEATURE ENGINEERING & MODEL TRAINING
# ---------------------
if __name__ == "__main__":
    # Establish SQLite connection and create tables
    conn = sqlite3.connect("mlb_predictions.db")
    create_tables(conn)

    # 5.1 Scrape team-level stats
    scrape_team_batting_stats(conn)
    scrape_team_pitching_stats(conn)

    # 5.2 Scrape historical game results
    season_start = "2025-03-28"  # Opening Day
    today = datetime.date.today().isoformat()
    scrape_game_results_mlb_api(conn, season_start, today)

    # 5.3 Preview the scraped data
    print("\n[INFO] Batting Stats Sample:")
    print(pd.read_sql_query("SELECT * FROM team_batting_stats ORDER BY date DESC LIMIT 5", conn))

    print("\n[INFO] Pitching Stats Sample:")
    print(pd.read_sql_query("SELECT * FROM team_pitching_stats ORDER BY date DESC LIMIT 5", conn))

    print("\n[INFO] Game Results Sample:")
    print(pd.read_sql_query("SELECT * FROM game_results ORDER BY game_date DESC, id DESC LIMIT 5", conn))

    # 5.4 Load data into pandas for modeling
    batting_df = pd.read_sql_query("SELECT * FROM team_batting_stats", conn)
    pitching_df = pd.read_sql_query("SELECT * FROM team_pitching_stats", conn)
    games_df = pd.read_sql_query("SELECT * FROM game_results", conn)

    # Utility to forward-fill team stats up to each game date
    def get_stats_on_date(stats_df, team_col, date_col, prefix):
        stats_df[date_col] = pd.to_datetime(stats_df[date_col])
        # Sort and forward-fill missing days
        stats_sorted = stats_df.sort_values([team_col, date_col])
        filled = stats_sorted.groupby(team_col).apply(
            lambda grp: grp.set_index(date_col).resample('D').ffill().reset_index()
        ).reset_index(drop=True)
        # Prefix column names for clarity
        filled.columns = [f"{prefix}_{c}" if c not in [team_col, date_col] else c for c in filled.columns]
        filled.rename(columns={team_col: f"{prefix}_{team_col}", date_col: "date"}, inplace=True)
        return filled

    # Prepare time-aligned stats for home and away teams
    bat_recent = get_stats_on_date(batting_df, 'team_name', 'date', 'bat')
    pit_recent = get_stats_on_date(pitching_df, 'team_name', 'date', 'pit')

    # Align stats with each game record
    games_df['date'] = pd.to_datetime(games_df['game_date'])
    merged = games_df.merge(
        bat_recent, left_on=['home_team', 'date'], right_on=['bat_team_name', 'date'], how='left'
    ).merge(
        bat_recent, left_on=['away_team', 'date'], right_on=['bat_team_name', 'date'], how='left', suffixes=('', '_away')
    ).merge(
        pit_recent, left_on=['home_team', 'date'], right_on=['pit_team_name', 'date'], how='left'
    ).merge(
        pit_recent, left_on=['away_team', 'date'], right_on=['pit_team_name', 'date'], how='left', suffixes=('', '_away')
    )

    # Create targets
    merged['home_win'] = (merged['home_score'] > merged['away_score']).astype(int)
    merged['margin'] = merged['home_score'] - merged['away_score']
    merged['total_runs'] = merged['home_score'] + merged['away_score']

    # Select core features and compute differences
    feature_cols = ['bat_AVG','bat_OBP','bat_SLG','bat_OPS','bat_AB','bat_R','bat_H','pit_ERA','pit_WHIP']
    for col in feature_cols:
        merged[f'diff_{col}'] = merged[col] - merged[f'{col}_away']

    # Prepare modeling matrices
    X = merged[[f'diff_{col}' for col in feature_cols]].dropna()
    y_win = merged.loc[X.index, 'home_win']
    y_margin = merged.loc[X.index, 'margin']
    y_total = merged.loc[X.index, 'total_runs']

    # Train/test split for win model
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_win_train, y_win_test = train_test_split(X, y_win, test_size=0.2, random_state=42)

    # 5.5 Train Gradient Boosting Classifier for win probability
    from sklearn.ensemble import GradientBoostingClassifier
    clf = GradientBoostingClassifier()
    clf.fit(X_train, y_win_train)
    print(f"[INFO] Win model accuracy: {clf.score(X_test, y_win_test):.3f}")

    # 5.6 Train Gradient Boosting Regressor for margin
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_squared_error
    reg_margin = GradientBoostingRegressor()
    reg_margin.fit(X_train, merged.loc[X_train.index, 'margin'])
    margin_rmse = mean_squared_error(merged.loc[X_test.index, 'margin'], reg_margin.predict(X_test), squared=False)
    print(f"[INFO] Margin RMSE: {margin_rmse:.3f}")

    # 5.7 Train Gradient Boosting Regressor for total runs
    reg_total = GradientBoostingRegressor()
    reg_total.fit(X_train, merged.loc[X_train.index, 'total_runs'])
    total_rmse = mean_squared_error(merged.loc[X_test.index, 'total_runs'], reg_total.predict(X_test), squared=False)
    print(f"[INFO] Total runs RMSE: {total_rmse:.3f}")

    print("[INFO] Model training complete.")
