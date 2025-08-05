"""
Database Setup Script for MLB Prediction System.
This script fetches MLB team information, the full 2025 schedule, and initial team stats (batting, pitching, win-loss records) using the MLB Stats API.
It builds a SQLite database (mlb_predictions.db) with tables for teams, games, and team_stats.
The MLB Stats API schedule endpoint requires sportId, startDate, and endDate parameters:contentReference[oaicite:0]{index=0}.
The teams endpoint requires sportId and season parameters:contentReference[oaicite:1]{index=1}.
Team stats are retrieved per team by specifying teamId, season, and stat group (e.g., hitting or pitching):contentReference[oaicite:2]{index=2}.
"""
import sqlite3
import requests

# Define MLB API endpoints and parameters
YEAR = 2025
SPORT_ID = 1  # MLB sportId
teams_url = f"https://statsapi.mlb.com/api/v1/teams?season={YEAR}&sportIds={SPORT_ID}"
# Use schedule endpoint to get all regular season games for 2025
schedule_url = f"https://statsapi.mlb.com/api/v1/schedule?sportId={SPORT_ID}&startDate=03/01/{YEAR}&endDate=12/01/{YEAR}&gameTypes=R"

# Fetch teams data from MLB API
response_teams = requests.get(teams_url)
teams_data = response_teams.json().get("teams", [])

# Connect to (or create) the SQLite database
conn = sqlite3.connect("mlb_predictions.db")
c = conn.cursor()

# Drop existing tables if they exist, to rebuild from scratch
c.execute("DROP TABLE IF EXISTS teams")
c.execute("DROP TABLE IF EXISTS games")
c.execute("DROP TABLE IF EXISTS team_stats")

# Create teams table
c.execute("""
    CREATE TABLE teams (
        id INTEGER PRIMARY KEY,
        name TEXT,
        abbreviation TEXT
    )
""")

# Create games table
c.execute("""
    CREATE TABLE games (
        game_id INTEGER PRIMARY KEY,
        date TEXT,
        home_team_id INTEGER,
        away_team_id INTEGER,
        home_score INTEGER,
        away_score INTEGER,
        status TEXT
    )
""")

# Create team_stats table
c.execute("""
    CREATE TABLE team_stats (
        team_id INTEGER PRIMARY KEY,
        wins INTEGER,
        losses INTEGER,
        runs_scored INTEGER,
        runs_allowed INTEGER,
        batting_avg REAL,
        era REAL
    )
""")

# Insert team data into teams table
teams_list = []
for team in teams_data:
    team_id = team.get("id")
    name = team.get("name")
    # Use abbreviation or a fallback field for short name
    abbr = team.get("abbreviation") or team.get("teamCode") or team.get("clubName")
    teams_list.append((team_id, name, abbr))
c.executemany("INSERT INTO teams (id, name, abbreviation) VALUES (?, ?, ?)", teams_list)

# Fetch full season schedule data
response_schedule = requests.get(schedule_url)
schedule_json = response_schedule.json()

# Prepare to accumulate team performance (wins, losses, runs) from completed games
team_performance = {team_id: [0, 0, 0, 0] for team_id, _, _ in teams_list}  # team_id -> [wins, losses, runs_scored, runs_allowed]

games_list = []
# Loop through each date in the schedule
for date_block in schedule_json.get("dates", []):
    date_str = date_block.get("date")  # format "YYYY-MM-DD"
    for game in date_block.get("games", []):
        game_id = game.get("gamePk")
        status = game.get("status", {}).get("detailedState", "")
        # Team IDs and scores
        home_team = game.get("teams", {}).get("home", {})
        away_team = game.get("teams", {}).get("away", {})
        home_team_id = home_team.get("team", {}).get("id")
        away_team_id = away_team.get("team", {}).get("id")
        home_score = home_team.get("score")
        away_score = away_team.get("score")
        # Add game record to list (scores may be None for future games)
        games_list.append((game_id, date_str, home_team_id, away_team_id, home_score, away_score, status))
        # If game is final, update team performance stats
        if home_score is not None and away_score is not None:
            if home_score > away_score:
                # Home win, away loss
                team_performance[home_team_id][0] += 1
                team_performance[away_team_id][1] += 1
            elif away_score > home_score:
                # Away win, home loss
                team_performance[away_team_id][0] += 1
                team_performance[home_team_id][1] += 1
            # Update runs scored and allowed
            team_performance[home_team_id][2] += home_score
            team_performance[home_team_id][3] += away_score
            team_performance[away_team_id][2] += away_score
            team_performance[away_team_id][3] += home_score

# Insert all games into games table
c.executemany("""
    INSERT INTO games (game_id, date, home_team_id, away_team_id, home_score, away_score, status)
    VALUES (?, ?, ?, ?, ?, ?, ?)
""", games_list)

# Build team_stats entries using performance data and additional stats from API
team_stats_list = []
for team_id, perf in team_performance.items():
    wins, losses, runs_scored, runs_allowed = perf
    # Fetch team batting and pitching stats (e.g., batting average and ERA)
    stats_batting_url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/stats?season={YEAR}&group=hitting&stats=season"
    stats_pitching_url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/stats?season={YEAR}&group=pitching&stats=season"
    batting_avg = 0.0
    era = 0.0
    try:
        r_bat = requests.get(stats_batting_url)
        r_pitch = requests.get(stats_pitching_url)
        stats_batting = r_bat.json().get("stats", [])
        stats_pitching = r_pitch.json().get("stats", [])
        if stats_batting:
            # The batting average is typically under 'stat' -> 'avg'
            batting_avg_str = stats_batting[0]["splits"][0]["stat"].get("avg")
            if batting_avg_str is not None:
                batting_avg = float(batting_avg_str)
        if stats_pitching:
            # The ERA is typically under 'stat' -> 'era'
            era_str = stats_pitching[0]["splits"][0]["stat"].get("era")
            if era_str is not None:
                era = float(era_str)
    except Exception:
        # In case of an API error, leave batting_avg and era as 0.0
        batting_avg = batting_avg
        era = era
    team_stats_list.append((team_id, wins, losses, runs_scored, runs_allowed, batting_avg, era))

# Insert all team stats into team_stats table
c.executemany("""
    INSERT INTO team_stats (team_id, wins, losses, runs_scored, runs_allowed, batting_avg, era)
    VALUES (?, ?, ?, ?, ?, ?, ?)
""", team_stats_list)

# Commit changes and close the database connection
conn.commit()
conn.close()
