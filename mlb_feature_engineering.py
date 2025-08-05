"""
Feature engineering and model training for MLB game predictions.
This module transforms team and game data into features and trains machine learning models:
- A classification model to predict the game winner.
- A regression model to predict the run score margin.
- A regression model to predict total runs scored.
The models are trained using only 2025 season data (team stats and game outcomes from 2025).
"""
import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def create_training_data(db_path="mlb_predictions.db"):
    """
    Create training feature matrix X and target vectors (y_win, y_margin, y_total) from the database.
    It uses team-level cumulative statistics up to each game (prior to that game) as features.
    
    Returns:
        X (pd.DataFrame): Feature matrix.
        y_win (pd.Series): Binary target for home team win (1 if home team wins, 0 if away wins).
        y_margin (pd.Series): Target for score margin (home_score - away_score).
        y_total (pd.Series): Target for total runs scored (home_score + away_score).
    """
    conn = sqlite3.connect(db_path)
    # Load all games that have been played (home_score not NULL) in chronological order
    games_df = pd.read_sql("SELECT * FROM games WHERE home_score IS NOT NULL ORDER BY date", conn)
    conn.close()
    X_features = []
    y_win = []
    y_margin = []
    y_total = []
    # Initialize cumulative stats for each team (games played, wins, losses, runs scored, runs allowed)
    cumulative_stats = {}
    team_ids = set(games_df['home_team_id']) | set(games_df['away_team_id'])
    for tid in team_ids:
        cumulative_stats[tid] = [0, 0, 0, 0, 0]  # [games, wins, losses, runs_scored, runs_allowed]
    # Iterate through each game in order to build features without using future data
    for _, game in games_df.iterrows():
        home_id = game['home_team_id']
        away_id = game['away_team_id']
        home_score = game['home_score']
        away_score = game['away_score']
        # Current cumulative stats for each team before this game
        games_home, wins_home, losses_home, runs_home, allowed_home = cumulative_stats[home_id]
        games_away, wins_away, losses_away, runs_away, allowed_away = cumulative_stats[away_id]
        # Compute features: win percentage and average runs scored/allowed prior to the game
        home_win_pct = wins_home / games_home if games_home > 0 else 0.0
        away_win_pct = wins_away / games_away if games_away > 0 else 0.0
        home_runs_per_game = runs_home / games_home if games_home > 0 else 0.0
        away_runs_per_game = runs_away / games_away if games_away > 0 else 0.0
        home_runs_allowed_per_game = allowed_home / games_home if games_home > 0 else 0.0
        away_runs_allowed_per_game = allowed_away / games_away if games_away > 0 else 0.0
        # Home field indicator (1 for home team)
        home_field = 1.0
        # Feature vector for this game
        features = [
            home_win_pct,
            away_win_pct,
            home_runs_per_game,
            away_runs_per_game,
            home_runs_allowed_per_game,
            away_runs_allowed_per_game,
            home_field
        ]
        X_features.append(features)
        # Targets for this game
        home_win = 1 if home_score > away_score else 0
        margin = home_score - away_score
        total_runs = home_score + away_score
        y_win.append(home_win)
        y_margin.append(margin)
        y_total.append(total_runs)
        # Update cumulative stats after this game
        cumulative_stats[home_id][0] += 1  # games played for home
        cumulative_stats[away_id][0] += 1  # games played for away
        if home_score > away_score:
            cumulative_stats[home_id][1] += 1  # home wins
            cumulative_stats[away_id][2] += 1  # away losses
        else:
            cumulative_stats[away_id][1] += 1  # away wins
            cumulative_stats[home_id][2] += 1  # home losses
        # Update runs scored and allowed
        cumulative_stats[home_id][3] += home_score
        cumulative_stats[home_id][4] += away_score
        cumulative_stats[away_id][3] += away_score
        cumulative_stats[away_id][4] += home_score
    # Convert feature list and targets to DataFrame/Series
    X = pd.DataFrame(X_features, columns=[
        "home_win_pct", "away_win_pct",
        "home_runs_per_game", "away_runs_per_game",
        "home_runs_allowed_per_game", "away_runs_allowed_per_game",
        "home_field"
    ])
    y_win = pd.Series(y_win, name="home_win")
    y_margin = pd.Series(y_margin, name="run_margin")
    y_total = pd.Series(y_total, name="total_runs")
    return X, y_win, y_margin, y_total

def train_models(db_path="mlb_predictions.db"):
    """
    Train machine learning models for game outcome prediction using 2025 data.
    
    Returns:
        clf (RandomForestClassifier): Model predicting whether the home team wins.
        margin_model (RandomForestRegressor): Model predicting the run margin (home - away).
        total_model (RandomForestRegressor): Model predicting total runs scored.
    """
    # Prepare training data
    X, y_win, y_margin, y_total = create_training_data(db_path)
    # Initialize models
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    margin_model = RandomForestRegressor(n_estimators=100, random_state=42)
    total_model = RandomForestRegressor(n_estimators=100, random_state=42)
    # Train models on the entire 2025 dataset
    clf.fit(X, y_win)
    margin_model.fit(X, y_margin)
    total_model.fit(X, y_total)
    return clf, margin_model, total_model
