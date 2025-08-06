"""
Feature engineering and model training for MLB game predictions.
This module transforms team and game data into features and trains machine learning models:
- A classification model to predict the game winner.
- A regression model to predict the run score margin.
- A regression model to predict total runs scored.
The models are trained using only 2025 season data (team stats and game outcomes from 2025).
"""
# mlb_feature_engineering.py
# ---------------------------
# Loads team stats + game results from database, trains models, predicts outcomes

import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.impute import SimpleImputer
import joblib


def load_data(db_path="mlb_predictions.db"):
    conn = sqlite3.connect(db_path)
    games = pd.read_sql("SELECT * FROM games WHERE home_score IS NOT NULL AND away_score IS NOT NULL", conn)
    team_stats = pd.read_sql("SELECT * FROM team_stats", conn)
    conn.close()

    # Build feature rows for each game using team-level stats
    rows = []
    for _, row in games.iterrows():
        home = team_stats[team_stats.name == row.home_team]
        away = team_stats[team_stats.name == row.away_team]
        if home.empty or away.empty:
            continue

        home = home.iloc[0]
        away = away.iloc[0]
        home_games = home.wins + home.losses
        away_games = away.wins + away.losses

        features = {
            "home_win_pct": home.wins / home_games if home_games > 0 else 0.0,
            "away_win_pct": away.wins / away_games if away_games > 0 else 0.0,
            "home_runs_scored_pg": home.runs_scored / home_games if home_games > 0 else 0.0,
            "away_runs_scored_pg": away.runs_scored / away_games if away_games > 0 else 0.0,
            "home_runs_allowed_pg": home.runs_allowed / home_games if home_games > 0 else 0.0,
            "away_runs_allowed_pg": away.runs_allowed / away_games if away_games > 0 else 0.0,
            "home_field": 1.0
        }
        label = {
            "home_score": row.home_score,
            "away_score": row.away_score
        }

        rows.append((features, label))

    # Convert to DataFrame
    X = pd.DataFrame([f for f, _ in rows])
    y = pd.DataFrame([l for _, l in rows])

    return X, y


def train_models(X, y):
    y_win = (y.home_score > y.away_score).astype(int)
    y_margin = y.home_score - y.away_score
    y_total = y.home_score + y.away_score

    imputer = SimpleImputer(strategy="mean")
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    X_train, X_test, y_win_train, y_win_test = train_test_split(X, y_win, test_size=0.2, random_state=42)
    _, _, y_margin_train, y_margin_test = train_test_split(X, y_margin, test_size=0.2, random_state=42)
    _, _, y_total_train, y_total_test = train_test_split(X, y_total, test_size=0.2, random_state=42)

    clf = GradientBoostingClassifier().fit(X_train, y_win_train)
    reg_margin = GradientBoostingRegressor().fit(X_train, y_margin_train)
    reg_total = GradientBoostingRegressor().fit(X_train, y_total_train)

    acc = accuracy_score(y_win_test, clf.predict(X_test))
    rmse_margin = np.sqrt(mean_squared_error(y_margin_test, reg_margin.predict(X_test)))
    rmse_total = np.sqrt(mean_squared_error(y_total_test, reg_total.predict(X_test)))

    print(f"[INFO] Accuracy={acc:.3f}, Margin RMSE={rmse_margin:.3f}, Total RMSE={rmse_total:.3f}")

    joblib.dump(clf, "model_winner.pkl")
    joblib.dump(reg_margin, "model_margin.pkl")
    joblib.dump(reg_total, "model_total.pkl")
    print("[INFO] Models saved.")

    return clf, reg_margin, reg_total


if __name__ == "__main__":
    X, y = load_data()
    train_models(X, y)
