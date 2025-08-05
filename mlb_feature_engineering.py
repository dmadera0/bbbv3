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
# Loads features from database, trains models, predicts outcomes

import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_squared_error

def load_data(db_path="mlb_predictions.db"):
    """
    Loads team_stats and games from SQLite, pivots into feature matrix X and outcome y.
    Ensures X and y have aligned indices by reindexing X to y's index.
    Returns:
      X: DataFrame of numeric features, indexed by game_id
      y: DataFrame with home_score and away_score indexed by game_id
    """
    conn = sqlite3.connect(db_path)
    stats = pd.read_sql_query("SELECT * FROM team_stats", conn)
    games = pd.read_sql_query("SELECT * FROM games", conn)
    conn.close()

    # Pivot stats into home and away features per game
    home = stats[stats.is_home == 1].copy()
    away = stats[stats.is_home == 0].copy()

    home = home.set_index("game_id").add_prefix("home_")
    away = away.set_index("game_id").add_prefix("away_")

    # Combine home & away into feature matrix X
    X = home.join(away, how="inner")

    # Remove potential duplicate game_id rows
    X = X[~X.index.duplicated(keep='first')]

    # Drop identifier columns if present
    for col in ["home_team_name", "away_team_name"]:
        X.drop(columns=[col], errors="ignore", inplace=True)

    # Load outcomes and keep only games with recorded scores
    y = games.set_index("game_id")[['home_score','away_score']].dropna()

    # Align features to the outcomes index
    X = X.reindex(index=y.index)

    # Keep only numeric feature columns
    X = X.select_dtypes(include=[np.number])

    # Debug lengths after alignment
    print(f"[DEBUG] load_data after align: X.len={len(X)}, y.len={len(y)}")

    return X, y


def train_models(X, y):
    """
    Train models for:
     - Home win classification
     - Score margin regression
     - Total runs regression
    Returns trained (clf, reg_margin, reg_total).
    """
    # Targets
    y_win = (y.home_score > y.away_score).astype(int)
    y_margin = (y.home_score - y.away_score)
    y_total = (y.home_score + y.away_score)

    # Debug target lengths
    print(f"[DEBUG] train_models: y_win.len={len(y_win)}, y_margin.len={len(y_margin)}, y_total.len={len(y_total)}")

    # Impute missing feature values if any
    all_nan = [c for c in X.columns if X[c].isna().all()]
    if all_nan:
        print(f"[INFO] Dropping all-NaN columns: {all_nan}")
        X = X.drop(columns=all_nan)
    imputer = SimpleImputer(strategy="mean")
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
    print(f"[INFO] After imputation, any NaNs left? {X.isna().any().any()}")

    # Unified split of features and three targets
    (X_train, X_test,
     y_win_train, y_win_test,
     y_margin_train, y_margin_test,
     y_total_train, y_total_test) = train_test_split(
        X, y_win, y_margin, y_total, test_size=0.2, random_state=42
    )

    # Train models
    clf = GradientBoostingClassifier().fit(X_train, y_win_train)
    reg_margin = GradientBoostingRegressor().fit(X_train, y_margin_train)
    reg_total = GradientBoostingRegressor().fit(X_train, y_total_train)

    # Evaluate
    acc = clf.score(X_test, y_win_test)
    rmse_margin = np.sqrt(mean_squared_error(y_margin_test, reg_margin.predict(X_test)))
    rmse_total = np.sqrt(mean_squared_error(y_total_test, reg_total.predict(X_test)))
    print(f"[INFO] Trained on {len(X_train)} games; Accuracy={acc:.3f}, "
          f"Margin RMSE={rmse_margin:.3f}, Total RMSE={rmse_total:.3f}")

    return clf, reg_margin, reg_total

if __name__ == "__main__":
    X, y = load_data()
    train_models(X, y)
