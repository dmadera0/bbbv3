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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_squared_error


def load_data(db_path="mlb_predictions.db"):
    conn = sqlite3.connect(db_path)
    stats = pd.read_sql_query("SELECT * FROM team_stats", conn)
    games = pd.read_sql_query("SELECT * FROM games", conn)
    conn.close()

    # Pivot stats into home and away
    home = stats[stats.is_home == 1].copy()
    away = stats[stats.is_home == 0].copy()

    home = home.set_index("game_id").add_prefix("home_")
    away = away.set_index("game_id").add_prefix("away_")

    X = home.join(away, lsuffix="_home", rsuffix="_away")
    X = X.drop(columns=["home_team_name", "away_team_name"], errors="ignore")

    # Drop rows with any missing values
    y = games.set_index("game_id")[["home_score", "away_score"]].dropna()
    common = X.index.intersection(y.index)
    X = X.loc[common]
    y = y.loc[common]

    X = X.select_dtypes(include=[np.number])  # keep numeric only

    return X, y


def train_models(X, y):
    # Targets
    y_win = (y.home_score > y.away_score).astype(int)
    y_margin = (y.home_score - y.away_score)
    y_total = (y.home_score + y.away_score)

    # Impute missing features
    all_nan = [c for c in X.columns if X[c].isna().all()]
    if all_nan:
        print(f"[INFO] Dropping all-NaN columns: {all_nan}")
        X = X.drop(columns=all_nan)
    imputer = SimpleImputer(strategy="mean")
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    print(f"[INFO] After imputation, any NaNs left? {X.isna().any().any()}")

    X_train, X_test, y_win_train, y_win_test = train_test_split(X, y_win, test_size=0.2, random_state=42)
    _, _, y_margin_train, y_margin_test = train_test_split(X, y_margin, test_size=0.2, random_state=42)
    _, _, y_total_train, y_total_test = train_test_split(X, y_total, test_size=0.2, random_state=42)

    clf = GradientBoostingClassifier().fit(X_train, y_win_train)
    reg_margin = GradientBoostingRegressor().fit(X_train, y_margin_train)
    reg_total = GradientBoostingRegressor().fit(X_train, y_total_train)

    acc = clf.score(X_test, y_win_test)
    rmse_margin = np.sqrt(mean_squared_error(y_margin_test, reg_margin.predict(X_test)))
    rmse_total = np.sqrt(mean_squared_error(y_total_test, reg_total.predict(X_test)))

    print(f"[INFO] Trained on {len(X_train)} games; Accuracy={acc:.3f}, Margin RMSE={rmse_margin:.3f}, Total RMSE={rmse_total:.3f}")

    return clf, reg_margin, reg_total


if __name__ == "__main__":
    X, y = load_data()
    train_models(X, y)
