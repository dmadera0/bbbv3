# mlb_feature_engineering.py
# -----------------------
# This module loads per-game boxscore stats, pivots features, imputes missing values,
# and trains ML models on real boxscore data.

import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

def load_and_train(db_path: str = "mlb_predictions.db"):
    conn = sqlite3.connect(db_path)
    stats = pd.read_sql_query(
        "SELECT game_pk, team_name, AB, R, H, HR, RBI, BB, K, ERA, WHIP FROM team_stats_by_date",
        conn
    )
    games = pd.read_sql_query(
        "SELECT game_pk, home_team, away_team, home_score, away_score FROM game_results",
        conn
    )
    merged = stats.merge(games, on="game_pk", how="inner")
    home_df = merged[merged.team_name == merged.home_team]
    home_feats = home_df.set_index("game_pk")[['AB','R','H','HR','RBI','BB','K','ERA','WHIP']]
    home_feats.columns = [f"home_{c}" for c in home_feats.columns]
    away_df = merged[merged.team_name == merged.away_team]
    away_feats = away_df.set_index("game_pk")[['AB','R','H','HR','RBI','BB','K','ERA','WHIP']]
    away_feats.columns = [f"away_{c}" for c in away_feats.columns]
    features = home_feats.join(away_feats, how='inner')
    outcomes = games.set_index('game_pk')[['home_score','away_score']]
    df = features.join(outcomes, how='inner')
    df['home_win']    = (df.home_score > df.away_score).astype(int)
    df['margin']      = df.home_score - df.away_score
    df['total_runs']  = df.home_score + df.away_score
    X = df.drop(columns=['home_score','away_score','home_win','margin','total_runs'])
    all_nan = [c for c in X.columns if X[c].isna().all()]
    if all_nan:
        print(f"[INFO] Dropping all-NaN columns: {all_nan}")
        X = X.drop(columns=all_nan)
    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
    print(f"[INFO] After imputation, any NaNs left? {X.isna().any().any()}")
    y_win    = df['home_win']
    y_margin = df['margin']
    y_total  = df['total_runs']
    common   = X.index.intersection(y_win.index)
    X        = X.loc[common]
    y_win    = y_win.loc[common]
    y_margin = y_margin.loc[common]
    y_total  = y_total.loc[common]
    if len(X) < 5:
        from sklearn.dummy import DummyClassifier, DummyRegressor
        clf        = DummyClassifier(strategy='uniform').fit(np.zeros((1, X.shape[1])), [0])
        reg_margin = DummyRegressor(strategy='mean').fit(np.zeros((1, X.shape[1])), [0])
        reg_total  = DummyRegressor(strategy='mean').fit(np.zeros((1, X.shape[1])), [0])
        print(f"[WARN] Only {len(X)} games available, using dummy models")
        return clf, reg_margin, reg_total
    (X_train, X_test,
     y_win_tr, y_win_te,
     y_mar_tr, y_mar_te,
     y_tot_tr, y_tot_te) = train_test_split(
        X, y_win, y_margin, y_total, test_size=0.2, random_state=42
    )
    clf        = GradientBoostingClassifier().fit(X_train, y_win_tr)
    reg_margin = GradientBoostingRegressor().fit(X_train, y_mar_tr)
    reg_total  = GradientBoostingRegressor().fit(X_train, y_tot_tr)
    acc         = clf.score(X_test, y_win_te)
    rmse_margin = np.sqrt(mean_squared_error(y_mar_te, reg_margin.predict(X_test)))
    rmse_total  = np.sqrt(mean_squared_error(y_tot_te,  reg_total.predict(X_test)))
    print(f"[INFO] Trained on {len(X_train)} games; Accuracy={acc:.3f}, "
          f"Margin RMSE={rmse_margin:.3f}, Total RMSE={rmse_total:.3f}")
    return clf, reg_margin, reg_total

def build_features_for_upcoming_games(games_df, db_path="mlb_predictions.db"):
    conn = sqlite3.connect(db_path)
    stats = pd.read_sql_query("SELECT * FROM team_stats_by_date", conn)
    stats['date'] = pd.to_datetime(stats['date'])
    most_recent = stats.groupby('team_name')['date'].max().reset_index()
    latest_stats = stats.merge(most_recent, on=['team_name','date'])

    home = latest_stats.rename(columns=lambda x: f"home_{x}" if x not in ['team_name','game_pk','date'] else x)
    away = latest_stats.rename(columns=lambda x: f"away_{x}" if x not in ['team_name','game_pk','date'] else x)

    home = home.rename(columns={"team_name":"home_team"}).set_index("home_team")
    away = away.rename(columns={"team_name":"away_team"}).set_index("away_team")

    meta = games_df[['game_pk','home_team','away_team']].set_index('game_pk')
    merged = meta.join(home, on='home_team').join(away, on='away_team')

    exclude = {'game_pk','date','home_team','away_team','team_name'}
    X = merged.drop(columns=[c for c in merged.columns if any(e in c for e in exclude)])

    all_nan = [c for c in X.columns if X[c].isna().all()]
    if all_nan:
        print(f"[INFO] Dropping all-NaN columns: {all_nan}")
        X = X.drop(columns=all_nan)

    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    X = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)
    return X, meta.reset_index()

if __name__ == "__main__":
    load_and_train()
