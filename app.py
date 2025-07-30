# MLB Game Predictor - Streamlit App Integration
# --------------------------------------------------
# This Streamlit app provides an interactive interface to predict MLB game outcomes
# using historical team statistics and machine learning models.
# Major functions:
# 1. load_data_and_models(): Load data from SQLite and train ML models
# 2. get_upcoming_games(): Fetch scheduled games for a selected date
# 3. predict_for_games(): Generate predictions for win probability, margin, and total runs
# 4. Streamlit UI: Date selector, game list, predictions display, save functionality

import warnings
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

import streamlit as st
import requests
import pandas as pd
import sqlite3
import datetime
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split

# ---------------------
# GLOBAL CONFIGURATION
# ---------------------
DB_PATH = "mlb_predictions.db"  # Path to the SQLite database file

# ---------------------
# 1. LOAD DATA AND TRAIN MODELS
# ---------------------
@st.cache_data(persist=True)
# Note: st.cache_data is the new caching decorator for data-intensive functions
def load_data_and_models():
    """
    Connect to SQLite database, load historical stats and results,
    perform feature engineering, train ML models, and return:
      - bat_recent: DataFrame of forward-filled batting stats
      - pit_recent: DataFrame of forward-filled pitching stats
      - clf: Trained classifier for home win probability
      - reg_margin: Trained regressor for margin of victory
      - reg_total: Trained regressor for total runs
    If there is insufficient historical data, returns dummy models.
    """
    # Connect to DB
    conn = sqlite3.connect(DB_PATH)

    # 1. Load tables into pandas
    batting_df = pd.read_sql_query("SELECT * FROM team_batting_stats", conn)
    pitching_df = pd.read_sql_query("SELECT * FROM team_pitching_stats", conn)
    games_df = pd.read_sql_query("SELECT * FROM game_results", conn)

    # 2. Feature engineering functions
    def get_stats_on_date(stats_df, team_col, date_col, prefix):
        stats_df[date_col] = pd.to_datetime(stats_df[date_col])
        sorted_df = stats_df.sort_values([team_col, date_col])
        filled = sorted_df.groupby(team_col).apply(
            lambda grp: grp.set_index(date_col).resample('D').ffill().reset_index()
        ).reset_index(drop=True)
        filled.columns = [f"{prefix}_{c}" if c not in [team_col, date_col] else c for c in filled.columns]
        filled.rename(columns={team_col: f"{prefix}_{team_col}", date_col: "date"}, inplace=True)
        return filled

    # Generate recent stats tables
    bat_recent = get_stats_on_date(batting_df, 'team_name', 'date', 'bat')
    pit_recent = get_stats_on_date(pitching_df, 'team_name', 'date', 'pit')

    # 3. Merge stats with game results
    games_df['date'] = pd.to_datetime(games_df['game_date'])
    merged = games_df.merge(
        bat_recent, left_on=['home_team','date'], right_on=['bat_team_name','date'], how='left'
    ).merge(
        bat_recent, left_on=['away_team','date'], right_on=['bat_team_name','date'], how='left', suffixes=('', '_away')
    ).merge(
        pit_recent, left_on=['home_team','date'], right_on=['pit_team_name','date'], how='left'
    ).merge(
        pit_recent, left_on=['away_team','date'], right_on=['pit_team_name','date'], how='left', suffixes=('', '_away')
    )

    # 4. Define targets
    merged['home_win'] = (merged['home_score'] > merged['away_score']).astype(int)
    merged['margin'] = merged['home_score'] - merged['away_score']
    merged['total_runs'] = merged['home_score'] + merged['away_score']

    # 5. Create feature differences
    feature_cols = ['bat_AVG','bat_OBP','bat_SLG','bat_OPS','bat_AB','bat_R','bat_H','pit_ERA','pit_WHIP']
    for col in feature_cols:
        merged[f'diff_{col}'] = merged[col] - merged[f'{col}_away']

    # 6. Prepare modeling matrices
    X = merged[[f'diff_{col}' for col in feature_cols]].dropna()
    y_win = merged.loc[X.index, 'home_win']
    y_margin = merged.loc[X.index, 'margin']
    y_total = merged.loc[X.index, 'total_runs']

    # 7. Handle insufficient data
    if X.shape[0] < 2:
        # Not enough samples to train; create dummy models
        from sklearn.dummy import DummyClassifier, DummyRegressor
        clf = DummyClassifier(strategy="uniform").fit([[0]*len(feature_cols)], [0])
        reg_margin = DummyRegressor(strategy="mean").fit([[0]*len(feature_cols)], [0])
        reg_total = DummyRegressor(strategy="mean").fit([[0]*len(feature_cols)], [0])
        return bat_recent, pit_recent, clf, reg_margin, reg_total

    # 8. Train/test split for classification
    X_train, X_test, y_train, y_test = train_test_split(X, y_win, test_size=0.2, random_state=42)

    # 9. Train models
    clf = GradientBoostingClassifier().fit(X_train, y_train)
    reg_margin = GradientBoostingRegressor().fit(X_train, y_margin.loc[X_train.index])
    reg_total = GradientBoostingRegressor().fit(X_train, y_total.loc[X_train.index])

    return bat_recent, pit_recent, clf, reg_margin, reg_total

# ---------------------
# 2. FETCH UPCOMING GAMES
# ---------------------
def get_upcoming_games(date):
    """
    Query MLB Stats API for scheduled games on a given date.
    Filters out games already completed.
    Returns a DataFrame of home and away teams.
    """
    api_url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date.isoformat()}"
    data = requests.get(api_url).json()
    games_list = []
    for day in data.get('dates', []):
        for game in day.get('games', []):
            status = game.get('status', {}).get('detailedState', '')
            # Exclude final games
            if status in ['Final', 'Game Over']:
                continue
            home = game['teams']['home']['team']['name']
            away = game['teams']['away']['team']['name']
            games_list.append({'home_team': home, 'away_team': away})
    return pd.DataFrame(games_list)

# ---------------------
# 3. PREDICTION LOGIC
# ---------------------
def predict_for_games(df_games, date, bat_recent, pit_recent, clf, reg_margin, reg_total):
    """
    Merge team stats with each upcoming game,
    compute feature differences, and apply models to predict:
      - win probability
      - margin
      - total runs
    Returns DataFrame with prediction columns added.
    """
    df = df_games.copy()
    df['date'] = pd.to_datetime(date)
    # Align home and away stats
    df = df.merge(bat_recent, left_on=['home_team','date'], right_on=['bat_team_name','date'], how='left')
    df = df.merge(bat_recent, left_on=['away_team','date'], right_on=['bat_team_name','date'], how='left', suffixes=('','_away'))
    df = df.merge(pit_recent, left_on=['home_team','date'], right_on=['pit_team_name','date'], how='left')
    df = df.merge(pit_recent, left_on=['away_team','date'], right_on=['pit_team_name','date'], how='left', suffixes=('','_away'))

    # Compute diff features
    feature_cols = ['bat_AVG','bat_OBP','bat_SLG','bat_OPS','bat_AB','bat_R','bat_H','pit_ERA','pit_WHIP']
    for col in feature_cols:
        df[f'diff_{col}'] = df[col] - df[f'{col}_away']

    X_pred = df[[f'diff_{col}' for col in feature_cols]]

    # Generate predictions
    # Handle classifiers with only one class gracefully
    proba = clf.predict_proba(X_pred)
    if proba.shape[1] > 1:
        df['win_prob'] = proba[:, 1]
    else:
        # Only one class present; assign 0.5 probability
        df['win_prob'] = 0.5
    # Always predict margin and total
    df['pred_margin'] = reg_margin.predict(X_pred)
    df['pred_total'] = reg_total.predict(X_pred)

    return df

# ---------------------
# 4. STREAMLIT USER INTERFACE
# ---------------------
# App title
st.title("MLB Game Outcome Predictor")

# Load cached data and models
bat_recent, pit_recent, clf, reg_margin, reg_total = load_data_and_models()

# Date selector
selected_date = st.date_input("Select game date", datetime.date.today())

# Display upcoming games
upcoming = get_upcoming_games(selected_date)
if upcoming.empty:
    st.write("No scheduled games on this date.")
else:
    st.write(f"### Games on {selected_date.isoformat()}")
    preds = predict_for_games(upcoming, selected_date, bat_recent, pit_recent, clf, reg_margin, reg_total)
    # Show prediction table
    st.dataframe(preds[['away_team', 'home_team', 'win_prob', 'pred_margin', 'pred_total']])

    # Button to save predictions
    if st.button("Save Predictions"):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        for _, row in preds.iterrows():
            cursor.execute(
                '''INSERT INTO predictions
                   (game_date, home_team, away_team, predicted_winner, win_confidence, predicted_margin, predicted_total_runs)
                   VALUES (?, ?, ?, ?, ?, ?, ?)''',
                (selected_date.isoformat(), row.home_team, row.away_team,
                 row.home_team if row.win_prob>0.5 else row.away_team,
                 float(row.win_prob), float(row.pred_margin), float(row.pred_total))
            )
        conn.commit()
        st.success("Predictions saved to database.")

# ---------------------
# README & RUNNING THE APP
# ---------------------
# 1. Ensure Python 3.8+ is installed on your system.
# 2. Install required packages:
#    pip install streamlit pandas requests beautifulsoup4 scikit-learn
# 3. If the 'streamlit' command is not found, run:
#    python3 -m streamlit run <your_script_name>.py
# 4. To launch the app (from your project directory):
#    streamlit run <your_script_name>.py
#    OR
#    python3 -m streamlit run <your_script_name>.py
# 5. A browser window will open with the app UI.
# 6. Select a date to view and predict upcoming games.
# 7. Click 'Save Predictions' to store them in the database.
# 8. To exit, press Ctrl+C in the terminal and close the browser tab.
