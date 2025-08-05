# app.py - Streamlit UI for MLB Game Predictions
# -------------------------------------------------
# Features:
# 1. Predicts all upcoming MLB games using trained ML models.
# 2. Displays historical predictions (stored in DB via predict_today.py).
# 3. Allows navigation and filtering by game date.

import streamlit as st
import pandas as pd
import sqlite3
from datetime import date
from mlb_feature_engineering import load_and_train, build_features_for_upcoming_games
from sklearn.ensemble import GradientBoostingClassifier

# Load and train models
clf, reg_margin, reg_total = load_and_train("mlb_predictions.db")

# App layout
st.title("⚾ MLB Game Predictions")
st.markdown("""
This app predicts the outcome of every MLB game using team boxscore stats. It allows you to:
- View **past predictions** (via stored database)
- Run new **predictions for upcoming games**
- Navigate to any day in the season to see game predictions
""")

# Database connection
conn = sqlite3.connect("mlb_predictions.db")

# ------------------------------
# 1. Show today's saved predictions (predict_today.py)
# ------------------------------
st.header("📅 Today's Predictions")
today = date.today().isoformat()
today_preds = pd.read_sql_query("SELECT * FROM predictions_today WHERE game_date = ?", conn, params=(today,))
if today_preds.empty:
    st.warning("No predictions stored for today. Run predict_today.py or use 'Predict Upcoming Games'.")
else:
    st.dataframe(today_preds, use_container_width=True)

# ------------------------------
# 2. Predict ALL future games
# ------------------------------
st.header("🔮 Predict Upcoming Games")
if st.button("Predict Future Games"):
    # Load upcoming games not yet played
    upcoming = pd.read_sql_query('''
        SELECT s.game_pk, s.game_date, s.home_team, s.away_team
        FROM game_schedule s
        LEFT JOIN game_results r ON s.game_pk = r.game_pk
        WHERE r.game_pk IS NULL
    ''', conn)

    if upcoming.empty:
        st.info("✅ All games have already been played.")
    else:
        X, meta = build_features_for_upcoming_games(upcoming, db_path="mlb_predictions.db")
        preds = meta.copy()
        preds['win_prob']    = clf.predict_proba(X)[:,1]
        preds['margin_pred'] = reg_margin.predict(X)
        preds['total_pred']  = reg_total.predict(X)
        preds['game_date']   = pd.to_datetime(preds['game_pk'].map(dict(zip(upcoming.game_pk, upcoming.game_date))))

        st.success(f"Predicted {len(preds)} upcoming games.")
        st.dataframe(preds[['game_date','home_team','away_team','win_prob','margin_pred','total_pred']], use_container_width=True)

# ------------------------------
# 3. Browse Predictions by Date
# ------------------------------
st.header("📅 View Historical Predictions")
all_dates = pd.read_sql_query("SELECT DISTINCT game_date FROM predictions_today ORDER BY game_date DESC", conn)
selected_date = st.selectbox("Select a game date:", all_dates['game_date'])

query = "SELECT * FROM predictions_today WHERE game_date = ?"
historic = pd.read_sql_query(query, conn, params=(selected_date,))
if historic.empty:
    st.info("No predictions stored for this date.")
else:
    st.dataframe(historic, use_container_width=True)

conn.close()
