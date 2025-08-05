"""
Streamlit web application for MLB Game Predictions.
This app allows users to:
- View today's game predictions.
- Generate predictions for all upcoming (future) games.
- Browse historical predictions by selecting a past date.
- View the full 2025 season schedule including future games.
"""
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import joblib

# Load trained models (if available) for on-demand predictions
try:
    clf = joblib.load("model_winner.pkl")
    margin_model = joblib.load("model_margin.pkl")
    total_model = joblib.load("model_total.pkl")
except Exception:
    # If model files not found, train models using latest data
    from mlb_feature_engineering import train_models
    clf, margin_model, total_model = train_models("mlb_predictions.db")

# Connect to the database (allowing thread access for Streamlit)
conn = sqlite3.connect("mlb_predictions.db", check_same_thread=False)

# Load team names for display
teams_df = pd.read_sql("SELECT id, name FROM teams", conn)
team_name_map = {int(row.id): row.name for _, row in teams_df.iterrows()}

st.title("MLB 2025 Game Prediction Dashboard")

# Tabs for different views
tab1, tab2, tab3, tab4 = st.tabs(["Today's Predictions", "Upcoming Games", "Historical Predictions", "Full Schedule"])

# Tab 1: Today's Predictions
with tab1:
    st.subheader("Today's Predictions")
    today_str = pd.Timestamp.now().strftime("%Y-%m-%d")
    query_today = """
        SELECT t1.name as away_team, t2.name as home_team,
               t3.name as predicted_winner,
               p.predicted_margin, p.predicted_total
        FROM predictions p
        JOIN games g ON p.game_id = g.game_id
        JOIN teams t1 ON g.away_team_id = t1.id
        JOIN teams t2 ON g.home_team_id = t2.id
        JOIN teams t3 ON p.predicted_winner_id = t3.id
        WHERE g.date = ?
    """
    df_today = pd.read_sql(query_today, conn, params=[today_str])
    if df_today.empty:
        st.write("No predictions for today are available yet.")
    else:
        st.table(df_today)

# Tab 2: Upcoming Games
with tab2:
    st.subheader("Predict Upcoming Games")
    if "predictions_all" not in st.session_state:
        st.session_state.predictions_all = None
    # Button to generate predictions for all remaining games
    if st.session_state.predictions_all is None:
        if st.button("Predict all remaining games"):
            # Get all unplayed games beyond today
            upcoming_df = pd.read_sql(
                "SELECT date, away_team_id, home_team_id FROM games WHERE home_score IS NULL AND date > ? ORDER BY date",
                conn, params=[pd.Timestamp.now().strftime("%Y-%m-%d")]
            )
            team_stats_df = pd.read_sql("SELECT team_id, wins, losses, runs_scored, runs_allowed FROM team_stats", conn)
            stats_map = {int(row.team_id): row for _, row in team_stats_df.iterrows()}
            predictions = []
            for _, game in upcoming_df.iterrows():
                home_id = int(game.home_team_id)
                away_id = int(game.away_team_id)
                game_date = game.date
                stats_home = stats_map.get(home_id)
                stats_away = stats_map.get(away_id)
                if stats_home is None or stats_away is None:
                    continue
                wins_home, losses_home = int(stats_home.wins), int(stats_home.losses)
                wins_away, losses_away = int(stats_away.wins), int(stats_away.losses)
                games_home = wins_home + losses_home
                games_away = wins_away + losses_away
                runs_home = stats_home.runs_scored
                runs_away = stats_away.runs_scored
                allowed_home = stats_home.runs_allowed
                allowed_away = stats_away.runs_allowed
                # Compute features for this game
                home_win_pct = wins_home / games_home if games_home > 0 else 0.0
                away_win_pct = wins_away / games_away if games_away > 0 else 0.0
                home_runs_per_game = runs_home / games_home if games_home > 0 else 0.0
                away_runs_per_game = runs_away / games_away if games_away > 0 else 0.0
                home_runs_allowed_per_game = allowed_home / games_home if games_home > 0 else 0.0
                away_runs_allowed_per_game = allowed_away / games_away if games_away > 0 else 0.0
                features = [
                    home_win_pct,
                    away_win_pct,
                    home_runs_per_game,
                    away_runs_per_game,
                    home_runs_allowed_per_game,
                    away_runs_allowed_per_game,
                    1.0  # home_field advantage
                ]
                win_pred = clf.predict([features])[0]
                margin_pred = margin_model.predict([features])[0]
                total_pred = total_model.predict([features])[0]
                predicted_winner_id = home_id if win_pred == 1 else away_id
                predictions.append({
                    "date": game_date,
                    "away_team": team_name_map.get(away_id, str(away_id)),
                    "home_team": team_name_map.get(home_id, str(home_id)),
                    "predicted_winner": team_name_map.get(predicted_winner_id, str(predicted_winner_id)),
                    "predicted_margin": round(float(margin_pred), 1),
                    "predicted_total": round(float(total_pred), 1)
                })
            if predictions:
                st.session_state.predictions_all = pd.DataFrame(predictions)
            else:
                st.write("No upcoming games found or prediction failed.")
    if st.session_state.predictions_all is not None:
        st.info("Predictions for all remaining games have been generated.")
        future_dates = sorted(st.session_state.predictions_all['date'].unique().tolist())
        date_choice = st.selectbox("Select date to view predictions", future_dates)
        df_future = st.session_state.predictions_all
        df_selected = df_future[df_future['date'] == date_choice].reset_index(drop=True)
        st.dataframe(df_selected)

# Tab 3: Historical Predictions
with tab3:
    st.subheader("Historical Predictions")
    pred_dates = pd.read_sql("SELECT DISTINCT date FROM predictions ORDER BY date", conn)['date'].tolist()
    if not pred_dates:
        st.write("No historical predictions are available yet.")
    else:
        default_idx = len(pred_dates) - 1  # default to latest date
        date_selected = st.selectbox("Select date", pred_dates, index=default_idx)
        query_hist = """
            SELECT t1.name as away_team, t2.name as home_team,
                   t3.name as predicted_winner,
                   p.predicted_margin, p.predicted_total,
                   g.away_score, g.home_score
            FROM predictions p
            JOIN games g ON p.game_id = g.game_id
            JOIN teams t1 ON g.away_team_id = t1.id
            JOIN teams t2 ON g.home_team_id = t2.id
            JOIN teams t3 ON p.predicted_winner_id = t3.id
            WHERE g.date = ?
        """
        df_hist = pd.read_sql(query_hist, conn, params=[date_selected])
        if df_hist.empty:
            st.write("No predictions available for this date.")
        else:
            # Determine actual winner and whether the prediction was correct
            df_hist['actual_winner'] = np.where(df_hist['home_score'] > df_hist['away_score'],
                                                df_hist['home_team'], df_hist['away_team'])
            df_hist['correct'] = df_hist['predicted_winner'] == df_hist['actual_winner']
            df_hist['correct'] = df_hist['correct'].apply(lambda x: "Yes" if x else "No")
            # Construct a score string for display
            df_hist['actual_score'] = df_hist['away_score'].astype(int).astype(str) + " - " + df_hist['home_score'].astype(int).astype(str)
            # Prepare display DataFrame
            df_display = df_hist.copy()
            df_display.insert(0, 'date', date_selected)
            df_display = df_display[['date', 'away_team', 'home_team', 'predicted_winner',
                                     'predicted_margin', 'predicted_total', 'actual_score', 'correct']]
            st.table(df_display)

# Tab 4: Full Schedule
with tab4:
    st.subheader("Full 2025 Schedule")
    schedule_df = pd.read_sql("""
        SELECT g.date, t1.name as away_team, t2.name as home_team,
               g.away_score, g.home_score, g.status
        FROM games g
        JOIN teams t1 ON g.away_team_id = t1.id
        JOIN teams t2 ON g.home_team_id = t2.id
        ORDER BY g.date
    """, conn)
    # Create a combined score column (blank for future games)
    schedule_df['score'] = schedule_df.apply(
        lambda x: f"{int(x.away_score)} - {int(x.home_score)}" if pd.notnull(x.away_score) else "",
        axis=1
    )
    schedule_df.drop(columns=['away_score', 'home_score'], inplace=True)
    teams_list = ["All Teams"] + sorted(teams_df['name'].tolist())
    team_filter = st.selectbox("Filter by team", teams_list)
    if team_filter and team_filter != "All Teams":
        sched_to_show = schedule_df[(schedule_df['away_team'] == team_filter) | (schedule_df['home_team'] == team_filter)]
    else:
        sched_to_show = schedule_df
    st.dataframe(sched_to_show.reset_index(drop=True))
