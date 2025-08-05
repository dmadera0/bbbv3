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

# Load models (or train if missing)
try:
    clf = joblib.load("model_winner.pkl")
    margin_model = joblib.load("model_margin.pkl")
    total_model = joblib.load("model_total.pkl")
except Exception:
    from mlb_feature_engineering import load_data, train_models
    X, y = load_data("mlb_predictions.db")
    clf, margin_model, total_model = train_models(X, y)

# Connect to DB
conn = sqlite3.connect("mlb_predictions.db", check_same_thread=False)

# Load all known team names
team_names = pd.read_sql("""
    SELECT DISTINCT name FROM team_stats
    UNION
    SELECT DISTINCT home_team AS name FROM games
    UNION
    SELECT DISTINCT away_team AS name FROM games
""", conn)
team_list = sorted(team_names['name'].unique())
team_name_map = {idx: name for idx, name in enumerate(team_list, start=1)}

st.title("⚾ MLB 2025 Game Prediction Dashboard")

tab1, tab2, tab3, tab4 = st.tabs([
    "Today's Predictions", "Upcoming Games",
    "Historical Predictions", "Full Schedule"
])

# --- TAB 1: TODAY ---
with tab1:
    st.subheader("Today's Predictions")
    today_str = pd.Timestamp.now().strftime("%Y-%m-%d")
    query_today = """
        SELECT away_team, home_team, predicted_winner, predicted_margin, predicted_total
        FROM predictions
        WHERE date = ?
    """
    df_today = pd.read_sql(query_today, conn, params=[today_str])
    if df_today.empty:
        st.info("No predictions for today are available yet.")
    else:
        st.table(df_today)

# --- TAB 2: UPCOMING ---
with tab2:
    st.subheader("Predict Upcoming Games")

    if "predictions_all" not in st.session_state:
        st.session_state.predictions_all = None

    if st.session_state.predictions_all is None:
        if st.button("📊 Predict All Remaining Games"):
            upcoming_df = pd.read_sql("""
                SELECT date, home_team, away_team FROM games
                WHERE home_score IS NULL AND away_score IS NULL AND date > ?
                ORDER BY date
            """, conn, params=[today_str])

            stats_df = pd.read_sql("SELECT * FROM team_stats", conn)
            stats_map = stats_df.set_index("name").to_dict("index")

            predictions = []
            for _, game in upcoming_df.iterrows():
                home = game.home_team
                away = game.away_team
                date = game.date
                if home not in stats_map or away not in stats_map:
                    continue

                s_home = stats_map[home]
                s_away = stats_map[away]

                def safe_div(n, d): return n / d if d > 0 else 0.0

                features = [
                    safe_div(s_home['wins'], s_home['wins'] + s_home['losses']),
                    safe_div(s_away['wins'], s_away['wins'] + s_away['losses']),
                    safe_div(s_home['runs_scored'], s_home['wins'] + s_home['losses']),
                    safe_div(s_away['runs_scored'], s_away['wins'] + s_away['losses']),
                    safe_div(s_home['runs_allowed'], s_home['wins'] + s_home['losses']),
                    safe_div(s_away['runs_allowed'], s_away['wins'] + s_away['losses']),
                    1.0  # home field advantage
                ]
                win = clf.predict([features])[0]
                margin = margin_model.predict([features])[0]
                total = total_model.predict([features])[0]
                predictions.append({
                    "date": date,
                    "away_team": away,
                    "home_team": home,
                    "predicted_winner": home if win == 1 else away,
                    "predicted_margin": round(float(margin), 1),
                    "predicted_total": round(float(total), 1)
                })

            if predictions:
                st.session_state.predictions_all = pd.DataFrame(predictions)
            else:
                st.warning("No games to predict.")

    if st.session_state.predictions_all is not None:
        st.success("Predictions generated.")
        date_list = sorted(st.session_state.predictions_all['date'].unique())
        selected_date = st.selectbox("Select date", date_list)
        filtered = st.session_state.predictions_all[st.session_state.predictions_all['date'] == selected_date]
        st.dataframe(filtered.reset_index(drop=True))

# --- TAB 3: HISTORICAL ---
with tab3:
    st.subheader("Historical Predictions")
    dates = pd.read_sql("SELECT DISTINCT date FROM predictions ORDER BY date", conn)['date'].tolist()

    if not dates:
        st.info("No predictions stored yet.")
    else:
        default_idx = len(dates) - 1
        date_chosen = st.selectbox("Select date", dates, index=default_idx)
        df = pd.read_sql("""
            SELECT away_team, home_team, predicted_winner, predicted_margin, predicted_total,
                   away_score, home_score
            FROM predictions
            JOIN games USING (game_id)
            WHERE date = ?
        """, conn, params=[date_chosen])

        if df.empty:
            st.warning("No predictions found.")
        else:
            df['actual_winner'] = np.where(df.home_score > df.away_score, df.home_team, df.away_team)
            df['correct'] = df['predicted_winner'] == df['actual_winner']
            df['correct'] = df['correct'].apply(lambda x: "✅" if x else "❌")
            df['actual_score'] = df.away_score.astype(int).astype(str) + " - " + df.home_score.astype(int).astype(str)
            df_final = df[['away_team', 'home_team', 'predicted_winner', 'actual_score',
                           'predicted_margin', 'predicted_total', 'actual_winner', 'correct']]
            st.dataframe(df_final.reset_index(drop=True))

# --- TAB 4: FULL SCHEDULE ---
with tab4:
    st.subheader("Full 2025 Schedule")
    df_sched = pd.read_sql("""
        SELECT date, away_team, home_team, away_score, home_score
        FROM games
        ORDER BY date
    """, conn)

    df_sched['score'] = df_sched.apply(
        lambda row: f"{int(row.away_score)} - {int(row.home_score)}" if pd.notnull(row.home_score) else "",
        axis=1
    )
    df_sched.drop(columns=['away_score', 'home_score'], inplace=True)

    team_filter = st.selectbox("Filter by team", ["All Teams"] + team_list)
    if team_filter != "All Teams":
        df_sched = df_sched[(df_sched['away_team'] == team_filter) | (df_sched['home_team'] == team_filter)]

    st.dataframe(df_sched.reset_index(drop=True))
