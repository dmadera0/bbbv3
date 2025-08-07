import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import joblib

with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

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

# Load team names
team_names = pd.read_sql("""
    SELECT DISTINCT home_team AS name FROM games
    UNION
    SELECT DISTINCT away_team AS name FROM games
""", conn)
teams_list = sorted(team_names['name'].tolist())

st.title("MLB 2025 Game Prediction Dashboard")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Today's Predictions", "Upcoming Games",
    "Historical Predictions", "Full Schedule",
    "All Stored Predictions"
])

# --- TAB 1: TODAY ---
with tab1:
    st.subheader("Today's Predictions")
    today_str = pd.Timestamp.now().strftime("%Y-%m-%d")
    query_today = """
        SELECT g.away_team, g.home_team, p.predicted_winner,
               p.predicted_margin, p.predicted_total
        FROM predictions p
        JOIN games g ON p.game_id = g.game_id
        WHERE p.date = ?
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
        if st.button("Predict All Remaining Games"):
            upcoming_df = pd.read_sql("""
                SELECT game_id, date, home_team, away_team
                FROM games
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
                g_id = game.game_id
                if home not in stats_map or away not in stats_map:
                    continue

                s_home = stats_map[home]
                s_away = stats_map[away]

                def safe_div(n, d): return n / d if d else 0.0

                features = [
                    safe_div(s_home['wins'], s_home['wins'] + s_home['losses']),
                    safe_div(s_away['wins'], s_away['wins'] + s_away['losses']),
                    safe_div(s_home['runs_scored'], s_home['wins'] + s_home['losses']),
                    safe_div(s_away['runs_scored'], s_away['wins'] + s_away['losses']),
                    safe_div(s_home['runs_allowed'], s_home['wins'] + s_home['losses']),
                    safe_div(s_away['runs_allowed'], s_away['wins'] + s_away['losses']),
                    1.0
                ]

                win_proba = clf.predict_proba([features])[0][1]  # probability home team wins
                win = clf.predict([features])[0]
                margin = margin_model.predict([features])[0]
                total = total_model.predict([features])[0]

                predicted = home if win == 1 else away
                confidence_val = round(100 * max(win_proba, 1 - win_proba), 1)
                predictions.append({
                    "game_id": g_id,
                    "date": date,
                    "away_team": away,
                    "home_team": home,
                    "predicted_winner": predicted,
                    "confidence": confidence_val,
                    "predicted_margin": round(float(margin), 1),
                    "predicted_total": round(float(total), 1)
                })

            if predictions:
                df_preds = pd.DataFrame(predictions)
                df_preds.sort_values(by="confidence", ascending=False, inplace=True)
                st.session_state.predictions_all = df_preds
            else:
                st.warning("No games to predict.")

    if st.session_state.predictions_all is not None:
        st.success("Predictions generated.")
        date_list = sorted(st.session_state.predictions_all['date'].unique())
        selected_date = st.selectbox("Select date", date_list)
        filtered = st.session_state.predictions_all[st.session_state.predictions_all['date'] == selected_date]

        def highlight_confidence(row):
            conf = row['confidence']
            if conf >= 75:
                color = "#d4edda"  # green
            elif conf <= 60:
                color = "#f8d7da"  # red
            else:
                color = ""
            return [f"background-color: {color}" for _ in row]

        st.markdown("""
            **Legend:**
            <span style='background-color:#d4edda;padding:4px'>High Confidence (≥ 75%)</span> &nbsp;
            <span style='background-color:#f8d7da;padding:4px'>Low Confidence (≤ 60%)</span>
        """, unsafe_allow_html=True)

        st.dataframe(
            filtered.reset_index(drop=True).style.apply(highlight_confidence, axis=1)
        )

        # --- SAVE PREDICTIONS BUTTON ---
        if st.button("💾 Save Predictions For This Date"):
            saved, skipped = 0, 0
            for _, row in filtered.iterrows():
                exists = conn.execute(
                    "SELECT 1 FROM predictions WHERE game_id = ? AND date = ?",
                    (row['game_id'], row['date'])
                ).fetchone()
                if exists:
                    skipped += 1
                    continue
                conn.execute("""
                    INSERT INTO predictions (game_id, date, predicted_winner, predicted_margin, predicted_total)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    row['game_id'], row['date'], row['predicted_winner'],
                    float(row['predicted_margin']), float(row['predicted_total'])
                ))
                saved += 1
            conn.commit()
            st.success(f"Saved {saved} predictions. Skipped {skipped} already saved.")

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
            SELECT g.away_team, g.home_team, p.predicted_winner, p.predicted_margin,
                   p.predicted_total, g.away_score, g.home_score
            FROM predictions p
            JOIN games g ON p.game_id = g.game_id
            WHERE p.date = ?
              AND g.away_score IS NOT NULL AND g.home_score IS NOT NULL
        """, conn, params=[date_chosen])

        if df.empty:
            st.warning("No predictions found or no completed games for this date.")
        else:
            df['actual_winner'] = np.where(
                df.home_score > df.away_score, df.home_team,
                np.where(df.away_score > df.home_score, df.away_team, "")
            )
            df['correct'] = df['predicted_winner'] == df['actual_winner']
            df['correct'] = df['correct'].apply(lambda x: "✅" if x else "❌")

            def safe_score(row):
                if pd.notnull(row['away_score']) and pd.notnull(row['home_score']):
                    return f"{int(row['away_score'])} - {int(row['home_score'])}"
                return ""

            df['actual_score'] = df.apply(safe_score, axis=1)

            df_final = df[['away_team', 'home_team', 'predicted_winner', 'actual_score',
                           'predicted_margin', 'predicted_total', 'actual_winner', 'correct']]

            accuracy = (df['correct'] == "✅").mean()
            st.markdown(f"**Prediction Accuracy:** {accuracy:.0%} ✅")
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

    team_filter = st.selectbox("Filter by team", ["All Teams"] + teams_list)
    if team_filter != "All Teams":
        df_sched = df_sched[(df_sched['away_team'] == team_filter) | (df_sched['home_team'] == team_filter)]

    st.dataframe(df_sched.reset_index(drop=True))

# --- TAB 5: ALL STORED PREDICTIONS ---
with tab5:
    st.subheader("All Stored Predictions")
    df_all_preds = pd.read_sql("""
        SELECT g.date, g.away_team, g.home_team, 
               p.predicted_winner, p.predicted_margin, p.predicted_total
        FROM predictions p
        JOIN games g ON p.game_id = g.game_id
        ORDER BY g.date DESC
    """, conn)

    if df_all_preds.empty:
        st.info("No stored predictions found.")
    else:
        st.dataframe(df_all_preds.reset_index(drop=True))
