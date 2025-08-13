import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import joblib
import os

from datetime import datetime

today_str = datetime.now().strftime("%Y-%m-%d")  # <-- FIXED

with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ---------- SETUP ----------
st.set_page_config(layout="wide")
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

DB_PATH = "mlb_predictions.db"
MODEL_UPDATE_FILE = "last_model_update.txt"
TODAY = pd.Timestamp.now().strftime("%Y-%m-%d")

# ---------- DATABASE CONNECTION ----------
conn = sqlite3.connect(DB_PATH, check_same_thread=False)

# ---------- AUTO-RETRAIN IF NEEDED ----------
def retrain_if_needed():
    if os.path.exists(MODEL_UPDATE_FILE):
        with open(MODEL_UPDATE_FILE, "r") as f:
            last_update = f.read().strip()
            if last_update == TODAY:
                return joblib.load("model_winner.pkl"), joblib.load("model_margin.pkl"), joblib.load("model_total.pkl"), f"✅ Model loaded from cache (last trained: {last_update})"

    from mlb_feature_engineering import load_data, train_models
    X, y = load_data(DB_PATH)
    clf, reg_margin, reg_total = train_models(X, y)
    joblib.dump(clf, "model_winner.pkl")
    joblib.dump(reg_margin, "model_margin.pkl")
    joblib.dump(reg_total, "model_total.pkl")
    with open(MODEL_UPDATE_FILE, "w") as f:
        f.write(TODAY)
    return clf, reg_margin, reg_total, f"✅ Model retrained today: {TODAY}"

clf, margin_model, total_model, retrain_msg = retrain_if_needed()

# ---------- TEAM LIST ----------
team_names = pd.read_sql("""
    SELECT DISTINCT home_team AS name FROM games
    UNION
    SELECT DISTINCT away_team AS name FROM games
""", conn)
teams_list = sorted(team_names['name'].tolist())

# ---------- UI ----------
st.title("MLB 2025 Game Prediction Dashboard")
st.caption(retrain_msg)

# ---------- TABS ----------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Today's Predictions", "Upcoming Games",
    "Historical Predictions", "Full Schedule",
    "All Stored Predictions"
])

# ---------- TAB 1: TODAY ----------
with tab1:
    st.subheader("Today's Predictions")
    df_today = pd.read_sql("""
        SELECT g.away_team, g.home_team, p.predicted_winner,
               p.predicted_margin, p.predicted_total
        FROM predictions p
        JOIN games g ON p.game_id = g.game_id
        WHERE p.date = ?
    """, conn, params=[TODAY])
    st.table(df_today) if not df_today.empty else st.info("No predictions for today are available yet.")

# ---------- TAB 2: UPCOMING ----------
with tab2:
    st.subheader("Predict Upcoming Games")
    if "predictions_all" not in st.session_state:
        st.session_state.predictions_all = None

    if st.session_state.predictions_all is None:
        upcoming_df = pd.read_sql("""
            SELECT game_id, date, home_team, away_team
            FROM games
            WHERE home_score IS NULL AND away_score IS NULL AND date >= ?
            ORDER BY date
        """, conn, params=[TODAY])

        stats_df = pd.read_sql("SELECT * FROM team_stats", conn)
        stats_map = stats_df.set_index("name").to_dict("index")

        predictions = []
        for _, game in upcoming_df.iterrows():
            home, away = game.home_team, game.away_team
            if home not in stats_map or away not in stats_map:
                continue

            h, a = stats_map[home], stats_map[away]
            def safe_div(n, d): return n / d if d else 0.0
            features = [
                safe_div(h['wins'], h['wins'] + h['losses']),
                safe_div(a['wins'], a['wins'] + a['losses']),
                safe_div(h['runs_scored'], h['wins'] + h['losses']),
                safe_div(a['runs_scored'], a['wins'] + a['losses']),
                safe_div(h['runs_allowed'], h['wins'] + h['losses']),
                safe_div(a['runs_allowed'], a['wins'] + a['losses']),
                1.0
            ]

            win_proba = clf.predict_proba([features])[0][1]
            win = clf.predict([features])[0]
            margin = margin_model.predict([features])[0]
            total = total_model.predict([features])[0]

            predictions.append({
                "game_id": game.game_id,
                "date": game.date,
                "away_team": away,
                "home_team": home,
                "predicted_winner": home if win == 1 else away,
                "confidence": round(100 * max(win_proba, 1 - win_proba), 1),
                "predicted_margin": round(float(margin), 1),
                "predicted_total": round(float(total), 1)
            })

        if predictions:
            df_preds = pd.DataFrame(predictions)
            st.session_state.predictions_all = df_preds.sort_values(by="confidence", ascending=False)
            saved, skipped = 0, 0
            for _, row in df_preds.iterrows():
                exists = conn.execute("SELECT 1 FROM predictions WHERE game_id = ? AND date = ?",
                                     (row['game_id'], row['date'])).fetchone()
                if exists:
                    skipped += 1
                    continue
                conn.execute("""
                    INSERT INTO predictions (game_id, date, predicted_winner, predicted_margin, predicted_total)
                    VALUES (?, ?, ?, ?, ?)
                """, (row['game_id'], row['date'], row['predicted_winner'], row['predicted_margin'], row['predicted_total']))
                saved += 1
            conn.commit()
            st.success(f"Saved {saved} new predictions. Skipped {skipped} already saved.")

    if st.session_state.predictions_all is not None:
        selected_date = st.selectbox("Select date", sorted(st.session_state.predictions_all['date'].unique()))
        filtered = st.session_state.predictions_all.query("date == @selected_date")

        def highlight_confidence(row):
            c = row['confidence']
            return [f"background-color: {'#d4edda' if c >= 75 else '#f8d7da' if c <= 60 else ''}" for _ in row]

        st.markdown("""
            **Legend:**
            <span style='background-color:#d4edda;padding:4px'>High Confidence (≥ 75%)</span> &nbsp;
            <span style='background-color:#f8d7da;padding:4px'>Low Confidence (≤ 60%)</span>
        """, unsafe_allow_html=True)

        st.dataframe(filtered.reset_index(drop=True).style.apply(highlight_confidence, axis=1))


# --- TAB 3: HISTORICAL ---
with tab3:
    st.subheader("Historical Predictions")
    dates = pd.read_sql("""
        SELECT DISTINCT date 
        FROM predictions 
        WHERE date < ? 
        ORDER BY date DESC
    """, conn, params=[today_str])['date'].tolist()

    if not dates:
        st.info("No past predictions available.")
    else:
        selected_date = st.selectbox("Select date", dates)
        df = pd.read_sql("""
            SELECT g.away_team, g.home_team, g.away_score, g.home_score,
                   p.predicted_winner, p.predicted_margin, p.predicted_total
            FROM predictions p
            JOIN games g ON p.game_id = g.game_id
            WHERE g.date = ?
              AND g.away_score IS NOT NULL 
              AND g.home_score IS NOT NULL
        """, conn, params=[selected_date])

        if df.empty:
            st.warning("No results found for this date.")
        else:
            df['actual_winner'] = np.where(df.home_score > df.away_score, df.home_team,
                                           np.where(df.home_score < df.away_score, df.away_team, "Tie"))
            df['correct'] = df['predicted_winner'] == df['actual_winner']
            df['correct'] = df['correct'].map({True: "✅", False: "❌"})
            df['actual_score'] = df.apply(
                lambda r: f"{int(r.away_score)} - {int(r.home_score)}", axis=1)

            df_final = df[['away_team', 'home_team', 'predicted_winner', 'actual_score',
                           'predicted_margin', 'predicted_total', 'actual_winner', 'correct']]

            acc = (df['correct'] == "✅").mean()
            st.markdown(f"**Prediction Accuracy:** {acc:.0%} ✅")
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
        lambda row: f"{int(row.away_score)} - {int(row.home_score)}"
        if pd.notnull(row.away_score) and pd.notnull(row.home_score) else "", axis=1)
    df_sched.drop(columns=['away_score', 'home_score'], inplace=True)

    team_filter = st.selectbox("Filter by team", ["All Teams"] + teams_list)
    if team_filter != "All Teams":
        df_sched = df_sched[(df_sched.away_team == team_filter) | (df_sched.home_team == team_filter)]

    st.dataframe(df_sched.reset_index(drop=True))

# --- TAB 5: ALL STORED PREDICTIONS ---
with tab5:
    st.subheader("All Stored Predictions")
    df_all_preds = pd.read_sql("""
        SELECT g.date, g.away_team, g.home_team,
               p.predicted_winner, p.predicted_margin, p.predicted_total,
               g.away_score, g.home_score
        FROM predictions p
        JOIN games g ON p.game_id = g.game_id
        ORDER BY g.date DESC
    """, conn)

    if df_all_preds.empty:
        st.info("No predictions found.")
    else:
        df_all_preds['actual_winner'] = np.where(df_all_preds.home_score > df_all_preds.away_score, df_all_preds.home_team,
                                                 np.where(df_all_preds.away_score > df_all_preds.home_score, df_all_preds.away_team, "Tie"))
        df_all_preds['correct'] = df_all_preds['predicted_winner'] == df_all_preds['actual_winner']
        df_all_preds['correct'] = df_all_preds['correct'].map({True: "✅", False: "❌"})
        df_all_preds['actual_score'] = df_all_preds.apply(
            lambda r: f"{int(r.away_score)} - {int(r.home_score)}" if pd.notnull(r.away_score) and pd.notnull(r.home_score) else "",
            axis=1)

        df_final = df_all_preds[['date', 'away_team', 'home_team', 'predicted_winner',
                                 'predicted_margin', 'predicted_total', 'actual_score', 'actual_winner', 'correct']]
        st.dataframe(df_final.reset_index(drop=True))
