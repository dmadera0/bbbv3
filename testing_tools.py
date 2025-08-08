import sqlite3
import random
import pandas as pd

DB_PATH = "mlb_predictions.db"

def add_dummy_game(date, home_team, away_team, home_score=None, away_score=None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO games (date, home_team, away_team, home_score, away_score, source)
        VALUES (?, ?, ?, ?, ?, 'dummy')
    """, (date, home_team, away_team, home_score, away_score))
    conn.commit()
    game_id = cursor.lastrowid
    conn.close()
    return game_id

def add_dummy_prediction(game_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    predicted_winner = random.choice(['home', 'away'])
    predicted_margin = round(random.uniform(1, 5), 1)
    predicted_total = round(random.uniform(6, 12), 1)

    cursor.execute("SELECT date FROM games WHERE game_id = ?", (game_id,))
    result = cursor.fetchone()
    if result is None:
        print(f"Game ID {game_id} not found.")
        conn.close()
        return

    date = result[0]
    cursor.execute("""
        INSERT INTO predictions (game_id, date, predicted_winner, predicted_margin, predicted_total)
        VALUES (?, ?, ?, ?, ?)
    """, (game_id, date, predicted_winner, predicted_margin, predicted_total))
    conn.commit()
    conn.close()

def delete_game_by_id(game_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM predictions WHERE game_id = ?", (game_id,))
    cursor.execute("DELETE FROM games WHERE game_id = ?", (game_id,))
    conn.commit()
    conn.close()

def edit_game(game_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM games WHERE game_id = ?", (game_id,))
    game = cursor.fetchone()
    if not game:
        print("Game not found.")
        return

    print(f"Current values: Date={game[1]}, Home={game[2]}, Away={game[3]}, Home Score={game[4]}, Away Score={game[5]}")
    date = input("New date (YYYY-MM-DD): ") or game[1]
    home = input("New home team: ") or game[2]
    away = input("New away team: ") or game[3]
    home_score = input("New home score (blank to leave unchanged): ") or game[4]
    away_score = input("New away score (blank to leave unchanged): ") or game[5]

    cursor.execute("""
        UPDATE games SET date=?, home_team=?, away_team=?, home_score=?, away_score=?
        WHERE game_id=?
    """, (date, home, away, home_score, away_score, game_id))
    conn.commit()
    conn.close()
    print("Game updated.")

def show_all_dummy_games():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("""
        SELECT * FROM games
        WHERE source = 'dummy'
        ORDER BY date DESC
    """, conn)
    df['date'] = df['date'] + " ***"
    print(df.to_string(index=False))
    conn.close()

def show_all_dummy_predictions():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("""
        SELECT p.*, g.home_team, g.away_team FROM predictions p
        JOIN games g ON p.game_id = g.game_id
        WHERE g.source = 'dummy'
        ORDER BY p.date DESC
    """, conn)
    print("\n--- Dummy Predictions ---")
    print(df.to_string(index=False))
    conn.close()

def dummy_game_cli():
    while True:
        print("\n--- Dummy Game CLI ---")
        print("1. Add new dummy game")
        print("2. Delete dummy game")
        print("3. Edit dummy game")
        print("4. Show all dummy games")
        print("5. Manage dummy predictions")
        print("6. Exit")
        choice = input("Choose an option: ")

        if choice == "1":
            date = input("Enter date (YYYY-MM-DD): ")
            home = input("Enter home team: ")
            away = input("Enter away team: ")
            home_score = input("Enter home score (optional): ") or None
            away_score = input("Enter away score (optional): ") or None
            home_score = int(home_score) if home_score else None
            away_score = int(away_score) if away_score else None
            gid = add_dummy_game(date, home, away, home_score, away_score)
            add_dummy_prediction(gid)
            print(f"✅ Dummy game added with Game ID {gid}")

        elif choice == "2":
            gid = input("Enter Game ID to delete: ")
            delete_game_by_id(int(gid))
            print("✅ Game deleted.")

        elif choice == "3":
            gid = input("Enter Game ID to edit: ")
            edit_game(int(gid))

        elif choice == "4":
            show_all_dummy_games()

        elif choice == "5":
            dummy_prediction_cli()

        elif choice == "6":
            break

        else:
            print("Invalid choice. Try again.")

def dummy_prediction_cli():
    while True:
        print("\n--- Dummy Prediction CLI ---")
        print("1. Show all dummy predictions")
        print("2. Back to main menu")
        choice = input("Choose an option: ")

        if choice == "1":
            show_all_dummy_predictions()
        elif choice == "2":
            break
        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    dummy_game_cli()
