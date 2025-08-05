"""
MLB Game Prediction System - Component 1: Data Pipeline and Database Setup
This module handles data collection from MLB Stats API and database operations.
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
import json

# You'll need to install: pip install MLB-StatsAPI pandas numpy
try:
    import statsapi
except ImportError:
    print("Please install MLB-StatsAPI: pip install MLB-StatsAPI")
    raise

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MLBDatabase:
    """Handles all database operations for the MLB prediction system."""
    
    def __init__(self, db_path: str = "mlb_predictions.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the SQLite database with required tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Teams table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS teams (
                team_id INTEGER PRIMARY KEY,
                team_name TEXT NOT NULL,
                team_abbrev TEXT NOT NULL,
                division TEXT,
                league TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Games table - stores historical and scheduled games
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS games (
                game_id INTEGER PRIMARY KEY,
                game_date DATE NOT NULL,
                home_team_id INTEGER NOT NULL,
                away_team_id INTEGER NOT NULL,
                home_score INTEGER,
                away_score INTEGER,
                game_status TEXT,
                season INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (home_team_id) REFERENCES teams (team_id),
                FOREIGN KEY (away_team_id) REFERENCES teams (team_id)
            )
        ''')
        
        # Team stats table - daily snapshots of team performance
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS team_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                team_id INTEGER NOT NULL,
                stat_date DATE NOT NULL,
                season INTEGER NOT NULL,
                -- Offensive stats
                runs_per_game REAL,
                batting_avg REAL,
                on_base_pct REAL,
                slugging_pct REAL,
                ops REAL,
                home_runs INTEGER,
                rbis INTEGER,
                stolen_bases INTEGER,
                walks INTEGER,
                strikeouts INTEGER,
                -- Pitching stats
                era REAL,
                whip REAL,
                strikeouts_per_9 REAL,
                walks_per_9 REAL,
                hits_per_9 REAL,
                home_runs_allowed INTEGER,
                runs_allowed_per_game REAL,
                -- Fielding stats
                fielding_pct REAL,
                errors INTEGER,
                -- Record
                wins INTEGER,
                losses INTEGER,
                win_pct REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (team_id) REFERENCES teams (team_id),
                UNIQUE(team_id, stat_date)
            )
        ''')
        
        # Predictions table - stores model predictions
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id INTEGER NOT NULL,
                predicted_winner_id INTEGER NOT NULL,
                win_probability REAL NOT NULL,
                predicted_margin REAL NOT NULL,
                predicted_total_runs REAL NOT NULL,
                model_version TEXT,
                prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (game_id) REFERENCES games (game_id),
                FOREIGN KEY (predicted_winner_id) REFERENCES teams (team_id)
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    
    def insert_teams(self, teams_data: List[Dict]):
        """Insert team data into the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for team in teams_data:
            cursor.execute('''
                INSERT OR REPLACE INTO teams (team_id, team_name, team_abbrev, division, league)
                VALUES (?, ?, ?, ?, ?)
            ''', (team['id'], team['name'], team['abbreviation'], 
                  team.get('division', ''), team.get('league', '')))
        
        conn.commit()
        conn.close()
        logger.info(f"Inserted {len(teams_data)} teams")
    
    def insert_games(self, games_data: List[Dict]):
        """Insert game data into the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for game in games_data:
            cursor.execute('''
                INSERT OR REPLACE INTO games 
                (game_id, game_date, home_team_id, away_team_id, home_score, away_score, game_status, season)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (game['game_id'], game['game_date'], game['home_team_id'], 
                  game['away_team_id'], game.get('home_score'), game.get('away_score'),
                  game['game_status'], game['season']))
        
        conn.commit()
        conn.close()
        logger.info(f"Inserted {len(games_data)} games")
    
    def insert_team_stats(self, stats_data: List[Dict]):
        """Insert team statistics into the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for stats in stats_data:
            cursor.execute('''
                INSERT OR REPLACE INTO team_stats 
                (team_id, stat_date, season, runs_per_game, batting_avg, on_base_pct, 
                 slugging_pct, ops, home_runs, rbis, stolen_bases, walks, strikeouts,
                 era, whip, strikeouts_per_9, walks_per_9, hits_per_9, home_runs_allowed,
                 runs_allowed_per_game, fielding_pct, errors, wins, losses, win_pct)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                stats['team_id'], stats['stat_date'], stats['season'],
                stats.get('runs_per_game'), stats.get('batting_avg'), stats.get('on_base_pct'),
                stats.get('slugging_pct'), stats.get('ops'), stats.get('home_runs'),
                stats.get('rbis'), stats.get('stolen_bases'), stats.get('walks'), stats.get('strikeouts'),
                stats.get('era'), stats.get('whip'), stats.get('strikeouts_per_9'),
                stats.get('walks_per_9'), stats.get('hits_per_9'), stats.get('home_runs_allowed'),
                stats.get('runs_allowed_per_game'), stats.get('fielding_pct'), stats.get('errors'),
                stats.get('wins'), stats.get('losses'), stats.get('win_pct')
            ))
        
        conn.commit()
        conn.close()
        logger.info(f"Inserted {len(stats_data)} team stat records")
    
    def get_games_for_prediction(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Get games that need predictions."""
        conn = sqlite3.connect(self.db_path)
        query = '''
            SELECT g.game_id, g.game_date, g.home_team_id, g.away_team_id,
                   ht.team_name as home_team_name, at.team_name as away_team_name,
                   g.game_status
            FROM games g
            JOIN teams ht ON g.home_team_id = ht.team_id
            JOIN teams at ON g.away_team_id = at.team_id
            WHERE g.game_date BETWEEN ? AND ?
            AND g.game_status IN ('Scheduled', 'Pre-Game')
            ORDER BY g.game_date
        '''
        df = pd.read_sql_query(query, conn, params=(start_date, end_date))
        conn.close()
        return df
    
    def get_team_stats_for_date(self, team_id: int, date: str) -> Dict:
        """Get team stats for a specific date (or most recent before that date)."""
        conn = sqlite3.connect(self.db_path)
        query = '''
            SELECT * FROM team_stats 
            WHERE team_id = ? AND stat_date <= ?
            ORDER BY stat_date DESC
            LIMIT 1
        '''
        cursor = conn.cursor()
        cursor.execute(query, (team_id, date))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            columns = [description[0] for description in cursor.description]
            return dict(zip(columns, result))
        return {}

class MLBDataCollector:
    """Handles data collection from MLB Stats API."""
    
    def __init__(self, db: MLBDatabase):
        self.db = db
        self.current_season = 2025
    
    def collect_teams(self):
        """Collect and store all MLB teams."""
        logger.info("Collecting team data...")
        teams = statsapi.get('teams', {'sportId': 1})  # MLB is sport ID 1
        
        teams_data = []
        for team in teams['teams']:
            teams_data.append({
                'id': team['id'],
                'name': team['name'],
                'abbreviation': team['abbreviation'],
                'division': team.get('division', {}).get('name', ''),
                'league': team.get('league', {}).get('name', '')
            })
        
        self.db.insert_teams(teams_data)
        return teams_data
    
    def collect_games_for_date_range(self, start_date: str, end_date: str):
        """Collect games for a date range."""
        logger.info(f"Collecting games from {start_date} to {end_date}...")
        
        # Get schedule for the date range
        schedule = statsapi.schedule(start_date=start_date, end_date=end_date)
        
        games_data = []
        for game in schedule:
            games_data.append({
                'game_id': game['game_id'],
                'game_date': game['game_date'],
                'home_team_id': game['home_id'],
                'away_team_id': game['away_id'],
                'home_score': game.get('home_score'),
                'away_score': game.get('away_score'),
                'game_status': game['status'],
                'season': self.current_season
            })
        
        self.db.insert_games(games_data)
        return games_data
    
    def collect_team_stats(self, season: int = 2025):
        """Collect current team statistics."""
        logger.info(f"Collecting team stats for season {season}...")
        
        # Get all teams first
        conn = sqlite3.connect(self.db.db_path)
        teams_df = pd.read_sql_query("SELECT team_id, team_abbrev FROM teams", conn)
        conn.close()
        
        stats_data = []
        current_date = datetime.now().strftime('%Y-%m-%d')
        
        for _, team in teams_df.iterrows():
            try:
                # Get team stats from API
                team_stats = statsapi.team_stats(team['team_id'], group="[hitting,pitching,fielding]", type="season")
                
                # Parse hitting stats
                hitting_stats = {}
                pitching_stats = {}
                fielding_stats = {}
                
                for stat_group in team_stats:
                    if stat_group['group'] == 'hitting':
                        stats = stat_group['stats']
                        hitting_stats = {
                            'batting_avg': float(stats.get('avg', 0)),
                            'on_base_pct': float(stats.get('onBasePercentage', 0)),
                            'slugging_pct': float(stats.get('slg', 0)),
                            'ops': float(stats.get('ops', 0)),
                            'home_runs': int(stats.get('homeRuns', 0)),
                            'rbis': int(stats.get('rbi', 0)),
                            'stolen_bases': int(stats.get('stolenBases', 0)),
                            'walks': int(stats.get('baseOnBalls', 0)),
                            'strikeouts': int(stats.get('strikeOuts', 0)),
                            'runs_per_game': float(stats.get('runs', 0)) / max(float(stats.get('gamesPlayed', 1)), 1)
                        }
                    
                    elif stat_group['group'] == 'pitching':
                        stats = stat_group['stats']
                        pitching_stats = {
                            'era': float(stats.get('era', 0)),
                            'whip': float(stats.get('whip', 0)),
                            'strikeouts_per_9': float(stats.get('strikeoutsPer9Inn', 0)),
                            'walks_per_9': float(stats.get('walksPer9Inn', 0)),
                            'hits_per_9': float(stats.get('hitsPer9Inn', 0)),
                            'home_runs_allowed': int(stats.get('homeRuns', 0)),
                            'runs_allowed_per_game': float(stats.get('runs', 0)) / max(float(stats.get('gamesPlayed', 1)), 1)
                        }
                    
                    elif stat_group['group'] == 'fielding':
                        stats = stat_group['stats']
                        fielding_stats = {
                            'fielding_pct': float(stats.get('fielding', 0)),
                            'errors': int(stats.get('errors', 0))
                        }
                
                # Get win-loss record
                standings = statsapi.standings_data(leagueId="103,104", season=season)
                wins, losses, win_pct = 0, 0, 0.0
                
                for division in standings.values():
                    for team_record in division['teams']:
                        if team_record['team_id'] == team['team_id']:
                            wins = team_record['w']
                            losses = team_record['l']
                            win_pct = team_record['w_pct']
                            break
                
                # Combine all stats
                team_stat_record = {
                    'team_id': team['team_id'],
                    'stat_date': current_date,
                    'season': season,
                    'wins': wins,
                    'losses': losses,
                    'win_pct': win_pct,
                    **hitting_stats,
                    **pitching_stats,
                    **fielding_stats
                }
                
                stats_data.append(team_stat_record)
                
            except Exception as e:
                logger.error(f"Error collecting stats for team {team['team_abbrev']}: {e}")
                continue
        
        self.db.insert_team_stats(stats_data)
        return stats_data
    
    def update_daily_data(self):
        """Daily update routine - collect yesterday's results and today's schedule."""
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        today = datetime.now().strftime('%Y-%m-%d')
        tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        
        # Collect yesterday's completed games
        self.collect_games_for_date_range(yesterday, yesterday)
        
        # Collect today and tomorrow's scheduled games
        self.collect_games_for_date_range(today, tomorrow)
        
        # Update team stats
        self.collect_team_stats()
        
        logger.info("Daily data update completed")

def initialize_system():
    """Initialize the complete system with historical data."""
    # Initialize database
    db = MLBDatabase()
    collector = MLBDataCollector(db)
    
    # Collect teams
    teams = collector.collect_teams()
    logger.info(f"Collected {len(teams)} teams")
    
    # Collect historical games (season start to now)
    season_start = "2025-03-20"  # Approximate season start
    today = datetime.now().strftime('%Y-%m-%d')
    
    games = collector.collect_games_for_date_range(season_start, today)
    logger.info(f"Collected {len(games)} games")
    
    # Collect current team stats
    stats = collector.collect_team_stats()
    logger.info(f"Collected stats for {len(stats)} teams")
    
    return db, collector

if __name__ == "__main__":
    # Initialize the system
    db, collector = initialize_system()
    
    # Test the data collection
    print("\n=== MLB Prediction System - Data Pipeline Test ===")
    
    # Show some teams
    conn = sqlite3.connect(db.db_path)
    teams_df = pd.read_sql_query("SELECT * FROM teams LIMIT 5", conn)
    print("\nSample Teams:")
    print(teams_df)
    
    # Show some games
    games_df = pd.read_sql_query("""
        SELECT g.game_date, ht.team_name as home_team, at.team_name as away_team, 
               g.home_score, g.away_score, g.game_status
        FROM games g
        JOIN teams ht ON g.home_team_id = ht.team_id  
        JOIN teams at ON g.away_team_id = at.team_id
        ORDER BY g.game_date DESC
        LIMIT 5
    """, conn)
    print("\nRecent Games:")
    print(games_df)
    
    # Show some team stats
    stats_df = pd.read_sql_query("""
        SELECT ts.team_id, t.team_name, ts.batting_avg, ts.era, ts.wins, ts.losses
        FROM team_stats ts
        JOIN teams t ON ts.team_id = t.team_id
        ORDER BY ts.win_pct DESC
        LIMIT 5
    """, conn)
    print("\nTop Teams by Win Percentage:")
    print(stats_df)
    
    conn.close()
    print("\n✅ Data pipeline setup complete!")