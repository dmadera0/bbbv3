"""
MLB Game Prediction System - Component 2: Machine Learning Models
This module handles feature engineering, model training, and predictions.
"""

import sqlite3
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error
import joblib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class MLBFeatureEngine:
    """Handles feature engineering for MLB game predictions."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def get_training_data(self, start_date: str = "2025-03-20") -> pd.DataFrame:
        """Get completed games with team stats for training."""
        conn = sqlite3.connect(self.db_path)
        
        # Get completed games with team stats
        query = '''
        SELECT 
            g.game_id,
            g.game_date,
            g.home_team_id,
            g.away_team_id,
            g.home_score,
            g.away_score,
            
            -- Home team stats
            ht_stats.batting_avg as home_batting_avg,
            ht_stats.on_base_pct as home_obp,
            ht_stats.slugging_pct as home_slg,
            ht_stats.ops as home_ops,
            ht_stats.runs_per_game as home_rpg,
            ht_stats.home_runs as home_hr,
            ht_stats.stolen_bases as home_sb,
            ht_stats.walks as home_bb,
            ht_stats.strikeouts as home_so,
            ht_stats.era as home_era,
            ht_stats.whip as home_whip,
            ht_stats.strikeouts_per_9 as home_k9,
            ht_stats.walks_per_9 as home_bb9,
            ht_stats.runs_allowed_per_game as home_rapg,
            ht_stats.fielding_pct as home_fpct,
            ht_stats.errors as home_errors,
            ht_stats.win_pct as home_win_pct,
            
            -- Away team stats
            at_stats.batting_avg as away_batting_avg,
            at_stats.on_base_pct as away_obp,
            at_stats.slugging_pct as away_slg,
            at_stats.ops as away_ops,
            at_stats.runs_per_game as away_rpg,
            at_stats.home_runs as away_hr,
            at_stats.stolen_bases as away_sb,
            at_stats.walks as away_bb,
            at_stats.strikeouts as away_so,
            at_stats.era as away_era,
            at_stats.whip as away_whip,
            at_stats.strikeouts_per_9 as away_k9,
            at_stats.walks_per_9 as away_bb9,
            at_stats.runs_allowed_per_game as away_rapg,
            at_stats.fielding_pct as away_fpct,
            at_stats.errors as away_errors,
            at_stats.win_pct as away_win_pct,
            
            -- Team names for reference
            ht.team_name as home_team_name,
            at.team_name as away_team_name
            
        FROM games g
        JOIN teams ht ON g.home_team_id = ht.team_id
        JOIN teams at ON g.away_team_id = at.team_id
        LEFT JOIN team_stats ht_stats ON g.home_team_id = ht_stats.team_id 
            AND ht_stats.stat_date <= g.game_date
        LEFT JOIN team_stats at_stats ON g.away_team_id = at_stats.team_id 
            AND at_stats.stat_date <= g.game_date
        WHERE g.game_date >= ?
        AND g.home_score IS NOT NULL 
        AND g.away_score IS NOT NULL
        AND ht_stats.team_id IS NOT NULL 
        AND at_stats.team_id IS NOT NULL
        ORDER BY g.game_date
        '''
        
        df = pd.read_sql_query(query, conn, params=(start_date,))
        conn.close()
        
        return df
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features from raw team stats."""
        features_df = df.copy()
        
        # Target variables
        features_df['home_win'] = (df['home_score'] > df['away_score']).astype(int)
        features_df['margin'] = df['home_score'] - df['away_score']
        features_df['total_runs'] = df['home_score'] + df['away_score']
        
        # Home field advantage (implicit - always 1 for home team)
        features_df['home_field'] = 1
        
        # Offensive differentials (home advantage)
        features_df['ops_diff'] = df['home_ops'] - df['away_ops']
        features_df['rpg_diff'] = df['home_rpg'] - df['away_rpg']
        features_df['obp_diff'] = df['home_obp'] - df['away_obp']
        features_df['slg_diff'] = df['home_slg'] - df['away_slg']
        features_df['hr_rate_diff'] = (df['home_hr'] / df['home_rpg'].clip(lower=1)) - (df['away_hr'] / df['away_rpg'].clip(lower=1))
        
        # Pitching differentials (lower is better, so away - home for advantage)
        features_df['era_diff'] = df['away_era'] - df['home_era']  # Positive means home has better ERA
        features_df['whip_diff'] = df['away_whip'] - df['home_whip']
        features_df['rapg_diff'] = df['away_rapg'] - df['home_rapg']
        features_df['k9_diff'] = df['home_k9'] - df['away_k9']  # Higher K/9 is better
        features_df['bb9_diff'] = df['away_bb9'] - df['home_bb9']  # Lower BB/9 is better
        
        # Overall team strength differentials
        features_df['win_pct_diff'] = df['home_win_pct'] - df['away_win_pct']
        features_df['run_diff_combined'] = (df['home_rpg'] - df['home_rapg']) - (df['away_rpg'] - df['away_rapg'])
        
        # Composite offensive power
        features_df['home_offensive_power'] = df['home_ops'] * df['home_rpg']
        features_df['away_offensive_power'] = df['away_ops'] * df['away_rpg']
        features_df['offensive_power_diff'] = features_df['home_offensive_power'] - features_df['away_offensive_power']
        
        # Composite pitching quality (lower is better)
        features_df['home_pitching_quality'] = df['home_era'] * df['home_whip']
        features_df['away_pitching_quality'] = df['away_era'] * df['away_whip']
        features_df['pitching_quality_diff'] = features_df['away_pitching_quality'] - features_df['home_pitching_quality']
        
        # Expected runs (simple Pythagorean-like)
        features_df['home_expected_runs'] = df['home_rpg'] * (1 + features_df['ops_diff'] * 0.1)
        features_df['away_expected_runs'] = df['away_rpg'] * (1 - features_df['ops_diff'] * 0.1)
        
        # Fielding differentials
        features_df['fielding_diff'] = df['home_fpct'] - df['away_fpct']
        features_df['errors_diff'] = df['away_errors'] - df['home_errors']  # Fewer errors is better
        
        # Fill any NaN values with 0
        numeric_columns = features_df.select_dtypes(include=[np.number]).columns
        features_df[numeric_columns] = features_df[numeric_columns].fillna(0)
        
        return features_df
    
    def get_feature_columns(self) -> List[str]:
        """Return list of feature columns for model training."""
        return [
            # Raw stats
            'home_ops', 'away_ops', 'home_rpg', 'away_rpg',
            'home_era', 'away_era', 'home_whip', 'away_whip',
            'home_win_pct', 'away_win_pct', 'home_fpct', 'away_fpct',
            
            # Differentials
            'ops_diff', 'rpg_diff', 'era_diff', 'whip_diff', 'win_pct_diff',
            'run_diff_combined', 'offensive_power_diff', 'pitching_quality_diff',
            'k9_diff', 'bb9_diff', 'fielding_diff', 'errors_diff',
            
            # Composite features
            'home_expected_runs', 'away_expected_runs',
            
            # Home field advantage
            'home_field'
        ]

class MLBPredictor:
    """Main prediction system for MLB games."""
    
    def __init__(self, db_path: str, model_version: str = "v1.0"):
        self.db_path = db_path
        self.model_version = model_version
        self.feature_engine = MLBFeatureEngine(db_path)
        
        # Models
        self.winner_model = None
        self.margin_model = None
        self.total_model = None
        self.scaler = StandardScaler()
        
        # Model performance tracking
        self.performance_metrics = {}
    
    def train_models(self, test_size: float = 0.2, random_state: int = 42):
        """Train all three prediction models."""
        logger.info("Starting model training...")
        
        # Get training data
        df = self.feature_engine.get_training_data()
        logger.info(f"Loaded {len(df)} completed games for training")
        
        if len(df) < 50:
            logger.warning("Limited training data available. Models may not be optimal.")
        
        # Create features
        features_df = self.feature_engine.create_features(df)
        feature_cols = self.feature_engine.get_feature_columns()
        
        X = features_df[feature_cols]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Targets
        y_winner = features_df['home_win']
        y_margin = features_df['margin']
        y_total = features_df['total_runs']
        
        # Train-test split
        X_train, X_test, y_win_train, y_win_test = train_test_split(
            X_scaled, y_winner, test_size=test_size, random_state=random_state
        )
        _, _, y_margin_train, y_margin_test = train_test_split(
            X_scaled, y_margin, test_size=test_size, random_state=random_state
        )
        _, _, y_total_train, y_total_test = train_test_split(
            X_scaled, y_total, test_size=test_size, random_state=random_state
        )
        
        # Train Winner Prediction Model (Classification)
        logger.info("Training winner prediction model...")
        self.winner_model = LogisticRegression(random_state=random_state, max_iter=1000)
        self.winner_model.fit(X_train, y_win_train)
        
        # Evaluate winner model
        win_pred = self.winner_model.predict(X_test)
        win_accuracy = accuracy_score(y_win_test, win_pred)
        win_prob = self.winner_model.predict_proba(X_test)[:, 1]
        
        self.performance_metrics['winner'] = {
            'accuracy': win_accuracy,
            'test_samples': len(y_win_test)
        }
        
        # Train Margin Prediction Model (Regression)
        logger.info("Training margin prediction model...")
        self.margin_model = RandomForestRegressor(n_estimators=100, random_state=random_state, max_depth=10)
        self.margin_model.fit(X_train, y_margin_train)
        
        # Evaluate margin model
        margin_pred = self.margin_model.predict(X_test)
        margin_mae = mean_absolute_error(y_margin_test, margin_pred)
        margin_rmse = np.sqrt(mean_squared_error(y_margin_test, margin_pred))
        
        self.performance_metrics['margin'] = {
            'mae': margin_mae,
            'rmse': margin_rmse,
            'test_samples': len(y_margin_test)
        }
        
        # Train Total Runs Prediction Model (Regression)
        logger.info("Training total runs prediction model...")
        self.total_model = RandomForestRegressor(n_estimators=100, random_state=random_state, max_depth=10)
        self.total_model.fit(X_train, y_total_train)
        
        # Evaluate total model
        total_pred = self.total_model.predict(X_test)
        total_mae = mean_absolute_error(y_total_test, total_pred)
        total_rmse = np.sqrt(mean_squared_error(y_total_test, total_pred))
        
        self.performance_metrics['total'] = {
            'mae': total_mae,
            'rmse': total_rmse,
            'test_samples': len(y_total_test)
        }
        
        logger.info("Model training completed!")
        self.print_performance_summary()
    
    def print_performance_summary(self):
        """Print model performance metrics."""
        print("\n" + "="*50)
        print("MODEL PERFORMANCE SUMMARY")
        print("="*50)
        
        if 'winner' in self.performance_metrics:
            print(f"Winner Prediction:")
            print(f"  Accuracy: {self.performance_metrics['winner']['accuracy']:.3f}")
            print(f"  Test samples: {self.performance_metrics['winner']['test_samples']}")
        
        if 'margin' in self.performance_metrics:
            print(f"\nMargin Prediction:")
            print(f"  MAE: {self.performance_metrics['margin']['mae']:.2f} runs")
            print(f"  RMSE: {self.performance_metrics['margin']['rmse']:.2f} runs")
            print(f"  Test samples: {self.performance_metrics['margin']['test_samples']}")
        
        if 'total' in self.performance_metrics:
            print(f"\nTotal Runs Prediction:")
            print(f"  MAE: {self.performance_metrics['total']['mae']:.2f} runs")
            print(f"  RMSE: {self.performance_metrics['total']['rmse']:.2f} runs")
            print(f"  Test samples: {self.performance_metrics['total']['test_samples']}")
        
        print("="*50)
    
    def predict_game(self, home_team_id: int, away_team_id: int, game_date: str) -> Dict:
        """Predict outcome for a single game."""
        if not all([self.winner_model, self.margin_model, self.total_model]):
            raise ValueError("Models not trained yet. Call train_models() first.")
        
        # Get team stats for the game date
        conn = sqlite3.connect(self.db_path)
        
        query = '''
        SELECT 
            -- Home team stats
            ht_stats.batting_avg as home_batting_avg,
            ht_stats.on_base_pct as home_obp,
            ht_stats.slugging_pct as home_slg,
            ht_stats.ops as home_ops,
            ht_stats.runs_per_game as home_rpg,
            ht_stats.home_runs as home_hr,
            ht_stats.era as home_era,
            ht_stats.whip as home_whip,
            ht_stats.strikeouts_per_9 as home_k9,
            ht_stats.walks_per_9 as home_bb9,
            ht_stats.runs_allowed_per_game as home_rapg,
            ht_stats.fielding_pct as home_fpct,
            ht_stats.errors as home_errors,
            ht_stats.win_pct as home_win_pct,
            
            -- Away team stats
            at_stats.batting_avg as away_batting_avg,
            at_stats.on_base_pct as away_obp,
            at_stats.slugging_pct as away_slg,
            at_stats.ops as away_ops,
            at_stats.runs_per_game as away_rpg,
            at_stats.home_runs as away_hr,
            at_stats.era as away_era,
            at_stats.whip as away_whip,
            at_stats.strikeouts_per_9 as away_k9,
            at_stats.walks_per_9 as away_bb9,
            at_stats.runs_allowed_per_game as away_rapg,
            at_stats.fielding_pct as away_fpct,
            at_stats.errors as away_errors,
            at_stats.win_pct as away_win_pct,
            
            -- Team names
            ht.team_name as home_team_name,
            at.team_name as away_team_name
            
        FROM teams ht, teams at
        LEFT JOIN team_stats ht_stats ON ht.team_id = ht_stats.team_id 
            AND ht_stats.stat_date <= ?
        LEFT JOIN team_stats at_stats ON at.team_id = at_stats.team_id 
            AND at_stats.stat_date <= ?
        WHERE ht.team_id = ? AND at.team_id = ?
        ORDER BY ht_stats.stat_date DESC, at_stats.stat_date DESC
        LIMIT 1
        '''
        
        result = pd.read_sql_query(query, conn, params=(game_date, game_date, home_team_id, away_team_id))
        conn.close()
        
        if result.empty:
            raise ValueError(f"No stats found for teams {home_team_id} vs {away_team_id} on {game_date}")
        
        # Create a temporary dataframe for feature engineering
        temp_df = result.copy()
        temp_df['home_score'] = 0  # Dummy values for feature creation
        temp_df['away_score'] = 0
        
        # Create features
        features_df = self.feature_engine.create_features(temp_df)
        feature_cols = self.feature_engine.get_feature_columns()
        
        X = features_df[feature_cols].iloc[0:1]  # Take first (and only) row
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        win_prob = self.winner_model.predict_proba(X_scaled)[0, 1]  # Probability home wins
        predicted_winner_id = home_team_id if win_prob > 0.5 else away_team_id
        
        predicted_margin = self.margin_model.predict(X_scaled)[0]
        predicted_total = self.total_model.predict(X_scaled)[0]
        
        # Calculate implied scores
        home_implied_score = (predicted_total + predicted_margin) / 2
        away_implied_score = (predicted_total - predicted_margin) / 2
        
        return {
            'home_team_id': home_team_id,
            'away_team_id': away_team_id,
            'home_team_name': result.iloc[0]['home_team_name'],
            'away_team_name': result.iloc[0]['away_team_name'],
            'predicted_winner_id': predicted_winner_id,
            'predicted_winner_name': result.iloc[0]['home_team_name'] if predicted_winner_id == home_team_id else result.iloc[0]['away_team_name'],
            'win_probability': win_prob if predicted_winner_id == home_team_id else (1 - win_prob),
            'predicted_margin': abs(predicted_margin),
            'predicted_total_runs': predicted_total,
            'home_implied_score': max(0, round(home_implied_score, 1)),
            'away_implied_score': max(0, round(away_implied_score, 1))
        }
    
    def predict_games_batch(self, game_date: str) -> List[Dict]:
        """Predict all games for a specific date."""
        conn = sqlite3.connect(self.db_path)
        
        # Get scheduled games for the date
        games_query = '''
        SELECT game_id, home_team_id, away_team_id, 
               ht.team_name as home_team_name, at.team_name as away_team_name
        FROM games g
        JOIN teams ht ON g.home_team_id = ht.team_id
        JOIN teams at ON g.away_team_id = at.team_id
        WHERE g.game_date = ?
        AND g.game_status IN ('Scheduled', 'Pre-Game')
        '''
        
        games_df = pd.read_sql_query(games_query, conn, params=(game_date,))
        conn.close()
        
        predictions = []
        for _, game in games_df.iterrows():
            try:
                prediction = self.predict_game(
                    game['home_team_id'], 
                    game['away_team_id'], 
                    game_date
                )
                prediction['game_id'] = game['game_id']
                predictions.append(prediction)
            except Exception as e:
                logger.error(f"Error predicting game {game['game_id']}: {e}")
                continue
        
        return predictions
    
    def save_predictions_to_db(self, predictions: List[Dict]):
        """Save predictions to the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for pred in predictions:
            cursor.execute('''
                INSERT OR REPLACE INTO predictions 
                (game_id, predicted_winner_id, win_probability, predicted_margin, 
                 predicted_total_runs, model_version)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                pred['game_id'],
                pred['predicted_winner_id'],
                pred['win_probability'],
                pred['predicted_margin'],
                pred['predicted_total_runs'],
                self.model_version
            ))
        
        conn.commit()
        conn.close()
        logger.info(f"Saved {len(predictions)} predictions to database")
    
    def save_models(self, filepath_prefix: str = "mlb_models"):
        """Save trained models to disk."""
        if self.winner_model:
            joblib.dump(self.winner_model, f"{filepath_prefix}_winner.pkl")
        if self.margin_model:
            joblib.dump(self.margin_model, f"{filepath_prefix}_margin.pkl")
        if self.total_model:
            joblib.dump(self.total_model, f"{filepath_prefix}_total.pkl")
        
        joblib.dump(self.scaler, f"{filepath_prefix}_scaler.pkl")
        
        logger.info(f"Models saved with prefix: {filepath_prefix}")
    
    def load_models(self, filepath_prefix: str = "mlb_models"):
        """Load trained models from disk."""
        try:
            self.winner_model = joblib.load(f"{filepath_prefix}_winner.pkl")
            self.margin_model = joblib.load(f"{filepath_prefix}_margin.pkl")
            self.total_model = joblib.load(f"{filepath_prefix}_total.pkl")
            self.scaler = joblib.load(f"{filepath_prefix}_scaler.pkl")
            logger.info(f"Models loaded from prefix: {filepath_prefix}")
        except FileNotFoundError as e:
            logger.error(f"Model files not found: {e}")
            raise

def main():
    """Main training and prediction workflow."""
    # Initialize predictor
    predictor = MLBPredictor("mlb_predictions.db")
    
    # Train models
    predictor.train_models()
    
    # Save models
    predictor.save_models()
    
    # Make predictions for tomorrow's games
    tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    predictions = predictor.predict_games_batch(tomorrow)
    
    if predictions:
        print(f"\n🔮 PREDICTIONS FOR {tomorrow}")
        print("="*60)
        
        for pred in predictions:
            confidence = pred['win_probability'] * 100
            print(f"{pred['away_team_name']} @ {pred['home_team_name']}")
            print(f"  Winner: {pred['predicted_winner_name']} ({confidence:.1f}% confidence)")
            print(f"  Predicted Score: {pred['home_team_name']} {pred['home_implied_score']}, {pred['away_team_name']} {pred['away_implied_score']}")
            print(f"  Total Runs: {pred['predicted_total_runs']:.1f}")
            print()
        
        # Save to database
        predictor.save_predictions_to_db(predictions)
    else:
        print(f"No games scheduled for {tomorrow}")

if __name__ == "__main__":
    main()