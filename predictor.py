import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import numpy as np
import json
import os
import time
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from dotenv import load_dotenv

"""
Predicts the outcome of an NBA game using an XGBoost Classifier optimized for higher accuracy (target: 70%+). 
Uses advanced team statistics and feature engineering to improve prediction quality.
"""

# Ensure summary_api import is correctly placed and attempted
try:
    from summary_api import generate_game_summary
    summary_api_available = True
    # ADDING DIAGNOSTIC PRINT HERE
    import summary_api # Import the module itself to check its path
    print(f"DEBUG: predictor.py - Successfully imported summary_api. Path: {summary_api.__file__}")
except ImportError as e:
    # MODIFIED DIAGNOSTIC PRINT HERE
    print(f"DEBUG: predictor.py - Failed to import summary_api. Error: {e}")
    summary_api_available = False
    summary_api = None # To avoid NameError if __file__ was attempted on a non-existent module object.

# Load environment variables from .env file (for OPENAI_API_KEY, DB_NAME, etc.)
load_dotenv()

# Initialize variables
DB_NAME = os.getenv("DB_NAME", "nba_data.db")
MODEL_FILENAME = "nba_predictor_model.joblib"
FEATURES_FILENAME = "feature_importances.joblib"
home_team_name = "Portland Trail Blazers"
away_team_name = "Oklahoma City Thunder"

def load_data():
    # ... (load_data function remains largely the same, ensuring teams_info_df is loaded) ...
    """Loads all necessary data from the SQLite database."""
    print(f"Loading data from {DB_NAME}...")
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME)
        
        games_df = pd.read_sql_query("SELECT * FROM games", conn)
        print(f"Loaded {len(games_df)} entries from 'games' table.")
        if games_df.empty:
            print("Warning: 'games' table is empty. Ensure data_collector.py has run successfully.")

        player_logs_df = pd.read_sql_query("SELECT * FROM player_game_logs", conn)
        print(f"Loaded {len(player_logs_df)} entries from 'player_game_logs' table.")
        if player_logs_df.empty:
            print("Warning: 'player_game_logs' table is empty.")

        team_logs_df = pd.read_sql_query("SELECT * FROM team_game_logs", conn)
        print(f"Loaded {len(team_logs_df)} entries from 'team_game_logs' table.")
        if team_logs_df.empty:
            print("Warning: 'team_game_logs' table is empty.")

        teams_info_df = pd.read_sql_query("SELECT * FROM teams", conn)
        print(f"Loaded {len(teams_info_df)} entries from 'teams' table.")
        if teams_info_df.empty:
            print("Warning: 'teams' table is empty (needed for team names and IDs).")

        return {
            "games": games_df,
            "player_logs": player_logs_df,
            "team_logs": team_logs_df,
            "teams_info": teams_info_df
        }

    except sqlite3.OperationalError as e:
        print(f"Database error: {e}. Make sure '{DB_NAME}' exists and data_collector.py has run successfully.")
        return None
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        return None
    finally:
        if conn:
            conn.close()

def calculate_team_rolling_stats(team_id, game_date, team_logs_df, window_size=10):
    """Calculate comprehensive team statistics including advanced metrics and trends."""
    team_logs_df['game_date'] = pd.to_datetime(team_logs_df['game_date'])
    game_date = pd.to_datetime(game_date)

    # Get past games sorted by date (most recent first)
    past_games = team_logs_df[
        (team_logs_df['team_id'] == team_id) &
        (team_logs_df['game_date'] < game_date)
    ].sort_values(by='game_date', ascending=False)
    
    # Get different window sizes for different metrics
    past_games_10 = past_games.head(min(10, len(past_games)))  # Last 10 games
    past_games_5 = past_games.head(min(5, len(past_games)))    # Last 5 games (more recent form)
    past_games_3 = past_games.head(min(3, len(past_games)))    # Last 3 games (very recent form)
    
    # Get home/away splits
    past_home_games = past_games[past_games['matchup'].str.contains('vs.', na=False)].head(min(5, len(past_games)))
    past_away_games = past_games[past_games['matchup'].str.contains('@', na=False)].head(min(5, len(past_games)))

    # Define all stats columns we want to calculate
    basic_stats = [
        'pts', 'fgm', 'fga', 'fg_pct', 'fg3m', 'fg3a', 'fg3_pct', 
        'ftm', 'fta', 'ft_pct', 'oreb', 'dreb', 'reb', 'ast', 
        'stl', 'blk', 'tov', 'pf', 'plus_minus'
    ]
    
    # Initialize result dict
    team_stats = {}
    
    # Ensure minimum number of games for meaningful stats
    if past_games.empty or len(past_games) < 3:
        # Create empty series with all expected column names 
        columns = [
            # Basic averages
            'avg_pts', 'avg_fg_pct', 'avg_fg3_pct', 'avg_ft_pct', 'avg_reb', 'avg_ast', 
            'avg_stl', 'avg_blk', 'avg_tov', 'avg_pf', 'avg_win_pct',
            # Recent form and trends
            'win_streak', 'recent_form_pts', 'momentum_fg_pct', 'momentum_fg3_pct', 
            'home_win_pct', 'away_win_pct', 'pts_last3', 'efficiency_rating',
            'defensive_rating', 'offensive_rating'
        ]
        return pd.Series({col: np.nan for col in columns})

    # BASIC AVERAGES - 10-game rolling averages for standard stats
    # Create explicit copy of the dataframe slice to avoid SettingWithCopyWarning
    if len(past_games_10) > 0:
        past_games_10_copy = past_games_10.copy()
        for col in basic_stats:
            if col in past_games_10_copy.columns:
                past_games_10_copy.loc[:, col] = pd.to_numeric(past_games_10_copy[col], errors='coerce')
                team_stats[f'avg_{col}'] = past_games_10_copy[col].mean()
    
    # WIN PERCENTAGE and STREAKS
    if 'wl' in past_games_10.columns:
        # Basic win percentages
        team_stats['avg_win_pct'] = (past_games_10['wl'] == 'W').mean()
        team_stats['home_win_pct'] = (past_home_games['wl'] == 'W').mean() if len(past_home_games) > 0 else np.nan
        team_stats['away_win_pct'] = (past_away_games['wl'] == 'W').mean() if len(past_away_games) > 0 else np.nan
        
        # Add win streak (positive for wins, negative for losses)
        recent_games = past_games.sort_values('game_date', ascending=False).head(10).copy()
        if not recent_games.empty:
            # Current streak (consecutive W or L)
            win_loss_streak = 0
            current_streak = None
            
            for _, game in recent_games.iterrows():
                result = game.get('wl')
                if result is None:
                    continue
                    
                if current_streak is None:
                    current_streak = result
                    win_loss_streak = 1 if result == 'W' else -1
                elif result == current_streak:
                    if result == 'W':
                        win_loss_streak += 1
                    else:
                        win_loss_streak -= 1
                else:
                    break
            
            team_stats['win_streak'] = win_loss_streak
            
            # Weighted recent form (more weight to recent games)
            if len(recent_games) >= 5:
                recent_5 = recent_games.head(5).copy()
                weights = np.array([0.35, 0.25, 0.20, 0.15, 0.05])[:len(recent_5)]
                weights = weights / weights.sum()  # normalize weights
                
                # Convert W/L to 1/0
                recent_5_wins = (recent_5['wl'] == 'W').astype(int).values
                team_stats['weighted_recent_form'] = np.sum(recent_5_wins * weights)
                
                # Momentum: Compare last 3 games to previous 7
                if len(recent_games) >= 10:
                    last_3_win_pct = (recent_games.iloc[:3]['wl'] == 'W').mean()
                    prev_7_win_pct = (recent_games.iloc[3:10]['wl'] == 'W').mean()
                    team_stats['momentum_win_pct'] = last_3_win_pct - prev_7_win_pct
            
            # Conference record
            if 'team_id' in recent_games.columns and 'matchup_conference' in recent_games.columns:
                conf_games = recent_games[recent_games['matchup_conference'] == True]
                if len(conf_games) > 0:
                    team_stats['conference_win_pct'] = (conf_games['wl'] == 'W').mean()

    # FORM AND MOMENTUM
    # Recent form (weighted average of last 5 games for scoring)
    if 'pts' in past_games_5.columns and len(past_games_5) > 0:
        past_games_5_copy = past_games_5.copy()
        weights = np.linspace(1.5, 0.5, len(past_games_5_copy))  # Higher weight to more recent games
        weights = weights / sum(weights)  # Normalize weights to sum to 1
        past_games_5_copy.loc[:, 'pts'] = pd.to_numeric(past_games_5_copy['pts'], errors='coerce')
        team_stats['recent_form_pts'] = np.average(past_games_5_copy['pts'], weights=weights)
    
    # Last 3 games average points (very recent form)
    if 'pts' in past_games_3.columns and len(past_games_3) > 0:
        team_stats['pts_last3'] = past_games_3['pts'].mean()
    
    # Shooting efficiency trends (comparing last 3 games vs. season average)
    for col in ['fg_pct', 'fg3_pct']:
        if len(past_games_3) > 0 and len(past_games) > 3:
            past_games_3_copy = past_games_3.copy()
            past_games_copy = past_games.copy()
            
            # Field goal percentage momentum
            if col in past_games_3_copy.columns and col in past_games_copy.columns:
                past_games_3_copy.loc[:, col] = pd.to_numeric(past_games_3_copy[col], errors='coerce')
                past_games_copy.loc[:, col] = pd.to_numeric(past_games_copy[col], errors='coerce')
                recent = past_games_3_copy[col].mean()
                season = past_games_copy[col].mean()
                team_stats[f'momentum_{col}'] = recent - season
    
    # Add seasonality features
    if 'game_date' in past_games.columns and len(past_games) > 0:
        latest_game = past_games['game_date'].max()
        if pd.notna(latest_game):
            latest_game = pd.to_datetime(latest_game)
            
            # Determine the season phase
            month = latest_game.month
            if 10 <= month <= 11:  # Oct-Nov
                team_stats['season_phase'] = 0  # Early season
            elif 12 <= month <= 2:  # Dec-Feb
                team_stats['season_phase'] = 0.5  # Mid season
            else:  # Mar-Jun
                team_stats['season_phase'] = 1  # Late season/playoff push
                
            # Days since last game (fatigue/rest indicator)
            if len(past_games) >= 2:
                past_games_sorted = past_games.sort_values('game_date', ascending=False)
                last_game_date = pd.to_datetime(past_games_sorted.iloc[0]['game_date'])
                prev_game_date = pd.to_datetime(past_games_sorted.iloc[1]['game_date'])
                team_stats['days_since_last_game'] = (last_game_date - prev_game_date).days

    # ADVANCED METRICS
    # Offensive efficiency (points per possession estimate)
    if all(col in team_stats for col in ['avg_pts', 'avg_tov']):
        possessions = team_stats['avg_fga'] - team_stats['avg_oreb'] + team_stats['avg_tov']
        if possessions > 0:
            team_stats['offensive_rating'] = team_stats['avg_pts'] / possessions * 100
    
    # Defensive rating (estimate)
    if all(col in past_games_10.columns for col in ['opp_pts', 'opp_fg_pct']):
        past_games_10['opp_pts'] = pd.to_numeric(past_games_10['opp_pts'], errors='coerce')
        past_games_10['opp_fg_pct'] = pd.to_numeric(past_games_10['opp_fg_pct'], errors='coerce')
        team_stats['defensive_rating'] = past_games_10['opp_pts'].mean() * past_games_10['opp_fg_pct'].mean()
    
    # Overall efficiency rating (estimate)
    if all(col in team_stats for col in ['avg_pts', 'avg_tov', 'avg_ast', 'avg_stl']):
        team_stats['efficiency_rating'] = (
            team_stats['avg_pts'] + team_stats['avg_ast'] + team_stats['avg_stl']) / \
            max(1, team_stats['avg_tov'])
    
    return pd.Series(team_stats)

def create_head_to_head_features(home_team_id, away_team_id, game_date, team_logs_df, lookback_years=3):
    """Create advanced features based on historical matchups between two teams.
    
    Args:
        home_team_id: ID of the home team
        away_team_id: ID of the away team
        game_date: Date of the game being predicted
        team_logs_df: DataFrame containing team game logs
        lookback_years: Number of years to look back for matchup history
        
    Returns:
        Dictionary of head-to-head features with advanced metrics
    """
    # Convert dates to datetime
    game_date = pd.to_datetime(game_date)
    team_logs_df = team_logs_df.copy()  # Avoid SettingWithCopyWarning
    team_logs_df['game_date'] = pd.to_datetime(team_logs_df['game_date'])
    
    # Set lookback window (increased to 3 years for more data)
    lookback_date = game_date - pd.Timedelta(days=365*lookback_years)
    
    # Find all games between these teams within the lookback period
    home_games = team_logs_df[
        (team_logs_df['team_id'] == home_team_id) &
        (team_logs_df['matchup'].str.contains(str(away_team_id), na=False)) &
        (team_logs_df['game_date'] < game_date) &
        (team_logs_df['game_date'] >= lookback_date)
    ].sort_values(by='game_date', ascending=False)
    
    away_games = team_logs_df[
        (team_logs_df['team_id'] == away_team_id) &
        (team_logs_df['matchup'].str.contains(str(home_team_id), na=False)) &
        (team_logs_df['game_date'] < game_date) &
        (team_logs_df['game_date'] >= lookback_date)
    ].sort_values(by='game_date', ascending=False)
    
    # If we don't have enough data, return empty dict
    if len(home_games) == 0 or len(away_games) == 0:
        return {}
    
    # Merge home and away games by game_id to create matchup dataset
    home_games = home_games.rename(columns={
        'pts': 'home_pts', 
        'wl': 'home_wl',
        'fg_pct': 'home_fg_pct',
        'fg3_pct': 'home_fg3_pct',
        'ft_pct': 'home_ft_pct',
        'reb': 'home_reb',
        'ast': 'home_ast',
        'stl': 'home_stl',
        'blk': 'home_blk',
        'tov': 'home_tov',
        'plus_minus': 'home_plus_minus'
    })
    
    away_games = away_games.rename(columns={
        'pts': 'away_pts', 
        'wl': 'away_wl',
        'fg_pct': 'away_fg_pct',
        'fg3_pct': 'away_fg3_pct',
        'ft_pct': 'away_ft_pct',
        'reb': 'away_reb',
        'ast': 'away_ast',
        'stl': 'away_stl',
        'blk': 'away_blk',
        'tov': 'away_tov',
        'plus_minus': 'away_plus_minus'
    })
    
    # Try to find common game_ids
    common_ids = set(home_games['game_id']).intersection(set(away_games['game_id']))
    
    # If we can't find exact game ID matches, return empty dict
    if not common_ids:
        return {}
        
    # Filter games to just those with matching IDs
    home_games_filtered = home_games[home_games['game_id'].isin(common_ids)]
    away_games_filtered = away_games[away_games['game_id'].isin(common_ids)]
    
    # Calculate features
    features = {}
    
    # Basic win percentage features
    if 'home_wl' in home_games_filtered.columns:
        # Home team overall win percentage against opponent
        features['h2h_home_win_pct'] = (home_games_filtered['home_wl'] == 'W').mean()
        
        # Compute winning streaks against this opponent
        win_streak = 0
        wl_series = home_games_filtered['home_wl'].values
        for wl in wl_series:
            if wl == 'W':
                win_streak += 1
            else:
                break
        features['h2h_home_win_streak'] = win_streak
        
        # Lose streak
        lose_streak = 0
        for wl in wl_series:
            if wl == 'L':
                lose_streak += 1
            else:
                break
        features['h2h_home_lose_streak'] = lose_streak
    
    # Recent matchup performance with weighted recency bias
    last_n = min(5, len(home_games_filtered))
    if last_n > 0 and 'home_wl' in home_games_filtered.columns:
        # Simple last 5 win percentage
        features['h2h_last5_win_pct'] = (home_games_filtered.head(last_n)['home_wl'] == 'W').mean()
        
        # Weighted recency win percentage (more weight to recent games)
        if last_n >= 3:
            weights = np.array([0.5, 0.3, 0.2, 0.1, 0.05][:last_n])
            weights = weights / weights.sum()  # normalize
            win_indicators = (home_games_filtered.head(last_n)['home_wl'] == 'W').astype(int).values
            features['h2h_weighted_win_pct'] = np.sum(win_indicators * weights)
            
            # Last game outcome (1 = win, 0 = loss)
            features['h2h_last_game'] = 1 if home_games_filtered.iloc[0]['home_wl'] == 'W' else 0
    
    # Merge game stats for comprehensive analysis
    merged_games = pd.DataFrame()
    stat_columns = [
        ('pts', 'pts'),
        ('fg_pct', 'fg_pct'),
        ('fg3_pct', 'fg3_pct'),
        ('ft_pct', 'ft_pct'),
        ('reb', 'reb'),
        ('ast', 'ast'),
        ('stl', 'stl'),
        ('blk', 'blk'),
        ('tov', 'tov'),
        ('plus_minus', 'plus_minus')
    ]
    
    # Dynamically merge stats based on what's available
    home_cols = ['game_id']
    away_cols = ['game_id']
    for home_col, away_col in stat_columns:
        home_col_name = f'home_{home_col}'
        away_col_name = f'away_{away_col}'
        
        if home_col_name in home_games_filtered.columns:
            home_cols.append(home_col_name)
        if away_col_name in away_games_filtered.columns:
            away_cols.append(away_col_name)
    
    # Merge the available columns
    if len(home_cols) > 1 and len(away_cols) > 1:
        merged_games = pd.merge(
            home_games_filtered[home_cols], 
            away_games_filtered[away_cols],
            on='game_id'
        )
    
    if not merged_games.empty:
        # Calculate matchup statistics with differentials
        for home_col, away_col in stat_columns:
            home_col_name = f'home_{home_col}'
            away_col_name = f'away_{away_col}'
            diff_name = f'diff_{home_col}'
            
            if home_col_name in merged_games.columns and away_col_name in merged_games.columns:
                # Calculate differential (home - away)
                merged_games[diff_name] = merged_games[home_col_name] - merged_games[away_col_name]
                
                # Overall average differential
                features[f'h2h_avg_{diff_name}'] = merged_games[diff_name].mean()
                
                # Recent trend (last 2 games vs previous 3+)
                if len(merged_games) >= 3:
                    recent = merged_games.iloc[:2][diff_name].mean()
                    previous = merged_games.iloc[2:][diff_name].mean()
                    features[f'h2h_trend_{diff_name}'] = recent - previous
                
                # Consistency/variance in this stat against this opponent
                features[f'h2h_std_{diff_name}'] = merged_games[diff_name].std() if len(merged_games) >= 3 else 0
                
                # Last game value - most recent insight
                features[f'h2h_last_{diff_name}'] = merged_games.iloc[0][diff_name] if not merged_games.empty else 0
        
        # Create composite matchup score based on multiple factors
        try:
            if 'diff_pts' in merged_games.columns and 'diff_plus_minus' in merged_games.columns:
                # Weighted composite of point differential and plus-minus
                features['h2h_composite_score'] = (merged_games['diff_pts'].mean() * 0.7 + 
                                                   merged_games['diff_plus_minus'].mean() * 0.3)
        except Exception:
            # Skip if columns missing
            pass
    
    # Court advantage - percentage difference in stats when playing home vs away
    if len(merged_games) >= 4:
        try:
            home_court_effect = features.get('h2h_home_win_pct', 0.5) - 0.5
            features['h2h_home_court_advantage'] = home_court_effect 
        except:
            pass
    
    return features


#TODO: Improve algorithm to achieve 70% accuracy
def feature_engineering(raw_data_dict):
    if raw_data_dict is None:
        print("Raw data is missing, cannot perform feature engineering.")
        return pd.DataFrame(), pd.Series(), pd.DataFrame()

    games_df = raw_data_dict.get("games")
    team_logs_df = raw_data_dict.get("team_logs")
    teams_info_df = raw_data_dict.get("teams_info")
    player_logs_df = raw_data_dict.get("player_logs")

    if not all(df is not None and not df.empty for df in [games_df, team_logs_df, teams_info_df]):
        print("One or more essential DataFrames (games, team_logs, teams_info) are missing or empty.")
        return pd.DataFrame(), pd.Series(), pd.DataFrame()

    print("Starting enhanced feature engineering...")

    # Convert dates and sort
    games_df['game_date'] = pd.to_datetime(games_df['game_date'])
    team_logs_df['game_date'] = pd.to_datetime(team_logs_df['game_date'], errors='coerce')
    games_df = games_df.sort_values(by=['game_date', 'game_id']).reset_index(drop=True)

    # Target variable: HOME_TEAM_WINS
    # Ensure scores are numeric, drop games with missing scores before calculating winner
    games_df['home_team_score'] = pd.to_numeric(games_df['home_team_score'], errors='coerce')
    games_df['away_team_score'] = pd.to_numeric(games_df['away_team_score'], errors='coerce')
    games_df.dropna(subset=['home_team_score', 'away_team_score'], inplace=True)
    games_df['HOME_TEAM_WINS'] = (games_df['home_team_score'] > games_df['away_team_score']).astype(int)

    team_id_to_name = teams_info_df.set_index('id')['full_name'].to_dict()
    window_size = 10
    all_features_list = []

    # Pre-calculate last game date for each team for rest day calculation
    team_last_game_date = {}

    for index, game_row in games_df.iterrows():
        game_id = game_row['game_id']
        home_team_id = game_row['home_team_id']
        away_team_id = game_row['away_team_id']
        current_game_date = game_row['game_date']

        # Calculate Rest Days and B2B
        home_last_played = team_last_game_date.get(home_team_id)
        away_last_played = team_last_game_date.get(away_team_id)

        home_rest_days = (current_game_date - home_last_played).days if home_last_played else 14 # Default to high rest if no prior game found (e.g. start of season)
        away_rest_days = (current_game_date - away_last_played).days if away_last_played else 14
        
        # Cap rest days (e.g., at 14) and handle B2B
        home_rest_days = min(home_rest_days, 14)
        away_rest_days = min(away_rest_days, 14)
        
        home_is_b2b = 1 if home_rest_days == 0 else 0 # Technically, rest_days would be 1 if played yesterday. If 0, means same day (error) or needs adjustment.
        # Let's adjust: if game_date is 1 day after last_played_date, it's B2B (0 rest days between). 
        # If last_played is current_game_date - 1 day, then rest_days = 1. This means B2B.
        # If rest_days = (current_game_date - (current_game_date - 1 day)).days = 1, then it's B2B.
        home_is_b2b = 1 if home_rest_days == 1 else 0
        away_is_b2b = 1 if away_rest_days == 1 else 0

        # Update last game date for teams
        team_last_game_date[home_team_id] = current_game_date
        team_last_game_date[away_team_id] = current_game_date

        # Rolling Stats
        home_team_stats = calculate_team_rolling_stats(home_team_id, current_game_date, team_logs_df, window_size)
        away_team_stats = calculate_team_rolling_stats(away_team_id, current_game_date, team_logs_df, window_size)

        game_features = {'game_id': game_id}
        game_features.update({f'home_{stat}': val for stat, val in home_team_stats.items()})
        game_features.update({f'away_{stat}': val for stat, val in away_team_stats.items()})
        
        game_features['home_rest_days'] = home_rest_days
        game_features['away_rest_days'] = away_rest_days
        game_features['home_is_b2b'] = home_is_b2b
        game_features['away_is_b2b'] = away_is_b2b

        # Difference Features (ensure stats are present before diffing)
        for stat_col_prefix in home_team_stats.index: # e.g. 'avg_pts'
            stat_name = stat_col_prefix.replace('avg_', '') # 'pts'
            if f'home_{stat_col_prefix}' in game_features and f'away_{stat_col_prefix}' in game_features:
                game_features[f'diff_{stat_name}'] = game_features[f'home_{stat_col_prefix}'] - game_features[f'away_{stat_col_prefix}']
            else:
                game_features[f'diff_{stat_name}'] = np.nan # Ensure column exists even if data was missing
        
        game_features['diff_rest_days'] = home_rest_days - away_rest_days

        # Add team names for context in full_game_details_df, not for model training directly
        game_features['home_team_name'] = team_id_to_name.get(home_team_id, 'Unknown')
        game_features['away_team_name'] = team_id_to_name.get(away_team_id, 'Unknown')
        game_features['home_team_id'] = home_team_id # Keep for full_game_details_df
        game_features['away_team_id'] = away_team_id # Keep for full_game_details_df

        all_features_list.append(game_features)

    if not all_features_list:
        print("No features generated. Check data and loops.")
        return pd.DataFrame(), pd.Series(), pd.DataFrame()

    features_df_with_details = pd.DataFrame(all_features_list)
    
    # Merge with target variable
    final_df = pd.merge(features_df_with_details, games_df[['game_id', 'HOME_TEAM_WINS']], on='game_id', how='inner')

    if final_df.empty:
        print("No data after merging features and target. Check game_id alignment.")
        return pd.DataFrame(), pd.Series(), pd.DataFrame()

    # Drop rows with any NaN values in crucial feature columns before splitting
    # Create head-to-head matchup features if possible
    try:
        # Ensure game_date exists in final_df (use index if necessary)
        if 'game_date' not in final_df.columns and isinstance(final_df.index, pd.DatetimeIndex):
            # If game_date is the index, add it as a column
            final_df['game_date'] = final_df.index
            
        if 'game_date' in final_df.columns:
            final_df['head_to_head_features'] = final_df.apply(
                lambda row: create_head_to_head_features(
                    row['home_team_id'], 
                    row['away_team_id'], 
                    game_date=row['game_date'], 
                    team_logs_df=team_logs_df
                ), axis=1
            )
        else:
            print("Warning: game_date column not found for head-to-head features")
            final_df['head_to_head_features'] = [{} for _ in range(len(final_df))]
        
        # Extract head-to-head features
        h2h_features = pd.json_normalize(final_df['head_to_head_features'].apply(lambda x: x if isinstance(x, dict) else {}))
        if not h2h_features.empty:
            final_df = pd.concat([final_df, h2h_features], axis=1)
            print(f"Added {len(h2h_features.columns)} head-to-head features")
    except Exception as e:
        print(f"Error creating head-to-head features: {e}")
    
    # Identify feature columns for the model (advanced features + diff features + B2B flags + streaks)
    model_feature_columns = [
        # Difference features
        'diff_pts', 'diff_fg_pct', 'diff_fg3_pct', 'diff_ft_pct', 'diff_reb', 
        'diff_ast', 'diff_tov', 'diff_stl', 'diff_win_pct', 'diff_rest_days',
        
        # Home/Away specific features
        'home_is_b2b', 'away_is_b2b', 'home_rest_days', 'away_rest_days',
        'home_win_streak', 'away_win_streak', 'home_home_win_pct', 'away_away_win_pct',
        
        # Advanced metrics
        'home_recent_form_pts', 'away_recent_form_pts',
        'home_momentum_fg_pct', 'away_momentum_fg_pct',
        'home_momentum_fg3_pct', 'away_momentum_fg3_pct',
        'home_efficiency_rating', 'away_efficiency_rating',
        'home_pts_last3', 'away_pts_last3',
        
        # Head-to-head features (if available)
        'h2h_home_win_pct', 'h2h_last5_win_pct', 'h2h_avg_point_diff'
    ]
    
    # Add feature interactions for key metrics
    final_df['fg_pct_interaction'] = final_df['home_avg_fg_pct'] * (1/final_df['away_avg_fg_pct'])
    final_df['fg3_pct_interaction'] = final_df['home_avg_fg3_pct'] * (1/final_df['away_avg_fg3_pct'])
    final_df['win_pct_interaction'] = final_df['home_avg_win_pct'] * (1/final_df['away_avg_win_pct'])
    
    # Add these interactions to model features
    model_feature_columns += ['fg_pct_interaction', 'fg3_pct_interaction', 'win_pct_interaction']
    
    # Ensure all selected model_feature_columns actually exist in final_df
    model_feature_columns = [col for col in model_feature_columns if col in final_df.columns]

    # Print final column count
    print(f"Final model features: {len(model_feature_columns)} columns")
    
    # Data cleaning: Handle non-numeric, NaN, and infinite values
    for col in model_feature_columns:
        # Convert object columns to numeric
        if final_df[col].dtype == 'object':
            try:
                final_df[col] = pd.to_numeric(final_df[col], errors='coerce')
            except Exception as e:
                print(f"Error converting column {col} to numeric: {e}")
                # Drop this column if we can't convert it
                model_feature_columns.remove(col)
                continue
        
        # Replace infinite values with NaN
        if final_df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            mask = np.isinf(final_df[col])
            if mask.any():
                inf_count = mask.sum()
                print(f"Replacing {inf_count} infinite values in {col} with NaN")
                final_df.loc[mask, col] = np.nan
    
    # Fill remaining NaN values with median values
    for col in model_feature_columns:
        if final_df[col].isna().any():
            nan_count = final_df[col].isna().sum()
            if nan_count > 0:
                print(f"Filling {nan_count} NaN values in {col} with median")
                median_val = final_df[col].median()
                if pd.isna(median_val):  # If median is also NaN, use 0
                    median_val = 0
                final_df[col].fillna(median_val, inplace=True)
    
    # Keep only rows with complete data for model_feature_columns after cleaning
    before_count = len(final_df)
    final_df = final_df.dropna(subset=model_feature_columns)
    after_count = len(final_df)
    if before_count > after_count:
        print(f"Dropped {before_count - after_count} rows with missing values after cleaning")
    
    print(f"After cleaning: {len(final_df)} complete rows remain with {len(model_feature_columns)} features")

    target_final = final_df['HOME_TEAM_WINS']
    features_for_model = final_df.set_index('game_id')[model_feature_columns]
    
    # full_game_details_df for context, prediction lookup, etc.
    # It should contain all generated features and identifiers
    full_game_details_cols = ['home_team_id', 'home_team_name', 'away_team_id', 'away_team_name', 'HOME_TEAM_WINS'] + model_feature_columns
    # Add original home/away stats if needed for inspection
    original_stat_cols = [col for col in features_df_with_details.columns if (col.startswith('home_avg_') or col.startswith('away_avg_')) and col not in model_feature_columns]
    full_game_details_cols.extend(original_stat_cols)
    full_game_details_cols.extend(['home_rest_days', 'away_rest_days']) # Ensure these are also in full_game_details_df
    full_game_details_cols = list(dict.fromkeys(full_game_details_cols)) # Remove duplicates, preserve order
    full_game_details_cols = [col for col in full_game_details_cols if col in final_df.columns] # Ensure all cols exist

    full_game_details_df = final_df.set_index('game_id')[full_game_details_cols]

    return features_for_model, target_final, full_game_details_df


def train_model(X_train, y_train):
    """Train an ensemble model with stacking for improved accuracy."""
    print("\nTraining model with advanced optimization...")
    
    # Important: Create a stratified k-fold for consistent evaluation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Advanced preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), X_train.columns)
        ]
    )
    
    # XGBoost with advanced hyperparameter search
    print("Optimizing XGBoost...")
    xgb_param_grid = {
        'xgb__n_estimators': [100, 200],
        'xgb__max_depth': [3, 5, 7],
        'xgb__learning_rate': [0.01, 0.1],
        'xgb__subsample': [0.8, 1.0],
        'xgb__colsample_bytree': [0.8, 1.0],
        'xgb__min_child_weight': [1, 3],
        'xgb__gamma': [0, 0.1]
    }
    
    # 1. Define base learners with different algorithms and configurations
    # XGBoost with optimized parameters
    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric='auc',
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1
    )
    
    # LightGBM model for different strengths
    lgb_model = LGBMClassifier(
        n_estimators=200,
        num_leaves=31,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    )
    
    # Random Forest for different algorithm approach
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1
    )
    
    # 2. Create feature-preprocessed models using Pipelines
    xgb_pipeline = Pipeline([
        ('scaler', StandardScaler()),  # XGBoost benefits from scaling
        ('xgb', xgb_model)
    ])
    
    xgb_search = RandomizedSearchCV(
        xgb_pipeline, 
        param_distributions=xgb_param_grid,
        n_iter=10,  # Try 10 random combinations
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    xgb_search.fit(X_train, y_train)
    print(f"Best XGBoost params: {xgb_search.best_params_}")
    print(f"Best XGBoost CV accuracy: {xgb_search.best_score_:.4f}")
    
    # LightGBM with advanced hyperparameter search
    print("\nOptimizing LightGBM...")
    lgb_param_grid = {
        'lgb__n_estimators': [100, 200],
        'lgb__max_depth': [3, 5, 7, -1],  # -1 for no limit
        'lgb__learning_rate': [0.01, 0.1],
        'lgb__subsample': [0.8, 1.0],
        'lgb__colsample_bytree': [0.8, 1.0],
        'lgb__min_child_samples': [20, 50]
    }
    
    lgb_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('lgb', LGBMClassifier(random_state=42))
    ])
    
    lgb_search = RandomizedSearchCV(
        lgb_pipeline, 
        param_distributions=lgb_param_grid,
        n_iter=10,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    lgb_search.fit(X_train, y_train)
    print(f"Best LightGBM params: {lgb_search.best_params_}")
    print(f"Best LightGBM CV accuracy: {lgb_search.best_score_:.4f}")
    
    # Random Forest with hyperparameter search
    print("\nOptimizing RandomForest...")
    rf_param_grid = {
        'rf__n_estimators': [100, 200],
        'rf__max_depth': [None, 5, 10],
        'rf__min_samples_split': [2, 5],
        'rf__min_samples_leaf': [1, 2],
        'rf__max_features': ['sqrt', 'log2']
    }
    
    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('rf', RandomForestClassifier(random_state=42))
    ])
    
    rf_search = RandomizedSearchCV(
        rf_pipeline, 
        param_distributions=rf_param_grid,
        n_iter=10,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    rf_search.fit(X_train, y_train)
    print(f"Best RandomForest params: {rf_search.best_params_}")
    print(f"Best RandomForest CV accuracy: {rf_search.best_score_:.4f}")
    
    # Gradient Boosting (new addition to our ensemble)
    print("\nOptimizing GradientBoosting...")
    gb_param_grid = {
        'gb__n_estimators': [100, 200],
        'gb__max_depth': [3, 5],
        'gb__learning_rate': [0.01, 0.1],
        'gb__subsample': [0.8, 1.0],
        'gb__min_samples_split': [2, 5]
    }
    
    gb_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('gb', GradientBoostingClassifier(random_state=42))
    ])
    
    gb_search = RandomizedSearchCV(
        gb_pipeline, 
        param_distributions=gb_param_grid,
        n_iter=10,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    gb_search.fit(X_train, y_train)
    print(f"Best GradientBoosting params: {gb_search.best_params_}")
    print(f"Best GradientBoosting CV accuracy: {gb_search.best_score_:.4f}")
    
    # Advanced Feature Selection
    print("\nPerforming recursive feature elimination...")
    selector = RFECV(
        estimator=xgb_search.best_estimator_.named_steps['xgb'],
        step=1,
        cv=cv,
        scoring='accuracy',
        min_features_to_select=10
    )
    
    # We need raw features for RFECV, not the preprocessed ones
    selector.fit(X_train, y_train)
    selected_features = X_train.columns[selector.support_]
    print(f"Selected {len(selected_features)} optimal features")
    print(f"Top 10 features: {selected_features[:10].tolist()}")
    
    # Create powerful ensemble using the best models
    print("\nBuilding advanced stacking ensemble...")
    stacking = StackingClassifier(
        estimators=[
            ('xgb_best', xgb_search.best_estimator_),
            ('lgb_best', lgb_search.best_estimator_),
            ('rf_best', rf_search.best_estimator_),
            ('gb_best', gb_search.best_estimator_)
        ],
        final_estimator=LogisticRegression(C=1.0, class_weight='balanced', random_state=42),
        cv=cv
    )
    
    # Alternative voting ensemble
    voting = VotingClassifier(
        estimators=[
            ('xgb_best', xgb_search.best_estimator_),
            ('lgb_best', lgb_search.best_estimator_),
            ('rf_best', rf_search.best_estimator_),
            ('gb_best', gb_search.best_estimator_)
        ],
        voting='soft',  # Use probability predictions
        weights=[3, 2, 1, 1]  # Weight more accurate models higher
    )
    
    # Train both ensemble models
    print("Training final stacking ensemble...")
    stacking.fit(X_train, y_train)
    
    print("Training final voting ensemble...")
    voting.fit(X_train, y_train)
    
    # Evaluate on training data with cross-validation
    stacking_cv_scores = cross_val_score(stacking, X_train, y_train, cv=cv)
    voting_cv_scores = cross_val_score(voting, X_train, y_train, cv=cv)
    
    stacking_cv_mean = stacking_cv_scores.mean()
    voting_cv_mean = voting_cv_scores.mean()
    
    print(f"Cross-validation Accuracy - Stacked: {stacking_cv_mean:.4f}, Voting: {voting_cv_mean:.4f}")
    
    # Choose the model with better CV accuracy
    if stacking_cv_mean >= voting_cv_mean:
        print("Selecting Stacked Ensemble Model as final model - more robust to overfitting")
        model = stacking
    else:
        print("Selecting Voting Ensemble Model as final model - better overall performance")
        model = voting
    
    # Extract feature importances (different handling for ensemble models)
    try:
        # For XGBoost or other models with feature_importances_ attribute
        if hasattr(model, 'feature_importances_'):
            feature_importances = model.feature_importances_
            feature_importances_series = pd.Series(feature_importances, index=X_train.columns).sort_values(ascending=False)
        # For stacked or voting models
        elif isinstance(model, (StackingClassifier, VotingClassifier)):
            print("\nGetting feature importance from ensemble model components...")
            
            # Initialize importances array with zeros
            combined_importances = np.zeros(len(X_train.columns))
            estimator_count = 0
            
            # Iterate through all estimators in the ensemble
            for name, estimator in model.named_estimators_.items():
                # Check if this is a pipeline with an estimator that has feature importances
                if hasattr(estimator, 'named_steps'):
                    # Extract the estimator from the pipeline
                    if 'xgb' in estimator.named_steps:
                        est = estimator.named_steps['xgb']
                        if hasattr(est, 'feature_importances_'):
                            print(f"  - Extracted importances from {name} (XGBoost)")
                            combined_importances += est.feature_importances_
                            estimator_count += 1
                    elif 'lgb' in estimator.named_steps:
                        est = estimator.named_steps['lgb']
                        if hasattr(est, 'feature_importances_'):
                            print(f"  - Extracted importances from {name} (LightGBM)")
                            combined_importances += est.feature_importances_
                            estimator_count += 1
                    elif 'rf' in estimator.named_steps:
                        est = estimator.named_steps['rf']
                        if hasattr(est, 'feature_importances_'):
                            print(f"  - Extracted importances from {name} (RandomForest)")
                            combined_importances += est.feature_importances_
                            estimator_count += 1
                # Direct estimator with feature_importances_
                elif hasattr(estimator, 'feature_importances_'):
                    print(f"  - Extracted importances from {name} (direct)")
                    combined_importances += estimator.feature_importances_
                    estimator_count += 1
            
            # Calculate average importance if we found at least one valid estimator
            if estimator_count > 0:
                combined_importances /= estimator_count
                feature_importances_series = pd.Series(combined_importances, index=X_train.columns).sort_values(ascending=False)
            else:
                # Fallback: create a series with equal values
                print("  - No valid feature importances found in ensemble components")
                feature_importances_series = pd.Series(np.ones(len(X_train.columns))/len(X_train.columns), index=X_train.columns)
        else:
            # Fallback: create a series with equal values
            feature_importances_series = pd.Series(np.ones(len(X_train.columns))/len(X_train.columns), index=X_train.columns)
    except Exception as e:
        print(f"Warning: Could not extract feature importances - {str(e)}")
        # Create a default series with equal importance
        feature_importances_series = pd.Series(np.ones(len(X_train.columns))/len(X_train.columns), index=X_train.columns)
    
    # Print top 10 features
    print("\nTop 10 most important features:")
    print(feature_importances_series.head(10))
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    feature_importances_series.head(15).plot(kind='barh')
    plt.title('Top 15 Most Important Features')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    
    # Save the model and feature importances
    print("\nSaving model and feature information...")
    joblib.dump(model, 'nba_predictor_model.joblib')
    joblib.dump(feature_importances_series, 'feature_importances.joblib')
    joblib.dump(list(X_train.columns), 'model_feature_columns.joblib')
    
    print("Model trained and saved.")
    
    # Print top 10 features
    print("\nTop 10 Features by Importance:")
    for i, (feature, importance) in enumerate(feature_importances_series.iloc[:10].items()):
        print(f"{i+1:2d}. {feature:25s}: {importance:.6f}")
    
    return model, feature_importances_series

def evaluate_model(model, X_test, y_test):
    """Evaluate and print model performance metrics."""
    print("\nEvaluating model performance:\n" + "-" * 30)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Get probabilities if model supports it
    try:
        probabilities = model.predict_proba(X_test)[:, 1]  # Probability of class 1 (home team win)
    except (AttributeError, IndexError):
        probabilities = None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    
    # Create a metrics dictionary
    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }
    
    # Add AUC if probability predictions are available
    if probabilities is not None:
        try:
            auc = roc_auc_score(y_test, probabilities)
            metrics['AUC'] = auc
        except Exception as e:
            print(f"Warning: Could not compute AUC: {str(e)}")
    
    # Print metrics
    print("\nModel Evaluation Results:")
    print("=" * 50)
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, predictions)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    
    # If probability estimates are available, plot ROC curve
    if probabilities is not None and 'AUC' in metrics:
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(y_test, probabilities)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {metrics["AUC"]:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.tight_layout()
        plt.savefig('roc_curve.png')
        
        # Calibration plot
        from sklearn.calibration import calibration_curve
        plt.figure(figsize=(8, 6))
        fraction_of_positives, mean_predicted_value = calibration_curve(y_test, probabilities, n_bins=10)
        plt.plot(mean_predicted_value, fraction_of_positives, 's-', label='Model')
        plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Observed Fraction of Positives')
        plt.title('Calibration Plot')
        plt.legend()
        plt.tight_layout()
        plt.savefig('calibration_plot.png')
    
    # Print class-specific metrics
    from sklearn.metrics import classification_report
    print("\nClassification Report:")
    print("=" * 50)
    print(classification_report(y_test, predictions))
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(cm)
    
    print("\nSaved evaluation visualizations to disk.")
    
    # Check if we hit our target accuracy
    if accuracy >= 0.70:
        print(f"\n✅ Success! Model achieved {accuracy:.1%} accuracy, exceeding the 70% target.")
    else:
        print(f"\n⚠ Model accuracy ({accuracy:.1%}) is below the 70% target.")
    
    return accuracy

def predict_game(model, game_features_df, home_team_name="Home Team", away_team_name="Away Team", top_n_features=5, feature_importances=None):
    """Predicts outcome, shows key factors, saves data for LLM, and calls summary_api to generate a game summary."""
    print(f"\nPredicting for: {home_team_name} (Home) vs. {away_team_name} (Away)")
    
    if isinstance(game_features_df, pd.Series):
        # If a Series is passed (e.g. from X.loc[game_id]), convert to DataFrame
        game_features_for_prediction = game_features_df.to_frame().T
    else:
        game_features_for_prediction = game_features_df

    # Load model feature columns to ensure alignment
    try:
        model_feature_columns = joblib.load('model_feature_columns.joblib')
        print(f"Loaded {len(model_feature_columns)} expected features for prediction")
        
        # Align features with model's expected features
        missing_cols = [col for col in model_feature_columns if col not in game_features_for_prediction.columns]
        extra_cols = [col for col in game_features_for_prediction.columns if col not in model_feature_columns]
        
        # Print feature alignment info
        if missing_cols:
            print(f"Adding {len(missing_cols)} missing features with zeros: {', '.join(missing_cols[:5])}{'...' if len(missing_cols) > 5 else ''}")
            for col in missing_cols:
                game_features_for_prediction[col] = 0
        
        if extra_cols:
            print(f"Dropping {len(extra_cols)} extra features: {', '.join(extra_cols[:5])}{'...' if len(extra_cols) > 5 else ''}")
        
        # Select and reorder columns to match model's expected features
        game_features_for_prediction = game_features_for_prediction[model_feature_columns]
    except FileNotFoundError:
        print("Warning: model_feature_columns.joblib not found. Features may be misaligned.")
    
    prediction = model.predict(game_features_for_prediction)
    probability = model.predict_proba(game_features_for_prediction)
    
    predicted_winner_numeric = prediction[0]
    winner_name = home_team_name if predicted_winner_numeric == 1 else away_team_name
    loser_name = away_team_name if predicted_winner_numeric == 1 else home_team_name
    win_probability = probability[0][1] if predicted_winner_numeric == 1 else probability[0][0]
    
    print(f"Prediction: {winner_name} wins (Probability: {win_probability:.2f})")

    llm_prompt_data = {
        "home_team_name": home_team_name,
        "away_team_name": away_team_name,
        "predicted_winner_name": winner_name,
        "predicted_loser_name": loser_name,
        "win_probability_percent": f"{win_probability*100:.0f}", # Changed to not include %% symbol for cleaner JSON
        "home_stats": {},
        "away_stats": {}
    }

    # Extract relevant stats from game_features_df for the prompt
    # Assuming features are named like 'home_avg_pts', 'away_avg_fg_pct' etc.
    current_game_values = game_features_for_prediction.iloc[0]
    
    # Debug: Print available columns to understand the data structure
    print(f"DEBUG: Available columns in game features: {list(current_game_values.index)[:10]}...")
    
    # Look for various possible column naming patterns
    for col_name, value in current_game_values.items():
        # Check multiple possible patterns for home team stats
        if col_name.startswith("home_avg_") or col_name.startswith("home_"):
            stat_key = col_name.replace("home_avg_", "").replace("home_", "")
            llm_prompt_data["home_stats"][stat_key] = f"{value:.2f}"
        # Check multiple possible patterns for away team stats  
        elif col_name.startswith("away_avg_") or col_name.startswith("away_"):
            stat_key = col_name.replace("away_avg_", "").replace("away_", "")
            llm_prompt_data["away_stats"][stat_key] = f"{value:.2f}"
    
    # Debug: Print what stats were extracted
    print(f"DEBUG: Extracted home_stats: {llm_prompt_data['home_stats']}")
    print(f"DEBUG: Extracted away_stats: {llm_prompt_data['away_stats']}")
    
    # Add top N global features that influenced prediction
    top_global_features_info = []
    if feature_importances is not None:
        print("\nKey global features influencing this type of matchup (and their values for this game):")
        count = 0
        for feature_name, importance_score in feature_importances.items():
            if feature_name in current_game_values.index:
                value = current_game_values[feature_name]
                print(f"- {feature_name}: {value:.2f} (Global Importance: {importance_score:.4f})")
                top_global_features_info.append(f"{feature_name} (value: {value:.2f}, importance: {importance_score:.4f})")
                count += 1
                if count >= top_n_features:
                    break
    llm_prompt_data["top_global_features"] = "; ".join(top_global_features_info) if top_global_features_info else "N/A"

    # Save the llm_prompt_data to a JSON file
    json_output_path = "llm_prompt_data.json"
    try:
        with open(json_output_path, 'w') as f:
            json.dump(llm_prompt_data, f, indent=4)
        print(f"\nLLM prompt data saved to: {json_output_path}")
    except IOError as e:
        print(f"Error saving LLM prompt data to JSON: {e}")

    # Call summary_api to generate and print summary
    if summary_api_available:
        print("\n--- Attempting to generate LLM Game Summary via summary_api.py ---")
        game_summary = generate_game_summary(llm_prompt_data) # Pass the dictionary
        if game_summary:
            print("\n--- LLM Generated Game Summary ---")
            print(game_summary)
            print("--- End of LLM Game Summary ---")
        else:
            print("LLM summary could not be generated by summary_api.py.")
    else:
        # Fallback to printing the old prompt format if API not available (optional)
        # For now, we just note it wasn't generated.
        print("\nSkipping LLM summary generation as summary_api.py is not available.")
            
    return predicted_winner_numeric, win_probability

# New helper function for team ID lookup
def get_team_info_by_name(team_name_query, teams_info_df):
    """Looks up team information by full name. Case-insensitive partial match first, then exact."""
    if teams_info_df is None or teams_info_df.empty:
        print("Teams info DataFrame is not available for lookup.")
        return None

    # Try case-insensitive exact match first
    match = teams_info_df[teams_info_df['full_name'].str.lower() == team_name_query.lower()]
    if not match.empty:
        return match.iloc[0] # Return the first match as a Series

    # Try case-insensitive partial match if no exact found
    match = teams_info_df[teams_info_df['full_name'].str.contains(team_name_query, case=False, na=False)]
    if not match.empty:
        print(f"Note: Partial match found for '{team_name_query}'. Using '{match.iloc[0]['full_name']}'.")
        return match.iloc[0]
        
    print(f"Warning: Team '{team_name_query}' not found in teams_info_df.")
    return None

def get_prediction_data_for_teams(home_team_name_input, away_team_name_input):
    """
    Get prediction data for two teams without printing or calling summary_api directly.
    Returns a dictionary with prediction data that can be used by the app.
    
    Args:
        home_team_name_input (str): Name of the home team
        away_team_name_input (str): Name of the away team
    
    Returns:
        dict: Dictionary with prediction data, or None if prediction fails
    """
    try:
        # Load data
        data = load_data()
        if data is None:
            print("Error: Could not load data")
            return None
        
        # Check for essential dataframes
        if not all(data.get(key) is not None and not data.get(key).empty for key in ["games", "team_logs", "teams_info"]):
            print("Error: Essential dataframes are missing or empty")
            return None

        # Feature engineering
        X, y, full_game_details_df = feature_engineering(data) 
        if X.empty or y.empty:
            print("Error: Feature engineering resulted in empty data")
            return None

        # Load or train model
        model = None
        feature_importances = None
        
        # Try to load the model and feature importances
        if os.path.exists(MODEL_FILENAME) and os.path.exists(FEATURES_FILENAME):
            try:
                model = joblib.load(MODEL_FILENAME)
                feature_importances = joblib.load(FEATURES_FILENAME)
                print("Model and feature importances loaded successfully.")
            except Exception as e:
                print(f"Error loading saved model or features: {e}")
                model = None
                feature_importances = None

        # If no model loaded, need to train (simplified for app use)
        if model is None or feature_importances is None:
            if len(X) < 2 or len(y) < 2:
                print("Error: Not enough data to train model")
                return None
            
            # Quick train for app use
            test_size = 0.2
            if len(X) * (1 - test_size) < 1:
                test_size = 0.5
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                                test_size=test_size, 
                                                                random_state=42)
            
            if X_train.empty:
                print("Error: Training set is empty")
                return None

            model, feature_importances = train_model(X_train, y_train)

        # Find game matching the teams
        if not model or full_game_details_df.empty or data.get("teams_info") is None:
            print("Error: Model or required data not available")
            return None
        
        # Find the first game that matches the home and away team names (case-insensitive)
        target_game_df = full_game_details_df[
            (full_game_details_df['home_team_name'].str.lower() == home_team_name_input.lower()) &
            (full_game_details_df['away_team_name'].str.lower() == away_team_name_input.lower())
        ]

        if target_game_df.empty:
            print(f"No past game found for {home_team_name_input} vs {away_team_name_input}")
            return None

        game_id_to_predict = target_game_df.index[0]
        
        # Get features for this specific game
        if game_id_to_predict not in X.index:
            print(f"Could not find features for game_id {game_id_to_predict}")
            return None
            
        single_game_features = X.loc[game_id_to_predict]
        
        # Get actual team names from full_game_details_df for consistent casing
        actual_home_name = target_game_df.loc[game_id_to_predict, 'home_team_name']
        actual_away_name = target_game_df.loc[game_id_to_predict, 'away_team_name']

        # Get prediction data (modified version of predict_game logic)
        return _get_prediction_data_internal(model, single_game_features, actual_home_name, actual_away_name, feature_importances)

    except Exception as e:
        print(f"Error in get_prediction_data_for_teams: {e}")
        return None


def _get_prediction_data_internal(model, game_features_df, home_team_name, away_team_name, feature_importances, top_n_features=5):
    """
    Internal function to get prediction data without printing or calling external APIs.
    Modified version of predict_game that returns data instead of printing/calling APIs.
    """
    if isinstance(game_features_df, pd.Series):
        game_features_for_prediction = game_features_df.to_frame().T
    else:
        game_features_for_prediction = game_features_df

    # Load model feature columns to ensure alignment
    try:
        model_feature_columns = joblib.load('model_feature_columns.joblib')
        
        # Align features with model's expected features
        missing_cols = [col for col in model_feature_columns if col not in game_features_for_prediction.columns]
        
        # Add missing features with zeros
        for col in missing_cols:
            game_features_for_prediction[col] = 0
        
        # Select and reorder columns to match model's expected features
        game_features_for_prediction = game_features_for_prediction[model_feature_columns]
    except FileNotFoundError:
        print("Warning: model_feature_columns.joblib not found. Features may be misaligned.")
    
    prediction = model.predict(game_features_for_prediction)
    probability = model.predict_proba(game_features_for_prediction)
    
    predicted_winner_numeric = prediction[0]
    winner_name = home_team_name if predicted_winner_numeric == 1 else away_team_name
    loser_name = away_team_name if predicted_winner_numeric == 1 else home_team_name
    win_probability = probability[0][1] if predicted_winner_numeric == 1 else probability[0][0]
    
    llm_prompt_data = {
        "home_team_name": home_team_name,
        "away_team_name": away_team_name,
        "predicted_winner_name": winner_name,
        "predicted_loser_name": loser_name,
        "win_probability_percent": f"{win_probability*100:.0f}",
        "home_stats": {},
        "away_stats": {}
    }

    # Extract relevant stats from game_features_df for the prompt
    current_game_values = game_features_for_prediction.iloc[0]
    
    # Debug: Print available columns to understand the data structure
    print(f"DEBUG: Available columns in game features: {list(current_game_values.index)[:10]}...")
    
    # Look for various possible column naming patterns
    for col_name, value in current_game_values.items():
        # Check multiple possible patterns for home team stats
        if col_name.startswith("home_avg_") or col_name.startswith("home_"):
            stat_key = col_name.replace("home_avg_", "").replace("home_", "")
            llm_prompt_data["home_stats"][stat_key] = f"{value:.2f}"
        # Check multiple possible patterns for away team stats  
        elif col_name.startswith("away_avg_") or col_name.startswith("away_"):
            stat_key = col_name.replace("away_avg_", "").replace("away_", "")
            llm_prompt_data["away_stats"][stat_key] = f"{value:.2f}"
    
    # Debug: Print what stats were extracted
    print(f"DEBUG: Extracted home_stats: {llm_prompt_data['home_stats']}")
    print(f"DEBUG: Extracted away_stats: {llm_prompt_data['away_stats']}")
    
    # Add top N global features that influenced prediction
    top_global_features_info = []
    if feature_importances is not None:
        count = 0
        for feature_name, importance_score in feature_importances.items():
            if feature_name in current_game_values.index:
                value = current_game_values[feature_name]
                top_global_features_info.append(f"{feature_name} (value: {value:.2f}, importance: {importance_score:.4f})")
                count += 1
                if count >= top_n_features:
                    break
    llm_prompt_data["top_global_features"] = "; ".join(top_global_features_info) if top_global_features_info else "N/A"

    return llm_prompt_data

if __name__ == "__main__":
    data = load_data()
    if data is None:
        exit("Exiting due to data loading issues.") # Simple exit for critical failure
    # More robust check for essential dataframes
    if not all(data.get(key) is not None and not data.get(key).empty for key in ["games", "team_logs", "teams_info"]):
        exit("Essential dataframes (games, team_logs, teams_info) are missing or empty. Exiting.")

    X, y, full_game_details_df = feature_engineering(data) 
    if X.empty or y.empty:
        exit("Feature engineering resulted in empty data. Cannot proceed. Exiting.")

    model = None
    feature_importances = None
    force_retrain = True # Set to True to force retraining

    # Try to load the model and feature importances
    if os.path.exists(MODEL_FILENAME) and os.path.exists(FEATURES_FILENAME) and not force_retrain:
        print(f"Loading saved model from {MODEL_FILENAME}...")
        try:
            model = joblib.load(MODEL_FILENAME)
            print(f"Loading saved feature importances from {FEATURES_FILENAME}...")
            feature_importances = joblib.load(FEATURES_FILENAME)
            print("Model and feature importances loaded successfully.")
        except Exception as e:
            print(f"Error loading saved model or features: {e}. Will retrain.")
            model = None # Ensure model is None if loading failed
            feature_importances = None

    if model is None or feature_importances is None: 
        if len(X) < 2 or len(y) < 2: # Ensure enough data for splitting and training
            # Correctly indented exit call
            exit(f"Not enough data to train model. Found {len(X)} samples. Needs at least 2. Exiting.")

        # Stratify logic improved slightly for robustness with small datasets
        # Ensure test_size doesn't result in an empty training set for very small N
        test_size = 0.2
        if len(X) * (1 - test_size) < 1: # If training set would be < 1 sample
            test_size = 1 / len(X) # Make test set 1 sample, train with rest (if len(X) > 1)
            if len(X) * (1-test_size) < 1 and len(X) > 1: # if len(X) == 2, test_size = 0.5, train will be 1
                 test_size = 0.5 # ensure train has at least 1
            elif len(X) <=1: # cannot split
                 exit(f"Cannot split data with only {len(X)} sample(s). Exiting")

        # Stratification check: ensure at least 1 sample per class in y_train AND y_test if stratifying
        # This is complex to guarantee perfectly for all tiny datasets with train_test_split.
        # A simpler check: stratify if more than 1 class and enough samples per class for the split.
        can_stratify = False
        if len(y.unique()) > 1:
            vc = y.value_counts()
            # Check if smallest class is large enough for at least one sample in test set (approx)
            if vc.min() >= max(1, int(len(y) * test_size)) and vc.min() > 1 : # also ensure smallest class > 1 for stratify
                 can_stratify = True
            else:
                print(f"Warning: Cannot reliably stratify with small class sizes (counts: {vc.to_dict()}). Proceeding without stratification.") 
        else: # Single class data
            print("Warning: Only one class present in the target variable. Stratification is not applicable.")

        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                            test_size=test_size, 
                                                            random_state=42, 
                                                            stratify=y if can_stratify else None)
        
        if X_train.empty: # Primary check for train set
             exit("Training set is empty after split. This can happen with very small datasets. Exiting.")

        print("No saved model found or retraining forced/needed. Training a new model...")
        model, feature_importances = train_model(X_train, y_train)
        print("\nTop Global Feature Importances (from new model):")
        print(feature_importances.head(10))

        if not X_test.empty and not y_test.empty:
            evaluate_model(model, X_test, y_test)
        else:
            print("\nSkipping model evaluation as the test set (or y_test) is empty.")
    else:
        print("\nTop Global Feature Importances (from loaded model):")
        print(feature_importances.head(10))

    # --- Prediction by Team Names ---
    if model and not full_game_details_df.empty and data.get("teams_info") is not None:
        home_team_name_input = home_team_name
        away_team_name_input = away_team_name
        
        print(f"\nAttempting to find a past game for {home_team_name_input} vs {away_team_name_input} for prediction...")

        # No need to use get_team_info_by_name if full_game_details_df already has names.
        # We need home_team_id and away_team_id if we strictly want to match on IDs.
        # full_game_details_df has home_team_name and away_team_name directly.
        
        # Find the first game that matches the home and away team names
        # Ensure case-insensitivity if team names in DB might vary from input
        target_game_df = full_game_details_df[
            (full_game_details_df['home_team_name'].str.lower() == home_team_name_input.lower()) &
            (full_game_details_df['away_team_name'].str.lower() == away_team_name_input.lower())
        ]

        if not target_game_df.empty:
            game_id_to_predict = target_game_df.index[0] # game_id is the index
            print(f"Found game_id: {game_id_to_predict} for the matchup.")

            # Get features for this specific game
            # X has game_id as index
            if game_id_to_predict in X.index:
                single_game_features = X.loc[game_id_to_predict]
                
                # Get actual team names from full_game_details_df for consistent casing
                actual_home_name = target_game_df.loc[game_id_to_predict, 'home_team_name']
                actual_away_name = target_game_df.loc[game_id_to_predict, 'away_team_name']

                predict_game(model, single_game_features, 
                             home_team_name=actual_home_name, 
                             away_team_name=actual_away_name, 
                             top_n_features=6, 
                             feature_importances=feature_importances)
            else:
                print(f"Could not find features for game_id {game_id_to_predict} in the feature set (X).")
        else:
            print(f"No past game found in the database for {home_team_name_input} (Home) vs {away_team_name_input} (Away).")
            print("Try different team names or ensure data_collector.py has run for games involving these teams.")
    elif not model:
        print("\nModel not trained or loaded. Cannot make predictions by team names.")
    else:
        print("\nCannot make predictions by team names due to missing data (full_game_details_df or teams_info).")

    print("\nPredictor script completed.") 