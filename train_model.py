import pandas as pd
from supabase_client import supabase
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.feature_selection import RFECV, SelectKBest, f_classif
import numpy as np
import joblib
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns
import sys

"""
This script handles the training of the NBA game prediction model.
It fetches data, performs feature engineering, trains a stacking classifier,
evaluates its performance, and saves the trained model and associated artifacts.
"""

# --- Configuration ---
load_dotenv()
MODEL_FILENAME = "nba_predictor_model.joblib"
SCALER_FILENAME = "scaler.joblib"
MODEL_COLUMNS_FILENAME = "model_columns.joblib"
FEATURE_IMPORTANCES_FILENAME = "feature_importances.joblib"

# --- Data Loading ---
def load_data():
    """Loads all necessary data from the Supabase database."""
    print("Loading data from Supabase...")
    try:
        games_response = supabase.table('games').select('*').execute()
        games_df = pd.DataFrame(games_response.data)
        team_logs_response = supabase.table('team_game_logs').select('*').execute()
        team_logs_df = pd.DataFrame(team_logs_response.data)
        teams_info_response = supabase.table('teams').select('*').execute()
        teams_info_df = pd.DataFrame(teams_info_response.data)
        print(f"Loaded {len(games_df)} games, {len(team_logs_df)} team logs, and {len(teams_info_df)} teams.")
        return {"games": games_df, "team_logs": team_logs_df, "teams_info": teams_info_df}
    except Exception as e:
        print(f"An error occurred during data loading from Supabase: {e}", file=sys.stderr)
        return None

# --- Feature Engineering Functions ---
def calculate_team_rolling_stats(team_id, game_date, team_logs_df, window_size=10):
    team_logs_df['game_date'] = pd.to_datetime(team_logs_df['game_date'])
    game_date = pd.to_datetime(game_date)
    past_games = team_logs_df[(team_logs_df['team_id'] == team_id) & (team_logs_df['game_date'] < game_date)].sort_values(by='game_date', ascending=False)
    if past_games.empty or len(past_games) < 3:
        return pd.Series()
    past_games_10 = past_games.head(min(10, len(past_games)))
    team_stats = {}
    basic_stats = ['pts', 'fgm', 'fga', 'fg_pct', 'fg3m', 'fg3a', 'fg3_pct', 'ftm', 'fta', 'ft_pct', 'oreb', 'dreb', 'reb', 'ast', 'stl', 'blk', 'tov', 'pf', 'plus_minus']
    if len(past_games_10) > 0:
        past_games_10_copy = past_games_10.copy()
        for col in basic_stats:
            if col in past_games_10_copy.columns:
                past_games_10_copy.loc[:, col] = pd.to_numeric(past_games_10_copy[col], errors='coerce')
                team_stats[f'avg_{col}'] = past_games_10_copy[col].mean()
    
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
        # team_stats['efficiency_rating'] = (team_stats['avg_pts'] + team_stats['avg_ast'] + team_stats['avg_stl']) / max(1, team_stats['avg_tov'])
        pass
    
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


#TODO: Improve algorithm to achieve 65% accuracy
def feature_engineering(raw_data_dict):
    if raw_data_dict is None:
        print("Raw data is missing, cannot perform feature engineering.")
        return pd.DataFrame(), pd.Series(), pd.DataFrame()

    games_df = raw_data_dict.get("games")
    team_logs_df = raw_data_dict.get("team_logs")
    teams_info_df = raw_data_dict.get("teams_info")

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

        # Rolling Stats
        home_team_stats = calculate_team_rolling_stats(home_team_id, current_game_date, team_logs_df, window_size)
        away_team_stats = calculate_team_rolling_stats(away_team_id, current_game_date, team_logs_df, window_size)

        game_features = {'game_id': game_id}
        game_features.update({f'home_{stat}': val for stat, val in home_team_stats.items()})
        game_features.update({f'away_{stat}': val for stat, val in away_team_stats.items()})
        
        # Difference Features (ensure stats are present before diffing)
        for stat_col_prefix in home_team_stats.index: # e.g. 'avg_pts'
            stat_name = stat_col_prefix.replace('avg_', '') # 'pts'
            if f'home_{stat_col_prefix}' in game_features and f'away_{stat_col_prefix}' in game_features:
                game_features[f'diff_{stat_name}'] = game_features[f'home_{stat_col_prefix}'] - game_features[f'away_{stat_col_prefix}']
            else:
                game_features[f'diff_{stat_name}'] = np.nan # Ensure column exists even if data was missing
        
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

    # Add interaction and ratio features
    # Ensure required columns exist before creating new features
    required_cols_for_interactions = [
        'home_avg_fg_pct', 'away_avg_fg_pct', 'home_avg_fg3_pct', 'away_avg_fg3_pct',
        'home_avg_efg_pct', 'away_avg_efg_pct', 'home_avg_ts_pct', 'away_avg_ts_pct',
        'home_avg_oreb_pct', 'away_avg_dreb_pct', 'home_avg_dreb_pct', 'away_avg_oreb_pct',
        'home_avg_ast_tov', 'away_avg_ast_tov', 'home_avg_pts', 'away_avg_pts',
        'home_avg_ast', 'away_avg_ast', 'home_avg_reb', 'away_avg_reb',
        'home_avg_tov', 'away_avg_tov', 'home_avg_stl', 'away_avg_stl',
        'home_avg_blk', 'away_avg_blk'
    ]

    # Check if all required columns are present
    missing_cols = [col for col in required_cols_for_interactions if col not in final_df.columns]
    if missing_cols:
        print(f"Warning: Missing columns required for interaction/ratio features: {missing_cols}. Skipping these features.")
    else:
        print("Adding interaction features...")
        epsilon = 1e-6 # Small constant to avoid division by zero

        final_df['fg_pct_interaction'] = final_df['home_avg_fg_pct'] * (1 / (final_df['away_avg_fg_pct'] + epsilon))
        final_df['fg3_pct_interaction'] = final_df['home_avg_fg3_pct'] * (1 / (final_df['away_avg_fg3_pct'] + epsilon))

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
    
    # Add these interactions to model features
    model_feature_columns += ['fg_pct_interaction', 'fg3_pct_interaction']
    
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

    final_df.set_index('game_id', inplace=True)

    target_final = final_df['HOME_TEAM_WINS']
    features_for_model = final_df[model_feature_columns]

    # full_game_details_df for context, prediction lookup, etc.
    # It should contain all generated features and identifiers
    full_game_details_cols = ['home_team_id', 'home_team_name', 'away_team_id', 'away_team_name', 'HOME_TEAM_WINS'] + model_feature_columns
    # Add original home/away stats if needed for inspection
    original_stat_cols = [col for col in features_df_with_details.columns if (col.startswith('home_avg_') or col.startswith('away_avg_')) and col not in model_feature_columns]
    full_game_details_cols.extend(original_stat_cols)
    full_game_details_cols.extend(['home_rest_days', 'away_rest_days']) # Ensure these are also in full_game_details_df
    full_game_details_cols = list(dict.fromkeys(full_game_details_cols)) # Remove duplicates, preserve order
    full_game_details_cols = [col for col in full_game_details_cols if col in final_df.columns] # Ensure all cols exist

    full_game_details_df = final_df[full_game_details_cols]

    return features_for_model, target_final, full_game_details_df

# --- Model Training ---
def train_model(X_train, y_train):
    # --- Class Imbalance Handling ---
    # Calculate scale_pos_weight for XGBoost, which is effective for imbalanced datasets.
    # It's the ratio of the number of negative class instances to positive class instances.
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
    print(f"\nClass balance check: Negative={neg_count}, Positive={pos_count}, XGBoost scale_pos_weight={scale_pos_weight:.2f}")

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
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight
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
        n_iter=30,  # Try 10 random combinations
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
        ('lgb', LGBMClassifier(random_state=42, class_weight='balanced'))
    ])
    
    lgb_search = RandomizedSearchCV(
        lgb_pipeline, 
        param_distributions=lgb_param_grid,
        n_iter=30,
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
        ('rf', RandomForestClassifier(random_state=42, class_weight='balanced'))
    ])
    
    rf_search = RandomizedSearchCV(
        rf_pipeline, 
        param_distributions=rf_param_grid,
        n_iter=30,
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
        'gb__learning_rate': [0.01, 0.01],
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
        n_iter=30,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    gb_search.fit(X_train, y_train)
    print(f"Best GradientBoosting params: {gb_search.best_params_}")
    print(f"Best GradientBoosting CV accuracy: {gb_search.best_score_:.4f}")
    
    # Define the Voting Classifier using best estimators from RandomizedSearchCV
    print("Defining VotingClassifier with best estimators...")
    voting_clf = VotingClassifier(
        estimators=[
            ('xgb', xgb_search.best_estimator_),
            ('lgb', lgb_search.best_estimator_),
            ('rf', rf_search.best_estimator_),
            ('gb', gb_search.best_estimator_)
        ],
        voting='soft'  # Use 'soft' for probability-based voting
    )

    print("Fitting VotingClassifier...")
    voting_clf.fit(X_train, y_train)
    print("VotingClassifier fitting complete.")

    # For feature importance, extract the XGBoost model from the pipeline if possible
    # This logic remains the same as it's about extracting from xgb_search.best_estimator_
    xgb_model_for_importance = None
    best_xgb_pipeline = xgb_search.best_estimator_
    if hasattr(best_xgb_pipeline, 'named_steps') and 'xgb' in best_xgb_pipeline.named_steps:
        xgb_model_for_importance = best_xgb_pipeline.named_steps['xgb']
    elif hasattr(best_xgb_pipeline, 'steps'): # Fallback for differently structured pipelines
        for step_name, step_model in best_xgb_pipeline.steps:
            if 'xgb' in step_name.lower(): # More robust check for 'xgb' step
                xgb_model_for_importance = step_model
                break
    if xgb_model_for_importance is None:
        print("Warning: Could not extract XGBoost model from pipeline for feature importances. Using the pipeline itself.")
        xgb_model_for_importance = best_xgb_pipeline
    
    return voting_clf, xgb_model_for_importance



# --- Model Saving --- 
def save_model_and_artifacts(model, X_train):
    """Save the trained model, scaler, and feature names."""
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
    print("\nTop 15 most important features:")
    print(feature_importances_series.head(15))
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    feature_importances_series.head(15).plot(kind='barh') # Already 15, no change needed here, but keeping for context
    plt.title('Top 15 Most Important Features')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    
    # Save the model and feature importances
    print("\nSaving model and feature information...")
    joblib.dump(model, MODEL_FILENAME)
    joblib.dump(feature_importances_series, FEATURE_IMPORTANCES_FILENAME)
    joblib.dump(list(X_train.columns), MODEL_COLUMNS_FILENAME)
    
    print("Model trained and saved.")
    
    # Print top 10 features
    print("\nTop 15 Features by Importance (Console Output):")
    for i, (feature, importance) in enumerate(feature_importances_series.iloc[:15].items()):
        print(f"{i+1:2d}. {feature:25s}: {importance:.6f}")
    
    return model, feature_importances_series

# --- Model Evaluation ---
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
    if accuracy >= 0.65:
        print(f"\n✅ Success! Model achieved {accuracy:.1%} accuracy, exceeding the 65% target.")
    else:
        print(f"\n⚠ Model accuracy ({accuracy:.1%}) is below the 65% target.")
    
    return accuracy

# --- Main Execution ---
if __name__ == "__main__":
    print("Starting model training process...")
    raw_data = load_data()
    if not raw_data:
        sys.exit("Failed to load data. Exiting.")

    features, target, full_game_details_df = feature_engineering(raw_data)
    if features.empty or target.empty:
        sys.exit("Feature engineering resulted in an empty DataFrame. Exiting.")

    # Align data for training - ensure that we use the same game_ids for features and target
    aligned_target = target.loc[features.index]

    # Split the data chronologically to prevent data leakage
    # A real-world scenario would use time-based splitting (e.g., by season)
    # For this dataset, a simple percentage split on the sorted data is a good approximation.
    split_index = int(len(features) * 0.8)
    X_train = features.iloc[:split_index]
    X_test = features.iloc[split_index:]
    y_train = aligned_target.iloc[:split_index]
    y_test = aligned_target.iloc[split_index:]

    # --- Feature Selection using SelectKBest --- #
    print("\nSelecting top 15 features...")
    # Scale data temporarily for feature selection, as SelectKBest works best with scaled data
    scaler_for_selection = StandardScaler()
    X_train_scaled_for_selection = scaler_for_selection.fit_transform(X_train)

    # Ensure k is not greater than the number of available features
    k = min(15, X_train.shape[1])
    if k < 15:
        print(f"Warning: Requested 15 features, but only {k} are available. Using k={k}.")

    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X_train_scaled_for_selection, y_train)

    selected_features_mask = selector.get_support()
    selected_features = X_train.columns[selected_features_mask]
    
    print(f"Selected features: {selected_features.tolist()}")

    # Filter the original dataframes to keep only selected features
    X_train = X_train[selected_features]
    X_test = X_test[selected_features]
    # --- End of Feature Selection --- #
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")

    print("Training a new model...")
    model, feature_importances = train_model(X_train, y_train)

    print("\n--- Model Evaluation ---")
    evaluate_model(model, X_test, y_test)

    print("\n--- Saving Model and Artifacts ---")
    save_model_and_artifacts(model, X_train) # X_train is used for column names

    print("\nScript finished. Model, feature importances, and artifacts have been saved.")

    print("\nTraining script completed.")
