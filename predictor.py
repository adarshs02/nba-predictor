import pandas as pd
import numpy as np
import joblib
import os
import json
from supabase_client import supabase
from datetime import datetime

# --- Constants ---
MODEL_FILENAME = "nba_predictor_model.joblib"
FEATURES_FILENAME = "model_columns.joblib"
IMPORTANCES_FILENAME = "feature_importances.joblib"

# --- Team Names ---
HOME_TEAM_NAME = "Golden State Warriors"
AWAY_TEAM_NAME = "Los Angeles Lakers"


def get_team_id(team_name, teams_df):
    """
    Finds the team ID for a given team name.
    Handles variations in team names (e.g., 'LAC' vs 'LA Clippers').
    """
    # Exact match first
    team_row = teams_df[teams_df['full_name'] == team_name]
    if not team_row.empty:
        return team_row.iloc[0]['id'], team_row.iloc[0]['abbreviation']

    # Partial match for common names
    team_row = teams_df[teams_df['full_name'].str.contains(team_name, case=False, na=False)]
    if not team_row.empty:
        return team_row.iloc[0]['id'], team_row.iloc[0]['abbreviation']

    # Match abbreviation
    team_row = teams_df[teams_df['abbreviation'] == team_name.upper()]
    if not team_row.empty:
        return team_row.iloc[0]['id'], team_row.iloc[0]['abbreviation']

    return None, None

def get_team_recent_stats(team_id, team_logs_df, num_games=10):
    """
    Fetches and calculates key performance indicators for a team from its last N games.
    This function is for display purposes in the Streamlit app.
    """
    # Filter for the specific team and sort by date to get the most recent games
    team_logs = team_logs_df[team_logs_df['team_id'] == team_id].copy()
    team_logs['game_date'] = pd.to_datetime(team_logs['game_date'])
    recent_games = team_logs.sort_values('game_date', ascending=False).head(num_games)

    if recent_games.empty:
        return {
            "Win Rate (Last 10)": "N/A",
            "Avg Points": "N/A",
            "Avg Rebounds": "N/A",
            "Avg Assists": "N/A",
            "Field Goal %": "N/A",
            "3-Point %": "N/A",
        }

    # Calculate stats
    # Ensure columns are numeric before calculating mean
    for col in ['pts', 'reb', 'ast', 'fg_pct', 'fg3_pct']:
        recent_games[col] = pd.to_numeric(recent_games[col], errors='coerce')

    win_rate = (recent_games['wl'] == 'W').mean()
    avg_pts = recent_games['pts'].mean()
    avg_reb = recent_games['reb'].mean()
    avg_ast = recent_games['ast'].mean()
    avg_fg_pct = recent_games['fg_pct'].mean()
    avg_fg3_pct = recent_games['fg3_pct'].mean()

    # Format for display
    return {
        "Win Rate (Last 10)": f"{win_rate:.1%}" if pd.notna(win_rate) else "N/A",
        "Avg Points": f"{avg_pts:.1f}" if pd.notna(avg_pts) else "N/A",
        "Avg Rebounds": f"{avg_reb:.1f}" if pd.notna(avg_reb) else "N/A",
        "Avg Assists": f"{avg_ast:.1f}" if pd.notna(avg_ast) else "N/A",
        "Field Goal %": f"{avg_fg_pct:.1%}" if pd.notna(avg_fg_pct) else "N/A",
        "3-Point %": f"{avg_fg3_pct:.1%}" if pd.notna(avg_fg3_pct) else "N/A",
    }

def generate_features_for_game(home_team_id, away_team_id, teams_df, team_logs_df):
    """
    Generates a single feature vector for a given matchup.
    This should mirror the feature engineering in train_model.py
    """
    # Helper to calculate rolling stats for a single team based on its history
    def _calculate_rolling_stats_for_team(team_id, logs_df, date_before=None):
        team_logs = logs_df[logs_df['team_id'] == team_id].copy()
        team_logs['game_date'] = pd.to_datetime(team_logs['game_date'])
        
        # If predicting for a future date, use all available historical data
        if date_before:
            team_logs = team_logs[team_logs['game_date'] < date_before]

        team_logs = team_logs.sort_values('game_date', ascending=False)

        # Define stats to calculate rolling averages for
        stat_cols = ['pts', 'fgm', 'fga', 'fg_pct', 'fg3m', 'fg3a', 'fg3_pct', 
                     'ftm', 'fta', 'ft_pct', 'oreb', 'dreb', 'reb', 'ast', 'stl', 'blk', 'tov', 'pf']
        
        for col in stat_cols:
            team_logs[col] = pd.to_numeric(team_logs[col], errors='coerce')

        # Get the latest stats, which are the rolling averages before the 'current' game
        # Here we take the mean of the last 10 games as a proxy for current form
        latest_stats = team_logs.head(10)[stat_cols].mean().to_dict()
        
        # Add win percentage for the last 10 games
        latest_stats['win_pct_last_10'] = (team_logs.head(10)['wl'] == 'W').mean()

        return latest_stats

    # Get the latest stats for both teams (as of today)
    home_stats = _calculate_rolling_stats_for_team(home_team_id, team_logs_df)
    away_stats = _calculate_rolling_stats_for_team(away_team_id, team_logs_df)

    if not home_stats or not away_stats:
        return pd.DataFrame() # Not enough data

    # Create the feature vector
    feature_vector = {}
    for stat, value in home_stats.items():
        feature_vector[f'HOME_AVG_{stat.upper()}'] = value
    
    for stat, value in away_stats.items():
        feature_vector[f'AWAY_AVG_{stat.upper()}'] = value

    # Add differential features
    for stat in home_stats.keys():
        home_val = home_stats.get(stat, 0)
        away_val = away_stats.get(stat, 0)
        feature_vector[f'AVG_{stat.upper()}_diff'] = home_val - away_val

    # TODO: Add head-to-head features if applicable

    return pd.DataFrame([feature_vector]) # Placeholder

def get_prediction_data_for_teams(home_team_name, away_team_name):
    """
    Main function to get prediction and stats for a given matchup.
    This will be called by the Streamlit app.
    """
    # 1. Load Model and Feature List
    try:
        model = joblib.load(MODEL_FILENAME)
        feature_names = joblib.load(FEATURES_FILENAME)
        feature_importances = joblib.load(IMPORTANCES_FILENAME)
    except FileNotFoundError as e:
        # If model artifacts aren't found, return a clear error.
        return {"error": f"Model artifacts not found: {e}. Please run train_model.py first."}
    except Exception as e:
        return {"error": f"Error loading model artifacts: {e}"} 

    # 2. Fetch Data from Supabase
    try:
        teams_response = supabase.table('teams').select('id, full_name, abbreviation, logo_url').execute()
        logs_response = supabase.table('team_game_logs').select('*').execute()
        
        if not teams_response.data or not logs_response.data:
            return {"error": "Could not fetch teams or game logs from the database."}

        teams_df = pd.DataFrame(teams_response.data)
        team_logs_df = pd.DataFrame(logs_response.data)
        team_logs_df['game_date'] = pd.to_datetime(team_logs_df['game_date'])
    except Exception as e:
        return {"error": f"Database error: {e}"}

    # 3. Get Team IDs
    home_team_id, home_team_abbr = get_team_id(home_team_name, teams_df)
    away_team_id, away_team_abbr = get_team_id(away_team_name, teams_df)

    if not home_team_id or not away_team_id:
        missing_team = home_team_name if not home_team_id else away_team_name
        return {"error": f"Could not find team: {missing_team}"}

    # 4. Generate Features for the Game
    feature_vector_df = generate_features_for_game(home_team_id, away_team_id, teams_df, team_logs_df)

    if feature_vector_df.empty:
        return {"error": "Could not generate features for the matchup. Check data availability."}

    # 5. Align Feature Vector with Model's Features
    for col in feature_names:
        if col not in feature_vector_df.columns:
            feature_vector_df[col] = 0
    feature_vector_df = feature_vector_df[feature_names]

    # 6. Make Prediction
    prediction = model.predict(feature_vector_df)
    probability = model.predict_proba(feature_vector_df)
    
    predicted_winner_id = home_team_id if prediction[0] == 1 else away_team_id
    winner_team_response = supabase.table('teams').select('full_name').eq('id', predicted_winner_id).execute()
    predicted_winner_name = winner_team_response.data[0]['full_name'] if winner_team_response.data else "Unknown"
    
    # Confidence is the probability of the predicted class
    confidence = probability[0][1] if prediction[0] == 1 else probability[0][0]

    # 7. Get Key Factors
    top_features = feature_importances.sort_values(ascending=False).head(5).index.tolist()
    key_factors_list = [feature.replace('_', ' ').replace('Avg', 'Average').title() for feature in top_features]
    key_factors_str = "; ".join(key_factors_list)

    # 8. Get Stats for UI Display
    home_stats = get_team_recent_stats(home_team_id, team_logs_df)
    away_stats = get_team_recent_stats(away_team_id, team_logs_df)

    # 9. Structure and Return Results for API and UI compatibility
    return {
        "home_team_name": home_team_name,
        "away_team_name": away_team_name,

        # For summary_api.py
        "predicted_winner_name": predicted_winner_name,
        "win_probability_percent": f"{confidence * 100:.2f}",
        "top_global_features": key_factors_str,

        # For app.py (with some duplication for compatibility)
        "predicted_winner": predicted_winner_name,
        "confidence_score": f"{confidence:.2%}",
        "key_factors": key_factors_str, # Pass as string to fix bug in app.py

        # Common data
        "home_stats": home_stats,
        "away_stats": away_stats,
        "home_logo_url": teams_df.loc[teams_df['id'] == home_team_id, 'logo_url'].iloc[0],
        "away_logo_url": teams_df.loc[teams_df['id'] == away_team_id, 'logo_url'].iloc[0],
    }

if __name__ == '__main__':
    # This block allows for direct testing of the predictor script.
    # Example usage:
    home_team = HOME_TEAM_NAME
    away_team = AWAY_TEAM_NAME

    print(f"Running prediction for: {home_team} vs. {away_team}")
    
    prediction_data = get_prediction_data_for_teams(home_team, away_team)

    if "error" in prediction_data:
        print(f"\n--- ERROR ---\n{prediction_data['error']}")
    else:
        print("\n--- PREDICTION RESULTS ---")
        print(f"Matchup: {prediction_data['home_team_name']} vs. {prediction_data['away_team_name']}")
        print(f"Predicted Winner: {prediction_data['predicted_winner']}")
        print(f"Confidence: {prediction_data['confidence_score']}")

        print("\n--- KEY FACTORS ---")
        print(prediction_data.get('key_factors', 'N/A'))

        print(f"\n--- {prediction_data['home_team_name']} Recent Stats ---")
        if prediction_data.get('home_stats'):
            for stat, value in prediction_data['home_stats'].items():
                print(f"{stat}: {value}")
        else:
            print("No home stats available.")

        print(f"\n--- {prediction_data['away_team_name']} Recent Stats ---")
        if prediction_data.get('away_stats'):
            for stat, value in prediction_data['away_stats'].items():
                print(f"{stat}: {value}")
        else:
            print("No away stats available.")