import pandas as pd
import numpy as np
import joblib
import os
import json
from supabase_client import supabase
from datetime import datetime
from feature_utils import calculate_team_rolling_stats, create_head_to_head_features, calculate_rest_b2b_features

# --- Constants ---
MODEL_FILENAME = "nba_predictor_model.joblib"
FEATURES_FILENAME = "model_columns.joblib"
IMPORTANCES_FILENAME = "feature_importances.joblib"

# --- Team Names ---
HOME_TEAM_NAME = "Oklahoma City Thunder"
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
    Generates a single feature vector for a given matchup using feature_utils,
    aligning with the feature names in model_columns.joblib.
    """
    game_date = datetime.now() # Use current date for predictions

    # DEBUG: Inspect inputs before rolling stats calculation
    print(f"[Debug generate_features_for_game] team_logs_df shape: {team_logs_df.shape}")
    if not team_logs_df.empty and 'game_date' in team_logs_df.columns:
        # Ensure game_date column is datetime for min/max operations
        temp_game_dates = pd.to_datetime(team_logs_df['game_date'], errors='coerce')
        print(f"[Debug generate_features_for_game] team_logs_df min game_date: {temp_game_dates.min()}, max game_date: {temp_game_dates.max()}")
    print(f"[Debug generate_features_for_game] game_date for rolling stats: {game_date}")

    # 1. Calculate rolling stats for home and away teams
    # These are Series with keys like 'avg_pts', 'avg_fg_pct', 'efficiency_rating', 'avg_win_pct'
    home_rolling_stats = calculate_team_rolling_stats(home_team_id, game_date, team_logs_df)
    away_rolling_stats = calculate_team_rolling_stats(away_team_id, game_date, team_logs_df)

    if home_rolling_stats.isnull().all() or away_rolling_stats.isnull().all():
        print(f"Warning: Rolling stats (all NaN) not available for one or both teams. home_id: {home_team_id}, away_id: {away_team_id}")
        # Return an empty DataFrame with expected columns if possible, or just empty
        # This helps the calling function's alignment step
        expected_cols = joblib.load(FEATURES_FILENAME)
        return pd.DataFrame(columns=expected_cols)

    # 2. Calculate rest and back-to-back features (dictionaries with 'rest_days', 'is_b2b')
    home_rest_b2b = calculate_rest_b2b_features(home_team_id, game_date, team_logs_df)
    away_rest_b2b = calculate_rest_b2b_features(away_team_id, game_date, team_logs_df)

    # 3. (Optional) Head-to-head features - not explicitly in model_columns.joblib from output
    # h2h_features = create_head_to_head_features(home_team_id, away_team_id, game_date, team_logs_df)

    # 4. Construct the feature dictionary according to model_columns.joblib
    features = {}

    # Direct mapping for rest and b2b
    features['home_rest_days'] = home_rest_b2b.get('rest_days', np.nan)
    features['away_rest_days'] = away_rest_b2b.get('rest_days', np.nan)
    features['home_is_b2b'] = home_rest_b2b.get('is_b2b', 0) # Default to 0 if not found
    features['away_is_b2b'] = away_rest_b2b.get('is_b2b', 0)
    features['diff_rest_days'] = features['home_rest_days'] - features['away_rest_days']

    # Efficiency ratings
    features['home_efficiency_rating'] = home_rolling_stats.get('efficiency_rating', np.nan)
    features['away_efficiency_rating'] = away_rolling_stats.get('efficiency_rating', np.nan)

    # Differential features from rolling stats
    # model_columns: ['diff_pts', 'diff_fg_pct', 'diff_fg3_pct', 'diff_ft_pct', 'diff_reb', 'diff_ast', 'diff_tov', 'diff_stl', 'diff_win_pct']
    stat_map = {
        'pts': 'avg_pts',
        'fg_pct': 'avg_fg_pct',
        'fg3_pct': 'avg_fg3_pct',
        'ft_pct': 'avg_ft_pct',
        'reb': 'avg_reb',
        'ast': 'avg_ast',
        'tov': 'avg_tov',
        'stl': 'avg_stl',
        'win_pct': 'avg_win_pct' # from calculate_team_rolling_stats
    }

    for model_key, util_key in stat_map.items():
        home_val = home_rolling_stats.get(util_key, np.nan)
        away_val = away_rolling_stats.get(util_key, np.nan)
        features[f'diff_{model_key}'] = home_val - away_val

    # Interaction terms
    # model_columns: ['fg_pct_interaction', 'fg3_pct_interaction', 'win_pct_interaction']
    home_fg_pct = home_rolling_stats.get('avg_fg_pct', np.nan)
    away_fg_pct = away_rolling_stats.get('avg_fg_pct', np.nan)
    features['fg_pct_interaction'] = home_fg_pct * away_fg_pct

    home_fg3_pct = home_rolling_stats.get('avg_fg3_pct', np.nan)
    away_fg3_pct = away_rolling_stats.get('avg_fg3_pct', np.nan)
    features['fg3_pct_interaction'] = home_fg3_pct * away_fg3_pct

    home_win_pct = home_rolling_stats.get('avg_win_pct', np.nan)
    away_win_pct = away_rolling_stats.get('avg_win_pct', np.nan)
    features['win_pct_interaction'] = home_win_pct * away_win_pct
    
    # Ensure column order matches model_columns.joblib
    df = pd.DataFrame([features])
    expected_cols = joblib.load(FEATURES_FILENAME)
    # Add any missing expected columns with NaN (or 0 if preferred, though alignment later handles 0s)
    for col in expected_cols:
        if col not in df.columns:
            df[col] = np.nan # Or 0, but subsequent alignment in get_prediction_data_for_teams uses 0
    return df[expected_cols] # Return DataFrame with columns in the exact expected order

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
        # Implement pagination to fetch all team_game_logs
        all_logs_data = []
        page_size = 1000  # Default/max limit per request by Supabase
        current_offset = 0
        while True:
            # Fetch a page of game logs
            page_response = supabase.table('team_game_logs') \
                                  .select('*') \
                                  .range(current_offset, current_offset + page_size - 1) \
                                  .execute()
            
            if not page_response.data:
                # No more data or an error occurred on this page fetch
                if current_offset == 0: # Failed on the very first fetch
                    return {"error": "Could not fetch any game logs from the database."}
                break # Stop if no data returned for the current page

            all_logs_data.extend(page_response.data)

            if len(page_response.data) < page_size:
                break  # Last page fetched
            
            current_offset += page_size

        if not teams_response.data or not all_logs_data:
            return {"error": "Could not fetch teams or game logs (after pagination) from the database."}

        teams_df = pd.DataFrame(teams_response.data)
        team_logs_df = pd.DataFrame(all_logs_data)
        team_logs_df['game_date'] = pd.to_datetime(team_logs_df['game_date'])

        # DEBUG: Check team_logs_df content for a specific team ID before passing it on
        print(f"[Debug predictor] Shape of full team_logs_df after loading: {team_logs_df.shape}")
        problematic_team_id_to_check = 1610612738 # Boston Celtics ID
        # Ensure 'team_id' column is of the correct type for comparison, Supabase might return it as string or number
        if 'team_id' in team_logs_df.columns:
            # Attempt conversion to int, coercing errors for robustness if mixed types exist
            team_logs_df['team_id'] = pd.to_numeric(team_logs_df['team_id'], errors='coerce').astype('Int64') # Use Int64 for nullable integers
            
            boston_logs_in_predictor_df = team_logs_df[team_logs_df['team_id'] == problematic_team_id_to_check]
            print(f"[Debug predictor] For team ID {problematic_team_id_to_check} (Boston), found {len(boston_logs_in_predictor_df)} rows in loaded team_logs_df.")
            if not boston_logs_in_predictor_df.empty:
                print(f"[Debug predictor] Date range for team {problematic_team_id_to_check} in loaded team_logs_df: {boston_logs_in_predictor_df['game_date'].min()} to {boston_logs_in_predictor_df['game_date'].max()}")
        else:
            print(f"[Debug predictor] 'team_id' column not found in team_logs_df.")

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

    # Ensure the columns attribute is a fresh pd.Index object from feature_names
    feature_vector_df.columns = pd.Index(feature_names)

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