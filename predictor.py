import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import json
import os
from dotenv import load_dotenv
import joblib

"""
Predicts the outcome of an NBA game using a Random Forest Classifier. Current accuracy is right under 60%. There are no player based predictions yet. It only takes into account team stats.
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
home_team_name = "Charlotte Hornets"
away_team_name = "Washington Wizards"

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
    # ... (calculate_team_rolling_stats function remains the same) ...
    team_logs_df['game_date'] = pd.to_datetime(team_logs_df['game_date'])
    game_date = pd.to_datetime(game_date)

    past_games = team_logs_df[
        (team_logs_df['team_id'] == team_id) &
        (team_logs_df['game_date'] < game_date)
    ].sort_values(by='game_date', ascending=False).head(window_size)

    if past_games.empty:
        return pd.Series(dtype='float64') 

    stats_to_average = ['pts', 'fg_pct', 'fg3_pct', 'ast', 'reb', 'tov']
    
    for col in stats_to_average:
        if col in past_games.columns:
            past_games[col] = pd.to_numeric(past_games[col], errors='coerce')
        else: 
            past_games[col] = pd.NA 
    
    rolling_stats = past_games[stats_to_average].mean(numeric_only=True)
    
    # Calculate win_pct if 'wl' column exists
    if 'wl' in past_games.columns:
        rolling_stats['win_pct'] = (past_games['wl'] == 'W').mean()
    else:
        rolling_stats['win_pct'] = pd.NA # Or 0 or some other default

    # Prefix stats with 'avg_' to match expected feature names if needed by LLM prompt structure
    rolling_stats = rolling_stats.add_prefix('avg_')
    return rolling_stats

#TODO: Improve alogirthm to have higher accuracy
def feature_engineering(raw_data_dict):
    # ... (feature_engineering function updated to set game_id as index for X) ...
    if raw_data_dict is None:
        print("Raw data is missing, cannot perform feature engineering.")
        return pd.DataFrame(), pd.Series(), pd.DataFrame()

    games_df = raw_data_dict.get("games")
    team_logs_df = raw_data_dict.get("team_logs")
    teams_info_df = raw_data_dict.get("teams_info")

    if games_df is None or games_df.empty or \
       team_logs_df is None or team_logs_df.empty or \
       teams_info_df is None or teams_info_df.empty:
        print("Games, Team Logs, or Teams Info DataFrame is missing or empty. Cannot proceed.")
        # Fallback to return empty structures matching the expected output
        return pd.DataFrame(), pd.Series(), pd.DataFrame()


    print("Starting feature engineering...")

    games_df['game_date'] = pd.to_datetime(games_df['game_date'])
    games_df = games_df.sort_values(by=['game_date', 'game_id'])

    games_df['home_team_score'] = pd.to_numeric(games_df['home_team_score'], errors='coerce').fillna(0)
    games_df['away_team_score'] = pd.to_numeric(games_df['away_team_score'], errors='coerce').fillna(0)
    games_df['HOME_TEAM_WINS'] = (games_df['home_team_score'] > games_df['away_team_score']).astype(int)

    team_id_to_name = teams_info_df.set_index('id')['full_name'].to_dict()

    all_features_list = []
    window_size = 10
    team_logs_df['game_date'] = pd.to_datetime(team_logs_df['game_date'], errors='coerce')

    for index, game_row in games_df.iterrows():
        game_id = game_row['game_id']
        home_team_id = game_row['home_team_id']
        away_team_id = game_row['away_team_id']
        current_game_date = game_row['game_date']

        home_team_stats = calculate_team_rolling_stats(home_team_id, current_game_date, team_logs_df, window_size)
        away_team_stats = calculate_team_rolling_stats(away_team_id, current_game_date, team_logs_df, window_size)

        game_features = {
            'game_id': game_id, # Keep game_id for now, will be index later
            'home_team_id': home_team_id,
            'away_team_id': away_team_id,
            'home_team_name': team_id_to_name.get(home_team_id, 'Unknown'),
            'away_team_name': team_id_to_name.get(away_team_id, 'Unknown')
        }
        for stat_name, value in home_team_stats.items(): # stat_name will be avg_pts etc.
            game_features[f'home_{stat_name}'] = value 
        for stat_name, value in away_team_stats.items(): # stat_name will be avg_pts etc.
            game_features[f'away_{stat_name}'] = value
        
        all_features_list.append(game_features)

    features_df_with_details = pd.DataFrame(all_features_list)
    features_df_with_details = features_df_with_details.fillna(0) # Fill NaNs after stat calculation

    print(f"Feature engineering complete. Generated {len(features_df_with_details)} feature sets.")
    
    final_df = pd.merge(features_df_with_details, games_df[['game_id', 'HOME_TEAM_WINS']], on='game_id', how='inner')
    
    if final_df.empty:
        print("No data after merging features and target. Check game_id alignment.")
        return pd.DataFrame(), pd.Series(), pd.DataFrame()

    target_final = final_df['HOME_TEAM_WINS']
    
    # Set game_id as index for features_for_model (X)
    # This X will be used for training and for looking up specific games for prediction
    features_for_model = final_df.set_index('game_id').drop(columns=[
        'HOME_TEAM_WINS', 
        'home_team_id', 'away_team_id',
        'home_team_name', 'away_team_name' 
    ])

    full_game_details_df = final_df[['game_id', 'home_team_id', 'home_team_name', 'away_team_id', 'away_team_name', 'HOME_TEAM_WINS']].set_index('game_id')

    return features_for_model, target_final, full_game_details_df


def train_model(X_train, y_train):
    """Trains the prediction model, saves it and feature importances, and returns them."""
    print("Model training...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    importances = model.feature_importances_
    feature_names = X_train.columns
    feature_importances_series = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    
    try:
        joblib.dump(model, MODEL_FILENAME)
        print(f"Model saved to {MODEL_FILENAME}")
        joblib.dump(feature_importances_series, FEATURES_FILENAME)
        print(f"Feature importances saved to {FEATURES_FILENAME}")
    except Exception as e:
        print(f"Error saving model or feature importances: {e}")
        
    return model, feature_importances_series

def evaluate_model(model, X_test, y_test):
    # ... (evaluate_model function remains the same) ...
    print("Model evaluation placeholder") # Placeholder, should be actual evaluation
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    return accuracy

def predict_game(model, game_features_df, home_team_name="Home Team", away_team_name="Away Team", top_n_features=5, feature_importances=None):
    """Predicts outcome, shows key factors, saves data for LLM, and calls summary_api to generate a game summary."""
    print(f"\nPredicting for: {home_team_name} (Home) vs. {away_team_name} (Away)")
    
    if isinstance(game_features_df, pd.Series):
        # If a Series is passed (e.g. from X.loc[game_id]), convert to DataFrame
        game_features_for_prediction = game_features_df.to_frame().T
    else:
        game_features_for_prediction = game_features_df

    # Ensure column order matches training data if game_features_for_prediction was reconstructed
    # This is usually handled if X_train.columns is used as a reference, but model.predict is robust if names match.
    
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
    for col_name, value in current_game_values.items():
        if col_name.startswith("home_avg_"):
            stat_key = col_name.replace("home_avg_", "")
            llm_prompt_data["home_stats"][stat_key] = f"{value:.2f}"
        elif col_name.startswith("away_avg_"):
            stat_key = col_name.replace("away_avg_", "")
            llm_prompt_data["away_stats"][stat_key] = f"{value:.2f}"

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
    force_retrain = False # Set to True if you want to force retraining

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
                             top_n_features=5, 
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