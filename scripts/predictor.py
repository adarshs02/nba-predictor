import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier # Example model
from sklearn.metrics import accuracy_score
import re # For parsing feature names

DB_NAME = "nba_data.db" # Define DB name, same as in data_collector.py

# Dictionary for interpreting features for advice generation
FEATURE_INTERPRETATIONS = {
    'pts': {
        'name': 'Points Scored',
        'strength_own': "Your team's ability to consistently score points has been a key factor. Keep up the offensive pressure.",
        'weakness_own': "Your team has struggled with scoring. Focus on improving offensive execution and shot selection.",
        'strength_opponent': "The opponent's high scoring average is a threat. Your defense needs to be tight and focused on limiting their opportunities.",
        'weakness_opponent': "The opponent has struggled to score. Capitalize on this by maintaining strong defensive pressure and forcing tough shots."
    },
    'fg_pct': {
        'name': 'Field Goal Percentage',
        'strength_own': "Excellent shooting efficiency (FG%) is a significant advantage. Continue to create and take high-percentage shots.",
        'weakness_own': "Improving shot selection and execution to boost your Field Goal Percentage should be a priority.",
        'strength_opponent': "The opponent shoots very efficiently. Contest every shot and work to disrupt their offensive rhythm.",
        'weakness_opponent': "The opponent has a low shooting percentage. Exploit this by forcing them into difficult shots and securing defensive rebounds."
    },
    'fg3_pct': {
        'name': 'Three-Point Percentage',
        'strength_own': "Your team's proficiency from beyond the arc (3PT%) is a major strength. Look for open three-point opportunities.",
        'weakness_own': "Struggles with three-point shooting are evident. Focus on better ball movement to create open looks or prioritize higher percentage shots.",
        'strength_opponent': "The opponent is dangerous from three-point range. Emphasize closing out on shooters and defending the perimeter.",
        'weakness_opponent': "The opponent is not a strong three-point shooting team. You might be able to play tighter inside, but don't leave shooters completely uncontested."
    },
    'ast': {
        'name': 'Assists',
        'strength_own': "Great teamwork and ball movement, reflected in high assists, are working well. Continue to share the ball effectively.",
        'weakness_own': "Low assist numbers suggest a need for better ball movement and creating scoring opportunities for teammates.",
        'strength_opponent': "The opponent moves the ball well, leading to many assists. Disrupt their passing lanes and play strong team defense.",
        'weakness_opponent': "The opponent doesn't generate many assists. This could indicate a more isolation-heavy offense, which your team defense can prepare for."
    },
    'reb': {
        'name': 'Rebounds',
        'strength_own': "Dominating the boards (Rebounds) gives your team crucial extra possessions. Continue to focus on boxing out and crashing the glass.",
        'weakness_own': "Improving rebounding on both ends of the floor is crucial to control possessions.",
        'strength_opponent': "The opponent is strong on the rebounds. Boxing out effectively and team rebounding will be key to limiting their second-chance points.",
        'weakness_opponent': "The opponent is weak on the boards. Aggressively pursue offensive rebounds and ensure you secure defensive rebounds."
    },
    'tov': { # Note: For TOV, 'strength' means low turnovers, 'weakness' means high turnovers.
        'name': 'Turnovers',
        'strength_own': "Low turnovers show your team is taking care of the ball. Maintain this discipline.", # 'Strength' in TOV means low value
        'weakness_own': "High turnover rates are hurting your possessions. Focus on ball security and making smart decisions.", # 'Weakness' in TOV means high value
        'strength_opponent': "The opponent protects the ball well (low turnovers). You'll need active hands and pressure to create takeaways.", # Opponent's 'strength' = low TOV
        'weakness_opponent': "The opponent is prone to turnovers. Apply defensive pressure to force mistakes and convert them into points." # Opponent's 'weakness' = high TOV
    },
    'win_pct': {
        'name': 'Recent Win Percentage',
        'strength_own': "Your team's strong recent winning record (Win %) is a positive momentum factor. Build on this confidence.",
        'weakness_own': "Recent losses (low Win %) might be affecting team morale. Focus on fundamentals and securing a win to turn things around.",
        'strength_opponent': "The opponent is on a winning streak. Be prepared for a confident and aggressive team.",
        'weakness_opponent': "The opponent has been struggling to win recently. This could be an opportunity to assert dominance early."
    }
    # Can add more stats like 'STL', 'BLK', 'PF', 'PTS_Allowed' if they are engineered as features
}

def parse_feature_name(feature_name):
    """Parses feature name like 'home_pts_last_10' or 'away_fg_pct'."""
    match = re.match(r"(home|away)_(avg_)?(pts|fg_pct|fg3_pct|ast|reb|tov|win_pct)(_last_\d+)?", feature_name)
    if match:
        team_scope = match.group(1) # home or away
        stat_category = match.group(3) # pts, fg_pct, etc.
        # window_info = match.group(4) # _last_10 or None
        return team_scope, stat_category
    return None, None

def load_data():
    """Loads all necessary data from the SQLite database."""
    print(f"Loading data from {DB_NAME}...")
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME)
        
        # Load games table - this will likely be our main prediction target table
        # Each row is a game that occurred.
        games_df = pd.read_sql_query("SELECT * FROM games", conn)
        print(f"Loaded {len(games_df)} entries from 'games' table.")
        if games_df.empty:
            print("Warning: 'games' table is empty. Ensure data_collector.py has run successfully.")

        # Load player game logs
        player_logs_df = pd.read_sql_query("SELECT * FROM player_game_logs", conn)
        print(f"Loaded {len(player_logs_df)} entries from 'player_game_logs' table.")
        if player_logs_df.empty:
            print("Warning: 'player_game_logs' table is empty.")

        # Load team game logs
        team_logs_df = pd.read_sql_query("SELECT * FROM team_game_logs", conn)
        print(f"Loaded {len(team_logs_df)} entries from 'team_game_logs' table.")
        if team_logs_df.empty:
            print("Warning: 'team_game_logs' table is empty.")

        # Potentially load 'teams' and 'players' static info if needed directly for feature engineering later
        teams_info_df = pd.read_sql_query("SELECT * FROM teams", conn)
        print(f"Loaded {len(teams_info_df)} entries from 'teams' table.")
        if teams_info_df.empty:
            print("Warning: 'teams' table is empty (needed for team names).")

        # players_info_df = pd.read_sql_query("SELECT * FROM players", conn) # Not used yet

        return {
            "games": games_df,
            "player_logs": player_logs_df,
            "team_logs": team_logs_df,
            "teams_info": teams_info_df
        }

    except sqlite3.OperationalError as e:
        print(f"Database error: {e}. Make sure '{DB_NAME}' exists and data_collector.py has run successfully.")
        return None # Return None if DB or tables are not found
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        return None
    finally:
        if conn:
            conn.close()

def calculate_team_rolling_stats(team_id, game_date, team_logs_df, window_size=10):
    """Calculates rolling stats for a team before a given game_date."""
    # Ensure game_date in team_logs_df is datetime for comparison
    team_logs_df['game_date'] = pd.to_datetime(team_logs_df['game_date'])
    game_date = pd.to_datetime(game_date)

    # Filter logs for the specific team and before the current game_date
    past_games = team_logs_df[
        (team_logs_df['team_id'] == team_id) &
        (team_logs_df['game_date'] < game_date)
    ].sort_values(by='game_date', ascending=False).head(window_size)

    if past_games.empty:
        return pd.Series() # Return empty series if no past games in window

    # Calculate rolling averages for specified stats
    stats_to_average = ['pts', 'fg_pct', 'fg3_pct', 'ast', 'reb', 'tov']
    # Ensure opponent points are calculated if not directly available
    # For now, let's assume `team_logs_df` contains opponent scores or we derive `pts_allowed`
    # If not, we'd need to join with games table or infer from +/- if available and makes sense
    
    # For simplicity, let's add a placeholder for points allowed if not in team_logs_df
    # In a real scenario, you would need to get opponent scores for each game in past_games.
    # For team_logs, WL (Win/Loss) is present. We can calculate win percentage.
    
    # Ensure stats_to_average columns are numeric, coercing errors
    for col in stats_to_average:
        if col in past_games.columns:
            past_games[col] = pd.to_numeric(past_games[col], errors='coerce')
        else: # If a stat is missing, create a dummy column of NaN to avoid KeyError
            past_games[col] = pd.NA 
    
    rolling_stats = past_games[stats_to_average].mean(numeric_only=True)
    rolling_stats['win_pct'] = (past_games['wl'] == 'W').mean()
    
    return rolling_stats

def feature_engineering(raw_data_dict):
    """Creates features for the model.
    
    Key features to develop:
    - Player form (e.g., rolling averages of key stats over last N games from player_logs_df)
    - Team form (e.g., rolling win percentage, avg points scored/allowed over last N games from team_logs_df)
    - Head-to-head stats (historical performance of Team A vs. Team B, derived from games_df and team_logs_df)
    - Home/Away advantage indicators (from games_df)
    - Rest days for each team (requires processing game_date)
    """
    if raw_data_dict is None:
        print("Raw data is missing, cannot perform feature engineering.")
        return pd.DataFrame(), pd.Series(), pd.DataFrame() # Added third return for full_game_details_df

    games_df = raw_data_dict.get("games")
    team_logs_df = raw_data_dict.get("team_logs")
    teams_info_df = raw_data_dict.get("teams_info") # Get teams_info

    if games_df is None or games_df.empty or \
       team_logs_df is None or team_logs_df.empty or \
       teams_info_df is None or teams_info_df.empty:
        print("Games, Team Logs, or Teams Info DataFrame is missing or empty. Cannot proceed with full feature engineering.")
        # Fallback logic (simplified for brevity, but should return three DataFrames if that's the new signature)
        if games_df is not None and not games_df.empty:
            games_df['home_team_score'] = pd.to_numeric(games_df['home_team_score'], errors='coerce').fillna(0)
            games_df['away_team_score'] = pd.to_numeric(games_df['away_team_score'], errors='coerce').fillna(0)
            target = (games_df['home_team_score'] > games_df['away_team_score']).astype(int)
            dummy_features = pd.DataFrame(index=games_df.index)
            dummy_features['game_id'] = games_df['game_id'] 
            for i in range(5): dummy_features[f'dummy_feature_{i}'] = 0.5
            # Create a dummy full_game_details_df for fallback
            dummy_details = games_df[['game_id']].copy()
            dummy_details['home_team_name'] = 'N/A'
            dummy_details['away_team_name'] = 'N/A'
            return dummy_features.drop(columns=['game_id']), target, dummy_details
        return pd.DataFrame(), pd.Series(), pd.DataFrame()

    print("Starting feature engineering with team names...")

    games_df['game_date'] = pd.to_datetime(games_df['game_date'])
    games_df = games_df.sort_values(by=['game_date', 'game_id'])

    games_df['home_team_score'] = pd.to_numeric(games_df['home_team_score'], errors='coerce').fillna(0)
    games_df['away_team_score'] = pd.to_numeric(games_df['away_team_score'], errors='coerce').fillna(0)
    games_df['HOME_TEAM_WINS'] = (games_df['home_team_score'] > games_df['away_team_score']).astype(int)

    # For easier lookup, create a mapping from team_id to team_name
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
            'game_id': game_id,
            'home_team_id': home_team_id,
            'away_team_id': away_team_id,
            'home_team_name': team_id_to_name.get(home_team_id, 'Unknown'), # Add home team name
            'away_team_name': team_id_to_name.get(away_team_id, 'Unknown')  # Add away team name
        }
        for stat_name, value in home_team_stats.items():
            game_features[f'home_{stat_name}'] = value
        for stat_name, value in away_team_stats.items():
            game_features[f'away_{stat_name}'] = value
        
        all_features_list.append(game_features)

    features_df_with_details = pd.DataFrame(all_features_list)
    print("\nSample of features_df_with_details BEFORE fillna(0):")
    print(features_df_with_details.head())
    
    features_df_with_details = features_df_with_details.fillna(0)

    print(f"Feature engineering complete. Generated {len(features_df_with_details)} feature sets.")
    if not features_df_with_details.empty:
        print("Sample of engineered features (with details):")
        print(features_df_with_details.head())
    
    final_df = pd.merge(features_df_with_details, games_df[['game_id', 'HOME_TEAM_WINS']], on='game_id', how='inner')
    
    if final_df.empty:
        print("No data after merging features and target. Check game_id alignment.")
        return pd.DataFrame(), pd.Series(), pd.DataFrame()

    target_final = final_df['HOME_TEAM_WINS']
    # Features for the model should not include IDs or names directly
    features_for_model = final_df.drop(columns=[
        'HOME_TEAM_WINS', 'game_id', 
        'home_team_id', 'away_team_id',
        'home_team_name', 'away_team_name' 
    ])

    # full_game_details_df will be used for lookup in __main__ for prediction output
    full_game_details_df = final_df[['game_id', 'home_team_name', 'away_team_name', 'HOME_TEAM_WINS']]

    return features_for_model, target_final, full_game_details_df

def generate_team_advice(top_features_for_game, home_team_name, away_team_name, predicted_winner_name, actual_feature_values):
    """Generates narrative advice for home and away teams based on top influential features."""
    home_advice = []
    away_advice = []

    # Define thresholds for what constitutes strength/weakness. These are illustrative and may need tuning.
    # For percentages (fg_pct, fg3_pct, win_pct), 0.5 is a neutral point.
    # For counts (pts, ast, reb), we might compare against league averages in a more advanced setup.
    # For turnovers (tov), lower is better.
    # These are simplified for now.
    thresholds = {
        'pts': {'high_is_strength': True, 'strength_threshold': 110, 'weakness_threshold': 100}, # Arbitrary points thresholds
        'fg_pct': {'high_is_strength': True, 'strength_threshold': 0.48, 'weakness_threshold': 0.42},
        'fg3_pct': {'high_is_strength': True, 'strength_threshold': 0.38, 'weakness_threshold': 0.32},
        'ast': {'high_is_strength': True, 'strength_threshold': 25, 'weakness_threshold': 20},
        'reb': {'high_is_strength': True, 'strength_threshold': 45, 'weakness_threshold': 40},
        'tov': {'high_is_strength': False, 'strength_threshold': 12, 'weakness_threshold': 16}, # Lower TOV is strength
        'win_pct': {'high_is_strength': True, 'strength_threshold': 0.6, 'weakness_threshold': 0.4}
    }

    for feature_name, _ in top_features_for_game.items():
        team_scope, stat_category = parse_feature_name(feature_name)
        feature_value = actual_feature_values.get(feature_name)

        if team_scope and stat_category and stat_category in FEATURE_INTERPRETATIONS and feature_value is not None:
            interp = FEATURE_INTERPRETATIONS[stat_category]
            thresh = thresholds.get(stat_category)
            if not thresh: continue # Skip if no threshold defined

            is_strength = False
            is_weakness = False

            if thresh['high_is_strength']:
                if feature_value >= thresh['strength_threshold']:
                    is_strength = True
                elif feature_value <= thresh['weakness_threshold']:
                    is_weakness = True
            else: # Low is strength (e.g., turnovers)
                if feature_value <= thresh['strength_threshold']:
                    is_strength = True
                elif feature_value >= thresh['weakness_threshold']:
                    is_weakness = True

            # Determine advice based on team_scope (home/away) and strength/weakness
            if team_scope == 'home':
                if is_strength and interp.get('strength_own'):
                    home_advice.append(f"({interp['name']} - Strength): {interp['strength_own']}")
                elif is_weakness and interp.get('weakness_own'):
                    home_advice.append(f"({interp['name']} - Weakness): {interp['weakness_own']}")
                
                # Advice for away team based on home team's stat
                if is_strength and interp.get('strength_opponent'):
                    away_advice.append(f"(Opponent's {interp['name']} - Strength): {interp['strength_opponent']}")
                elif is_weakness and interp.get('weakness_opponent'):
                    away_advice.append(f"(Opponent's {interp['name']} - Weakness): {interp['weakness_opponent']}")

            elif team_scope == 'away':
                if is_strength and interp.get('strength_own'):
                    away_advice.append(f"({interp['name']} - Strength): {interp['strength_own']}")
                elif is_weakness and interp.get('weakness_own'):
                    away_advice.append(f"({interp['name']} - Weakness): {interp['weakness_own']}")
                
                # Advice for home team based on away team's stat
                if is_strength and interp.get('strength_opponent'):
                    home_advice.append(f"(Opponent's {interp['name']} - Strength): {interp['strength_opponent']}")
                elif is_weakness and interp.get('weakness_opponent'):
                    home_advice.append(f"(Opponent's {interp['name']} - Weakness): {interp['weakness_opponent']}")
    
    # Deduplicate advice while preserving order (simple list to set to list conversion)
    unique_home_advice = list(dict.fromkeys(home_advice))
    unique_away_advice = list(dict.fromkeys(away_advice))

    # Limit to a few pieces of advice to avoid overwhelming the user
    max_advice_pieces = 3
    final_home_advice = "\n".join(unique_home_advice[:max_advice_pieces])
    final_away_advice = "\n".join(unique_away_advice[:max_advice_pieces])

    advice_output = f"\n--- Strategic Focus for {home_team_name} ---"
    if final_home_advice:
        advice_output += f"\n{final_home_advice}"
    else:
        advice_output += "\nNo specific strong/weak signals from top features for focused advice."

    advice_output += f"\n\n--- Strategic Focus for {away_team_name} ---"
    if final_away_advice:
        advice_output += f"\n{final_away_advice}"
    else:
        advice_output += "\nNo specific strong/weak signals from top features for focused advice."
        
    return advice_output

def train_model(X_train, y_train):
    """Trains the prediction model and returns it along with feature importances."""
    print("Model training...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    importances = model.feature_importances_
    feature_names = X_train.columns
    feature_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    
    return model, feature_importances # Return model and importances

def evaluate_model(model, X_test, y_test):
    """Evaluates the model's performance."""
    print("Model evaluation placeholder")
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    return accuracy

def predict_game(model, game_features_df, home_team_name="Home Team", away_team_name="Away Team", top_n_features=5, feature_importances=None):
    """Predicts the outcome of a single game, includes team names, shows top influential features, and generates strategic advice."""
    print(f"\nPredicting for: {home_team_name} (Home) vs. {away_team_name} (Away)")
    
    if isinstance(game_features_df, pd.Series):
        game_features_df = game_features_df.to_frame().T

    prediction = model.predict(game_features_df)
    probability = model.predict_proba(game_features_df)
    
    predicted_winner_numeric = prediction[0]
    winner_name = home_team_name if predicted_winner_numeric == 1 else away_team_name
    win_probability = probability[0][1] if predicted_winner_numeric == 1 else probability[0][0]
    print(f"Prediction: {winner_name} wins (Probability: {win_probability:.2f})")

    if feature_importances is not None and not game_features_df.empty:
        print("\nKey factors for this prediction (based on global feature importance):")
        current_game_feature_values = game_features_df.iloc[0]
        
        # Get the top N features by importance score that are actually present in this game's features
        # This ensures we only consider features that have values for the current game
        available_top_features = {}
        count = 0
        for feature_name, importance_score in feature_importances.items():
            if feature_name in current_game_feature_values.index:
                available_top_features[feature_name] = importance_score
                value = current_game_feature_values[feature_name]
                print(f"- {feature_name}: {value:.2f} (Global Importance: {importance_score:.4f})")
                count += 1
                if count >= top_n_features:
                    break
            # else: # This case should ideally not happen if features align, but good for debug
            #     print(f"- {feature_name} (Global Importance: {importance_score:.4f}) - Value not found for this game.")

        if available_top_features:
            # Pass the actual values of these top features to the advice generator
            advice_text = generate_team_advice(available_top_features, home_team_name, away_team_name, winner_name, current_game_feature_values)
            print(advice_text)
        else:
            print("No top features with available values found for this game to generate advice.")
            
    return predicted_winner_numeric, win_probability # Return numeric prediction and probability

if __name__ == "__main__":
    raw_data_dict = load_data()
    
    if raw_data_dict:
        features, target, full_game_details_df = feature_engineering(raw_data_dict)
        
        if features is not None and not features.empty and \
           target is not None and not target.empty and \
           full_game_details_df is not None and not full_game_details_df.empty:
            
            can_stratify = False
            if target.nunique() > 1:
                if all(target.value_counts() >= 2):
                    can_stratify = True
                else:
                    print("Warning: Cannot stratify due to insufficient samples in some classes.")
            
            X_train, X_test, y_train, y_test = train_test_split(
                features, 
                target, 
                test_size=0.2, 
                random_state=42, 
                stratify=target if can_stratify else None
            )

            print(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")
            if X_train.empty or (not X_test.empty and y_train.empty) or (X_test.empty and not y_train.empty and len(X_test)>0) :
                 print("Training or testing data is empty or mismatched after split. Check data and feature engineering.")
            elif X_train.empty and X_test.empty:
                 print("Both Training and testing data are empty after split. Check data and feature engineering.")
            else:
                model, feature_importances = train_model(X_train, y_train) # Get feature_importances
                print("\nTop Global Feature Importances:")
                print(feature_importances.head(10)) # Print top 10 global importances
                
                evaluate_model(model, X_test, y_test)
                
                if not X_test.empty:
                    sample_game_features_for_model = X_test.head(1)
                    sample_game_original_index = X_test.head(1).index[0]
                    
                    game_details_for_sample = full_game_details_df.loc[sample_game_original_index]
                    print(f"\nSample prediction for game details:")
                    print(game_details_for_sample)
                    print(f"\nFeature values for this sample game (from X_test.head(1)):")
                    print(sample_game_features_for_model)

                    home_name = game_details_for_sample['home_team_name']
                    away_name = game_details_for_sample['away_team_name']
                                        
                    if not sample_game_features_for_model.empty:
                        predict_game(model, sample_game_features_for_model, home_name, away_name, feature_importances=feature_importances)
                    else:
                        print("Sample game features from X_test are empty.")
                else:
                    print("X_test is empty, cannot run sample prediction.")
        else:
            print("Feature engineering did not produce valid features, target, or details. Exiting.")
    else:
        print("Data loading failed. Exiting predictor.") 